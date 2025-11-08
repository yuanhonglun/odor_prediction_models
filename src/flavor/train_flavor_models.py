# -*- coding: utf-8 -*-
"""
Multiclass flavor-category prediction (scaffold-aware split, HPO, optional GCN)
- Canonical SMILES de-duplication (majority label; drop 'Odorless' unless all Odorless)
- Murcko-scaffold split for train/validation
- Fingerprints: ECFP4/ECFP6 (nbits), MACCS (167) + basic physchem features
- Models: RandomForest / GBDT (XGBoost if available else HistGBDT) / MLP / GCN (2-layer, PyG)
- Class imbalance: class_weight via sample_weight or RandomOverSampler (optional)
- HPO: RandomizedSearchCV on TRAIN (macro-F1); optionally scaffold-group CV
- GCN: lightweight random search on a hold-out validation (no sklearn CV)
- Evaluation: original hold-out validation artifacts + per-fold CV report on TRAIN:
    * Fold-wise metrics (Macro-F1, Weighted-F1, Accuracy, Macro-Precision, Macro-Recall)
    * Mean ± standard deviation across folds
- Resume: --resume skips completed FP×Model combinations (including GCN)

Author: Honglun Yuan
"""
from __future__ import annotations
import argparse, json, os, random, inspect, time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Limit BLAS threads to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import joblib

# RDKit
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MACCSkeys
RDLogger.DisableLog('rdApp.*')

# sklearn
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone

# xgboost (optional backend for GBDT)
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# imblearn (optional for ROS)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler
    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False

# plotting (PNG only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======== Torch / PyG for GCN (optional) ========
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.loader import DataLoader as GeoLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

# ======== Reproducibility ========
def set_global_seed(seed: int = 42):
    """Set random seeds for Python, NumPy, and (if available) Torch."""
    random.seed(seed); np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# ======== Lightweight runtime logging ========
try:
    import psutil
    _HAS_PSUTIL = True
    _PROC = psutil.Process()
except Exception:
    _HAS_PSUTIL = False
    _PROC = None

def _mem():
    if not _HAS_PSUTIL: return "n/a"
    try:
        rss = _PROC.memory_info().rss / 1024 / 1024
        return f"{rss:.1f} MB"
    except Exception:
        return "n/a"

def log(msg: str):
    """Print a timestamped, flush-true log line with current RSS memory."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | mem={_mem()}", flush=True)

@contextmanager
def log_stage(name: str):
    """Context manager to time a stage and log start/end with elapsed time."""
    t0 = time.perf_counter()
    log(f"▶ {name} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {name} done in {dt/60:.2f} min")

# ======== utils ========
def write_done_marker(outdir: Path, fp_kind: str, model_name: str):
    (outdir / f"done__{fp_kind}__{model_name}.marker").write_text("OK", encoding="utf-8")

def is_done(outdir: Path, fp_kind: str, model_name: str) -> bool:
    return (outdir / f"done__{fp_kind}__{model_name}.marker").exists()

# ======== SMILES & Scaffold Utils ========
def canonical_smiles(smiles: str) -> Optional[str]:
    """Return canonical isomeric SMILES or None if parsing fails."""
    if not isinstance(smiles, str) or not smiles.strip(): return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

def get_scaffold_smiles(smiles: str) -> str:
    """Return isomeric SMILES of the Murcko scaffold (fallback to original if empty)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None: return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.MolToSmiles(scaf, isomericSmiles=True)

# ======== Deduplication ========
def deduplicate_by_smiles(df: pd.DataFrame, smiles_col: str, label_col: str,
                          odorless_label: str, seed: int = 42) -> pd.DataFrame:
    """
    Canonical-SMILES grouping rules:
      - Drop missing/empty labels.
      - If any non-Odorless exists in a group → drop all 'Odorless' rows in that group.
      - Keep the majority label; ties (only when exactly two rows) are broken randomly (seeded).
      - If a group contains only 'Odorless' → keep exactly one row.
    """
    set_global_seed(seed)
    df = df.copy()
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""].copy()

    with log_stage("Canonicalize SMILES"):
        df["_canonical_smiles"] = df[smiles_col].apply(canonical_smiles)
        df = df[~df["_canonical_smiles"].isna()].copy()
        log(f"[dedup] input rows after canon: {len(df)}")

    kept_rows = []
    n_groups = int(df["_canonical_smiles"].nunique())
    log(f"[dedup] groups to process: {n_groups}")

    with log_stage("Deduplicate by canonical SMILES"):
        for i, (can_smi, grp) in enumerate(df.groupby("_canonical_smiles", sort=False), start=1):
            if i % 10000 == 0:
                log(f"[dedup] progress {i}/{n_groups}")
            non_odor = grp[grp[label_col] != odorless_label]
            odor_only = grp[grp[label_col] == odorless_label]

            if len(non_odor) > 0:
                cand = non_odor
                vc = cand[label_col].value_counts()
                if vc.empty:
                    keep_idx = random.choice(cand.index.tolist())
                    kept_rows.append(df.loc[keep_idx]); continue
                top = vc.max(); top_labels = vc[vc == top].index.tolist()
                if len(cand) == 2 and len(top_labels) > 1:
                    keep_idx = random.choice(cand.index.tolist())
                else:
                    chosen_label = random.choice(top_labels)
                    keep_idx = cand[cand[label_col] == chosen_label].index[0]
                kept_rows.append(df.loc[keep_idx])
            else:
                if len(odor_only) > 0:
                    keep_idx = odor_only.index[0] if len(odor_only) == 1 else random.choice(odor_only.index.tolist())
                    kept_rows.append(df.loc[keep_idx])

    out = pd.DataFrame(kept_rows).drop_duplicates(subset=["_canonical_smiles"]).rename(columns={"_canonical_smiles":"SMILES_canonical"})
    log(f"[dedup] kept rows: {len(out)} (unique SMILES)")
    return out.reset_index(drop=True)

# ======== Features ========
def morgan_bits(mol: Chem.Mol, radius: int, nbits: int = 1024) -> np.ndarray:
    """Return ECFP bit vector (as numpy array of int8)."""
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def maccs_bits(mol: Chem.Mol) -> np.ndarray:
    """Return MACCS 167-bit vector (as numpy array of int8)."""
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def physchem_features(mol: Chem.Mol) -> Dict[str, float]:
    """Basic physicochemical descriptors used as global features."""
    return {
        "MolWeight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def build_feature_matrices(df, smiles_col, extra_physchem_cols, fp_kinds=("ECFP4","ECFP6","MACCS"), nbits=1024):
    """Compute FP+physchem features for requested kinds; return dict[kind] -> (DataFrame, columns)."""
    kind_to_radius = {"ECFP4":2, "ECFP6":3}
    results = {}
    for kind in fp_kinds:
        bit_cols = None
        feats, phys = [], []
        n = len(df)

        if kind == "MACCS":
            nbits_local = 167
            bit_cols = [f"MACCS_{i}" for i in range(nbits_local)]
            with log_stage(f"Build MACCS (167 bits) for {n} molecules"):
                for idx, smi in enumerate(df[smiles_col].tolist(), start=1):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        feats.append(np.zeros((nbits_local,), dtype=np.int8))
                        phys.append({"MolWeight":0.0,"LogP":0.0,"TPSA":0.0,"MolarRefractivity":0.0})
                    else:
                        feats.append(maccs_bits(mol))
                        phys.append(physchem_features(mol))
                    if idx % 5000 == 0:
                        log(f"[MACCS] {idx}/{n}")
        else:
            if kind not in kind_to_radius:
                continue
            radius = kind_to_radius[kind]
            nbits_local = nbits
            bit_cols = [f"{kind}_{i}" for i in range(nbits_local)]
            with log_stage(f"Build {kind} ({nbits_local} bits) for {n} molecules"):
                for idx, smi in enumerate(df[smiles_col].tolist(), start=1):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        feats.append(np.zeros((nbits_local,), dtype=np.int8))
                        phys.append({"MolWeight":0.0,"LogP":0.0,"TPSA":0.0,"MolarRefractivity":0.0})
                    else:
                        feats.append(morgan_bits(mol, radius, nbits_local))
                        phys.append(physchem_features(mol))
                    if idx % 5000 == 0:
                        log(f"[{kind}] {idx}/{n}")

        with log_stage(f"Assemble {kind} feature table"):
            bits_df = pd.DataFrame(np.vstack(feats), columns=bit_cols, index=df.index)
            phys_df = pd.DataFrame(phys, index=df.index)
            extras = [c for c in extra_physchem_cols if c in df.columns]
            extra_df = df[extras] if extras else None
            X = pd.concat([bits_df, phys_df, (extra_df if extra_df is not None else pd.DataFrame(index=df.index))], axis=1)
            X = X.astype(np.float32)  # memory/speed
            results[kind] = (X, X.columns.tolist())
            log(f"[{kind}] X shape: {X.shape}, dtype={X.dtypes.iloc[0] if len(X.columns)>0 else 'n/a'}")
    return results

# ======== Scaffold split & optional undersampling ========
def scaffold_split(df, smiles_col, label_col, val_size=0.2, seed=42):
    """Greedy Murcko-scaffold split to approximate requested validation ratio."""
    set_global_seed(seed)
    df = df.copy()
    with log_stage("Compute Murcko scaffolds"):
        df["_scaffold"] = df[smiles_col].apply(get_scaffold_smiles)
    sizes = df["_scaffold"].value_counts().sort_values(ascending=False)
    total = len(df); target_val = int(round(total * val_size))
    val_idx, current = set(), 0
    for scaf, size in sizes.items():
        if current + size <= target_val:
            idxs = df.index[df["_scaffold"] == scaf].tolist()
            val_idx.update(idxs); current += size
    val_df = df.loc[sorted(list(val_idx))].copy()
    train_df = df.drop(index=val_df.index).copy()
    train_df.drop(columns=["_scaffold"], inplace=True)
    val_df.drop(columns=["_scaffold"], inplace=True)
    log(f"[split] train={len(train_df)}, val={len(val_df)} (ratio ~ {len(val_df)/total:.2f})")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def undersample_majority_scaffold_prop(train_df: pd.DataFrame, label_col: str, majority_label: str,
                                       smiles_col: str = "SMILES", target_n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Downsample the majority class in TRAIN proportionally to scaffold sizes to preserve scaffold diversity.
    """
    rng = np.random.default_rng(seed)
    df = train_df.copy()
    maj_mask = (df[label_col].astype(str) == majority_label)
    maj_df = df[maj_mask].copy(); oth_df = df[~maj_mask].copy()
    if len(maj_df) <= target_n:
        return train_df

    def _scaf(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return Chem.MolToSmiles(scaf, isomericSmiles=True)

    with log_stage("[undersample] compute scaffolds for majority"):
        maj_df["_scaffold"] = maj_df[smiles_col].apply(_scaf)

    with log_stage("[undersample] scaffold-proportional allocation"):
        sizes = maj_df["_scaffold"].value_counts()
        total = int(sizes.sum())
        alloc = (sizes * (target_n / total)).round().astype(int)
        alloc[alloc == 0] = 1
        diff = int(alloc.sum() - target_n)
        if diff > 0:
            order = alloc.sort_values(ascending=False).index.tolist(); i = 0
            while diff > 0 and i < len(order):
                k = order[i]
                if alloc[k] > 1:
                    alloc[k] -= 1; diff -= 1
                else:
                    i += 1
        elif diff < 0:
            order = alloc.sort_values(ascending=True).index.tolist(); i = 0
            while diff < 0 and i < len(order):
                k = order[i]
                alloc[k] += 1; diff += 1

    with log_stage(f"[undersample] sample majority to target_n≈{target_n}"):
        picked_idx = []
        for scaf, k in alloc.items():
            grp_idx = maj_df.index[maj_df["_scaffold"] == scaf].tolist()
            if not grp_idx: continue
            if len(grp_idx) <= k:
                picked_idx.extend(grp_idx)
            else:
                picked_idx.extend(rng.choice(grp_idx, size=k, replace=False).tolist())
        picked_idx = picked_idx[:target_n]

    maj_down = maj_df.loc[sorted(set(picked_idx))].drop(columns=["_scaffold"])
    out = pd.concat([oth_df, maj_down], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log(f"[undersample] majority {majority_label}: {len(maj_df)} → {len(maj_down)}; train total {len(train_df)} → {len(out)}")
    return out

# ======== Imbalance helpers ========
def compute_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights normalized to average 1.0."""
    classes, counts = np.unique(y, return_counts=True)
    inv = {c: 1.0/cnt for c, cnt in zip(classes, counts)}
    w = np.array([inv[c] for c in y], dtype=np.float32)
    return w * (len(y)/np.sum(w))

# ======== Models (sklearn) ========
def make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600, max_depth=30, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", n_jobs=6, random_state=seed, bootstrap=True, oob_score=False
    )

def make_gbdt(n_classes: int, seed: int = 42):
    if _HAS_XGB:
        if n_classes is None or n_classes <= 2:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=600, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, eval_metric="logloss"
            )
        else:
            return xgb.XGBClassifier(
                objective="multi:softprob", num_class=int(n_classes),
                n_estimators=600, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, eval_metric="mlogloss"
            )
    else:
        return HistGradientBoostingClassifier(
            loss="log_loss", max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=seed
        )

def make_mlp(seed: int = 42):
    return SkPipeline([
        ("scaler", MaxAbsScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512,128), activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=300, random_state=seed, verbose=False
        ))
    ])

# ======== Eval & Plot ========
def evaluate_multiclass(y_true, y_prob, labels):
    """Compute multiclass metrics given class-probabilities."""
    y_pred = np.argmax(y_prob, axis=1)
    return dict(
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
        acc=accuracy_score(y_true, y_pred),
        macro_prec=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_rec=recall_score(y_true, y_pred, average="macro", zero_division=0),
        y_pred=y_pred
    )

def plot_confusion(cm, classes, title, outpath, normalize=False):
    """Save confusion matrix heatmap (counts or row-normalized)."""
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(np.float64)/cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90); plt.yticks(ticks, classes)
    fmt = '.2f' if normalize else 'd'
    thr = cm.max()/2. if cm.size>0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i,j]
            plt.text(j,i,format(v,fmt), ha="center",
                     color="white" if v>thr else "black", fontsize=8)
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight"); plt.close()

# ======== HPO helpers (sklearn) ========
def get_param_distributions(model_name: str, xgb_ok: bool = False):
    if model_name == "rf":
        return {
            "clf__n_estimators": [200, 300, 400, 600],
            "clf__max_depth": [10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        }
    if model_name == "gbdt":
        if xgb_ok:
            return {
                "clf__n_estimators": [300, 600, 900],
                "clf__max_depth": [4, 6, 8],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
                "clf__reg_lambda": [0.0, 1.0, 3.0],
            }
        else:
            return {
                "clf__max_iter": [200, 300, 500],
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__max_leaf_nodes": [31, 63, 127],
                "clf__min_samples_leaf": [10, 20, 50],
                "clf__l2_regularization": [0.0, 0.01, 0.1],
            }
    if model_name == "mlp":
        return {
            "clf__hidden_layer_sizes": [(512,128), (256,128), (256,64)],
            "clf__alpha": list(np.logspace(-5, -3, 5)),
            "clf__learning_rate_init": list(np.logspace(-4, -3, 5)),
            "clf__batch_size": [128, 256, 512],
        }
    return {}

def _supports_sample_weight(estimator) -> bool:
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def _route_sample_weight(estimator, sample_weight):
    """Map sample_weight to the final estimator inside (im)balanced pipelines when supported."""
    if sample_weight is None: return {}
    try:
        from imblearn.pipeline import Pipeline as ImbPL
    except Exception:
        ImbPL = tuple()
    from sklearn.pipeline import Pipeline as SkPL
    if isinstance(estimator, (SkPL, ImbPL)):
        final_est = estimator.named_steps.get("clf", None)
        if final_est is not None and _supports_sample_weight(final_est):
            return {"clf__sample_weight": sample_weight}
        else:
            return {}
    else:
        return {"sample_weight": sample_weight} if _supports_sample_weight(estimator) else {}

def build_estimator_for_hpo(model_name, n_classes, seed, imb_strategy):
    """Construct pipeline for HPO; attach ROS when requested and available."""
    use_ros = (imb_strategy == "oversample") and _HAS_IMBLEARN
    if model_name == "rf":
        base = make_rf(seed=seed)
        est = ImbPipeline([("ros", RandomOverSampler(random_state=seed)), ("clf", base)]) if use_ros else SkPipeline([("clf", base)])
    elif model_name == "gbdt":
        base = make_gbdt(n_classes=n_classes, seed=seed)
        est = ImbPipeline([("ros", RandomOverSampler(random_state=seed)), ("clf", base)]) if use_ros else SkPipeline([("clf", base)])
    elif model_name == "mlp":
        base = make_mlp(seed=seed)
        if use_ros:
            est = ImbPipeline([("scaler", base.named_steps["scaler"]), ("ros", RandomOverSampler(random_state=seed)), ("clf", base.named_steps["clf"])])
        else:
            est = SkPipeline([("scaler", base.named_steps["scaler"]), ("clf", base.named_steps["clf"])])
    else:
        est = None
    return est

def run_random_search(estimator, param_distributions, X, y, groups, scoring, n_iter, cv_folds,
                      sample_weight: Optional[np.ndarray], use_groups: bool, out_csv: Path, seed: int):
    """RandomizedSearchCV wrapper that logs and persists cv_results_."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    X = X.astype(np.float32, copy=False)
    y = np.asarray(y)

    if use_groups:
        cv = GroupKFold(n_splits=cv_folds)
        fit_kwargs = {"groups": groups}
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fit_kwargs = {}

    total_fits = n_iter * cv_folds
    log(f"[HPO] randomized search: {n_iter} candidates × {cv_folds} folds = {total_fits} fits")

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=3,
        pre_dispatch='6',
        refit=True,
        random_state=seed,
        verbose=2,
        return_train_score=False
    )

    with log_stage("RandomizedSearchCV.fit"):
        rs.fit(X, y, **fit_kwargs)

    pd.DataFrame(rs.cv_results_).to_csv(out_csv, index=False)
    log(f"[HPO] best CV macro-F1={rs.best_score_:.4f}")
    return rs

# ======== K-fold CV reporting ========
def _fold_splitter(n_splits: int, seed: int, use_groups: bool, groups: Optional[np.ndarray], y: np.ndarray):
    """Return a generator of (train_idx, val_idx) for GroupKFold or StratifiedKFold."""
    if use_groups:
        splitter = GroupKFold(n_splits=n_splits)
        return splitter.split(np.zeros_like(y), y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(np.zeros_like(y), y)

def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Helper to compute the standard metric set."""
    m = evaluate_multiclass(y_true, y_prob, labels=list(range(y_prob.shape[1])))
    return {
        "MacroF1": m["macro_f1"],
        "WeightedF1": m["weighted_f1"],
        "Acc": m["acc"],
        "MacroPrec": m["macro_prec"],
        "MacroRec": m["macro_rec"],
    }

def run_cv_report_sklearn(estimator, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray],
                          n_splits: int, seed: int, sample_weight: Optional[np.ndarray],
                          out_csv_path: Path, out_summary_path: Path) -> Dict[str, float]:
    """
    Train/predict per fold for sklearn estimator and write:
      - out_csv_path: per-fold metrics table
      - out_summary_path: single-row mean/std summary
    Returns dict of mean/std metrics for aggregation.
    """
    use_groups = groups is not None
    records = []
    fold_iter = _fold_splitter(n_splits, seed, use_groups, groups, y)

    for fold_idx, (tr, va) in enumerate(fold_iter, start=1):
        est = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            sw = sample_weight[tr]
            fit_kwargs = _route_sample_weight(est, sw)

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        with log_stage(f"[CV][sklearn] fold {fold_idx}/{n_splits} fit"):
            est.fit(Xtr, ytr, **fit_kwargs)

        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(Xva)
        else:
            y_pred = est.predict(Xva)
            y_prob = np.zeros((len(y_pred), int(np.max(y)+1)), dtype=np.float32)
            y_prob[np.arange(len(y_pred)), y_pred] = 1.0

        met = _compute_metrics(yva, y_prob)
        met["Fold"] = fold_idx
        records.append(met)
        log(f"[CV][sklearn] fold {fold_idx} metrics: {met}")

    cv_df = pd.DataFrame(records, columns=["Fold","MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"])
    cv_df.to_csv(out_csv_path, index=False)

    mean_vals = cv_df[["MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"]].mean()
    std_vals  = cv_df[["MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"]].std(ddof=1)
    summary = {f"{k}_mean": float(mean_vals[k]) for k in mean_vals.index}
    summary.update({f"{k}_std": float(std_vals[k]) for k in std_vals.index})
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_summary_path, index=False)
    log(f"[CV][sklearn] summary: {summary}")
    return summary

# ======== GCN utilities ========
_ATOM_SYMBOLS = ["B","C","N","O","F","Si","P","S","Cl","Br","I"]
def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Simple atom featurization: element one-hot + degree + formal charge + aromatic + total H."""
    sym = atom.GetSymbol()
    onehot = [1.0 if sym==s else 0.0 for s in _ATOM_SYMBOLS] + [1.0 if sym not in _ATOM_SYMBOLS else 0.0]
    feat = [atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic()), atom.GetTotalNumHs()]
    return np.array(onehot + feat, dtype=np.float32)

def mol_to_geodata(mol: Chem.Mol, y_label: int) -> Optional[GeoData]:
    """Convert an RDKit Mol to a torch_geometric Data object."""
    if mol is None: return None
    x = np.vstack([atom_features(a) for a in mol.GetAtoms()]) if mol.GetNumAtoms() > 0 else np.zeros((1, len(_ATOM_SYMBOLS)+1+4), dtype=np.float32)
    edge_index = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_index.append((u,v)); edge_index.append((v,u))
    if not edge_index:
        edge_index = [(0,0)]
    edge_index = np.array(edge_index, dtype=np.int64).T
    data = GeoData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.tensor([y_label], dtype=torch.long)
    )
    return data

class SimpleGCN(nn.Module):
    """Minimal 2-layer GCN with global-mean graph pooling."""
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index); x = self.act(x); x = self.dropout(x)
        x = self.conv2(x, edge_index); x = self.act(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        logits = self.lin(x)
        return logits

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))

@torch.no_grad()
def eval_on_loader(model, loader, device, n_classes: int):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        ps.append(prob)
        ys.append(batch.y.view(-1).detach().cpu().numpy())
    if not ys:
        return 0.0, 0.0, 0.0, 0.0, np.zeros((0,n_classes), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    m = evaluate_multiclass(y_true, y_prob, labels=list(range(n_classes)))
    return m["macro_f1"], m["weighted_f1"], m["acc"], m["macro_rec"], y_prob, y_true

def build_gcn_datasets(train_df, val_df, label_col: str, le: LabelEncoder):
    """Create torch_geometric Data lists for TRAIN and VAL dataframes."""
    def df_to_list(df):
        datas = []
        for smi, ystr in zip(df["SMILES"].tolist(), df[label_col].astype(str).tolist()):
            y = int(le.transform([ystr])[0])
            mol = Chem.MolFromSmiles(smi)
            d = mol_to_geodata(mol, y)
            if d is not None:
                datas.append(d)
        return datas
    train_list = df_to_list(train_df)
    val_list = df_to_list(val_df)
    return train_list, val_list

def run_gcn_branch(args, outdir: Path, train_df, val_df, le: LabelEncoder, class_names: List[str]) -> Optional[Dict]:
    """Train/validate GCN by random-search on the hold-out validation, export artifacts and return summary."""
    if not (_HAS_TORCH and _HAS_PYG):
        log("[GCN] WARN: torch/torch_geometric not available; skipping GCN.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(class_names)
    train_list, val_list = build_gcn_datasets(train_df, val_df, args.label_col, le)
    if len(train_list)==0 or len(val_list)==0:
        log("[GCN] Empty dataset; skip."); return None

    in_dim = train_list[0].x.size(1)
    log(f"[GCN] device={device}; in_dim={in_dim}; n_classes={n_classes}; train_items={len(train_list)}; val_items={len(val_list)}")

    hidden_choices = [64, 128, 256]
    dropout_choices = [0.1, 0.3, 0.5]
    lr_choices = [1e-3, 3e-3, 5e-4]
    wd_choices = [0.0, 1e-4, 5e-4]
    bs_choices = [64, 128, 256]

    def make_loader(lst, batch_size, shuffle):
        return GeoLoader(lst, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    results = []
    best = None
    best_prob = None
    best_true = None

    n_trials = args.n_iter if args.tune=="random" else 1
    log(f"[GCN] HPO trials = {n_trials} (val-based)")

    if args.resume and is_done(outdir, "Graph", "gcn"):
        log("[GCN] Done marker detected; skipping.")
        return None

    for t in range(1, n_trials+1):
        hidden = random.choice(hidden_choices) if args.tune=="random" else args.gcn_hidden
        dropout = random.choice(dropout_choices) if args.tune=="random" else args.gcn_dropout
        lr = random.choice(lr_choices) if args.tune=="random" else args.gcn_lr
        wd = random.choice(wd_choices) if args.tune=="random" else args.gcn_weight_decay
        bs = random.choice(bs_choices) if args.tune=="random" else args.gcn_batch

        model = SimpleGCN(in_dim=in_dim, hidden=hidden, n_classes=n_classes, dropout=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        train_loader = make_loader(train_list, batch_size=bs, shuffle=True)
        val_loader = make_loader(val_list, batch_size=bs, shuffle=False)

        best_val = -1.0
        best_state = None
        patience = args.gcn_patience
        stale = 0
        epochs = args.gcn_epochs

        with log_stage(f"[GCN] trial {t}/{n_trials}: hidden={hidden}, dropout={dropout}, lr={lr}, wd={wd}, bs={bs}, epochs={epochs}"):
            for ep in range(1, epochs+1):
                tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
                val_macro, val_weighted, val_acc, val_rec, y_prob, y_true = eval_on_loader(model, val_loader, device, n_classes)
                if val_macro > best_val:
                    best_val = val_macro
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    best_prob = y_prob
                    best_true = y_true
                    stale = 0
                else:
                    stale += 1
                if ep % 5 == 0 or ep==1:
                    log(f"[GCN][trial {t}] epoch {ep}/{epochs} | tr_loss={tr_loss:.4f} | val_macroF1={val_macro:.4f} | best={best_val:.4f} (stale={stale}/{patience})")
                if stale >= patience:
                    break

        results.append({
            "trial": t,
            "hidden": hidden, "dropout": dropout, "lr": lr, "weight_decay": wd, "batch_size": bs,
            "best_val_macroF1": float(best_val)
        })

        if best is None or best_val > best["best_val_macroF1"]:
            best = dict(results[-1])
            torch.save(best_state, outdir/"gcn_best.pt")
            with open(outdir/"hpo_gcn_best.json", "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(outdir/"hpo_gcn.csv", index=False)

    if best is not None and (outdir/"gcn_best.pt").exists():
        model = SimpleGCN(in_dim=in_dim, hidden=best["hidden"], n_classes=n_classes, dropout=best["dropout"]).to(device)
        state = torch.load(outdir/"gcn_best.pt", map_location=device)
        model.load_state_dict(state); model.eval()

        val_loader = GeoLoader(val_list, batch_size=best["batch_size"], shuffle=False, num_workers=0)
        _, _, _, _, y_prob, y_true = eval_on_loader(model, val_loader, device, n_classes)

        class_names_local = list(le.classes_)
        val_probs = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in class_names_local])
        val_meta = val_df[["SMILES", args.label_col]].copy().reset_index(drop=True)
        y_pred = np.argmax(y_prob, axis=1)
        val_meta["true_label"] = val_meta[args.label_col].astype(str); val_meta.drop(columns=[args.label_col], inplace=True)
        val_meta["pred_label"] = [class_names_local[i] for i in y_pred]
        val_pred = pd.concat([val_meta, val_probs], axis=1)
        val_pred.to_csv(outdir/"val_pred_best_gcn.csv", index=False)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        plot_confusion(cm, class_names_local, f"Confusion (best: Graph+GCN)", outdir/"confusion_best_counts_gcn.png", normalize=False)
        plot_confusion(cm, class_names_local, f"Confusion Normalized (best: Graph+GCN)", outdir/"confusion_best_normalized_gcn.png", normalize=True)

        m = evaluate_multiclass(y_true=y_true, y_prob=y_prob, labels=list(range(n_classes)))
        row = dict(FP="Graph", Model="GCN", MacroF1=m["macro_f1"], WeightedF1=m["weighted_f1"],
                   Acc=m["acc"], MacroPrec=m["macro_prec"], MacroRec=m["macro_rec"])
        write_done_marker(outdir, "Graph", "gcn")
        return {"row": row, "y_prob": y_prob, "best_hparams": best}
    else:
        return None

def run_gcn_cv_report(args, outdir: Path, df_train: pd.DataFrame, le: LabelEncoder,
                      best_hparams: Dict, n_classes: int, n_splits: int, seed: int,
                      use_groups: bool, label_col: str, class_names: List[str]):
    """
    Perform K-fold CV on TRAIN for GCN using best hyperparameters discovered on hold-out VAL.
    Save per-fold table and mean±std summary.
    """
    if not (_HAS_TORCH and _HAS_PYG):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups = None
    if use_groups:
        groups = np.array([get_scaffold_smiles(s) for s in df_train["SMILES"].tolist()])

    y_all = le.transform(df_train[label_col].astype(str).values)
    splitter_iter = _fold_splitter(n_splits, seed, use_groups, groups, y_all)

    hidden = best_hparams["hidden"]; dropout = best_hparams["dropout"]
    lr = best_hparams["lr"]; weight_decay = best_hparams["weight_decay"]
    batch_size = best_hparams["batch_size"]
    epochs = args.gcn_epochs
    patience = args.gcn_patience

    _one_mol = Chem.MolFromSmiles(df_train["SMILES"].iloc[0])
    _tmp_data = mol_to_geodata(_one_mol, int(y_all[0]))
    in_dim = _tmp_data.x.size(1) if _tmp_data is not None else len(_ATOM_SYMBOLS)+1+4

    records = []
    for fold_idx, (tr, va) in enumerate(splitter_iter, start=1):
        tr_df = df_train.iloc[tr].reset_index(drop=True)
        va_df = df_train.iloc[va].reset_index(drop=True)

        tr_list, va_list = build_gcn_datasets(tr_df, va_df, label_col, le)
        if len(tr_list) == 0 or len(va_list) == 0:
            log(f"[CV][GCN] fold {fold_idx} has empty data; skip.")
            continue

        model = SimpleGCN(in_dim=in_dim, hidden=hidden, n_classes=n_classes, dropout=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        tr_loader = GeoLoader(tr_list, batch_size=batch_size, shuffle=True, num_workers=0)
        va_loader = GeoLoader(va_list, batch_size=batch_size, shuffle=False, num_workers=0)

        best_val = -1.0
        best_state = None
        stale = 0

        with log_stage(f"[CV][GCN] fold {fold_idx}/{n_splits} fit"):
            for ep in range(1, epochs+1):
                _ = train_one_epoch(model, tr_loader, device, optimizer, criterion)
                val_macro, _, _, _, y_prob, y_true = eval_on_loader(model, va_loader, device, n_classes)
                if val_macro > best_val:
                    best_val = val_macro
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, _, _, _, y_prob, y_true = eval_on_loader(model, va_loader, device, n_classes)
        met = _compute_metrics(y_true, y_prob)
        met["Fold"] = fold_idx
        records.append(met)
        log(f"[CV][GCN] fold {fold_idx} metrics: {met}")

    if not records:
        return None

    tag = "Graph__gcn"
    out_csv_path = outdir / f"cv_report_{tag}.csv"
    out_summary_path = outdir / f"cv_report_{tag}_summary.csv"

    cv_df = pd.DataFrame(records, columns=["Fold","MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"])
    cv_df.to_csv(out_csv_path, index=False)

    mean_vals = cv_df[["MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"]].mean()
    std_vals  = cv_df[["MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"]].std(ddof=1)
    summary = {f"{k}_mean": float(mean_vals[k]) for k in mean_vals.index}
    summary.update({f"{k}_std": float(std_vals[k]) for k in std_vals.index})
    pd.DataFrame([summary]).to_csv(out_summary_path, index=False)
    log(f"[CV][GCN] summary: {summary}")
    return summary

# ======== Orchestration ========
def run(args):
    set_global_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    live_csv = outdir/"performance_summary_live.csv"

    with log_stage("Load CSV"):
        df = pd.read_csv(args.csv)
        log(f"read rows={len(df)}; cols={list(df.columns)[:6]}...")

    assert args.smiles_col in df.columns, f"Missing SMILES column: {args.smiles_col}"
    assert args.label_col in df.columns, f"Missing label column: {args.label_col}"

    with log_stage("Deduplicate"):
        df_dd = deduplicate_by_smiles(df, args.smiles_col, args.label_col, args.odorless_label, args.seed)
        df_dd["SMILES"] = df_dd["SMILES_canonical"]

    le = LabelEncoder()
    y_all = le.fit_transform(df_dd[args.label_col].astype(str).values)
    class_names = le.classes_.tolist(); n_classes = len(class_names)
    counts = pd.Series(y_all).value_counts().sort_index()
    ir = float(counts.max()/max(1, counts.min()))
    (outdir/"class_balance.json").write_text(json.dumps({
        "class_names": class_names,
        "counts": {class_names[i]: int(counts[i]) for i in range(len(counts))},
        "total": int(len(y_all)),
        "imbalance_ratio_max_min": ir
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[labels] classes={class_names}; IR(max/min)={ir:.2f}")

    with log_stage("Scaffold split"):
        train_df, val_df = scaffold_split(df_dd, smiles_col="SMILES", label_col=args.label_col,
                                          val_size=args.val_size, seed=args.seed)
        maj_label = args.majority_label
        if isinstance(maj_label, str) and maj_label.strip().upper() == "AUTO":
            maj_label = train_df[args.label_col].value_counts().idxmax()
        if args.majority_target_n and args.majority_target_n > 0:
            train_df = undersample_majority_scaffold_prop(
                train_df=train_df, label_col=args.label_col, majority_label=maj_label,
                smiles_col="SMILES", target_n=args.majority_target_n, seed=args.seed
            )
        train_df.to_csv(outdir/"train.csv", index=False)
        val_df.to_csv(outdir/"val.csv", index=False)

    extra_physchem_cols = ["HenrysConstant", "Solubility"]
    fp_list = [s.strip().upper() for s in args.fingerprints.split(",") if s.strip()]
    fp_list = [k for k in fp_list if k in ("ECFP4","ECFP6","MACCS")] or ["ECFP4","ECFP6","MACCS"]

    with log_stage(f"Build feature matrices (ECFP nbits={args.nbits}; MACCS=167)"):
        feat_all = build_feature_matrices(df_dd, "SMILES", extra_physchem_cols, fp_kinds=fp_list, nbits=args.nbits)

    def align_xy(X_all: pd.DataFrame):
        """Align feature matrix to train/val rows by SMILES join to ensure consistent ordering."""
        XA = X_all.copy(); XA["SMILES"] = df_dd["SMILES"]
        Xtr = train_df[["SMILES"]].merge(XA, on="SMILES", how="inner").drop(columns=["SMILES"])
        Xva = val_df[["SMILES"]].merge(XA, on="SMILES", how="inner").drop(columns=["SMILES"])
        Xtr = Xtr.astype(np.float32); Xva = Xva.astype(np.float32)
        ytr = le.transform(train_df[args.label_col].astype(str).values)
        yva = le.transform(val_df[args.label_col].astype(str).values)
        log(f"[align] Xtr={Xtr.shape}, Xva={Xva.shape}, dtype={Xtr.dtypes.iloc[0] if Xtr.shape[1]>0 else 'n/a'}")
        return Xtr, ytr, Xva, yva

    groups_tr = np.array([get_scaffold_smiles(s) for s in train_df["SMILES"].tolist()]) if args.cv_group_scaffold else None
    scoring = make_scorer(f1_score, average="macro")

    model_names = [s.strip().lower() for s in args.models.split(',') if s.strip()]
    allowed = {"rf", "gbdt", "mlp", "gcn"}
    model_names = [m for m in model_names if m in allowed] or ["rf", "gbdt", "mlp"]

    perf_rows, all_results = [], []
    cv_summary_rows = []

    if "gcn" in model_names:
        gcn_ret = run_gcn_branch(args, outdir, train_df, val_df, le, class_names)
        if gcn_ret is not None:
            row = gcn_ret["row"]
            perf_rows.append(row); all_results.append((row, ("gcn", None, gcn_ret["y_prob"])))
            try:
                pd.DataFrame(perf_rows).to_csv(live_csv, index=False)
            except Exception as e:
                log(f"[WARN] write live summary failed: {e}")
            gcn_cv_sum = run_gcn_cv_report(
                args=args, outdir=outdir, df_train=train_df, le=le, best_hparams=gcn_ret["best_hparams"],
                n_classes=len(class_names), n_splits=args.cv_folds, seed=args.seed,
                use_groups=(groups_tr is not None), label_col=args.label_col, class_names=class_names
            )
            if gcn_cv_sum is not None:
                cv_summary_rows.append(dict(FP="Graph", Model="GCN", **gcn_cv_sum))

    for fp_kind, (X_all, feat_names) in feat_all.items():
        Xtr, ytr, Xva, yva = align_xy(X_all)
        sw = compute_sample_weight(ytr) if args.imbalance_strategy == "class_weight" else None

        for mname in [m for m in model_names if m != "gcn"]:
            tag = f"{fp_kind}__{mname}"
            if args.resume and is_done(outdir, fp_kind, mname):
                log(f"[resume] Skip finished: {tag}")
                continue

            log(f"----- [{tag}] begin -----")
            t_combo0 = time.perf_counter()

            est = build_estimator_for_hpo(mname, n_classes, args.seed, args.imbalance_strategy)
            if est is None:
                log(f"[WARN] Estimator for {mname} unavailable; skip.")
                continue

            if args.tune == "random":
                pdist = get_param_distributions(mname, xgb_ok=_HAS_XGB)
                out_csv = outdir / f"hpo_{fp_kind}_{mname}.csv"
                use_groups = args.cv_group_scaffold and (groups_tr is not None)
                rs = run_random_search(estimator=est, param_distributions=pdist,
                                       X=Xtr, y=ytr, groups=groups_tr, scoring=scoring,
                                       n_iter=args.n_iter, cv_folds=args.cv_folds,
                                       sample_weight=sw, use_groups=use_groups, out_csv=out_csv, seed=args.seed)
                est_final = rs.best_estimator_
                (outdir/f"hpo_{fp_kind}_{mname}_best.json").write_text(
                    json.dumps({"best_params": rs.best_params_, "best_score_macroF1_cv": rs.best_score_}, indent=2),
                    encoding="utf-8"
                )
            else:
                est_final = est

            with log_stage(f"Final fit [{tag}]"):
                fit_kwargs = _route_sample_weight(est_final, sw)
                Xtr_np = Xtr.to_numpy(dtype=np.float32, copy=False) if isinstance(Xtr, pd.DataFrame) else Xtr.astype(np.float32, copy=False)
                est_final.fit(Xtr_np, ytr, **fit_kwargs)

            try:
                joblib.dump(est_final, outdir/f"model_checkpoint_{tag}.joblib")
            except Exception as e:
                log(f"[WARN] save checkpoint failed: {e}")

            with log_stage(f"Predict on val [{tag}]"):
                Xva_np = Xva.to_numpy(dtype=np.float32, copy=False) if isinstance(Xva, pd.DataFrame) else Xva.astype(np.float32, copy=False)
                if hasattr(est_final, "predict_proba"):
                    y_prob = est_final.predict_proba(Xva_np)
                else:
                    y_pred = est_final.predict(Xva_np)
                    y_prob = np.zeros((len(y_pred), n_classes), dtype=np.float32)
                    y_prob[np.arange(len(y_pred)), y_pred] = 1.0

            m = evaluate_multiclass(y_true=yva, y_prob=y_prob, labels=list(range(n_classes)))
            row = dict(FP=fp_kind, Model=mname.upper(), MacroF1=m["macro_f1"], WeightedF1=m["weighted_f1"],
                       Acc=m["acc"], MacroPrec=m["macro_prec"], MacroRec=m["macro_rec"])
            perf_rows.append(row); all_results.append((row, ("sk", est_final, y_prob)))
            log(f"[{tag}] Eval (hold-out VAL): {row}")

            try:
                out_cv = outdir / f"cv_report_{tag}.csv"
                out_cv_sum = outdir / f"cv_report_{tag}_summary.csv"
                use_groups = (groups_tr is not None)
                Xtr_np = Xtr.to_numpy(dtype=np.float32, copy=False) if isinstance(Xtr, pd.DataFrame) else Xtr.astype(np.float32, copy=False)
                cv_sum = run_cv_report_sklearn(
                    estimator=est_final, X=Xtr_np, y=ytr, groups=groups_tr if use_groups else None,
                    n_splits=args.cv_folds, seed=args.seed, sample_weight=sw,
                    out_csv_path=out_cv, out_summary_path=out_cv_sum
                )
                cv_summary_rows.append(dict(FP=fp_kind, Model=mname.upper(), **cv_sum))
            except Exception as e:
                log(f"[WARN] CV report failed for {tag}: {e}")

            try:
                pd.DataFrame(perf_rows).to_csv(live_csv, index=False)
            except Exception as e:
                log(f"[WARN] write live summary failed: {e}")

            write_done_marker(outdir, fp_kind, mname)
            log(f"----- [{tag}] end in {(time.perf_counter()-t_combo0)/60:.2f} min -----")

    perf_df = pd.DataFrame(perf_rows).sort_values(by=["MacroF1","Acc","MacroPrec","MacroRec"], ascending=False)
    if len(perf_df)==0:
        raise RuntimeError("No models were trained.")
    perf_df.to_csv(outdir/"performance_summary.csv", index=False)
    log(f"[summary] saved: {outdir/'performance_summary.csv'}")

    best = perf_df.iloc[0].to_dict()
    best_fp, best_model = best["FP"], best["Model"]
    log(f"=== Best (by Macro-F1; tie-break Acc->Prec->Rec) === {best}")

    if cv_summary_rows:
        pd.DataFrame(cv_summary_rows).to_csv(outdir/"cv_report_summary_all.csv", index=False)
        log(f"[CV] aggregated summary saved: {outdir/'cv_report_summary_all.csv'}")

    best_tuple = None
    for row, payload in all_results:
        if row["FP"] == best_fp and row["Model"] == best_model:
            best_tuple = payload; break

    if best_tuple is not None and best_tuple[0] == "sk":
        _, best_clf, best_prob = best_tuple
        pkg = {
            "label_encoder": le, "class_names": class_names,
            "fingerprint_kind": best_fp, "nbits": args.nbits,
            "physchem_used": ["MolWeight","LogP","TPSA","MolarRefractivity"],
            "extra_physchem_included": [c for c in ["HenrysConstant","Solubility"] if c in df_dd.columns],
            "sk_model": best_clf
        }
        joblib.dump(pkg, outdir/"best_model.joblib")
        (outdir/"best_model_summary.json").write_text(json.dumps({"best":best}, indent=2), encoding="utf-8")

        val_probs = pd.DataFrame(best_prob, columns=[f"prob_{c}" for c in class_names])
        val_meta = val_df[["SMILES", args.label_col]].copy()
        val_meta["true_label"] = val_meta[args.label_col].astype(str); val_meta.drop(columns=[args.label_col], inplace=True)
        y_true = le.transform(val_df[args.label_col].astype(str).values)
        y_pred = np.argmax(best_prob, axis=1)
        val_pred = pd.concat([val_meta.reset_index(drop=True), val_probs.reset_index(drop=True)], axis=1)
        val_pred["pred_label"] = [class_names[i] for i in y_pred]
        val_pred.to_csv(outdir/"val_pred_best.csv", index=False)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        plot_confusion(cm, class_names, f"Confusion (best: {best_fp}+{best_model})", outdir/"confusion_best_counts.png", normalize=False)
        plot_confusion(cm, class_names, f"Confusion Normalized (best: {best_fp}+{best_model})", outdir/"confusion_best_normalized.png", normalize=True)
        log("[best] saved sklearn best artifacts")
    else:
        log("[best] Top-1 model is GCN (its artifacts were exported as gcn_* files).")

    df_dd.to_csv(outdir/"deduplicated_dataset.csv", index=False)
    (outdir/"run_meta.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[INFO] All artifacts saved to: {outdir}")

# ======== CLI ========
def build_argparser():
    ap = argparse.ArgumentParser(description="Multiclass flavor prediction (scaffold split + HPO + optional GCN)")
    ap.add_argument("--csv", required=True, help="Input CSV (must include SMILES & label columns).")
    ap.add_argument("--smiles_col", default="SMILES", help="SMILES column name.")
    ap.add_argument("--label_col", default="Major_Category", help="Label column name.")
    ap.add_argument("--odorless_label", default="Odorless", help="Odorless category name (for dedup rule).")
    ap.add_argument("--outdir", default="model_out_multiclass", help="Output directory.")
    ap.add_argument("--val_size", type=float, default=0.2, help="Validation ratio for the hold-out split.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--resume", action="store_true", help="Resume mode: skip FP×Model combos already finished (marker files).")

    ap.add_argument("--imbalance_strategy", choices=["none","class_weight","oversample"], default="class_weight",
                    help="Imbalance handling: sample_weight or RandomOverSampler on TRAIN.")

    ap.add_argument("--fingerprints", default="ECFP4,ECFP6,MACCS", help="Comma-separated: ECFP4,ECFP6,MACCS.")
    ap.add_argument("--models", default="rf,gbdt,mlp", help="Comma-separated: rf,gbdt,mlp,gcn")
    ap.add_argument("--nbits", type=int, default=1024, help="ECFP fingerprint bits (1024 or 2048).")

    ap.add_argument("--tune", choices=["none","random"], default="random",
                    help="Enable RandomizedSearchCV (sklearn) / random trials (GCN).")
    ap.add_argument("--n_iter", type=int, default=30, help="RandomizedSearch iterations per FP×model; also GCN trials.")
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of folds for CV (HPO and CV report).")
    ap.add_argument("--cv_group_scaffold", action="store_true",
                    help="Use scaffold-aware GroupKFold in CV (both HPO and per-fold reports).")

    ap.add_argument("--majority_label", default="AUTO", help="Name of the largest class to downsample in TRAIN only.")
    ap.add_argument("--majority_target_n", type=int, default=5000,
                    help="Target number to keep for the majority class in TRAIN (scaffold-proportional). Set <=0 to disable.")

    ap.add_argument("--gcn_hidden", type=int, default=128, help="GCN hidden dim (used when --tune none).")
    ap.add_argument("--gcn_dropout", type=float, default=0.3, help="GCN dropout (used when --tune none).")
    ap.add_argument("--gcn_lr", type=float, default=1e-3, help="GCN learning rate (used when --tune none).")
    ap.add_argument("--gcn_weight_decay", type=float, default=1e-4, help="GCN weight decay (used when --tune none).")
    ap.add_argument("--gcn_batch", type=int, default=128, help="GCN batch size (used when --tune none).")
    ap.add_argument("--gcn_epochs", type=int, default=50, help="GCN max epochs per trial/fold.")
    ap.add_argument("--gcn_patience", type=int, default=10, help="GCN early stopping patience.")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
