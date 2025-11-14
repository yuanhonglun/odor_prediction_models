# -*- coding: utf-8 -*-
"""
Threshold (ODT) Regression with Scaffold-aware Split, HPO, and Band-wise Evaluation
Author: (Your Name) | License: MIT

Pipeline:
  - Input CSV: columns = ["SMILES", "threshold"]
  - Clean: drop rows with any NaN; drop threshold<=0; canonicalize & dedup by SMILES
  - Target: y = -log10(threshold)
  - Split: Greedy Murcko-scaffold hold-out (val_size=0.2 by default)
  - Features:
      * ECFP4 (1024 bits), ECFP6 (1024 bits), MACCS (167 bits)
      * Physchem: MolWeight, LogP, TPSA, MolarRefractivity
      - Concatenate (bits + physchem)
  - Models: RF / GBDT (XGBoostRegressor if available else HistGradientBoostingRegressor) / MLPRegressor
  - HPO: RandomizedSearchCV (n_iter × 5-fold), scoring='r2'
          * Optional GroupKFold by scaffold to avoid scaffold leakage
  - Final: retrain best params on TRAIN, evaluate on VAL; rank by (R2 desc, RMSE asc)
  - Export:
      * hpo_*.csv & *_best.json per combo
      * cv_report_*.csv & cv_report_*_summary.csv per combo (train-fold metrics)
      * val_pred_*.csv & val_metrics_*.json per combo
      * performance_summary_val.csv (all combos ranking)
      * cv_report_summary_all.csv (all combos train-CV mean±std)
      * best_model.joblib & best_model_summary.json & run_meta.json
      * band_eval_best.json: band-wise R2/RMSE on VAL (low/mid/high by threshold tertiles)
      * prediction_interval_best.json: 95% PI radius and coverage on VAL (best model)
      * val_pred_PI_*: VAL predictions with 95% PI (best model)
      * prediction_interval_scatter__*__val.(svg/pdf): observed vs predicted with PI band

Usage:
  python train_threshold_odt.py --csv threshold_data.csv --outdir model_out_threshold
"""

from __future__ import annotations
import argparse, json, os, random, time, inspect
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib

# ====== RDKit ======
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

# ====== sklearn ======
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone

# ====== xgboost (optional) ======
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ====== matplotlib for PI plots ======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ====== Logging / Repro ======
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        return f"{_PROC.memory_info().rss/1024/1024:.1f} MB"
    except Exception:
        return "n/a"

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | mem={_mem()}", flush=True)

@contextmanager
def log_stage(name: str):
    t0 = time.perf_counter()
    log(f"▶ {name} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {name} done in {dt/60:.2f} min")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

# small helper for plotting style
def _style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.tick_params(width=1.0)

# ====== SMILES & Scaffold ======
def canonical_smiles(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi.strip(): return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

def murcko_scaffold_smi(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None: return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.MolToSmiles(scaf, isomericSmiles=True)

def greedy_scaffold_split(df: pd.DataFrame, smiles_col: str, val_size: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy Murcko-scaffold split to approximate val_size.
    Pick scaffolds in descending frequency into VAL until reaching target size.
    """
    set_seed(seed)
    tmp = df.copy()
    with log_stage("Compute Murcko scaffolds"):
        tmp["_scaf"] = tmp[smiles_col].apply(murcko_scaffold_smi)
    sizes = tmp["_scaf"].value_counts().sort_values(ascending=False)
    total = len(tmp); target = int(round(total * val_size))
    val_idx, cur = set(), 0
    for scaf, sz in sizes.items():
        if cur + sz <= target:
            val_idx.update(tmp.index[tmp["_scaf"] == scaf].tolist())
            cur += sz
    val_df = tmp.loc[sorted(val_idx)].drop(columns=["_scaf"]).reset_index(drop=True)
    train_df = tmp.drop(index=val_df.index).drop(columns=["_scaf"]).reset_index(drop=True)
    log(f"[split] train={len(train_df)}, val={len(val_df)} (ratio ≈ {len(val_df)/total:.2f})")
    return train_df, val_df

# ====== Features ======
def fp_morgan_bits(mol: Chem.Mol, radius: int, nbits: int = 1024) -> np.ndarray:
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def fp_maccs_bits(mol: Chem.Mol) -> np.ndarray:
    bv = GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def physchem(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MolWeight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def build_feature_tables(df: pd.DataFrame, smiles_col: str,
                         fp_kinds=("ECFP4","ECFP6","MACCS"), nbits=1024) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    """
    Return dict: kind -> (X_df, columns)
    """
    results = {}
    kind2radius = {"ECFP4": 2, "ECFP6": 3}
    for kind in fp_kinds:
        feats, phys = [], []
        if kind == "MACCS":
            bit_cols = [f"MACCS_{i}" for i in range(167)]
        else:
            bit_cols = [f"{kind}_{i}" for i in range(nbits)]
        with log_stage(f"Build {kind} bits + physchem for {len(df)} molecules"):
            for smi in df[smiles_col].tolist():
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    if kind == "MACCS":
                        feats.append(np.zeros((167,), dtype=np.int8))
                    else:
                        feats.append(np.zeros((nbits,), dtype=np.int8))
                    phys.append({"MolWeight":0.0, "LogP":0.0, "TPSA":0.0, "MolarRefractivity":0.0})
                    continue
                if kind == "MACCS":
                    feats.append(fp_maccs_bits(mol))
                else:
                    radius = kind2radius[kind]
                    feats.append(fp_morgan_bits(mol, radius=radius, nbits=nbits))
                phys.append(physchem(mol))
        with log_stage(f"Assemble {kind} feature table"):
            bits_df = pd.DataFrame(np.vstack(feats), columns=bit_cols, index=df.index)
            phys_df = pd.DataFrame(phys, index=df.index)
            X = pd.concat([bits_df, phys_df], axis=1).astype(np.float32)
            results[kind] = (X, X.columns.tolist())
            log(f"[{kind}] X shape = {X.shape}")
    return results

# ====== Models ======
def make_rf(seed: int = 42) -> SkPipeline:
    return SkPipeline([("clf", RandomForestRegressor(
        n_estimators=600, max_depth=30, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", n_jobs=6, random_state=seed, bootstrap=True
    ))])

def make_gbdt(seed: int = 42) -> SkPipeline:
    if _HAS_XGB:
        return SkPipeline([("clf", xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1
        ))])
    else:
        # HistGradientBoostingRegressor fallback
        return SkPipeline([("clf", HistGradientBoostingRegressor(
            loss="squared_error", max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=seed
        ))])

def make_mlp(seed: int = 42) -> SkPipeline:
    return SkPipeline([
        ("scaler", MaxAbsScaler()),
        ("clf", MLPRegressor(
            hidden_layer_sizes=(512,128), activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=400, random_state=seed, verbose=False,
            early_stopping=True, n_iter_no_change=20
        ))
    ])

# ====== HPO Spaces ======
def get_param_distributions(model_name: str) -> Dict[str, List]:
    if model_name == "rf":
        return {
            "clf__n_estimators": [200, 300, 400, 600],
            "clf__max_depth": [10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        }
    if model_name == "gbdt":
        if _HAS_XGB:
            return {
                "clf__n_estimators": [300, 600, 900],
                "clf__max_depth": [4, 6, 8],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
                "clf__reg_lambda": [0.0, 1.0, 3.0],
            }
        else:
            # HistGradientBoostingRegressor fallback space (rough mapping)
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

# ====== HPO Wrapper ======
def _supports_sample_weight(estimator) -> bool:
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def _route_sample_weight(estimator, sample_weight):
    # not used in this regression (no class imbalance), but kept for API consistency
    return {}

def run_random_search(estimator, param_distributions, X, y, groups, n_iter, cv_folds,
                      use_groups: bool, out_csv: Path, seed: int):
    if isinstance(X, pd.DataFrame): X = X.to_numpy(dtype=np.float32, copy=False)
    X = X.astype(np.float32, copy=False); y = np.asarray(y, dtype=np.float32)
    if use_groups:
        cv = GroupKFold(n_splits=cv_folds)
        fit_kwargs = {"groups": groups}
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fit_kwargs = {}

    total = n_iter * cv_folds
    log(f"[HPO] RandomizedSearchCV: {n_iter} candidates × {cv_folds} folds = {total} fits")

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        n_jobs=3,
        pre_dispatch="6",
        refit=True,
        random_state=seed,
        verbose=2,
        return_train_score=False
    )
    with log_stage("RandomizedSearchCV.fit"):
        rs.fit(X, y, **fit_kwargs)

    pd.DataFrame(rs.cv_results_).to_csv(out_csv, index=False)
    log(f"[HPO] best CV R2={rs.best_score_:.4f}")
    return rs

# ====== CV Report (train folds) ======
def run_cv_report(estimator, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray],
                  n_splits: int, seed: int, out_csv: Path, out_summary: Path):
    use_groups = groups is not None
    if use_groups:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    records = []
    for k, (tr, va) in enumerate(split_iter, 1):
        est = clone(estimator)  # safe clone
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        with log_stage(f"[CV] fold {k}/{n_splits} fit"):
            est.fit(Xtr, ytr)
        yhat = est.predict(Xva)
        r2  = float(r2_score(yva, yhat))
        rmse = float(np.sqrt(mean_squared_error(yva, yhat)))
        records.append({"Fold": k, "R2": r2, "RMSE": rmse})
        log(f"[CV] fold {k}: R2={r2:.4f}, RMSE={rmse:.4f}")

    df = pd.DataFrame(records, columns=["Fold","R2","RMSE"])
    df.to_csv(out_csv, index=False)
    mR2, sR2 = float(df["R2"].mean()), float(df["R2"].std(ddof=1))
    mE, sE = float(df["RMSE"].mean()), float(df["RMSE"].std(ddof=1))
    pd.DataFrame([{"R2_mean":mR2, "R2_std":sR2, "RMSE_mean":mE, "RMSE_std":sE}]).to_csv(out_summary, index=False)
    log(f"[CV] summary: R2={mR2:.4f}±{sR2:.4f}; RMSE={mE:.4f}±{sE:.4f}")
    return {"R2_mean": mR2, "R2_std": sR2, "RMSE_mean": mE, "RMSE_std": sE}

# ====== Band-wise evaluation on VAL ======
def band_eval_on_val(val_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                     threshold_col: str = "threshold") -> Dict[str, Dict[str, float]]:
    """
    Split VAL samples into low/mid/high bands by tertiles of the original threshold (not -log10).
    Report R2 & RMSE per band.
    """
    thr = val_df[threshold_col].to_numpy(dtype=float)
    q1, q2 = np.quantile(thr, [1/3, 2/3])
    bands = {
        "Low": (thr <= q1),
        "Mid": (thr > q1) & (thr <= q2),
        "High": (thr > q2)
    }
    out = {"q1": float(q1), "q2": float(q2)}
    for name, mask in bands.items():
        idx = np.where(mask)[0]
        if len(idx) < 2:
            out[name] = {"n": int(len(idx)), "R2": float("nan"), "RMSE": float("nan")}
            continue
        r2  = float(r2_score(y_true[idx], y_pred[idx]))
        rmse = float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
        out[name] = {"n": int(len(idx)), "R2": r2, "RMSE": rmse}
    return out

# ====== Prediction interval plotting (best model on VAL) ======
def plot_prediction_interval_scatter(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     q95: float,
                                     title: str,
                                     out_prefix: Path):
    """
    Scatter plot of observed vs predicted −log10(ODT) with a symmetric ±q95 band
    around the identity line (95% PI approximation).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.1 * (hi - lo if hi > lo else 1.0)
    x_line = np.linspace(lo - pad, hi + pad, 200)
    diag = x_line
    upper = x_line + q95
    lower = x_line - q95

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    pt_color = plt.cm.Set2(0)
    band_color = plt.cm.Set2(1)

    ax.scatter(y_true, y_pred, s=10, alpha=0.7, color=pt_color, edgecolors="none", label="Compounds")
    ax.plot(x_line, diag, linestyle="--", color="grey", lw=1.0, label="Perfect")
    ax.plot(x_line, upper, linestyle=":", color=band_color, lw=1.0,
            label=f"±q$_{{0.95}}$ = {q95:.2f}")
    ax.plot(x_line, lower, linestyle=":", color=band_color, lw=1.0)

    ax.set_xlabel("Observed −log10(ODT [mg/L])")
    ax.set_ylabel("Predicted −log10(ODT [mg/L])")
    ax.set_title(title, pad=6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()

    for ext in (".svg", ".pdf"):
        fig.savefig(str(out_prefix) + ext, bbox_inches="tight")
    plt.close(fig)

# ====== Orchestrator ======
def run(args):
    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    live_csv = outdir / "performance_summary_val_live.csv"

    # ---- Load & Clean ----
    with log_stage("Load CSV"):
        df = pd.read_csv(args.csv)
        assert args.smiles_col in df.columns, f"Missing column: {args.smiles_col}"
        assert args.threshold_col in df.columns, f"Missing column: {args.threshold_col}"
        df = df[[args.smiles_col, args.threshold_col]].copy()

    with log_stage("Drop NaN / invalid threshold / canonicalize & dedup"):
        df = df.dropna(subset=[args.smiles_col, args.threshold_col]).copy()
        df[args.threshold_col] = pd.to_numeric(df[args.threshold_col], errors="coerce")
        df = df.dropna(subset=[args.threshold_col]).copy()
        df = df[df[args.threshold_col] > 0].copy()  # log10 valid
        # canonical smiles
        df["SMILES_canonical"] = df[args.smiles_col].apply(canonical_smiles)
        df = df[~df["SMILES_canonical"].isna()].copy()
        df["SMILES"] = df["SMILES_canonical"]
        df = df.groupby("SMILES", as_index=False)[args.threshold_col] \
            .apply(lambda s: 10 ** (-np.mean(-np.log10(s.values))))
        df.rename(columns={args.threshold_col: "threshold"}, inplace=True)
        # targets
        df["y"] = -np.log10(df[args.threshold_col].astype(float))
        log(f"rows after cleaning/dedup: {len(df)}")

    # ---- Split: greedy scaffold ----
    with log_stage("Scaffold-aware greedy split"):
        train_df, val_df = greedy_scaffold_split(df, smiles_col="SMILES", val_size=args.val_size, seed=args.seed)
        train_df.to_csv(outdir/"train.csv", index=False)
        val_df.to_csv(outdir/"val.csv", index=False)

    # ---- Features for all FP kinds ----
    fp_list = [s.strip().upper() for s in args.fingerprints.split(",") if s.strip()]
    fp_list = [k for k in fp_list if k in ("ECFP4","ECFP6","MACCS")] or ["ECFP4","ECFP6","MACCS"]

    with log_stage(f"Build features (ECFP bits={args.nbits}; MACCS=167)"):
        feat_all = build_feature_tables(df, "SMILES", fp_kinds=fp_list, nbits=args.nbits)

    def _align_Xy(X_all: pd.DataFrame, src_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # align by SMILES for consistent row ordering
        XA = X_all.copy(); XA["SMILES"] = df["SMILES"]
        X_src = src_df[["SMILES"]].merge(XA, on="SMILES", how="inner").drop(columns=["SMILES"])
        y_src = src_df["y"].to_numpy(dtype=np.float32)
        thr_src = src_df[args.threshold_col].to_numpy(dtype=np.float64)
        return X_src.to_numpy(dtype=np.float32, copy=False), y_src, thr_src

    # groups for scaffold-aware CV (TRAIN only)
    groups_train = np.array([murcko_scaffold_smi(s) for s in train_df["SMILES"].tolist()]) if args.cv_group_scaffold else None

    model_names = [s.strip().lower() for s in args.models.split(",") if s.strip()]
    allowed = {"rf", "gbdt", "mlp"}
    model_names = [m for m in model_names if m in allowed] or ["rf","gbdt","mlp"]

    perf_rows = []
    cv_summary_rows = []

    # For ranking
    best_combo = None
    best_val = None  # tuple (R2, RMSE), higher R2, lower RMSE
    best_payload = None  # (fp_kind, est_final, feat_columns)

    # ---- Iterate FP × Model ----
    for fp_kind, (X_all, feat_cols) in feat_all.items():
        X_tr, y_tr, thr_tr = _align_Xy(X_all, train_df)
        X_va, y_va, thr_va = _align_Xy(X_all, val_df)

        for mname in model_names:
            tag = f"{fp_kind}__{mname}"
            log(f"----- [{tag}] begin -----")
            t0 = time.perf_counter()

            # build base estimator
            if mname == "rf":
                est = make_rf(seed=args.seed)
            elif mname == "gbdt":
                est = make_gbdt(seed=args.seed)
            elif mname == "mlp":
                est = make_mlp(seed=args.seed)
            else:
                log(f"[WARN] unknown model: {mname}; skip"); continue

            # HPO
            if args.tune == "random":
                pdist = get_param_distributions(mname)
                out_hpo_csv = outdir / f"hpo_{fp_kind}_{mname}.csv"
                rs = run_random_search(
                    estimator=est, param_distributions=pdist,
                    X=X_tr, y=y_tr, groups=groups_train,
                    n_iter=args.n_iter, cv_folds=args.cv_folds,
                    use_groups=(groups_train is not None),
                    out_csv=out_hpo_csv, seed=args.seed
                )
                est_final = rs.best_estimator_
                (outdir / f"hpo_{fp_kind}_{mname}_best.json").write_text(
                    json.dumps({"best_params": rs.best_params_, "best_cv_R2": rs.best_score_}, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
            else:
                est_final = est

            # Final fit on TRAIN
            with log_stage(f"Final fit on TRAIN [{tag}]"):
                est_final.fit(X_tr, y_tr)

            # Predict on VAL
            with log_stage(f"Predict on VAL [{tag}]"):
                yhat_va = est_final.predict(X_va)
            R2 = float(r2_score(y_va, yhat_va))
            RMSE = float(np.sqrt(mean_squared_error(y_va, yhat_va)))
            perf_rows.append({"FP": fp_kind, "Model": mname.upper(), "Val_R2": R2, "Val_RMSE": RMSE})
            pd.DataFrame(perf_rows).to_csv(live_csv, index=False)
            (outdir / f"val_metrics_{fp_kind}_{mname}.json").write_text(
                json.dumps({"Val_R2": R2, "Val_RMSE": RMSE}, indent=2), encoding="utf-8"
            )

            # save VAL predictions for traceability
            val_pred = pd.DataFrame({
                "SMILES": val_df["SMILES"].values,
                "threshold": thr_va,
                "y_true(-log10)": y_va,
                "y_pred(-log10)": yhat_va,
                "abs_err": np.abs(y_va - yhat_va)
            })
            val_pred.to_csv(outdir / f"val_pred_{fp_kind}_{mname}.csv", index=False)

            # Train-CV report (using best estimator)
            try:
                out_cv = outdir / f"cv_report_{fp_kind}__{mname}.csv"
                out_cv_sum = outdir / f"cv_report_{fp_kind}__{mname}_summary.csv"
                cv_sum = run_cv_report(estimator=est_final, X=X_tr, y=y_tr, groups=groups_train,
                                       n_splits=args.cv_folds, seed=args.seed,
                                       out_csv=out_cv, out_summary=out_cv_sum)
                cv_summary_rows.append({"FP": fp_kind, "Model": mname.upper(), **cv_sum})
            except Exception as e:
                log(f"[WARN] CV report failed for {tag}: {e}")

            # global best update (by Val R2 desc; tie-break Val RMSE asc)
            cur_key = (R2, -RMSE)
            if (best_val is None) or (cur_key > best_val):
                best_val = cur_key
                best_combo = {"FP": fp_kind, "Model": mname.upper(), "Val_R2": R2, "Val_RMSE": RMSE}
                best_payload = (fp_kind, est_final, feat_cols)

            log(f"----- [{tag}] end in {(time.perf_counter()-t0)/60:.2f} min -----")

    # ---- Final ranking & save artifacts ----
    perf_df = pd.DataFrame(perf_rows).sort_values(by=["Val_R2","Val_RMSE"], ascending=[False, True])
    perf_df.to_csv(outdir / "performance_summary_val.csv", index=False)
    if cv_summary_rows:
        pd.DataFrame(cv_summary_rows).to_csv(outdir / "cv_report_summary_all.csv", index=False)

    if best_payload is None:
        raise RuntimeError("No trained models found.")
    fp_best, est_best, feat_cols_best = best_payload
    (outdir / "best_model_summary.json").write_text(json.dumps(best_combo, indent=2), encoding="utf-8")
    log(f"=== Global Best === {best_combo}")

    # Save best model package
    pkg = {
        "fingerprint_kind": fp_best,
        "nbits": int(args.nbits),
        "physchem_used": ["MolWeight","LogP","TPSA","MolarRefractivity"],
        "sk_model": est_best,
        "feature_columns": feat_cols_best,   # for sanity check
        "build_info": {"xgboost_used": _HAS_XGB}
    }
    joblib.dump(pkg, outdir / "best_model.joblib")
    log("[best] saved: best_model.joblib")

    # ---- Band-wise evaluation + prediction intervals on VAL for the best model ----
    best_mname_lower = best_combo["Model"].lower()
    best_val_pred_path = outdir / f"val_pred_{fp_best}_{best_mname_lower}.csv"
    if best_val_pred_path.exists():
        vp = pd.read_csv(best_val_pred_path)

        # band-wise metrics
        band_eval = band_eval_on_val(
            val_df=val_df,
            y_true=vp["y_true(-log10)"].to_numpy(),
            y_pred=vp["y_pred(-log10)"].to_numpy(),
            threshold_col="threshold"
        )
        (outdir/"band_eval_best.json").write_text(json.dumps(band_eval, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[band] saved band_eval_best.json: {band_eval}")

        # ---- 95% prediction interval based on absolute residuals (conformal-style) ----
        y_true_val = vp["y_true(-log10)"].to_numpy(dtype=float)
        y_pred_val = vp["y_pred(-log10)"].to_numpy(dtype=float)
        abs_err = np.abs(y_true_val - y_pred_val)
        q95 = float(np.quantile(abs_err, 0.95))  # 95th percentile of |residual|
        lower = y_pred_val - q95
        upper = y_pred_val + q95
        coverage = float(((y_true_val >= lower) & (y_true_val <= upper)).mean())

        vp["PI_lower(-log10)"] = lower
        vp["PI_upper(-log10)"] = upper
        vp.to_csv(outdir / f"val_pred_PI_{fp_best}_{best_mname_lower}.csv", index=False)

        pi_info = {
            "FP": fp_best,
            "Model": best_combo["Model"],
            "q_abs_err_0.95": q95,
            "coverage_val": coverage,
            "level": 0.95
        }
        (outdir / "prediction_interval_best.json").write_text(
            json.dumps(pi_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"[PI] saved prediction_interval_best.json: {pi_info}")

        # scatter plot with PI band
        pi_title = f"{fp_best}+{best_combo['Model']} (val) prediction intervals"
        pi_prefix = outdir / f"prediction_interval_scatter__{fp_best}__{best_mname_lower}__val"
        plot_prediction_interval_scatter(
            y_true=y_true_val,
            y_pred=y_pred_val,
            q95=q95,
            title=pi_title,
            out_prefix=pi_prefix
        )
        log("[PI] saved prediction_interval_scatter__*.svg/pdf")

    # meta
    (outdir / "run_meta.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[INFO] All artifacts saved to: {outdir}")

# ====== CLI ======
def build_argparser():
    ap = argparse.ArgumentParser(description="ODT threshold regression (scaffold split + HPO + CV report)")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: SMILES, threshold")
    ap.add_argument("--smiles_col", default="SMILES", help="SMILES column name")
    ap.add_argument("--threshold_col", default="threshold", help="Threshold column name (positive numeric)")
    ap.add_argument("--outdir", default="model_out_threshold", help="Output directory")
    ap.add_argument("--val_size", type=float, default=0.2, help="Scaffold-aware hold-out ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    ap.add_argument("--fingerprints", default="ECFP4,ECFP6,MACCS", help="Comma separated: ECFP4,ECFP6,MACCS")
    ap.add_argument("--nbits", type=int, default=1024, help="ECFP bit size (1024 or 2048)")
    ap.add_argument("--models", default="rf,gbdt,mlp", help="Comma separated: rf,gbdt,mlp")

    ap.add_argument("--tune", choices=["none","random"], default="random", help="Enable RandomizedSearchCV")
    ap.add_argument("--n_iter", type=int, default=30, help="RandomizedSearch iterations per FP×model")
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds for HPO and CV report")
    ap.add_argument("--cv_group_scaffold", action="store_true",
                    help="Use GroupKFold by scaffold on TRAIN during CV/HPO")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
