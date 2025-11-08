# -*- coding: utf-8 -*-
"""
Standalone validation for saved checkpoints (single or a small set).

What's new:
- --subset {auto,val,train,all} to choose evaluation split
  * auto  : use outdir/val.csv if present, else fall back to --csv
  * val   : force use outdir/val.csv
  * train : force use outdir/train.csv
  * all   : use the --csv you pass in (original behavior)
- Export per-class metrics table (precision, recall, f1, support) for each class.
- Confusion matrices saved as vector (SVG/PDF), plus a metrics bar chart.

Usage example (strictly match training validation set):
python validate_models.py --outdir model_out_multiclass --only MACCS-RF --subset val

If you want to validate multiple combos:
--only MACCS-RF,ECFP4-MLP,ECFP6-GBDT
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Quiet RDKit warnings
from rdkit import RDLogger
lg = RDLogger.logger(); lg.setLevel(RDLogger.CRITICAL)

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, MACCSkeys

import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---- Global plotting style (fonts, linewidths, vector-friendly) ----
plt.rcParams.update({
    "font.family": "Arial",      # If Arial is unavailable, use "DejaVu Sans"
    "font.size": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,          # Embed TrueType for vector editors/journals
    "ps.fonttype": 42,
})

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def _trunc_cmap(cmap, minval=0.1, maxval=0.7, n=256):
    """Use a lower–mid brightness range to obtain softer pastel tones."""
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}", cmap(np.linspace(minval, maxval, n))
    )

# Pastel heatmap colormap (can switch to PuBuGn/YlGnBu, etc.)
PASTEL_CMAP = _trunc_cmap(plt.cm.Blues, 0.1, 0.7)


# --------------------------
# Helpers
# --------------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_class_names(outdir: Path) -> Optional[List[str]]:
    js = outdir / "class_balance.json"
    if js.exists():
        try:
            obj = json.loads(js.read_text(encoding="utf-8"))
            cn = obj.get("class_names", None)
            if isinstance(cn, list) and len(cn) > 0:
                return cn
        except Exception:
            pass
    return None

def load_split_df(outdir: Path, subset: str, fallback_csv: Optional[str],
                  smiles_col: str, label_col: str) -> pd.DataFrame:
    """
    Select dataset according to --subset.
    """
    if subset == "val":
        p = outdir / "val.csv"
        if not p.exists():
            raise SystemExit(f"--subset val specified but not found: {p}")
        log(f"Using validation split: {p}")
        df = pd.read_csv(p)
    elif subset == "train":
        p = outdir / "train.csv"
        if not p.exists():
            raise SystemExit(f"--subset train specified but not found: {p}")
        log(f"Using training split: {p}")
        df = pd.read_csv(p)
    elif subset == "auto":
        p = outdir / "val.csv"
        if p.exists():
            log(f"Auto-selected: {p}")
            df = pd.read_csv(p)
        else:
            if fallback_csv is None:
                raise SystemExit("auto fallback needs --csv, but none provided.")
            log(f"Auto fallback to --csv: {fallback_csv}")
            df = pd.read_csv(fallback_csv)
    else:  # "all"
        if fallback_csv is None:
            raise SystemExit("--subset all requires --csv to be provided.")
        log(f"Using full CSV: {fallback_csv}")
        df = pd.read_csv(fallback_csv)

    if smiles_col not in df.columns:
        raise SystemExit(f"Missing SMILES column: {smiles_col}")
    if label_col not in df.columns:
        raise SystemExit(f"Missing label column: {label_col}")

    return df

def clean_and_align_labels(df: pd.DataFrame, label_col: str, class_names: List[str]) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Drop rows with missing labels and those not present in training class_names.
    Return cleaned df and a LabelEncoder fitted on training class_names (ensures consistent mapping).
    """
    # drop NaN and empty
    n0 = len(df)
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""].copy()
    dropped_missing = n0 - len(df)

    # drop unseen labels
    allowed = set(class_names)
    mask_seen = df[label_col].isin(allowed)
    dropped_unseen = int((~mask_seen).sum())
    if dropped_missing > 0 or dropped_unseen > 0:
        log(f"[labels] dropped rows -> missing: {dropped_missing}, unseen: {dropped_unseen}")

    df = df[mask_seen].reset_index(drop=True)

    # build label encoder with TRAINING classes (fixed order)
    le = LabelEncoder()
    le.fit(class_names)
    return df, le

def morgan_bits(mol: Chem.Mol, radius: int, nbits: int = 1024) -> np.ndarray:
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def maccs_bits(mol: Chem.Mol) -> np.ndarray:
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def physchem_features(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MolWeight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def build_X(df: pd.DataFrame, smiles_col: str, fp_kind: str, nbits: int,
            extra_physchem_cols: List[str]) -> pd.DataFrame:
    """
    Reproduce training feature order:
      - For MACCS: columns ["MACCS_0"..."MACCS_166"]
      - For ECFP4/6: columns [f"{fp_kind}_{0}"...f"{fp_kind}_{nbits-1}"]
      - Append physchem ["MolWeight","LogP","TPSA","MolarRefractivity"]
      - Append any present extras among extra_physchem_cols in input df (zeros if absent)
    """
    feats = []; phys = []
    if fp_kind == "MACCS":
        bit_cols = [f"MACCS_{i}" for i in range(167)]
        for smi in df[smiles_col].tolist():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                feats.append(np.zeros((167,), dtype=np.int8))
                phys.append({k:0.0 for k in ["MolWeight","LogP","TPSA","MolarRefractivity"]})
            else:
                feats.append(maccs_bits(mol))
                phys.append(physchem_features(mol))
    else:
        radius = 2 if fp_kind.upper()=="ECFP4" else 3
        bit_cols = [f"{fp_kind}_{i}" for i in range(nbits)]
        for smi in df[smiles_col].tolist():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                feats.append(np.zeros((nbits,), dtype=np.int8))
                phys.append({k:0.0 for k in ["MolWeight","LogP","TPSA","MolarRefractivity"]})
            else:
                feats.append(morgan_bits(mol, radius, nbits))
                phys.append(physchem_features(mol))

    bits_df = pd.DataFrame(np.vstack(feats), columns=bit_cols, index=df.index)
    phys_df = pd.DataFrame(phys, index=df.index)

    # extras (if present in df)
    extras_present = [c for c in extra_physchem_cols if c in df.columns]
    extra_df = df[extras_present].astype(np.float32) if extras_present else pd.DataFrame(index=df.index)

    X = pd.concat([bits_df, phys_df, extra_df], axis=1).astype(np.float32)
    return X

def evaluate_multiclass(y_true, y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    return dict(
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
        acc=accuracy_score(y_true, y_pred),
        macro_prec=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_rec=recall_score(y_true, y_pred, average="macro", zero_division=0),
        y_pred=y_pred
    )

def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_prefix: Path):
    """Save confusion matrix as counts and normalized (SVG/PDF)."""
    def _figure_size(n_classes: int) -> Tuple[float, float]:
        # Scale with number of classes to avoid label overlap; increase 'scale' if needed
        scale = 0.55
        w = max(6.0, scale * n_classes + 2.0)
        h = max(4.5, scale * n_classes + 2.0)
        return w, h

    def _style_axes(ax):
        # Unified 1 pt linewidth
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
        ax.tick_params(axis="both", width=1.0)

    # ---------- Counts ----------
    cm_counts = cm.copy()
    w, h = _figure_size(len(classes))
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(cm_counts, interpolation="nearest", cmap=PASTEL_CMAP, vmin=0, aspect="equal")
    ax.set_title(title + " (counts)", pad=6)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    thr = cm_counts.max() / 2.0 if cm_counts.size > 0 else 0.0
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            v = int(cm_counts[i, j])
            ax.text(j, i, f"{v:d}", ha="center", va="center",
                    color=("white" if v > thr else "black"), fontsize=8)

    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    _style_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(width=1.0)

    fig.tight_layout()
    for ext in (".svg", ".pdf"):
        fig.savefig(str(out_prefix) + "_counts" + ext, bbox_inches="tight")
    plt.close(fig)

    # ---------- Normalized ----------
    with np.errstate(all="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    w, h = _figure_size(len(classes))
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=PASTEL_CMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_title(title + " (norm)", pad=6)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    thr = cm_norm.max() / 2.0 if cm_norm.size > 0 else 0.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=("white" if v > thr else "black"), fontsize=8)

    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    _style_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(width=1.0)

    fig.tight_layout()
    for ext in (".svg", ".pdf"):
        fig.savefig(str(out_prefix) + "_norm" + ext, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(row: Dict[str, float], title: str, out_prefix: Path):
    """Bar chart for metrics (MacroF1, WeightedF1, Acc, MacroPrec, MacroRec)."""
    metric_keys = ["MacroF1","WeightedF1","Acc","MacroPrec","MacroRec"]
    vals = [row[k] for k in metric_keys]

    # Slightly larger figsize, pastel palette, unified 1 pt linewidth
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(metric_keys)))
    bars = ax.bar(metric_keys, vals, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_title(title, pad=6)
    ax.set_ylim(0, 1.0)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.tick_params(width=1.0)

    fig.tight_layout()
    for ext in (".svg", ".pdf"):
        fig.savefig(str(out_prefix) + ext, bbox_inches="tight")
    plt.close(fig)


def export_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], out_csv: Path):
    """Export per-class precision/recall/f1/support table as CSV."""
    labels = list(range(len(class_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame({
        "Class": class_names,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Support": sup.astype(int)
    })
    df.to_csv(out_csv, index=False)

# --------------------------
# Main
# --------------------------
def parse_only(s: str) -> List[Tuple[str,str]]:
    """
    Parse --only like "MACCS-RF,ECFP4-MLP" into list of (fp_kind, model_lower).
    FP is case-insensitive; model is lowered ("rf","gbdt","mlp").
    """
    out = []
    if not s: return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" not in tok:
            raise ValueError(f"--only entry must be like FP-Model, got: {tok}")
        fp, mdl = tok.split("-", 1)
        fp = fp.strip().upper()
        mdl = mdl.strip().lower()
        if fp not in {"MACCS","ECFP4","ECFP6"}:
            raise ValueError(f"Unsupported FP in --only: {fp}")
        if mdl not in {"rf","gbdt","mlp"}:
            raise ValueError(f"Unsupported model in --only: {mdl}")
        out.append((fp, mdl))
    return out

def run(args):
    outdir = Path(args.outdir)
    if not outdir.exists():
        raise SystemExit(f"Outdir not found: {outdir}")

    # class names from training
    class_names = load_class_names(outdir)
    if not class_names:
        raise SystemExit("class_balance.json not found in outdir; cannot align labels safely.")

    # choose evaluation dataset according to --subset
    df = load_split_df(outdir, args.subset, args.csv, args.smiles_col, args.label_col)

    df, le = clean_and_align_labels(df, args.label_col, class_names)
    if len(df) == 0:
        raise SystemExit("No rows left after dropping missing/unseen labels. Aborting.")

    combos = parse_only(args.only) if args.only else []
    if not combos:
        raise SystemExit("Please specify --only like 'MACCS-RF' or 'ECFP4-MLP'.")

    # extras possibly present in CSV (zeros if missing)
    extra_physchem_cols = ["HenrysConstant","Solubility"]

    for fp_kind, model_name in combos:
        tag = f"{fp_kind}__{model_name}"
        ckpt = outdir / f"model_checkpoint_{fp_kind}__{model_name}.joblib"
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found for {tag}: {ckpt}")
        log(f"== Validate: {tag} on subset={args.subset} ==")

        # load estimator
        est = joblib.load(ckpt)

        # Build features in training order
        X = build_X(df, args.smiles_col, fp_kind, args.nbits, extra_physchem_cols)
        y_true = le.transform(df[args.label_col].astype(str).values)

        # Predict
        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X.to_numpy(dtype=np.float32, copy=False))
        else:
            y_pred = est.predict(X.to_numpy(dtype=np.float32, copy=False))
            y_prob = np.zeros((len(y_pred), len(class_names)), dtype=np.float32)
            y_prob[np.arange(len(y_pred)), y_pred.astype(int)] = 1.0

        # Eval
        m = evaluate_multiclass(y_true, y_prob, labels=list(range(len(class_names))))
        row = dict(FP=fp_kind, Model=model_name.upper(),
                   MacroF1=m["macro_f1"], WeightedF1=m["weighted_f1"],
                   Acc=m["acc"], MacroPrec=m["macro_prec"], MacroRec=m["macro_rec"])
        log(f"[Eval] {row}")

        # Save summary CSV
        out_csv = outdir / f"validate_summary__{fp_kind}__{model_name}__{args.subset}.csv"
        pd.DataFrame([row]).to_csv(out_csv, index=False)

        # Confusion matrices (vector)
        y_pred = np.argmax(y_prob, axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        plot_confusion(cm, class_names, f"Confusion {fp_kind}+{model_name.upper()}", outdir / f"confusion__{fp_kind}__{model_name}__{args.subset}")

        # Metric bars (vector)
        plot_metric_bars(row, f"Metrics {fp_kind}+{model_name.upper()} ({args.subset})", outdir / f"metrics__{fp_kind}__{model_name}__{args.subset}")

        # Per-class metrics CSV
        per_class_csv = outdir / f"per_class_metrics__{fp_kind}__{model_name}__{args.subset}.csv"
        export_per_class_metrics(y_true, y_pred, class_names, per_class_csv)

        log(f"[OK] Saved: {out_csv}, confusion_* (SVG/PDF), metrics_* (SVG/PDF), {per_class_csv}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Validate saved FP×Model checkpoints on a chosen split.")
    ap.add_argument("--csv", help="Input CSV with SMILES and labels (used if --subset all or as auto fallback)")
    ap.add_argument("--smiles_col", default="SMILES", help="SMILES column name")
    ap.add_argument("--label_col", default="Major_Category", help="Label column name")
    ap.add_argument("--outdir", required=True, help="Training output directory containing checkpoints & splits")
    ap.add_argument("--only", required=True, help="One or more combos like MACCS-RF,ECFP4-MLP")
    ap.add_argument("--nbits", type=int, default=1024, help="ECFP bit-length used during training (if ECFP*)")
    ap.add_argument("--subset", choices=["auto","val","train","all"], default="auto",
                    help="Which dataset to evaluate on (default: auto=prefer val.csv)")
    return ap

if __name__ == "__main__":
    run(build_argparser().parse_args())
