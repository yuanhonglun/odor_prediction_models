# -*- coding: utf-8 -*-
"""
Binary flavor contribution classifier (positive=1 / negative=0)
- Computes MACCS (167 bits) + MW + logP + TPSA + MolarRefractivity
- Trains RF / GBDT / MLP
- Evaluates on train/test
- Saves processed dataset, splits, all models, performance summary
- NEW: Selects the best model by Test F1 (tie-break: Acc -> Prec -> Rec), prints it,
       and saves best_model.joblib + best_model_<NAME>.joblib + best_model_summary.json
"""

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MACCSkeys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score,
    precision_score, recall_score
)
    # note: seaborn/matplotlib not used here
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# how to use: cmd: python train_and_save_binary.py --csv contribution_data.csv

# --------------------------
# Data processing
# --------------------------
class DataProcessor:
    def __init__(self, file_path, smiles_col, label_col):
        self.file_path = file_path
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.df = pd.read_csv(file_path)

    def compute_descriptors(self, smiles):
        """
        Return 167 MACCS bits + [MolWeight, logP, TPSA, MolarRefractivity] (total 171 features).
        If SMILES invalid/missing, return zeros.
        Note: RDKit's Descriptors.MolMR is molar refractivity (not dipole moment).
        """
        if isinstance(smiles, str) and smiles.strip():
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                maccs_bits = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]  # 167 bits
                mol_weight = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                molar_refractivity = Descriptors.MolMR(mol)
                return maccs_bits + [mol_weight, logp, tpsa, molar_refractivity]
        return [0] * 167 + [0, 0, 0, 0]

    def load_and_process_data(self):
        print(f"[INFO] Loading data from: {self.file_path}")
        df = self.df.copy()

        if self.smiles_col not in df.columns:
            raise ValueError(f"Missing SMILES column: '{self.smiles_col}'")
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column: '{self.label_col}'")

        # ensure 0/1 integers
        df[self.label_col] = df[self.label_col].astype(int)

        # class counts
        c1 = int((df[self.label_col] == 1).sum())
        c0 = int((df[self.label_col] == 0).sum())
        print(f"[INFO] Class counts -> 1: {c1}, 0: {c0}")

        # descriptors
        print("[INFO] Computing molecular descriptors (MACCS+physchem)...")
        df["_desc_vec"] = df[self.smiles_col].apply(self.compute_descriptors)
        desc_df = pd.DataFrame(df["_desc_vec"].tolist(), index=df.index)

        # feature names
        maccs_features = [f"MACCS_{i}" for i in range(1, 168)]  # 1..167
        other_features = ["MolWeight", "LogP", "TPSA", "MolarRefractivity"]
        feature_names = maccs_features + other_features
        desc_df.columns = feature_names

        # merge
        df = pd.concat([df.drop(columns=["_desc_vec"]), desc_df], axis=1)

        return df, feature_names

    def split_and_save(self, df, feature_cols, outdir, test_size=0.2, random_state=42):
        X = df[feature_cols]
        y = df[self.label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        train_df = pd.concat([X_train.reset_index(drop=True),
                              y_train.reset_index(drop=True).rename(self.label_col)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True),
                             y_test.reset_index(drop=True).rename(self.label_col)], axis=1)

        outdir.mkdir(parents=True, exist_ok=True)
        train_path = outdir / "train_data_binary.csv"
        test_path = outdir / "test_data_binary.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[INFO] Train split saved: {train_path}")
        print(f"[INFO] Test split saved : {test_path}")

        return X_train, X_test, y_train, y_test


# --------------------------
# Modeling
# --------------------------
class ModelTrainer:
    def train_random_forest(self, X_train, y_train, seed=100):
        model = RandomForestClassifier(
            n_estimators=100, max_depth=25, min_samples_split=5,
            min_samples_leaf=1, random_state=seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_gbdt(self, X_train, y_train, seed=42):
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed
        )
        model.fit(X_train, y_train)
        return model

    def train_mlp(self, X_train, y_train, seed=42):
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=300,
            activation='relu', solver='adam', random_state=seed
        )
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X, y, tag):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        print(f"\n[{tag}] Confusion Matrix:\n{cm}")
        print(f"[{tag}] Classification Report:\n{classification_report(y, y_pred)}")

        return dict(
            f1=f1_score(y, y_pred),
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred)
        )


# --------------------------
# Utils
# --------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Train binary flavor contribution classifier and save models & reports."
    )
    ap.add_argument("--csv", required=True, help="Path to input CSV (must include SMILES and label columns).")
    ap.add_argument("--smiles_col", default="SMILES", help="SMILES column name (default: SMILES).")
    ap.add_argument("--label_col", default="Classification",
                    help="Label column name (0/1) (default: Classification).")
    ap.add_argument("--outdir", default="model_out_binary",
                    help="Output directory (default: model_out_binary).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split ratio (default: 0.2).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return ap


def save_artifacts(outdir: Path, df_processed: pd.DataFrame, feature_cols, reports, models):
    # processed dataset
    processed_path = outdir / "processed_dataset.csv"
    df_processed.to_csv(processed_path, index=False)
    print(f"[INFO] Processed dataset saved: {processed_path}")

    # feature meta
    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "notes": "Features = 167 MACCS bits + MolWeight + LogP + TPSA + MolarRefractivity"
    }
    (outdir / "feature_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Feature meta saved: {outdir / 'feature_meta.json'}")

    # reports
    perf_df = pd.DataFrame(reports)
    perf_csv = outdir / "performance_summary.csv"
    perf_df.to_csv(perf_csv, index=False)
    print(f"[INFO] Performance summary saved: {perf_csv}")

    # all models
    for name, model in models.items():
        path = outdir / f"{name}_model.joblib"
        joblib.dump(model, path)
        print(f"[INFO] Saved model: {path}")


def rank_and_pick_best(reports: list, models: dict):
    """
    Pick best by Test F1, then Test Acc, then Test Prec, then Test Rec (all descending).
    """
    def key_fn(r):
        return (r["Test F1"], r["Test Acc"], r["Test Prec"], r["Test Rec"])
    sorted_reports = sorted(reports, key=key_fn, reverse=True)
    best_info = sorted_reports[0]
    best_name = best_info["Model"]
    best_model = models[best_name]
    return best_name, best_model, best_info, sorted_reports


def save_best(outdir: Path, best_name: str, best_model, best_info: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    # save model with tag and a canonical name
    tagged = outdir / f"best_model_{best_name}.joblib"
    generic = outdir / "best_model.joblib"
    joblib.dump(best_model, tagged)
    joblib.dump(best_model, generic)

    # save summary
    summary = {
        "best_model": best_name,
        "criteria": "Test F1 (tie-break: Test Acc -> Test Prec -> Test Rec)",
        "metrics": best_info
    }
    (outdir / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Best Model Selected ===")
    print(f"Best model: {best_name}")
    print(f"Test F1={best_info['Test F1']:.4f}, "
          f"Acc={best_info['Test Acc']:.4f}, "
          f"Prec={best_info['Test Prec']:.4f}, "
          f"Rec={best_info['Test Rec']:.4f}")
    print(f"Saved: {tagged}")
    print(f"Saved: {generic}")
    print(f"Summary: {outdir / 'best_model_summary.json'}")


def interactive_predict(dp: DataProcessor, feature_cols, models: dict):
    try:
        raw = input("\nEnter SMILES (comma-separated) to predict [or press Enter to skip]: ").strip()
    except Exception:
        return
    if not raw:
        return
    smiles_list = [s.strip() for s in raw.split(",") if s.strip()]
    if not smiles_list:
        return

    desc_rows = [dp.compute_descriptors(smi) for smi in smiles_list]
    X_new = pd.DataFrame(desc_rows, columns=feature_cols)

    preds = {name: mdl.predict(X_new) for name, mdl in models.items()}
    for i, smi in enumerate(smiles_list):
        line = " | ".join([f"{name}:{int(preds[name][i])}" for name in models.keys()])
        print(f"SMILES: {smi} -> {line}")


def main():
    ap = build_argparser()
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    dp = DataProcessor(args.csv, args.smiles_col, args.label_col)
    df_proc, feature_cols = dp.load_and_process_data()
    X_train, X_test, y_train, y_test = dp.split_and_save(
        df_proc, feature_cols, outdir, test_size=args.test_size, random_state=args.seed
    )

    # Train
    mt = ModelTrainer()
    rf = mt.train_random_forest(X_train, y_train, seed=args.seed)
    gbdt = mt.train_gbdt(X_train, y_train, seed=args.seed)
    mlp = mt.train_mlp(X_train, y_train, seed=args.seed)

    # Evaluate
    reports = []
    for name, model in [("RandomForest", rf), ("GBDT", gbdt), ("MLP", mlp)]:
        tr = mt.evaluate(model, X_train, y_train, f"{name} - Train")
        te = mt.evaluate(model, X_test, y_test, f"{name} - Test")
        reports.append({
            "Model": name,
            "Train F1": tr["f1"], "Test F1": te["f1"],
            "Train Acc": tr["accuracy"], "Test Acc": te["accuracy"],
            "Train Prec": tr["precision"], "Test Prec": te["precision"],
            "Train Rec": tr["recall"], "Test Rec": te["recall"],
        })

    # Save all artifacts
    models = {"RandomForest": rf, "GBDT": gbdt, "MLP": mlp}
    save_artifacts(outdir, df_proc, feature_cols, reports, models)

    # Rank and pick best model
    best_name, best_model, best_info, sorted_reports = rank_and_pick_best(reports, models)
    # Save best model and summary
    save_best(outdir, best_name, best_model, best_info)

    # Optional: show ranking table
    print("\n=== Model Ranking (by Test F1, then Acc, Prec, Rec) ===")
    print(pd.DataFrame(sorted_reports))

    # Optional interactive prediction
    interactive_predict(dp, feature_cols, models)


if __name__ == "__main__":
    main()
