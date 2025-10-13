# -*- coding: utf-8 -*-
"""
Train ODT regression models and save the BEST model for threshold prediction.
- Input: CSV with target column '-log#odt' and feature columns starting with 'MACCS'
- Models: RandomForestRegressor / GradientBoostingRegressor / MLPRegressor
- Select best by highest Test R² (tie-break: smaller Test RMSE)
- Save ONLY: model_out_threshold/best_model.joblib
- Also save: model_out_threshold/maccs_cols.json, model_out_threshold/training_report.json
"""

import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# how to use: cmd: python train_and_save_threshold.py threshold_data.csv --outdir model_out_threshold

RANDOM_SEED = 100

def data_processor(file_path: str):
    """Load CSV, extract MACCS_* features and target '-log#odt'."""
    df = pd.read_csv(file_path)
    if '-log#odt' not in df.columns:
        raise ValueError("Target column '-log#odt' is missing in the dataset.")

    # drop rows with missing target
    df = df.dropna(subset=['-log#odt'])

    # features: any column starting with 'MACCS'
    maccs_cols = [c for c in df.columns if c.startswith('MACCS')]
    if not maccs_cols:
        raise ValueError("No feature columns starting with 'MACCS' were found.")

    X = df[maccs_cols].fillna(0).astype(int)
    y = df['-log#odt'].astype(float)
    return X, y, maccs_cols

def train_and_select_best(X, y):
    """Train 3 regressors, evaluate on train/test, and select the best model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=10, max_depth=11, min_samples_leaf=2, random_state=RANDOM_SEED
        ),
        "GBDT": GradientBoostingRegressor(
            n_estimators=50, max_depth=5, loss='squared_error', random_state=RANDOM_SEED
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(20,), max_iter=200, learning_rate_init=0.03, random_state=RANDOM_SEED
        )
    }

    report = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        ytr = model.predict(X_train)
        yte = model.predict(X_test)

        train_r2 = float(r2_score(y_train, ytr))
        test_r2  = float(r2_score(y_test, yte))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, ytr)))
        test_rmse  = float(np.sqrt(mean_squared_error(y_test, yte)))

        report.append({
            "Model": name,
            "Train R2": train_r2,
            "Test R2":  test_r2,
            "Train RMSE": train_rmse,
            "Test RMSE":  test_rmse,
            "model_obj": model
        })

    # Select best by Test R²; tie-break by smaller Test RMSE
    report_sorted = sorted(report, key=lambda d: (d["Test R2"], -d["Test RMSE"]), reverse=True)
    best = report_sorted[0]
    return best, report_sorted

def main():
    ap = argparse.ArgumentParser(description="Train ODT regressors and save the best model.")
    ap.add_argument("csv", help="Path to training CSV (must include '-log#odt' and MACCS_* columns)")
    ap.add_argument("--outdir", default="model_out_threshold", help="Output directory (default: model_out_threshold)")
    args = ap.parse_args()

    X, y, maccs_cols = data_processor(args.csv)
    best, report_sorted = train_and_select_best(X, y)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save ONLY the best model as best_model.joblib
    best_model_path = outdir / "best_model.joblib"
    joblib.dump(best["model_obj"], best_model_path)

    # Save feature column names for alignment at prediction time
    cols_path = outdir / "maccs_cols.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(maccs_cols, f, ensure_ascii=False, indent=2)

    # Save a lightweight training report
    meta_path = outdir / "training_report.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in d.items() if k != "model_obj"} for d in report_sorted],
            f, ensure_ascii=False, indent=2
        )

    # Console summary
    print("\n=== Training complete ===")
    print(f"Best model selected: {best['Model']}")
    print(f"Test R² = {best['Test R2']:.4f}, Test RMSE = {best['Test RMSE']:.4f}")
    print(f"Saved best model : {best_model_path}")
    print(f"Saved feature cols: {cols_path}")
    print(f"Saved report     : {meta_path}")

if __name__ == "__main__":
    main()
