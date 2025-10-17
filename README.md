# Odor Prediction Models

Two small tools for odor modeling:

- **Threshold regression** (predict `-log10(ODT)` and `ODT (mg/L)`):
  - Train: `threshold/train_and_save_threshold.py`
  - Predict (GUI/CLI): `threshold/predict_threshold.py`

- **Binary contribution classification** (`1=positive`, `0=negative`):
  - Train: `contribution/train_and_save_binary.py`
  - Predict (GUI/CLI): `contribution/predict_contribution.py`

> Both predictors have simple GUIs and also support command-line usage.

---

## Repo layout

odor_prediction_models/
- threshold/
  - train_and_save_threshold.py
  - predict_threshold.py
  - model_out_threshold/ # saved best model & feature columns
  - best_model.joblib
  - maccs_cols.json
  - training_report.json
- contribution/
  - train_and_save_binary.py
  - predict_contribution.py
  - model_out_binary/ # saved best classification model & feature meta
  - best_model.joblib
  - feature_meta.json
  - best_model_summary.json
- data/
  - threshold_data.csv
  - contribution_data.csv
- README.md



---

## Requirements

- Python 3.8+
- Install packages:
  ```bash
  pip install rdkit-pypi scikit-learn pandas numpy joblib pubchempy
pubchempy is only needed when predicting by Name (it resolves Name → SMILES via PubChem).

Train
1) Threshold regression
Input CSV must contain:

target column: -log#odt

MACCS_* feature columns: any column whose name starts with MACCS

Run:


python threshold/train_and_save_threshold.py data/threshold_data.csv
This will create:


threshold/model_out_threshold/
  -  best_model.joblib
  -  maccs_cols.json
  -  training_report.json
2) Contribution classification
Input CSV must contain:

SMILES (structure)

Classification (label 0/1)

Run:


python contribution/train_and_save_binary.py --csv data/contribution_data.csv
This will create:


contribution/model_out_binary/
  -  best_model.joblib
  -  feature_meta.json
  -  best_model_summary.json
Predict (GUI)
Threshold

python threshold/predict_threshold.py
Auto-loads ./threshold/model_out_threshold/best_model.joblib and maccs_cols.json if present.

If your training labels were mg/kg, set density ρ (kg/L) to convert to mg/L output (water ≈ 1.0).

Contribution

python contribution/predict_contribution.py
Auto-loads ./contribution/model_out_binary/best_model.joblib and feature_meta.json.

Input Name (resolved via PubChem) or SMILES; the GUI shows the predicted label (1/0) and positive probability.

Predict (CLI examples)
Threshold

# by name
python threshold/predict_threshold.py --name "vanillin"

# by SMILES
python threshold/predict_threshold.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
Contribution

# by name
python contribution/predict_contribution.py --name "vanillin"

# by SMILES
python contribution/predict_contribution.py --smiles "COC1=CC=C(C=O)C=C1O"