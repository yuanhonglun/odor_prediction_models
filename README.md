# Odor Prediction Models
**Aroma contribution (classification) + Odor threshold (regression)**

This repository hosts two pipelines:
1) **Aroma contribution** classification (multiclass + binary) from molecular structure.
2) **Odor threshold (ODT)** regression predicting `-log10(ODT)` and `ODT (mg/L)`.

Large datasets, trained models, and full result artifacts should be archived on a generalist repository (e.g., Zenodo) for permanence.  
**Zenodo record (example):** `10.5281/zenodo.17559514` (replace with your final DOI if different).

---

## 📁 Suggested Layout
```
odor_prediction_models/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ data/                                 # small demo slices; full sets via Zenodo
│  ├─ merged_classified_multi.csv
│  ├─ merged_classified_binary.csv
│  └─ threshold_data.csv
├─ src/
│  ├─ flavor/                            # classification
│  │  ├─ name_to_smiles.py
│  │  ├─ classify_descriptors.py
│  │  ├─ train_flavor_models.py          # multiclass training
│  │  ├─ validate_models.py              # validate saved FP×Model combos
│  │  └─ app_predict_gui.py              # binary prediction (GUI/CLI)
│  └─ threshold/                         # ODT regression
│     ├─ train_threshold_odt.py
│     └─ predict_threshold.py
└─ outputs/
   ├─ model_out_all/                     # multiclass artifacts (recommended outdir)
   ├─ model_out_binary_all/              # binary artifacts (your existing folder name)
   └─ model_out_threshold/               # regression artifacts
```
> Keep **code small** on GitHub. Put heavy data/models/results on Zenodo and cite the DOI in your manuscript’s Data Availability Statement.

---

## 🛠️ Environment

### Option A (Recommended) — Conda (RDKit via conda-forge)
```bash
conda env create -f environment.yml
conda activate odor
```

### Option B — Pure pip (RDKit not included)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Install RDKit separately (see RDKit official docs)
```

> **Optional (GCN)**: If you plan to run the GCN baseline in `train_flavor_models.py` (PyTorch + PyG),
install PyTorch (CPU/CUDA per your machine) and `torch_geometric` following their official instructions.

---

## 📦 Data & Models
- `data/merged_classified_multi.csv` — multiclass dataset (SMILES + label column, default: `Major_Category`)
- `data/merged_classified_binary.csv` — binary dataset for contribution (`Contribution` column, e.g., `pos/neg` or 1/0)
- `data/threshold_data.csv` — ODT dataset (SMILES + `threshold` column, positive numeric)

> Put **full** datasets and trained models on **Zenodo** (e.g., `10.5281/zenodo.17559514`).  
> Download to `data/` or `outputs/` as needed.

---

## 🚀 Workflows

### 0) Name → SMILES (batch, with cache-resume)
```bash
python src/flavor/name_to_smiles.py \
  -i data/your_names.csv \
  -o data/your_names_with_smiles.csv \
  --name-col Name --smiles-col SMILES --resume
```

### 1) Descriptor-based rule classifier (multi/binary)
```bash
# multi-class descriptors → categories
python src/flavor/classify_descriptors.py -i data/raw.csv -o data/multi_labeled.csv --mode multi

# binary contribution (Pos/Neg)
python src/flavor/classify_descriptors.py -i data/raw.csv -o data/binary_labeled.csv --mode binary
```

### 2) Classification — Train (multiclass; scaffold split + HPO + CV report)
```bash
python src/flavor/train_flavor_models.py \
  --csv data/merged_classified_multi.csv \
  --outdir outputs/model_out_all \
  --val_size 0.20 \
  --seed 42 \
  --fingerprints ECFP4,ECFP6,MACCS \
  --models rf,gbdt,mlp \
  --nbits 1024 \
  --tune random --n_iter 30 \
  --cv_folds 5 --cv_group_scaffold \
  --imbalance_strategy class_weight \
  --resume \
  --majority_label AUTO --majority_target_n 5000
```
**Notes**
- Imbalance handling: `--imbalance_strategy` ∈ `{none, class_weight, oversample}`
- GCN baseline (optional): add `gcn` to `--models` (install PyTorch + PyG first)
- Exports (in `--outdir`): CV reports (per-fold & mean±std), confusion matrices (counts/normalized, SVG/PDF),
  metrics tables, deduplicated dataset, and best-model artifacts for sklearn models.

### 3) Classification — Validate saved FP×Model combos
```bash
# strictly use the validation split recorded under outdir
python src/flavor/validate_models.py \
  --outdir outputs/model_out_all \
  --only MACCS-RF \
  --subset val

# evaluate multiple combos
python src/flavor/validate_models.py \
  --outdir outputs/model_out_all \
  --only MACCS-RF,ECFP4-MLP,ECFP6-GBDT \
  --subset auto

# evaluate on a custom CSV (subset=all)
python src/flavor/validate_models.py \
  --csv data/merged_classified_multi.csv \
  --outdir outputs/model_out_all \
  --only ECFP6-GBDT \
  --subset all
```
**Exports**
- Per-class metrics table (precision/recall/F1/support), confusion matrices (SVG/PDF), and a metrics bar chart.

### 4) Classification — Binary prediction (GUI/CLI)
```bash
# GUI (no text input → GUI opens)
python src/flavor/app_predict_gui.py --model outputs/model_out_binary_all/best_model.joblib

# CLI by SMILES
python src/flavor/app_predict_gui.py --model outputs/model_out_binary_all/best_model.joblib --smiles "CCO"

# CLI by name (requires pubchempy)
python src/flavor/app_predict_gui.py --model outputs/model_out_binary_all/best_model.joblib --name "linalool"
```

### 5) ODT Regression — Train (scaffold split + HPO + CV report)
```bash
python src/threshold/train_threshold_odt.py \
  --csv data/threshold_data.csv \
  --outdir outputs/model_out_threshold \
  --val_size 0.20 \
  --seed 42 \
  --fingerprints ECFP4,ECFP6,MACCS \
  --nbits 1024 \
  --models rf,gbdt,mlp \
  --tune random --n_iter 30 \
  --cv_folds 5 --cv_group_scaffold
```
**Exports (in `--outdir`)**
- `performance_summary_val.csv` (R²/RMSE ranking across FP×Model)  
- CV report CSVs (fold-wise & summary: mean ± std)  
- `val_pred_*.csv`, `val_metrics_*.json`, band-wise eval JSON (tertiles)  
- **`best_model.joblib`** + `best_model_summary.json` (contains feature column order etc.)

### 6) ODT Regression — Predict (CLI/GUI; supports name→SMILES)
```bash
# CLI with SMILES; training unit mg/L
python src/threshold/predict_threshold.py \
  --model outputs/model_out_threshold/best_model.joblib \
  --smiles "CCO" \
  --train_unit mgL

# Show packaged model meta
python src/threshold/predict_threshold.py \
  --model outputs/model_out_threshold/best_model.joblib \
  --show_info
```
If neither `--smiles` nor `--name` is provided, a small GUI pops up.

---

## 🔁 Reproducibility & Archival
- Use **Zenodo** (DOI) to deposit **data + code + models + results** for long-term access.
- Cite the DOI in your paper’s Data Availability Statement.
- Prefer vector outputs (SVG/PDF) from validation scripts for figures.

---

## 📜 License
Add a permissive license (MIT or Apache-2.0) in `LICENSE`.
