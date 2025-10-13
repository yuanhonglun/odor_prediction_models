# -*- coding: utf-8 -*-
"""
Predict binary flavor contribution (1=positive, 0=negative) using the best saved model.
- Loads best_model.joblib and feature_meta.json from training outputs
- Accepts a chemical name (via PubChem) or a SMILES string
- Outputs predicted contribution (0/1) and positive class probability
- Supports both CLI and a Tkinter GUI (copy-friendly outputs)
"""

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MACCSkeys

# GUI
import tkinter as tk
from tkinter import messagebox, filedialog, ttk


# --------------------------
# IO & feature preparation
# --------------------------
def load_resources(model_path: str, feature_meta_path: str):
    """Load trained model and feature column names from feature_meta.json."""
    model = joblib.load(model_path)
    meta = json.loads(Path(feature_meta_path).read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    return model, feature_cols


def name_to_smiles_via_pubchem(name: str):
    """Resolve a chemical name to SMILES via PubChem (pip install pubchempy)."""
    try:
        import pubchempy as pcp
    except Exception:
        raise RuntimeError("Missing dependency: pubchempy is required to search by name. Run: pip install pubchempy")

    try:
        hits = pcp.get_compounds(name, "name")
        if not hits:
            return None
        smi = hits[0].canonical_smiles or hits[0].isomeric_smiles
        return smi
    except Exception:
        return None


def smiles_to_feature_vector_aligned(smiles: str, feature_cols):
    """
    Compute descriptors and align them to training feature order.
    Training features were: 167 MACCS bits + [MolWeight, LogP, TPSA, MolarRefractivity]
    Feature names: MACCS_1..MACCS_167, MolWeight, LogP, TPSA, MolarRefractivity
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # MACCS: 167 bits (keep all bits, order consistent with training)
    bitstr = MACCSkeys.GenMACCSKeys(mol).ToBitString()  # "0101..."
    maccs_bits = [int(b) for b in bitstr]               # len=167

    # Physchem descriptors
    mol_weight = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    molar_refractivity = Descriptors.MolMR(mol)

    # Build a name→value map, then align to feature_cols
    fmap = {f"MACCS_{i+1}": maccs_bits[i] for i in range(167)}
    fmap.update({
        "MolWeight": mol_weight,
        "LogP": logp,
        "TPSA": tpsa,
        "MolarRefractivity": molar_refractivity
    })

    vec = [fmap.get(col, 0) for col in feature_cols]
    return np.array(vec, dtype=float).reshape(1, -1)


def predict_label_and_proba(model, X_row):
    """Return (pred_label, positive_probability) for one row."""
    y_pred = int(model.predict(X_row)[0])
    pos_prob = None
    if hasattr(model, "predict_proba"):
        try:
            classes = list(model.classes_)
            if 1 in classes:
                pos_idx = classes.index(1)
                pos_prob = float(model.predict_proba(X_row)[0, pos_idx])
            else:
                # fallback if class order is unexpected
                pos_prob = float(model.predict_proba(X_row)[0, -1])
        except Exception:
            pos_prob = None
    return y_pred, pos_prob


# --------------------------
# CLI
# --------------------------
def cli_main():
    ap = argparse.ArgumentParser(
        description="Predict binary flavor contribution using a saved best model."
    )
    ap.add_argument("--model", help="Path to best model (.joblib). Defaults to ./model_out_binary/best_model.joblib")
    ap.add_argument("--feature_meta", help="Path to feature_meta.json. Defaults to ./model_out_binary/feature_meta.json")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smiles", help="SMILES string")
    g.add_argument("--name", help="Chemical name (resolved to SMILES via PubChem)")
    args = ap.parse_args()

    # If no input provided, launch the GUI
    if not (args.smiles or args.name):
        gui_main(args.model, args.feature_meta)
        return

    # Defaults if not provided
    model_path = args.model or str(Path("model_out_binary") / "best_model.joblib")
    meta_path = args.feature_meta or str(Path("model_out_binary") / "feature_meta.json")

    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not Path(meta_path).exists():
        raise SystemExit(f"Feature meta not found: {meta_path}")

    model, feature_cols = load_resources(model_path, meta_path)

    # Resolve input
    if args.name:
        smi = name_to_smiles_via_pubchem(args.name)
        if not smi:
            raise SystemExit(f"Not found on PubChem by name: {args.name}")
        smiles = smi
        shown_input = args.name
    else:
        smiles = args.smiles
        shown_input = args.smiles

    X = smiles_to_feature_vector_aligned(smiles, feature_cols)
    y_pred, pos_prob = predict_label_and_proba(model, X)

    print("\n=== Contribution Prediction ===")
    print(f"Input : {shown_input}")
    print(f"SMILES: {smiles}")
    print(f"Predicted contribution: {y_pred}  (1=positive, 0=negative)")
    if pos_prob is not None:
        print(f"Positive probability : {pos_prob:.4f}")
    print("")


# --------------------------
# GUI
# --------------------------
class PredictApp(tk.Tk):
    def __init__(self, model_path=None, feature_meta_path=None):
        super().__init__()
        self.title("Contribution Prediction")
        self.geometry("960x560")
        self.resizable(True, True)
        self.minsize(720, 420)

        self.model_path = tk.StringVar(value=model_path or str(Path("model_out_binary") / "best_model.joblib"))
        self.meta_path = tk.StringVar(value=feature_meta_path or str(Path("model_out_binary") / "feature_meta.json"))

        self.input_mode = tk.StringVar(value="name")  # "name" or "smiles"
        self.input_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")

        self.model = None
        self.feature_cols = None

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        # Model files
        frm_files = ttk.LabelFrame(self, text="Model Files")
        frm_files.pack(fill="x", expand=False, **pad)

        ttk.Label(frm_files, text="Model (.joblib):").grid(row=0, column=0, sticky="e", **pad)
        ent_model = ttk.Entry(frm_files, textvariable=self.model_path)
        ent_model.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm_files, text="Browse...", command=self.browse_model).grid(row=0, column=2, **pad)

        ttk.Label(frm_files, text="Feature meta (feature_meta.json):").grid(row=1, column=0, sticky="e", **pad)
        ent_meta = ttk.Entry(frm_files, textvariable=self.meta_path)
        ent_meta.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm_files, text="Browse...", command=self.browse_meta).grid(row=1, column=2, **pad)

        frm_files.columnconfigure(1, weight=1)

        # Input
        frm_in = ttk.LabelFrame(self, text="Input")
        frm_in.pack(fill="x", expand=False, **pad)

        ttk.Radiobutton(frm_in, text="Name", value="name", variable=self.input_mode).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_in, text="SMILES", value="smiles", variable=self.input_mode).grid(row=0, column=1, sticky="w", **pad)

        ent_input = ttk.Entry(frm_in, textvariable=self.input_text)
        ent_input.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)

        frm_in.columnconfigure(0, weight=1)
        frm_in.columnconfigure(1, weight=1)
        frm_in.columnconfigure(2, weight=1)

        # Buttons
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", expand=False, **pad)
        ttk.Button(frm_btn, text="Predict", command=self.on_predict).pack(side="left", padx=8)
        ttk.Button(frm_btn, text="Exit", command=self.destroy).pack(side="right", padx=8)

        # Output (copy-friendly)
        frm_out = ttk.LabelFrame(self, text="Output (copy-friendly)")
        frm_out.pack(fill="both", expand=True, **pad)

        self.var_input = tk.StringVar(value="-")
        self.var_smiles = tk.StringVar(value="-")
        self.var_label = tk.StringVar(value="-")  # 0/1
        self.var_prob = tk.StringVar(value="-")   # probability for positive class

        ttk.Label(frm_out, text="Input:").grid(row=0, column=0, sticky="e", **pad)
        ent_in = ttk.Entry(frm_out, textvariable=self.var_input, state="readonly")
        ent_in.grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="SMILES:").grid(row=1, column=0, sticky="e", **pad)
        ent_sm = ttk.Entry(frm_out, textvariable=self.var_smiles, state="readonly")
        ent_sm.grid(row=1, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="Predicted contribution (1=positive, 0=negative):").grid(row=2, column=0, sticky="e", **pad)
        ent_lab = ttk.Entry(frm_out, textvariable=self.var_label, state="readonly")
        ent_lab.grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="Positive probability:").grid(row=3, column=0, sticky="e", **pad)
        ent_pb = ttk.Entry(frm_out, textvariable=self.var_prob, state="readonly")
        ent_pb.grid(row=3, column=1, sticky="ew", **pad)

        frm_out.columnconfigure(1, weight=1)

        # Copy & status
        frm_copy = ttk.Frame(self)
        frm_copy.pack(fill="x", expand=False, **pad)
        ttk.Button(frm_copy, text="Copy All", command=self.copy_all).pack(side="left", padx=8)
        ttk.Label(frm_copy, textvariable=self.status_text).pack(side="right", padx=8)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model file (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def browse_meta(self):
        path = filedialog.askopenfilename(
            title="Select feature meta (feature_meta.json)",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.meta_path.set(path)

    def ensure_loaded(self):
        if self.model is not None and self.feature_cols is not None:
            return
        mpath = self.model_path.get().strip()
        fpath = self.meta_path.get().strip()
        if not mpath or not fpath:
            raise RuntimeError("Please select both model file and feature meta file.")
        if not Path(mpath).exists():
            raise RuntimeError(f"Model file not found: {mpath}")
        if not Path(fpath).exists():
            raise RuntimeError(f"Feature meta file not found: {fpath}")
        self.model, self.feature_cols = load_resources(mpath, fpath)

    def on_predict(self):
        try:
            self.ensure_loaded()

            text = self.input_text.get().strip()
            if not text:
                messagebox.showerror("Input Error", "Please enter a chemical name or SMILES.")
                return

            if self.input_mode.get() == "name":
                smiles = name_to_smiles_via_pubchem(text)
                if not smiles:
                    messagebox.showerror("Not Found", f"No result on PubChem for: {text}")
                    return
                shown_input = text
            else:
                smiles = text
                shown_input = text

            X = smiles_to_feature_vector_aligned(smiles, self.feature_cols)
            y_pred, pos_prob = predict_label_and_proba(self.model, X)

            self.var_input.set(shown_input)
            self.var_smiles.set(smiles)
            self.var_label.set(str(y_pred))
            self.var_prob.set(f"{pos_prob:.6f}" if pos_prob is not None else "-")

            self.status_text.set("Prediction done. You can copy the results.")

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except RuntimeError as re:
            messagebox.showerror("Runtime Error", str(re))
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}")

    def copy_all(self):
        text = (
            f"Input: {self.var_input.get()}\n"
            f"SMILES: {self.var_smiles.get()}\n"
            f"Predicted contribution (1=positive, 0=negative): {self.var_label.get()}\n"
            f"Positive probability: {self.var_prob.get()}\n"
        )
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_text.set("Copied to clipboard.")
        except Exception:
            self.status_text.set("Copy failed.")


def gui_main(model_path=None, feature_meta_path=None):
    app = PredictApp(model_path, feature_meta_path)
    app.mainloop()


# Entry
if __name__ == "__main__":
    cli_main()
