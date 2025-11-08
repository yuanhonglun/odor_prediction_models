# -*- coding: utf-8 -*-
"""
Predict binary flavor contribution (1=positive, 0=negative) using a saved model (NEW format only).

This version supports only the packaged `best_model.joblib` produced by the new training script.
Expected keys inside that file:
  - "sk_model": sklearn estimator or pipeline (must support predict / predict_proba)
  - "class_names": list[str] (labels used during training; in binary typically ["neg","pos"] or similar)
  - "fingerprint_kind": "ECFP4" | "ECFP6" | "MACCS"
  - "nbits": int (ECFP bits, e.g., 1024/2048). Ignored for MACCS.
  - "physchem_used": ordered subset of ["MolWeight","LogP","TPSA","MolarRefractivity"]
  - "extra_physchem_included": extra fields included during training (e.g., ["HenrysConstant","Solubility"]).
        These cannot be computed here and will be filled with 0.0 as placeholders.

GUI & CLI are both supported. Chemical names are resolved to SMILES via PubChem (install: pip install pubchempy).

Usage
-----
# 1) GUI (double-click or run with no --smiles/--name):
python app_predict_gui.py --model path/to/best_model.joblib

# 2) CLI with SMILES:
python app_predict_gui.py --model path/to/best_model.joblib --smiles "CCO"

# 3) CLI with chemical name (PubChem required):
python app_predict_gui.py --model path/to/best_model.joblib --name "linalool"
"""
from __future__ import annotations
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MACCSkeys, rdMolDescriptors
from rdkit import DataStructs

# GUI
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# --------------------------
# Utilities
# --------------------------
def _is_pkg_model(obj: Any) -> bool:
    """True if obj looks like the new packaged dict saved as best_model.joblib."""
    required = {"sk_model", "fingerprint_kind", "class_names"}
    return isinstance(obj, dict) and required.issubset(set(obj.keys()))

def _find_positive_index(class_names: List[str]) -> int:
    """
    Find the index of the positive class among `class_names` (strings).
    We prefer aliases like 'pos','positive','1','yes','true' (case-insensitive).
    Fallbacks are designed for binary tasks.
    """
    pos_keys = {"pos","positive","1","yes","true"}
    neg_keys = {"neg","negative","0","no","false"}
    lc = [str(x).strip().lower() for x in class_names]
    for i, s in enumerate(lc):
        if s in pos_keys:
            return i
    if len(lc) == 2:
        if lc[0] in neg_keys and lc[1] not in neg_keys:
            return 1
        if lc[1] in neg_keys and lc[0] not in neg_keys:
            return 0
        return 1
    return len(class_names) - 1

# --------------------------
# IO & feature preparation
# --------------------------
def load_packaged_model(model_path: str) -> Dict[str, Any]:
    """
    Load the packaged best_model.joblib and return a dict with:
      - model, class_names, fingerprint_kind, nbits, physchem_used, extra_physchem_included
    """
    pkg = joblib.load(model_path)
    if not _is_pkg_model(pkg):
        raise RuntimeError("This file is not a packaged best_model.joblib produced by the new training script.")
    model = pkg["sk_model"]
    class_names = list(pkg.get("class_names", []))
    fp_kind = str(pkg.get("fingerprint_kind", "MACCS")).upper()
    if fp_kind not in {"MACCS","ECFP4","ECFP6"}:
        raise RuntimeError(f"Unsupported fingerprint_kind in package: {fp_kind}")
    nbits = int(pkg.get("nbits", 1024))
    physchem_used = list(pkg.get("physchem_used", ["MolWeight","LogP","TPSA","MolarRefractivity"]))
    extra_phys = list(pkg.get("extra_physchem_included", []))
    return {
        "model": model,
        "class_names": class_names,
        "fingerprint_kind": fp_kind,
        "nbits": nbits,
        "physchem_used": physchem_used,
        "extra_physchem_included": extra_phys,
    }

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

def _physchem_from_mol(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MolWeight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def _fp_bits_from_mol(mol: Chem.Mol, fp_kind: str, nbits: int) -> np.ndarray:
    if fp_kind == "MACCS":
        bv = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.int8); DataStructs.ConvertToNumpyArray(bv, arr)
        return arr.astype(np.float32, copy=False)
    radius = 2 if fp_kind == "ECFP4" else 3
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=int(nbits))
    arr = np.zeros((int(nbits),), dtype=np.int8); DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32, copy=False)

def smiles_to_feature_row(smiles: str, fp_kind: str, nbits: int,
                          physchem_used: List[str],
                          extra_physchem_included: List[str]) -> np.ndarray:
    """
    Build a single-row feature vector aligned to training order:
      [FP bits] + [physchem_used in order] + [extra_physchem_included (zeros)]
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = _fp_bits_from_mol(mol, fp_kind, nbits)
    phys_map = _physchem_from_mol(mol)
    phys_vals = [float(phys_map.get(k, 0.0)) for k in physchem_used]
    extras_vals = [0.0 for _ in extra_physchem_included]  # cannot compute here → zeros
    vec = np.concatenate([fp.astype(np.float32, copy=False),
                          np.asarray(phys_vals, dtype=np.float32),
                          np.asarray(extras_vals, dtype=np.float32)], axis=0)
    return vec.reshape(1, -1)

def predict_label_and_proba(model, X_row: np.ndarray, class_names: List[str]) -> Tuple[int, Optional[float]]:
    """Return (pred_label_int, positive_probability) for one row."""
    y_pred = model.predict(X_row)
    pred_int = int(y_pred[0]) if np.ndim(y_pred) else int(y_pred)
    pos_idx = None
    try:
        # If it's a pipeline, try to fetch final estimator
        clf = getattr(model, "named_steps", {}).get("clf", model)
        model_classes = getattr(clf, "classes_", None)
        if model_classes is not None:
            model_classes_str = [str(x) for x in model_classes]
            pos_idx = _find_positive_index(model_classes_str)
    except Exception:
        pos_idx = None
    if pos_idx is None:
        pos_idx = _find_positive_index([str(x) for x in class_names])
    pos_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_row)
            pos_prob = float(probs[0, pos_idx])
        except Exception:
            pos_prob = None
    return pred_int, pos_prob

# --------------------------
# CLI
# --------------------------
def cli_main():
    ap = argparse.ArgumentParser(description="Predict binary flavor contribution using a packaged best_model.joblib (GUI if no input)")
    ap.add_argument("--model", help="Path to best_model.joblib", default=str(Path('model_out_binary')/'best_model.joblib'))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smiles", help="SMILES string")
    g.add_argument("--name", help="Chemical name (resolved to SMILES via PubChem)")
    args = ap.parse_args()

    # If no input provided, launch the GUI
    if not (args.smiles or args.name):
        gui_main(args.model)
        return

    model_path = args.model
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")

    res = load_packaged_model(model_path)

    # Resolve input
    if args.name:
        smi = name_to_smiles_via_pubchem(args.name)
        if not smi:
            raise SystemExit(f"Not found on PubChem by name: {args.name}")
        smiles = smi; shown_input = args.name
    else:
        smiles = args.smiles; shown_input = args.smiles

    X = smiles_to_feature_row(smiles, res["fingerprint_kind"], res["nbits"], res["physchem_used"], res["extra_physchem_included"])
    y_pred, pos_prob = predict_label_and_proba(res["model"], X, res["class_names"])

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
    def __init__(self, model_path=None):
        super().__init__()
        self.title("Contribution Prediction")
        self.geometry("900x520")
        self.resizable(True, True)
        self.minsize(700, 420)

        self.model_path = tk.StringVar(value=model_path or str(Path("model_out_binary") / "best_model.joblib"))
        self.input_mode = tk.StringVar(value="name")  # "name" or "smiles"
        self.input_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")

        self.model_res: Optional[Dict[str,Any]] = None  # cached package

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        # Model file
        frm_files = ttk.LabelFrame(self, text="Model File (best_model.joblib)")
        frm_files.pack(fill="x", expand=False, **pad)

        ttk.Label(frm_files, text="Model (.joblib):").grid(row=0, column=0, sticky="e", **pad)
        ent_model = ttk.Entry(frm_files, textvariable=self.model_path)
        ent_model.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm_files, text="Browse...", command=self.browse_model).grid(row=0, column=2, **pad)

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
        self.var_fp = tk.StringVar(value="-")     # fingerprint kind
        self.var_nbits = tk.StringVar(value="-")  # nbits display
        self.var_classes = tk.StringVar(value="-")# class names

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

        ttk.Label(frm_out, text="Model FP kind:").grid(row=4, column=0, sticky="e", **pad)
        ent_fp = ttk.Entry(frm_out, textvariable=self.var_fp, state="readonly")
        ent_fp.grid(row=4, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="ECFP bits:").grid(row=5, column=0, sticky="e", **pad)
        ent_nb = ttk.Entry(frm_out, textvariable=self.var_nbits, state="readonly")
        ent_nb.grid(row=5, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="Class names:").grid(row=6, column=0, sticky="e", **pad)
        ent_cls = ttk.Entry(frm_out, textvariable=self.var_classes, state="readonly")
        ent_cls.grid(row=6, column=1, sticky="ew", **pad)

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
            self.model_res = None  # reset cache

    def ensure_loaded(self):
        if self.model_res is not None:
            return
        mpath = self.model_path.get().strip()
        if not mpath:
            raise RuntimeError("Please select a model file (.joblib).")
        if not Path(mpath).exists():
            raise RuntimeError(f"Model file not found: {mpath}")
        self.model_res = load_packaged_model(mpath)

        # Display model meta in UI
        self.var_fp.set(self.model_res["fingerprint_kind"])
        self.var_nbits.set(str(self.model_res["nbits"]) if self.model_res["fingerprint_kind"].startswith("ECFP") else "-")
        self.var_classes.set(", ".join([str(x) for x in self.model_res["class_names"]]) or "-")

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

            X = smiles_to_feature_row(
                smiles,
                self.model_res["fingerprint_kind"],
                self.model_res["nbits"],
                self.model_res["physchem_used"],
                self.model_res["extra_physchem_included"],
            )
            y_pred, pos_prob = predict_label_and_proba(self.model_res["model"], X, self.model_res["class_names"])

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
            f"Model FP kind: {self.var_fp.get()}\n"
            f"ECFP bits: {self.var_nbits.get()}\n"
            f"Class names: {self.var_classes.get()}\n"
        )
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_text.set("Copied to clipboard.")
        except Exception:
            self.status_text.set("Copy failed.")

def gui_main(model_path=None):
    app = PredictApp(model_path)
    app.mainloop()

# Entry
if __name__ == "__main__":
    cli_main()
