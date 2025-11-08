# -*- coding: utf-8 -*-
"""
Predict -log10(ODT) and ODT (mg/L) using the *new-format* best_model.joblib.

Works with the training artifact produced by train_threshold_odt.py:
joblib.dump({
    "fingerprint_kind": "ECFP4" | "ECFP6" | "MACCS",
    "nbits": 1024,
    "physchem_used": ["MolWeight","LogP","TPSA","MolarRefractivity"],
    "sk_model": <sklearn Pipeline>,
    "feature_columns": [...],
    "build_info": {"xgboost_used": bool}
}, "best_model.joblib")

- Supports input by --smiles or --name (via PubChem; optional dependency).
- No need for maccs_cols.json anymore; feature alignment uses feature_columns in joblib.
- Optional unit conversion: if training target was mg/kg, convert to mg/L with density rho.

Usage (CLI):
  python predict_threshold_new.py --model model_out_threshold_med/best_model.joblib \
      --smiles "CCO" --train_unit mgL

If you don't pass --smiles/--name, a small GUI will open.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import joblib
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys

# Silence RDKit logs
RDLogger.DisableLog('rdApp.*')

# -------------------- Helpers --------------------
def canonical_smiles(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

def name_to_smiles_via_pubchem(name: str) -> str | None:
    """Resolve a chemical name to SMILES via PubChem (pip install pubchempy)."""
    try:
        import pubchempy as pcp
    except Exception:
        raise RuntimeError("Missing dependency: install 'pubchempy' to resolve chemical names (pip install pubchempy).")
    try:
        hits = pcp.get_compounds(name, "name")
        if not hits:
            return None
        return hits[0].canonical_smiles or hits[0].isomeric_smiles
    except Exception:
        return None

def fp_morgan_bits(mol: Chem.Mol, radius: int, nbits: int) -> np.ndarray:
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def fp_maccs_bits(mol: Chem.Mol) -> np.ndarray:
    bv = GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def physchem(mol: Chem.Mol) -> dict[str, float]:
    from rdkit.Chem import Descriptors, Crippen
    return {
        "MolWeight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def features_from_smiles_aligned(smi: str, pkg: dict) -> tuple[np.ndarray, str]:
    """
    Build a 1×D feature vector aligned to pkg['feature_columns'].
    Only the fingerprint kind used by the best model is computed.
    Missing columns (should not happen) are filled with 0.
    """
    cano = canonical_smiles(smi)
    if not cano:
        raise ValueError(f"Invalid SMILES: {smi}")

    mol = Chem.MolFromSmiles(cano)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    kind = str(pkg.get("fingerprint_kind", "")).upper()
    nbits = int(pkg.get("nbits", 1024))
    cols: list[str] = list(pkg["feature_columns"])

    # Compute the required fingerprint
    feats: dict[str, float] = {}

    if kind in ("ECFP4", "ECFP6"):
        radius = 2 if kind == "ECFP4" else 3
        bits = fp_morgan_bits(mol, radius=radius, nbits=nbits)
        for i, v in enumerate(bits):
            feats[f"{kind}_{i}"] = float(v)
    elif kind == "MACCS":
        bits = fp_maccs_bits(mol)  # length 167
        for i, v in enumerate(bits):
            feats[f"MACCS_{i}"] = float(v)
    else:
        raise RuntimeError(f"Unsupported fingerprint_kind in model package: {kind}")

    # Physchem (fixed 4 keys)
    p = physchem(mol)
    feats.update({k: float(p.get(k, 0.0)) for k in ("MolWeight", "LogP", "TPSA", "MolarRefractivity")})

    # Align to training column order
    vec = np.array([feats.get(c, 0.0) for c in cols], dtype=np.float32).reshape(1, -1)
    return vec, cano

def convert_neglog_to_mgL(pred_neglog: float, train_unit: str = "mgkg", rho: float = 1.0) -> tuple[float, float]:
    """
    Convert model output -log10(ODT) to mg/L.
    If training target was mg/kg: y_mgL = y_mgkg - log10(rho), rho in kg/L.
    If training target was mg/L: no conversion.
    Returns: (y_mgL, odt_mgL)
    """
    if train_unit.lower() == "mgkg":
        y_mgL = pred_neglog - math.log10(rho)
    else:
        y_mgL = pred_neglog
    odt_mgL = 10 ** (-y_mgL)
    return y_mgL, odt_mgL

# -------------------- CLI --------------------
def cli_main():
    ap = argparse.ArgumentParser(description="Predict -log10(ODT) & ODT (mg/L) with new-format best_model.joblib")
    ap.add_argument("--model", default=str(Path("model_out_threshold_med")/"best_model.joblib"),
                    help="Path to best_model.joblib (default: model_out_threshold_med/best_model.joblib)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smiles", help="SMILES string")
    g.add_argument("--name", help="Chemical name (resolved via PubChem)")
    ap.add_argument("--train_unit", choices=["mgkg", "mgL"], default="mgkg",
                    help="Training target unit (default: mgkg)")
    ap.add_argument("--rho", type=float, default=1.0, help="Density ρ in kg/L (for mg/kg→mg/L; default 1.0)")
    ap.add_argument("--show_info", action="store_true", help="Print model meta info and exit")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    pkg: dict = joblib.load(model_path)
    model = pkg["sk_model"]

    if args.show_info:
        print("\n=== Model Info ===")
        print(f"fingerprint_kind : {pkg.get('fingerprint_kind')}")
        print(f"nbits            : {pkg.get('nbits')}")
        print(f"n_features       : {len(pkg.get('feature_columns', []))}")
        print(f"xgboost_used     : {pkg.get('build_info', {}).get('xgboost_used')}")
        print(f"physchem_used    : {pkg.get('physchem_used')}\n")
        return

    # If no textual input provided, open GUI
    if not (args.smiles or args.name):
        try:
            gui_main(str(model_path), args.train_unit, args.rho)
            return
        except Exception as e:
            raise SystemExit(f"GUI failed to start: {e}")

    if args.name:
        smi = name_to_smiles_via_pubchem(args.name)
        if not smi:
            raise SystemExit(f"Not found on PubChem by name: {args.name}")
        shown_input = args.name
        smiles = smi
    else:
        smiles = args.smiles
        shown_input = args.smiles

    X, cano = features_from_smiles_aligned(smiles, pkg)
    pred_neglog = float(model.predict(X)[0])
    y_mgL, odt_mgL = convert_neglog_to_mgL(pred_neglog, args.train_unit, args.rho)

    print("\n=== Threshold Prediction ===")
    print(f"Input : {shown_input}")
    print(f"SMILES: {cano}")
    print(f"-log10(ODT): {y_mgL:.4f}")
    print(f"ODT (mg/L):  {odt_mgL:.6g}\n")

# -------------------- Minimal GUI (optional) --------------------
def gui_main(model_path: str, train_unit: str = "mgkg", rho: float = 1.0):
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Threshold Prediction (new model package)")
            self.geometry("860x480")
            self.resizable(True, True)

            self.model_path = tk.StringVar(value=model_path)
            self.input_mode = tk.StringVar(value="name")   # "name" or "smiles"
            self.input_text = tk.StringVar()
            self.train_unit = tk.StringVar(value=train_unit)  # "mgkg"/"mgL"
            self.rho = tk.DoubleVar(value=rho)

            self.var_input = tk.StringVar(value="-")
            self.var_smiles = tk.StringVar(value="-")
            self.var_ylog = tk.StringVar(value="-")
            self.var_odt = tk.StringVar(value="-")
            self.status_text = tk.StringVar(value="Ready")

            self.pkg = None
            self.model = None
            self._build()

        def _build(self):
            pad = {'padx': 8, 'pady': 6}

            frm_files = ttk.LabelFrame(self, text="Model File")
            frm_files.pack(fill="x", **pad)
            ttk.Label(frm_files, text="best_model.joblib:").grid(row=0, column=0, sticky="e", **pad)
            ttk.Entry(frm_files, textvariable=self.model_path).grid(row=0, column=1, sticky="ew", **pad)
            ttk.Button(frm_files, text="Browse...", command=self.browse_model).grid(row=0, column=2, **pad)
            frm_files.columnconfigure(1, weight=1)

            frm_unit = ttk.LabelFrame(self, text="Unit Settings")
            frm_unit.pack(fill="x", **pad)
            ttk.Label(frm_unit, text="Training target unit:").grid(row=0, column=0, sticky="e", **pad)
            ttk.Radiobutton(frm_unit, text="mg/kg", value="mgkg", variable=self.train_unit).grid(row=0, column=1, sticky="w", **pad)
            ttk.Radiobutton(frm_unit, text="mg/L",  value="mgL",  variable=self.train_unit).grid(row=0, column=2, sticky="w", **pad)
            ttk.Label(frm_unit, text="Density ρ (kg/L):").grid(row=0, column=3, sticky="e", **pad)
            ttk.Entry(frm_unit, textvariable=self.rho, width=10).grid(row=0, column=4, sticky="w", **pad)

            frm_in = ttk.LabelFrame(self, text="Input")
            frm_in.pack(fill="x", **pad)
            ttk.Radiobutton(frm_in, text="Name", value="name", variable=self.input_mode).grid(row=0, column=0, sticky="w", **pad)
            ttk.Radiobutton(frm_in, text="SMILES", value="smiles", variable=self.input_mode).grid(row=0, column=1, sticky="w", **pad)
            ttk.Entry(frm_in, textvariable=self.input_text).grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
            frm_in.columnconfigure(0, weight=1)

            frm_btn = ttk.Frame(self); frm_btn.pack(fill="x", **pad)
            ttk.Button(frm_btn, text="Predict", command=self.on_predict).pack(side="left", padx=8)
            ttk.Button(frm_btn, text="Model Info", command=self.on_info).pack(side="left", padx=8)
            ttk.Button(frm_btn, text="Exit", command=self.destroy).pack(side="right", padx=8)

            frm_out = ttk.LabelFrame(self, text="Output")
            frm_out.pack(fill="both", expand=True, **pad)
            ttk.Label(frm_out, text="Input:").grid(row=0, column=0, sticky="e", **pad)
            ttk.Entry(frm_out, textvariable=self.var_input, state="readonly").grid(row=0, column=1, sticky="ew", **pad)
            ttk.Label(frm_out, text="SMILES:").grid(row=1, column=0, sticky="e", **pad)
            ttk.Entry(frm_out, textvariable=self.var_smiles, state="readonly").grid(row=1, column=1, sticky="ew", **pad)
            ttk.Label(frm_out, text="-log10(ODT):").grid(row=2, column=0, sticky="e", **pad)
            ttk.Entry(frm_out, textvariable=self.var_ylog, state="readonly").grid(row=2, column=1, sticky="ew", **pad)
            ttk.Label(frm_out, text="ODT (mg/L):").grid(row=3, column=0, sticky="e", **pad)
            ttk.Entry(frm_out, textvariable=self.var_odt, state="readonly").grid(row=3, column=1, sticky="ew", **pad)
            frm_out.columnconfigure(1, weight=1)

            frm_copy = ttk.Frame(self); frm_copy.pack(fill="x", **pad)
            ttk.Button(frm_copy, text="Copy All", command=self.copy_all).pack(side="left", padx=8)
            ttk.Label(frm_copy, textvariable=self.status_text).pack(side="right", padx=8)

        def browse_model(self):
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title="Select best_model.joblib",
                filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")]
            )
            if path:
                self.model_path.set(path)

        def ensure_loaded(self):
            mpath = Path(self.model_path.get().strip())
            if not mpath.exists():
                raise RuntimeError("Model file not found. Please select best_model.joblib.")
            self.pkg = joblib.load(mpath)
            self.model = self.pkg["sk_model"]

        def on_info(self):
            try:
                self.ensure_loaded()
                info = (
                    f"fingerprint_kind : {self.pkg.get('fingerprint_kind')}\n"
                    f"nbits            : {self.pkg.get('nbits')}\n"
                    f"n_features       : {len(self.pkg.get('feature_columns', []))}\n"
                    f"xgboost_used     : {self.pkg.get('build_info', {}).get('xgboost_used')}\n"
                    f"physchem_used    : {self.pkg.get('physchem_used')}"
                )
                messagebox.showinfo("Model Info", info)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def on_predict(self):
            try:
                self.ensure_loaded()
                text = self.input_text.get().strip()
                if not text:
                    messagebox.showerror("Input Error", "Please enter a chemical name or SMILES.")
                    return
                if self.input_mode.get() == "name":
                    smi = name_to_smiles_via_pubchem(text)
                    if not smi:
                        messagebox.showerror("Not Found", f"No result on PubChem for: {text}")
                        return
                    shown_input = text
                else:
                    smi = text
                    shown_input = text

                X, cano = features_from_smiles_aligned(smi, self.pkg)
                pred_neglog = float(self.model.predict(X)[0])

                tu = self.train_unit.get()
                rho_val = float(self.rho.get()) if self.rho.get() else 1.0
                y_mgL, odt_mgL = convert_neglog_to_mgL(pred_neglog, tu, rho_val)

                self.var_input.set(shown_input)
                self.var_smiles.set(cano)
                self.var_ylog.set(f"{y_mgL:.4f}")
                self.var_odt.set(f"{odt_mgL:.6g}")
                self.status_text.set("Prediction done.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def copy_all(self):
            text = (
                f"Input: {self.var_input.get()}\n"
                f"SMILES: {self.var_smiles.get()}\n"
                f"-log10(ODT): {self.var_ylog.get()}\n"
                f"ODT (mg/L): {self.var_odt.get()}\n"
            )
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                self.status_text.set("Copied.")
            except Exception:
                self.status_text.set("Copy failed.")

    App().mainloop()

# -------------------- Entry --------------------
if __name__ == "__main__":
    cli_main()
