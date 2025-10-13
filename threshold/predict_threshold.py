# -*- coding: utf-8 -*-
"""
Predict -log10(ODT) and ODT (mg/L) using the best saved threshold model.
- Auto-loads ./model_out_threshold/best_model.joblib and ./model_out_threshold/maccs_cols.json
  (if missing, GUI asks you to browse)
- Supports Name or SMILES input
- GUI is resizable and copy-friendly; -log10(ODT) shows no unit; ODT is in mg/L
- Optional unit conversion: if your training target was mg/kg, convert to mg/L with density rho
"""

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

import json
import math
import argparse
import joblib
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import MACCSkeys

# GUI
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# ------------ Loading & features ------------
def load_resources(model_path: str, cols_path: str):
    model = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        maccs_cols = json.load(f)
    return model, maccs_cols

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
        return hits[0].canonical_smiles or hits[0].isomeric_smiles
    except Exception:
        return None

def smiles_to_aligned_features(smiles: str, maccs_cols):
    """
    Convert SMILES to MACCS features aligned to training columns.
    RDKit MACCS is 167 bits. The original training code used fp.ToBitString()[1:] (drop the first bit -> 166 bits).
    We follow the same convention, then pick values according to maccs_cols (missing -> 0).
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    bitstr = fp.ToBitString()[1:]   # 166 bits to match training convention
    full = {f"MACCS_{i+1}": int(b) for i, b in enumerate(bitstr)}  # names: MACCS_1..MACCS_166
    vec = [full.get(col, 0) for col in maccs_cols]
    return np.array(vec, dtype=int).reshape(1, -1)

def convert_neglog_to_mgL(pred_neglog: float, train_unit: str = "mgkg", rho: float = 1.0):
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

# ------------ CLI ------------
def cli_main():
    ap = argparse.ArgumentParser(description="Predict -log10(ODT) and ODT (mg/L) using the best saved model.")
    # default paths under ./model_out_threshold
    ap.add_argument("--model", help="Path to best model (.joblib). Defaults to ./model_out_threshold/best_model.joblib")
    ap.add_argument("--maccs_cols", help="Path to maccs_cols.json. Defaults to ./model_out_threshold/maccs_cols.json")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smiles", help="SMILES string")
    g.add_argument("--name", help="Chemical name (resolved to SMILES via PubChem)")
    ap.add_argument("--train_unit", choices=["mgkg", "mgL"], default="mgkg",
                    help="Unit of the training target (default: mgkg)")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="Medium density in kg/L for mg/kg→mg/L conversion (default: 1.0)")
    args = ap.parse_args()

    # If no input provided, open the GUI (it will try auto-load; otherwise ask to browse)
    if not (args.smiles or args.name):
        gui_main(args.model, args.maccs_cols, args.train_unit, args.rho)
        return

    # Resolve default paths
    model_path = args.model or str(Path("model_out_threshold") / "best_model.joblib")
    cols_path  = args.maccs_cols or str(Path("model_out_threshold") / "maccs_cols.json")

    if not Path(model_path).exists() or not Path(cols_path).exists():
        raise SystemExit(
            "Model or feature file not found.\n"
            f"Checked:\n  {model_path}\n  {cols_path}\n"
            "Run GUI (no args) to browse files, or supply --model and --maccs_cols explicitly."
        )

    model, maccs_cols = load_resources(model_path, cols_path)

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

    X = smiles_to_aligned_features(smiles, maccs_cols)
    pred_neglog = float(model.predict(X)[0])
    y_mgL, odt_mgL = convert_neglog_to_mgL(pred_neglog, args.train_unit, args.rho)

    # Copy-friendly CLI output (no unit for -log10(ODT))
    print("\n=== Threshold Prediction ===")
    print(f"Input : {shown_input}")
    print(f"SMILES: {smiles}")
    print(f"-log10(ODT): {y_mgL:.4f}")
    print(f"ODT (mg/L):  {odt_mgL:.6g}\n")

# ------------ GUI ------------
class PredictApp(tk.Tk):
    def __init__(self, model_path=None, maccs_cols_path=None, train_unit="mgkg", rho=1.0):
        super().__init__()
        self.title("Threshold Prediction")
        self.geometry("960x560")
        self.resizable(True, True)
        self.minsize(720, 420)

        # Defaults to ./model_out_threshold/...
        default_model = Path("model_out_threshold") / "best_model.joblib"
        default_cols  = Path("model_out_threshold") / "maccs_cols.json"

        self.model_path = tk.StringVar(value=str(model_path) if model_path else str(default_model))
        self.cols_path  = tk.StringVar(value=str(maccs_cols_path) if maccs_cols_path else str(default_cols))
        self.input_mode = tk.StringVar(value="name")  # "name" or "smiles"
        self.input_text = tk.StringVar()
        self.train_unit = tk.StringVar(value=train_unit)  # "mgkg" / "mgL"
        self.rho = tk.DoubleVar(value=rho)
        self.status_text = tk.StringVar(value="Ready")

        self.model = None
        self.maccs_cols = None

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

        ttk.Label(frm_files, text="Features (maccs_cols.json):").grid(row=1, column=0, sticky="e", **pad)
        ent_cols = ttk.Entry(frm_files, textvariable=self.cols_path)
        ent_cols.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm_files, text="Browse...", command=self.browse_cols).grid(row=1, column=2, **pad)

        frm_files.columnconfigure(1, weight=1)

        # Unit & density
        frm_unit = ttk.LabelFrame(self, text="Unit Settings")
        frm_unit.pack(fill="x", expand=False, **pad)

        ttk.Label(frm_unit, text="Training target unit:").grid(row=0, column=0, sticky="e", **pad)
        ttk.Radiobutton(frm_unit, text="mg/kg", value="mgkg", variable=self.train_unit).grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm_unit, text="mg/L",  value="mgL",  variable=self.train_unit).grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(frm_unit, text="Density ρ (kg/L):").grid(row=0, column=3, sticky="e", **pad)
        ttk.Entry(frm_unit, textvariable=self.rho, width=10).grid(row=0, column=4, sticky="w", **pad)
        ttk.Label(frm_unit, text="(used for mg/kg→mg/L; water≈1.0)").grid(row=0, column=5, sticky="w", **pad)

        frm_unit.columnconfigure(1, weight=1)
        frm_unit.columnconfigure(2, weight=1)

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
        self.var_ylog = tk.StringVar(value="-")  # -log10(ODT), no unit
        self.var_odt = tk.StringVar(value="-")   # ODT (mg/L)

        ttk.Label(frm_out, text="Input:").grid(row=0, column=0, sticky="e", **pad)
        ent_in = ttk.Entry(frm_out, textvariable=self.var_input, state="readonly")
        ent_in.grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="SMILES:").grid(row=1, column=0, sticky="e", **pad)
        ent_sm = ttk.Entry(frm_out, textvariable=self.var_smiles, state="readonly")
        ent_sm.grid(row=1, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="-log10(ODT):").grid(row=2, column=0, sticky="e", **pad)
        ent_y = ttk.Entry(frm_out, textvariable=self.var_ylog, state="readonly")
        ent_y.grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(frm_out, text="ODT (mg/L):").grid(row=3, column=0, sticky="e", **pad)
        ent_o = ttk.Entry(frm_out, textvariable=self.var_odt, state="readonly")
        ent_o.grid(row=3, column=1, sticky="ew", **pad)

        frm_out.columnconfigure(1, weight=1)

        # Copy & status
        frm_copy = ttk.Frame(self)
        frm_copy.pack(fill="x", expand=False, **pad)
        ttk.Button(frm_copy, text="Copy All", command=self.copy_all).pack(side="left", padx=8)
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(frm_copy, textvariable=self.status_text).pack(side="right", padx=8)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model file (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def browse_cols(self):
        path = filedialog.askopenfilename(
            title="Select feature file (maccs_cols.json)",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.cols_path.set(path)

    def ensure_loaded(self):
        # Try auto-load; if missing, ask to browse
        mpath = Path(self.model_path.get().strip())
        cpath = Path(self.cols_path.get().strip())
        if not mpath.exists() or not cpath.exists():
            raise RuntimeError(
                "Model or feature file not found. Please click 'Browse...' to select "
                "best_model.joblib and maccs_cols.json."
            )
        self.model, self.maccs_cols = load_resources(str(mpath), str(cpath))

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

            X = smiles_to_aligned_features(smiles, self.maccs_cols)
            pred_neglog = float(self.model.predict(X)[0])

            # Convert to mg/L if needed
            tu = self.train_unit.get()
            rho_val = float(self.rho.get()) if self.rho.get() else 1.0
            y_mgL, odt_mgL = convert_neglog_to_mgL(pred_neglog, tu, rho_val)

            # Update copy-friendly fields
            self.var_input.set(shown_input)
            self.var_smiles.set(smiles)
            self.var_ylog.set(f"{y_mgL:.4f}")    # no unit
            self.var_odt.set(f"{odt_mgL:.6g}")   # mg/L
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
            f"-log10(ODT): {self.var_ylog.get()}\n"
            f"ODT (mg/L): {self.var_odt.get()}\n"
        )
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_text.set("Copied to clipboard.")
        except Exception:
            self.status_text.set("Copy failed.")

def gui_main(model_path=None, maccs_cols_path=None, train_unit="mgkg", rho=1.0):
    app = PredictApp(model_path, maccs_cols_path, train_unit, rho)
    app.mainloop()

# Entry
if __name__ == "__main__":
    cli_main()
