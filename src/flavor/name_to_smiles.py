import argparse
import json
import os
import time
from typing import Optional, Dict

import pandas as pd

try:
    import pubchempy as pcp
except Exception as e:
    raise SystemExit("Please install pubchempy first: pip install pubchempy") from e


def lookup_pubchem_smiles(name: str, sleep: float = 0.12) -> Optional[str]:
    """
    Query SMILES from PubChem using a compound name.
    - Try get_compounds(name, 'name'); if empty, try get_cids(...) then Compound.from_cid(...)
    - Prefer 'smiles' attribute; fallback to 'canonical_smiles'
    """
    if not isinstance(name, str):
        return None
    q = name.strip()
    if not q:
        return None

    try:
        comps = pcp.get_compounds(q, namespace="name")
        time.sleep(sleep)
        if comps:
            comp = comps[0]
            smi = getattr(comp, "smiles", None) or getattr(comp, "canonical_smiles", None)
            if smi:
                return smi

        cids = pcp.get_cids(q, namespace="name")
        time.sleep(sleep)
        if cids:
            comp = pcp.Compound.from_cid(cids[0])
            smi = getattr(comp, "smiles", None) or getattr(comp, "canonical_smiles", None)
            if smi:
                return smi
    except Exception:
        # Network/rate-limit/temporary failure: return None for later retry or manual handling
        return None

    return None


def main():
    ap = argparse.ArgumentParser(description="Batch query PubChem SMILES from a name column and write back to CSV.")
    ap.add_argument("-i", "--input", required=True, help="Input CSV path, e.g., flavor_net.csv")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: <input>_with_smiles.csv)")
    ap.add_argument("--name-col", default="Name", help="Column name for compound names (default: Name)")
    ap.add_argument("--smiles-col", default="SMILES", help="Output column name for SMILES (default: SMILES)")
    ap.add_argument("--sleep", type=float, default=0.12, help="Delay (sec) between queries (default: 0.12)")
    ap.add_argument("--resume", action="store_true", help="Enable cache-based resume (default: off)")
    ap.add_argument("--cache", default=".pubchem_name2smiles_cache.json", help="Cache file path (default: .pubchem_name2smiles_cache.json)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.name_col not in df.columns:
        raise SystemExit(f"Name column not found: {args.name_col}. Use --name-col to set it. Existing columns: {list(df.columns)}")

    out_path = args.output or os.path.splitext(args.input)[0] + "_with_smiles.csv"

    # Load cache (optional)
    cache: Dict[str, str] = {}
    if args.resume and os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache, "r", encoding="utf-8"))
            print(f"[INFO] Loaded cache: {args.cache} ({len(cache)} entries)")
        except Exception:
            print("[WARN] Failed to read cache; will ignore.")

    smiles_list = []
    unresolved = []

    total = len(df)
    for i, name in enumerate(df[args.name_col].astype(str).tolist()):
        key = name.strip()
        smi = cache.get(key)

        if not smi:
            smi = lookup_pubchem_smiles(key, sleep=args.sleep)
            if smi:
                cache[key] = smi

        smiles_list.append(smi)
        if not smi:
            unresolved.append({"row_index": i, "name": name})

        if (i + 1) % 25 == 0 or i == total - 1:
            print(f"[PROGRESS] {i + 1}/{total} processed; unresolved {len(unresolved)}")

        # Periodic cache flush
        if (i + 1) % 100 == 0:
            try:
                json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            except Exception:
                pass

    df[args.smiles_col] = smiles_list
    df.to_csv(out_path, index=False)
    print(f"[DONE] Written: {out_path}")

    if unresolved:
        unres_path = "unresolved_names.csv"
        pd.DataFrame(unresolved).to_csv(unres_path, index=False)
        print(f"[WARN] {len(unresolved)} names unresolved; written to: {unres_path}")

    # Final cache save
    try:
        json.dump(cache, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
