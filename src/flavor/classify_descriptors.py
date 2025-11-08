# -*- coding: utf-8 -*-
r"""
Flavor-descriptor classifier (built-in rules; simple CLI)

Modes:
- multi  : multi-class assignment (after category merging)
- binary : contribution classification ("Pos"/"Neg") via mapping table

Outputs:
- Always:
    * All_Matched_Categories  (unique normalized categories, '|' joined)
    * Major_Category          (final normalized class; "Odorless" only if no other match)
- In binary mode:
    * Contribution            ("Pos"/"Neg"/"NA")

Default output file name:
    <input>_classified_multi.csv   (mode=multi)
    <input>_classified_binary.csv  (mode=binary)
"""

import re
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Pattern
import pandas as pd

# -----------------------------
# Built-in category rules (English only)
# -----------------------------
BASE_RULES: Dict[str, List[str]] = {
    "Fruity": [
        "fruit","fruity","apple","pear","banana","berry","strawberry","raspberry","blueberry","blackberry",
        "peach","apricot","plum","cherry","grape","raisin","pineapple","mango","papaya","guava","passion",
        "lychee","longan","melon","watermelon","cantaloupe","coconut","tropical","stone fruit",
        "peachy","bananay","berry-like","appley","grapey","blackcurrant"
    ],
    "Citrus": [
        "citrus","lemon","lime","orange","grapefruit","mandarin","tangerine","yuzu","bergamot",
        "citrusy","lemony","orangey","limey","grapefruity"
    ],
    "Floral": [
        "floral","flower","flowery","rose","violet","jasmine","lily","geranium","honeysuckle",
        "lavender","osmanthus","neroli","ylang","rosy","gardenia"
    ],
    "Green/Herbal": [
        "green","grassy","grass","herb","herbal","herbaceous","leaf","leafy","fresh cut grass","fresh","tea","hay","straw",
        "basil","parsley","cilantro","coriander leaf","thyme","rosemary","sage",
        "cucumber","bell pepper","green pepper","tomato leaf","pepper leaf","beany","bean-like","pea","pea-like",
        "legume","leguminous","celery","celery-like","alliaceous"
    ],
    "Mint/Cool": [
        "mint","menthol","mentholic","peppermint","spearmint","cooling","eucalyptus",
        "camphor","camphorous","camphoraceous","turpentine","minty"
    ],
    "Woody/Resinous": [
        "woody","wood","cedar","pine","oak","sandalwood","resin","resinous","balsamic","incense","parchment"
    ],
    "Spicy": [
        "spice","spicy","pepper","black pepper","white pepper","peppery","clove","clovey","cinnamon","cinnamony",
        "ginger","gingery","nutmeg","allspice","anise","anisic","licorice","cardamom","pungent",
        "chili","chilli","capsicum","hot"
    ],
    "Roasted/Nutty/Toasty": [
        "roasted","roast","toasty","toast","nut","nutty","hazelnut","almond","walnut","peanut",
        "coffee","mocha","cocoa","chocolate","chocolatey","chocolaty","malty","baked",
        "bread","bready","biscuity","cereal","grain","grainy","oat","oaty","burnt sugar"
    ],
    "Sweet/Vanilla/Caramel": [
        "sweet","sweetish","vanilla","vanillic","caramel","caramelized","caramellic","toffee","honey","sugar",
        "maple","syrup","cotton candy","candy","confection","popcorn","butterscotch","marshmallow","savory"
    ],
    "Dairy/Buttery": [
        "butter","buttery","dairy","creamy","cream","cheese","cheesy","milk","milky","yogurt","yogurty","buttermilk"
    ],
    "Fatty/Lipidic": [
        "fat","fatty","oily","oil","wax","waxy","alkane","alkanes","paraffin","paraffinic","hydrocarbon","aliphatic",
        "lipid","lipidic","greasy","tallow","lard","lardy","soap","soapy","oleic","palmitic","stearic","lanolin","sebum",
        "wool","wool-like","wooly"
    ],
    "Fermented/Yeasty/Alcoholic": [
        "fermented","yeast","yeasty","beer","winey","wine","alcohol","alcoholic","ethereal","estery","sourdough",
        "pickled","brined","blackcurrant"
    ],
    "Acidic/Sour/Vinegar": [
        "acidic","sour","tart","vinegar","vinegary","acetic","butyric","isovaleric","propionic"
    ],
    "Bitter": [
        "bitter","bitterish"
    ],
    "Sulfur/Allium": [
        "sulfur","sulphur","garlic","onion","leek","chive","cabbage","radish","skunky","thiol","mercaptan","sulfide","allium",
        "cooked cabbage","cooked broccoli","sulfurous","sulfury"
    ],
    "Earthy/Musty": [
        "earthy","soil","earth","musty","mold","mould","mushroom","geosmin","beet","rooty","potato","petrichor","clay","clayey","dusty"
    ],
    "Smoky/Burnt": [
        "smoky","smoke","smokey","burnt","char","charred","ash","ashy","bacon","phenolic","tar","smoldering","soot","smokiness","tobacco"
    ],
    "Marine/Fishy": [
        "fishy","fish","marine","seaweed","oceanic","sea","shellfish","iodine","amine","trimethylamine","seafood"
    ],
    "Meaty": [
        "meaty","meat","beefy","beef"
    ],
    "Metallic": [
        "metal","metallic","metal-like"
    ],
    "Chemical/Solvent/Plastic": [
        "chemical","solvent","paint","glue","plastic","plasticine","rubber","medicinal","hospital","pharmaceutical","adhesive",
        "varnish","gasoline","diesel","petroleum","band-aid","bandaid","antiseptic","thinner",
        "camphoreous","ammoniacal","ether","phenol","lactone"
    ],
    "Rancid/Sweaty/Animalic": [
        "rancid","sweaty","sweat","fecal","faecal","goaty","barnyard","animal","urine","skatole","indolic","indole","dirty",
        "leather","leathery","musky","musk"
    ],
}

ODORLESS_KEYWORDS: List[str] = [
    "odorless","odourless","no odor","no odour","no smell","none detected","bland","no aroma"
]

# -----------------------------
# Category merging and priority
# -----------------------------
MERGE_MAP: Dict[str, str] = {
    "Smoky/Burnt": "Roasted/Nutty/Toasty",
    "Dairy/Buttery": "Sweet/Vanilla/Caramel",
    "Metallic": "Off-flavor",
    "Acidic/Sour/Vinegar": "Off-flavor",
    "Marine/Fishy": "Off-flavor",
    "Meaty": "Off-flavor",
    "Chemical/Solvent/Plastic": "Off-flavor",
    "Earthy/Musty": "Off-flavor",
    "Rancid/Sweaty/Animalic": "Off-flavor",
    "Fermented/Yeasty/Alcoholic": "Off-flavor",
}

PRIORITY_NORM: List[str] = [
    "Fruity","Citrus","Floral","Green/Herbal","Mint/Cool","Woody/Resinous",
    "Spicy","Roasted/Nutty/Toasty","Sweet/Vanilla/Caramel",
    "Fatty/Lipidic","Sulfur/Allium","Bitter","Off-flavor"
]

BINARY_MAP: Dict[str, str] = {
    "Sulfur/Allium": "Neg",
    "Sweet/Vanilla/Caramel": "Pos",
    "Off-flavor": "Neg",
    "Green/Herbal": "Neg",
    "Spicy": "Neg",
    "Woody/Resinous": "Neg",
    "Floral": "Pos",
    "Mint/Cool": "Pos",
    "Roasted/Nutty/Toasty": "Pos",
    "Fruity": "Pos",
    "Fatty/Lipidic": "Neg",
    "Citrus": "Pos",
    "Odorless": "Neg",
    "Bitter": "Neg",
}

# -----------------------------
# Regex helpers
# -----------------------------
def _kw_pattern(kw: str) -> Pattern:
    """Builds a case-insensitive whole-word regex for alphabetic keywords."""
    if re.search(r"[A-Za-z]", kw):
        return re.compile(r"(?<![A-Za-z])" + re.escape(kw) + r"(?![A-Za-z])", re.I)
    return re.compile(re.escape(kw), re.I)

CATEGORY_PATTERNS = {cat: [_kw_pattern(k) for k in kws] for cat, kws in BASE_RULES.items()}
ODORLESS_PATTERNS = [_kw_pattern(k) for k in ODORLESS_KEYWORDS]

# -----------------------------
# Text normalization & tokenization
# -----------------------------
SEP_RE = re.compile(r"\s*(?:;|,|/|\\|\|)\s*")
AND_RE = re.compile(r"\s+(?:and|&)\s+", re.I)

def normalize_text(x: str) -> str:
    """Normalizes spacing and separators; lowercases for matching."""
    if x is None:
        return ""
    t = str(x).strip().lower()
    t = AND_RE.sub(";", t)
    t = SEP_RE.sub(";", t)
    t = re.sub(r"\s+", " ", t)
    return t

def split_descriptors(t: str) -> List[str]:
    """Splits normalized descriptor string into tokens."""
    if not t:
        return []
    return [p for p in t.split(";") if p]

# -----------------------------
# Column detection
# -----------------------------
def detect_desc_col(df: pd.DataFrame, override: str = None) -> str:
    """Chooses the descriptor column by heuristic or uses the provided one."""
    if override and override in df.columns:
        return override
    cols = list(df.columns)
    cands = [c for c in cols if re.search(r"(flavo?r|odor|odour|aroma|descri|note)", c, re.I)]
    if cands:
        cands.sort(key=lambda c: -df[c].notna().sum())
        return cands[0]
    return cols[1] if len(cols) > 1 else cols[0]

# -----------------------------
# Classification helpers
# -----------------------------
def classify_token(token: str) -> List[str]:
    """Returns all matched raw categories for a single token (excluding Odorless)."""
    hits = []
    for cat, pats in CATEGORY_PATTERNS.items():
        if any(p.search(token) for p in pats):
            hits.append(cat)
    return hits

def is_odorless_token(token: str) -> bool:
    """Checks if token expresses odorless/blank aroma."""
    return any(p.search(token) for p in ODORLESS_PATTERNS)

def decide_major(votes_norm: Counter) -> str:
    """Resolves final class using majority vote on normalized categories (PRIORITY_NORM as tie-breaker)."""
    if not votes_norm:
        return "NA"
    max_ct = max(votes_norm.values())
    tied = [c for c, v in votes_norm.items() if v == max_ct]
    if len(tied) == 1:
        return tied[0]
    prio_pos = {c: i for i, c in enumerate(PRIORITY_NORM)}
    tied.sort(key=lambda c: prio_pos.get(c, 10**9))
    return tied[0]

def normalize_category(cat: str) -> str:
    """Maps a raw category to its normalized category via MERGE_MAP."""
    return MERGE_MAP.get(cat, cat)

# -----------------------------
# Core row classification
# -----------------------------
def classify_descriptors_cell(text: str) -> Dict[str, str]:
    """
    Classifies a single cell of descriptors.
    Returns:
        {
            "All_Matched_Categories": "<norm1|norm2|...>",
            "Major_Category": "<normMajor or Odorless/NA>"
        }
    """
    t = normalize_text(text)
    tokens = split_descriptors(t)

    votes_raw = Counter()
    odorless_flag = False

    for tok in tokens:
        if not tok:
            continue
        cats = classify_token(tok)
        if cats:
            for c in set(cats):
                votes_raw[c] += 1
        else:
            if is_odorless_token(tok):
                odorless_flag = True

    votes_norm = Counter()
    for raw_cat, ct in votes_raw.items():
        votes_norm[normalize_category(raw_cat)] += ct

    if votes_norm:
        major = decide_major(votes_norm)
        ordered = sorted(
            votes_norm.keys(),
            key=lambda c: PRIORITY_NORM.index(c) if c in PRIORITY_NORM else 10**9
        )
        all_matched = "|".join(ordered)
    else:
        if odorless_flag:
            major = "Odorless"
            all_matched = "Odorless"
        else:
            major = "NA"
            all_matched = ""

    return {
        "All_Matched_Categories": all_matched,
        "Major_Category": major
    }

# -----------------------------
# File-level routine
# -----------------------------
def classify_file(input_path: str, output_path: str = None, desc_col: str = None, mode: str = "multi") -> Path:
    """Reads input CSV, classifies each row, and writes output CSV."""
    in_path = Path(input_path)
    if output_path is None:
        suffix = "_classified_binary.csv" if mode == "binary" else "_classified_multi.csv"
        output_path = str(in_path.with_name(in_path.stem + suffix))

    df = pd.read_csv(in_path, encoding="utf-8", low_memory=False)
    col = detect_desc_col(df, desc_col)

    all_matched_col: List[str] = []
    major_col: List[str] = []
    contrib_col: List[str] = []

    for raw in df[col].astype(str).tolist():
        result = classify_descriptors_cell(raw)
        all_matched_col.append(result["All_Matched_Categories"])
        major_col.append(result["Major_Category"])

        if mode == "binary":
            mj = result["Major_Category"]
            contrib_col.append(BINARY_MAP.get(mj, "NA"))

    df["All_Matched_Categories"] = all_matched_col
    df["Major_Category"] = major_col

    if mode == "binary":
        df["Contribution"] = contrib_col

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return Path(output_path)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV path")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default depends on --mode)")
    parser.add_argument("--desc-col", default=None, help="Descriptor column name (optional)")
    parser.add_argument("--mode", choices=["multi", "binary"], default="multi",
                        help="Classification mode: 'multi' (default) or 'binary' (Pos/Neg)")
    args = parser.parse_args()

    out = classify_file(input_path=args.input, output_path=args.output, desc_col=args.desc_col, mode=args.mode)
    print(f"Wrote: {out}")
