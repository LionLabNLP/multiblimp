import argparse
import os
import sys
from collections import Counter

import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.treebank import Treebank
from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import get_ud_langs, remove_diacritics_langs

UPOS_TARGETS = ("NOUN", "DET", "ADJ")
PREFERRED_COL_ORDER = ["Masc", "Fem", "Neut", "Com"]


def scan_language(lang: str, resource_dir: str) -> dict:
    """UPOS -> Counter(Gender value or 'missing' -> token count) over every token of the pickled treebank (uncapped)."""
    treebank = Treebank(
        lang,
        remove_diacritics=(lang in remove_diacritics_langs),
        resource_dir=resource_dir,
        remove_typo=True,
        load_from_pickle=True,
    )
    counts = {upos: Counter() for upos in UPOS_TARGETS}
    for tree in treebank:
        for tok in tree:
            upos = tok.get("upos")
            if upos not in counts:
                continue
            feats = tok.get("feats") or {}
            gender = feats.get("Gender", "missing")
            counts[upos][gender] += 1
    return counts


def build_overview(all_counts: dict) -> pd.DataFrame:
    value_cols = sorted(
        {val for counts in all_counts.values() for c in counts.values() for val in c},
        key=lambda v: (v == "missing", PREFERRED_COL_ORDER.index(v) if v in PREFERRED_COL_ORDER else 99, v),
    )
    rows = []
    for lang, counts in all_counts.items():
        for upos, c in counts.items():
            total = sum(c.values())
            if total == 0:
                continue
            row = {"lang": lang, "upos": upos}
            for col in value_cols:
                n = c.get(col, 0)
                row[col] = n
                row[f"{col}_pct"] = round(n / total * 100, 2)
            row["total"] = total
            rows.append(row)
    pct_cols = [f"{col}_pct" for col in value_cols]
    ordered = ["lang", "upos"]
    for col, pct_col in zip(value_cols, pct_cols):
        ordered += [col, pct_col]
    ordered.append("total")
    return pd.DataFrame(rows, columns=ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tally the Gender values annotated on NOUN/DET/ADJ tokens of every UD language's "
                     "full pickled treebank (all tokens of those UPOS, not only those in NP pairs)."
    )
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    parser.add_argument("--out", default=os.path.join(_repo_root, "output", "overview", "gender_overview.csv"))
    args = parser.parse_args()

    langs = args.langs or get_ud_langs(RESOURCE_DIR)

    all_counts = {}
    for lang in langs:
        print(lang)
        counts = scan_language(lang, RESOURCE_DIR)
        if any(counts.values()):
            all_counts[lang] = counts

    df = build_overview(all_counts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")
