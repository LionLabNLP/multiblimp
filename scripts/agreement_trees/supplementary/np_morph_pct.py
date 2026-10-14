import argparse
import glob
import os
import sys
from collections import Counter

import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.npa.np_types import np_type_morph_counts
from multiblimp.config import NP_INSTANCES_DIR, NP_REPORT_DIR
from agreement_trees.extract.per_treebank import split_excluded

INSTANCES_DIR = NP_INSTANCES_DIR
TYPES_DIR = NP_REPORT_DIR


CORE_COLUMNS = ["np_type", "count", "proportion"]


def add_morph_pct_columns(csv_path: str, per_type_counts: dict) -> None:
    """Merge {np_type: (total, {col: count})} onto an np_types csv, adding one percentage column per
    (role[-role2]_feature, value), e.g. "HEAD_Gender_Fem". Percentages are of the row's "count".
    Overwrites csv_path; only CORE_COLUMNS are kept from the old file, so reruns replace the pct columns."""
    base = pd.read_csv(csv_path)[CORE_COLUMNS]
    pct_rows = []
    for np_type in base["np_type"]:
        total, counts = per_type_counts.get(np_type, (0, {}))
        row = {"np_type": np_type}
        if total:
            row.update({col: round(n / total * 100, 2) for col, n in counts.items()})
        pct_rows.append(row)
    pct_df = pd.DataFrame(pct_rows)
    base.merge(pct_df, on="np_type", how="left").to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add per-value morphological-feature percentage columns "
                     "(e.g. HEAD_Gender_Fem, HEAD-DET_Case_Nom) to the "
                     "existing np_types csv files (written by cache_treebank_features.py when np.report is true "
                     "in resources/agreement_config.json), computed from treebank_features/npa/np_instances/*.parquet."
    )
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    args = parser.parse_args()

    parquet_paths = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.parquet")))
    if args.langs:
        wanted = set(args.langs)
        parquet_paths = [
            p for p in parquet_paths
            if os.path.basename(p)[: -len(".parquet")] in wanted
        ]

    # pooled over languages (instance-weighted), not an average of per-language percentages
    macro_counts: dict[str, list] = {}

    for parquet_path in parquet_paths:
        lang = os.path.basename(parquet_path)[: -len(".parquet")]
        csv_path = os.path.join(TYPES_DIR, f"{lang}.csv")
        if not os.path.exists(csv_path):
            print(f"skip {lang}: no np_types csv found")
            continue

        print(lang)
        df, _ = split_excluded(pd.read_parquet(parquet_path))
        if "np_type" not in df.columns:
            # no head had a qualifying dependent: 0-column frame, nothing to compute pct columns from
            print(f"  skip: empty/degenerate parquet (no np_type column)")
            continue
        per_type_counts = np_type_morph_counts(df)
        add_morph_pct_columns(csv_path, per_type_counts)

        for np_type, (total, counts) in per_type_counts.items():
            entry = macro_counts.setdefault(np_type, [0, Counter()])
            entry[0] += total
            entry[1].update(counts)

    macro_path = os.path.join(TYPES_DIR, "_macro.csv")
    macro_per_type = {k: (total, dict(c)) for k, (total, c) in macro_counts.items()}
    add_morph_pct_columns(macro_path, macro_per_type)
    print("\ndone")
