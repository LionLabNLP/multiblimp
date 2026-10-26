"""Count, per language and condition, the pairs the old surface check dropped (no known contracted form and an
unattested uncontracted sequence): the languages whose pairs need rebuilding now that such pairs are kept.

A pair directory written by the surface-aware pair creation holds them in invalid_surface.parquet; an older one
(no surface_status column) is validated here with the current code, where they show up as uncontracted_unattested.
Writes a CSV (language, target, rows, dropped, share) sorted by dropped and prints the affected languages.

Usage (any directory):
  python3 count_invalid_surface.py [--pairs_dir output/minimal_pairs] [--langs Turkish German]
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

import pandas as pd
import pyarrow.parquet as pq

from agreement_trees.eval.benchmark import pair_files
from agreement_trees.eval.surface import read_pair_frame, validate_frame
from multiblimp.config import OUTPUT_DIAGNOSTICS_DIR, OUTPUT_MINIMAL_PAIRS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIAGNOSTICS_DIR, "mwt", "invalid_surface_counts.csv"))
    args = parser.parse_args()

    records = []
    for path, lang, cond, deprel in pair_files(args.pairs_dir, args.langs):
        meta = pq.ParquetFile(path)
        bucket = path.parent / "invalid_surface.parquet"
        if "surface_status" in meta.schema_arrow.names:
            rows = meta.metadata.num_rows
            dropped = pq.ParquetFile(bucket).metadata.num_rows if bucket.exists() else 0
            source = "bucket"
        elif meta.metadata.num_rows:
            frame = read_pair_frame(path)
            _, tally = validate_frame(frame)
            rows, dropped, source = len(frame), int(tally.get("uncontracted_unattested", 0)), "validated"
        else:
            continue
        records.append({"lang": lang, "target": os.path.relpath(path.parent, args.pairs_dir), "rows": rows, "dropped": dropped, "source": source})
        print(lang, records[-1]["target"], rows, dropped, source, flush=True)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.assign(share=(df["dropped"] / (df["rows"] + df["dropped"]).clip(lower=1)).round(4)).sort_values("dropped", ascending=False).to_csv(args.out, index=False)
    per_lang = df.groupby("lang")[["rows", "dropped"]].sum()
    print("\nAffected languages:")
    print(per_lang[per_lang["dropped"] > 0].sort_values("dropped", ascending=False).to_string())
    print(f"\nWrote {args.out}")
