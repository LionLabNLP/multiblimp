"""How many minimal pairs per language/condition came from the strict leaf
threshold (default 95%) vs. the relaxed second_chance retry.

Reads only what create_pairs already wrote under output/minimal_pairs
(meta.json + the "second_chance" column of correct_swaps.parquet), so it
needs no refit or rebuild. Works for any layout <condition...>/<language>/
(SVA "svNa/svNa_nsubj", NPA "npa/HEAD-DET_N/...", per-treebank variants).

Usage (from repo root):
    python scripts/agreement_trees/supplementary/second_chance_report.py                 # summaries to stdout
    python scripts/agreement_trees/supplementary/second_chance_report.py --only-relaxed  # drop all-strict rows
    python scripts/agreement_trees/supplementary/second_chance_report.py --lang French --csv out.csv
"""
import argparse
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.config import OUTPUT_MINIMAL_PAIRS_DIR


def _pair_counts(lang_dir):
    fn = os.path.join(lang_dir, "correct_swaps.parquet")
    if not os.path.exists(fn):
        return 0, 0
    pf = pq.ParquetFile(fn)
    total = pf.metadata.num_rows
    if "second_chance" not in pf.schema_arrow.names:
        return total, 0
    relaxed = int(pd.read_parquet(fn, columns=["second_chance"])["second_chance"].sum())
    return total, relaxed


def collect(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        if "meta.json" not in files:
            continue
        rel = os.path.relpath(dirpath, root).split(os.sep)
        if len(rel) < 2:
            continue
        with open(os.path.join(dirpath, "meta.json")) as f:
            meta = json.load(f)
        pairs, pairs_relaxed = _pair_counts(dirpath)
        keep = meta.get("num_ud_candidates_keep") or 0
        keep_relaxed = meta.get("num_ud_candidates_keep_second_chance") or 0
        rows.append({
            "condition": "/".join(rel[:-1]),
            "language": rel[-1],
            "pairs": pairs,
            "pairs_strict": pairs - pairs_relaxed,
            "pairs_relaxed": pairs_relaxed,
            "keep": keep,
            "keep_strict": keep - keep_relaxed,
            "keep_relaxed": keep_relaxed,
            "retried": "second_chance_info" in meta or pairs_relaxed > 0 or keep_relaxed > 0,
        })
    return pd.DataFrame(rows)


def _summarize(df, by):
    g = df.groupby(by).agg(
        pairs=("pairs", "sum"), strict=("pairs_strict", "sum"), relaxed=("pairs_relaxed", "sum"),
        keep=("keep", "sum"), keep_relaxed=("keep_relaxed", "sum"),
        retried=("retried", "sum"), n=("language" if by != "language" else "condition", "count"),
    ).reset_index()
    g["%relaxed_pairs"] = (100 * g["relaxed"] / g["pairs"].where(g["pairs"] > 0)).round(1)
    g["%relaxed_keep"] = (100 * g["keep_relaxed"] / g["keep"].where(g["keep"] > 0)).round(1)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    ap.add_argument("--lang", help="restrict to one language (as in the directory name)")
    ap.add_argument("--only-relaxed", action="store_true", help="only rows with any relaxed pairs/keep")
    ap.add_argument("--csv", help="also write the per-language-per-condition table here")
    args = ap.parse_args()

    df = collect(args.dir)
    if df.empty:
        sys.exit(f"no meta.json found under {args.dir}")
    if args.lang:
        df = df[df["language"] == args.lang]
    if args.only_relaxed:
        df = df[(df["pairs_relaxed"] > 0) | (df["keep_relaxed"] > 0)]
    if df.empty:
        sys.exit("nothing matched")

    df = df.assign(**{"%relaxed_pairs": (100 * df["pairs_relaxed"] / df["pairs"].where(df["pairs"] > 0)).round(1)})
    df = df.sort_values(["condition", "language"])

    pd.set_option("display.width", 200, "display.max_rows", None, "display.max_columns", None)
    print("== per condition x language ==")
    print(df[["condition", "language", "pairs_strict", "pairs_relaxed", "pairs", "%relaxed_pairs", "retried"]]
          .to_string(index=False))
    print("\n== per condition (sum over languages; n = languages) ==")
    print(_summarize(df, "condition").to_string(index=False))
    print("\n== per language (sum over conditions; n = conditions) ==")
    print(_summarize(df, "language").sort_values("%relaxed_pairs", ascending=False).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
