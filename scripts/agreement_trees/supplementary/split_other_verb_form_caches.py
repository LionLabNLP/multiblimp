"""Derive the per-VerbForm subject caches (treebank_features/nsubj_inf, nsubj_vnoun, ...) from the combined
nsubj_other cache, without re-reading any treebank: a form's cache holds the rows whose head VerbForm is that form,
which is exactly what extraction with that form's head filter would keep. Rows with no VerbForm or several values
(and forms outside agreement_trees.taxonomy.OTHER_VERB_FORMS) belong to no form and are left out.

A language with nothing for a form gets an .empty.json sentinel there; up-to-date outputs are skipped.

Usage (any directory):
  python3 split_other_verb_form_caches.py [--src DIR] [--langs Turkish Hindi] [--force]
"""
import argparse
import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from agreement_trees.extract.prediction_target import filter_head_feats
from agreement_trees.pipeline_utils import parquet_current, sentinel_current, write_sentinel, write_table
from agreement_trees.taxonomy import OTHER_VERB_FORMS
from multiblimp.config import TREEBANK_FEATURES_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(TREEBANK_FEATURES_DIR, "nsubj_other"))
    parser.add_argument("--dst_root", default=TREEBANK_FEATURES_DIR, help="where nsubj_<form> directories are written")
    parser.add_argument("--langs", nargs="*", default=None, help="Default: every language in --src.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dirs = {prefix: os.path.join(args.dst_root, f"nsubj_{prefix[2:]}") for prefix in OTHER_VERB_FORMS}
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    langs = sorted(f[: -len(".parquet")] for f in os.listdir(args.src) if f.endswith(".parquet"))
    sentinels = sorted(f[: -len(".empty.json")] for f in os.listdir(args.src) if f.endswith(".empty.json"))
    for lang in sorted(set(langs + sentinels)):
        if args.langs and lang not in args.langs:
            continue
        for prefix, (form, _) in OTHER_VERB_FORMS.items():
            out = os.path.join(dirs[prefix], f"{lang}.parquet")
            marker = os.path.join(dirs[prefix], f"{lang}.empty.json")
            if not args.force and ((os.path.exists(out) and parquet_current(out)) or sentinel_current(marker)):
                continue
            for stale in (out, marker):
                if os.path.exists(stale):
                    os.remove(stale)
            if lang in sentinels:
                with open(os.path.join(args.src, f"{lang}.empty.json")) as f:
                    write_sentinel(marker, {k: v for k, v in json.load(f).items() if k != "extraction_version"})
                continue
            table = pq.read_table(os.path.join(args.src, f"{lang}.parquet"))
            values = table["head_VerbForm"].to_pylist() if "head_VerbForm" in table.column_names else []
            keep = [filter_head_feats(v, require=form) for v in values]
            if any(keep):
                write_table(table.filter(pa.array(keep)), out)
                print(f"{lang} {prefix}: {sum(keep)} of {table.num_rows} rows", flush=True)
            else:
                write_sentinel(marker, {"predictor_var": "head_nsubj_Number_agreement"})
