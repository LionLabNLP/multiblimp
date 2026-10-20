"""Multiword tokens (UD ID ranges such as 3-4 "du" = "de le") of every UD language: how many sentences and spans,
which component UPOS patterns, and the surface -> component forms table.

Writes mwt_by_language.csv, mwt_types.csv and mwt_upos_patterns.csv to output/diagnostics/mwt/.

Usage (any directory):
  python3 ud_multiword_tokens.py
"""
import argparse
import collections
import os
import sys
from pathlib import Path

import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.conllu_spans import span_stats
from multiblimp.config import OUTPUT_DIAGNOSTICS_DIR, RESOURCE_DIR, UD_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ud_dir", default=os.path.join(RESOURCE_DIR, UD_PATH))
    parser.add_argument("--out_dir", default=os.path.join(OUTPUT_DIAGNOSTICS_DIR, "mwt"))
    args = parser.parse_args()

    files_by_lang = collections.defaultdict(list)
    for tb_dir in sorted(Path(args.ud_dir).glob("UD_*")):
        files_by_lang[tb_dir.name[3:].split("-")[0]] += sorted(tb_dir.glob("*.conllu"))

    lang_rows, type_rows = [], []
    for lang, files in sorted(files_by_lang.items()):
        stats, types = span_stats(files)
        lang_rows.append({"lang": lang, "treebanks": len({p.parent for p in files}), **stats})
        type_rows += [(lang, s, c, u, len(c.split()), n) for (s, c, u), n in types.most_common()]

    types_df = pd.DataFrame(type_rows, columns=["lang", "surface", "components", "upos", "n_components", "count"])
    ambiguous = (types_df.groupby(["lang", "components"]).surface.nunique() > 1).groupby("lang").sum()
    patterns = types_df.groupby(["lang", "upos"]).agg(count=("count", "sum"), types=("surface", "size")).reset_index()
    patterns = patterns.sort_values(["lang", "count"], ascending=[True, False])

    lg = pd.DataFrame(lang_rows).set_index("lang").fillna(0)
    lg = lg[lg.mwt_tokens > 0].astype(int)
    lg["mwt_types"] = types_df.drop_duplicates(["lang", "surface", "components"]).groupby("lang").size()
    lg["ambiguous_component_strings"] = ambiguous
    lg["top_upos_patterns"] = patterns.groupby("lang").apply(
        lambda g: "; ".join(f"{u} {n}" for u, n in zip(g.upos[:4], g["count"][:4])), include_groups=False
    )
    lg["pct_sent_with_mwt"] = (100 * lg.sent_with_mwt / lg.sentences).round(1)
    lg = lg.sort_values("pct_sent_with_mwt", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    lg.to_csv(os.path.join(args.out_dir, "mwt_by_language.csv"))
    types_df.to_csv(os.path.join(args.out_dir, "mwt_types.csv"), index=False)
    patterns.to_csv(os.path.join(args.out_dir, "mwt_upos_patterns.csv"), index=False)
    print(f"{len(lg)} languages with multiword tokens")
