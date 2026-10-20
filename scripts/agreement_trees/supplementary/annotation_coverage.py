"""Write the annotation coverage statistics (resources/annotation_coverage.json and
_overview.tsv, see multiblimp.annotation_coverage) from the existing UD treebank pickles
and the parquet tables (um_tables, ud_tables), without rebuilding any table.
scripts/data_caching/ud_tables.py and scripts/data_caching/um2ud.py write the table halves when they run.

One worker, one fresh process per language, RAM-capped.

Usage (any directory):
  python3 annotation_coverage.py [--langs German French] [--max_worker_mem_gb 8]
"""
import argparse
import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.pipeline_utils import run_in_pool, worker_mem_bytes
from multiblimp.annotation_coverage import (
    finish_ud_stats, new_ud_stats, table_coverage, update_coverage, write_overview,
)
from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import get_ud_langs, lang2langcode, lang2unimorph_lang, skip_langs


def coverage_for(lang: str) -> dict:
    from multiblimp.treebank import Treebank
    from multiblimp.ud_lexicon import collect_ud_rows
    from multiblimp.ud_table import table_path
    from multiblimp.um2ud import um_table_path

    treebank = Treebank(lang.replace(" ", "_"), remove_diacritics=False, load_from_pickle=True,
                        resource_dir=RESOURCE_DIR, remove_typo=False, pickle_path="ud/ud_typo_pickles")
    stats = new_ud_stats()
    with open(os.path.join(RESOURCE_DIR, "data_prep_config.json")) as f:
        cfg = json.load(f)["ud_tables"]
    collect_ud_rows(treebank, dup_form_threshold=cfg["dup_form_threshold"],
                    dup_feat_threshold=cfg["dup_feat_threshold"], coverage=stats)
    del treebank
    langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
    return {
        "ud": finish_ud_stats(stats),
        "um_table": table_coverage(um_table_path(langcode)),
        "ud_table": table_coverage(table_path(langcode)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", "-l", nargs="*", default=None)
    parser.add_argument("--max_worker_mem_gb", type=float, default=8)
    args = parser.parse_args()
    langs = [l for l in (args.langs or get_ud_langs(RESOURCE_DIR)) if l not in skip_langs]

    def done(lang, result, error):
        if error is not None:
            print(f"FAILED {lang}: {error}", flush=True)
            return
        for key, data in result.items():
            update_coverage(lang, key, data)
        print(lang, flush=True)

    run_in_pool(coverage_for, {lang: (lang,) for lang in langs}, 1,
                worker_mem_bytes(1, args.max_worker_mem_gb), 1, on_done=done)
    write_overview()


if __name__ == "__main__":
    main()
