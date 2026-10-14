"""Convert every language's UniMorph file to a UD-notation table (resources/um_tables/<code>.parquet,
see multiblimp.um2ud). Reads the raw files, writes only the new tables.

One worker, one fresh process per language, RAM-capped. UniMorph codes with no UD value are kept
as "UM:<code>" values and listed per language in output/logs/um2ud.log; a language that fails
or is killed goes to output/logs/cache_error_log.txt.

Usage (any directory):
  python3 um2ud.py [--langs German Finnish] [--max_worker_mem_gb 12]
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.pipeline_utils import run_in_pool, worker_mem_bytes
from multiblimp.annotation_coverage import table_coverage, update_coverage, write_overview
from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import get_ud_langs, lang2langcode, lang2unimorph_lang, skip_langs
from multiblimp.script_log import log_cache_error, script_logger
from multiblimp.um2ud import UM_TABLE_DIR, um_table_path, write_language


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", "-l", nargs="*", default=None)
    parser.add_argument("--max_worker_mem_gb", type=float, default=12)
    parser.add_argument("--out_dir", default=UM_TABLE_DIR)
    args = parser.parse_args()
    log = script_logger("um2ud")
    langs = [l for l in (args.langs or get_ud_langs(RESOURCE_DIR)) if l not in skip_langs]
    totals = {"written": 0, "no_unimorph_file": 0, "failed": 0}

    def done(lang, result, error):
        if error is None:
            langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
            update_coverage(lang, "um_table", table_coverage(um_table_path(langcode, args.out_dir)))
        if error is not None:
            totals["failed"] += 1
            log.info(f"FAILED {lang}: {error}")
            log_cache_error("um2ud", lang, f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error))
        elif not result["rows"]:
            totals["no_unimorph_file"] += 1
        else:
            totals["written"] += 1
            top = sorted(result["unmapped"].items(), key=lambda kv: -kv[1])
            log.info(f"{lang}: {result['rows']:,} rows" + (f"; no UD value: " + ", ".join(f"{k} {n:,}" for k, n in top[:15]) if top else "")
                     + (f"; past+PRF -> Tense=Pqp: {result['pluperfect']:,}" if result.get("pluperfect") else ""))

    run_in_pool(write_language, {l: (l, args.out_dir) for l in langs}, 1,
                worker_mem_bytes(1, args.max_worker_mem_gb), 1, on_done=done)
    write_overview()
    log.info(f"\n{totals}")


if __name__ == "__main__":
    main()
