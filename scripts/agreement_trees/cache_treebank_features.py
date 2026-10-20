"""Extraction-only caching sweep for the treebank feature caches.

  vp  verb-phrase caches: sv/sp/sa/ov/op/oa/iov/iop/ioa word_order parquets
      (Pipeline._build_df, the code the condition scripts use, without the
      tree fit, minimal pairs or second-chance retries)
  np  noun-phrase cache: treebank_features/npa/np_instances parquets
      (agreement_trees.npa.np_types)

Settings live in the shared agreement config (resources/agreement_config.json;
modes, langs, workers, vp, np; see agreement_trees.config), so a rerun
uses the same settings as the last one. Command line, for one-off or
machine-specific use: --config, --vp / --np (only that mode), --n_jobs and
--langs (override the config), --recache, --dry_run, --force.

Annotation filling (fetch_all: UniMorph and its UD-derived fallback) is on. A cache
holds all features, so it serves every target.

Usage (any directory):
  python3 cache_treebank_features.py
  python3 cache_treebank_features.py --langs German --recache
  python3 cache_treebank_features.py --dry_run

Languages already cached for everything requested are answered in the main
process (parquet metadata / sentinel files). The rest go to one worker each,
which runs only the still-uncached parts; VP families share one treebank load.
Languages with no matching instances get an empty sentinel so they aren't
re-parsed. Workers are RAM-guarded; a language that fails or kills its worker
is logged to output/logs/cache_error_log.txt and the sweep continues. Refuses
to start if another instance is running (unless --force). Run log:
output/logs/cache_treebank_features.log.
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.config import AGREEMENT_CONFIG_PATH
from agreement_trees.extract.cache_tools import run_cache_sweep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=AGREEMENT_CONFIG_PATH, help="Agreement config JSON (see module docstring).")
    parser.add_argument("--vp", action="store_true", help="Cache only the verb-phrase families.")
    parser.add_argument("--np", action="store_true", help="Cache only the noun-phrase instances.")
    parser.add_argument("--langs", "-l", nargs="*", default=None, help="Override the config's languages.")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Languages processed in parallel; overrides the config's n_jobs "
                             "(machine-dependent, so usually set per run).")
    parser.add_argument("--recache", action="store_true", help="Re-extract even where a cache already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report which languages and caches would be extracted; write nothing.")
    parser.add_argument("--force", action="store_true", help="Skip the check for other running instances.")
    return parser.parse_args()


if __name__ == "__main__":
    run_cache_sweep(parse_args())
