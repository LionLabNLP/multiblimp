"""Build the per-language multiword-token statistics (resources/ud_mwt, see agreement_trees.eval.mwt_cache): the
contraction map and the multiword spans of each sentence, from the non-excluded treebanks' raw CoNLL-U. Pair creation
reads them to write the sentences' surface forms (de le -> du). Up-to-date caches are skipped; they are also built
on demand when missing.

Usage (any directory):
  python3 mwt_stats.py [--langs French German] [--force]
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.mwt_cache import load_stats, treebank_files
from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import get_ud_langs
from multiblimp.script_log import script_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", "-l", nargs="*", default=None, help="Default: every language.")
    parser.add_argument("--force", action="store_true", help="Rebuild even where the cache is current.")
    args = parser.parse_args()

    log = script_logger("mwt_stats")
    for lang in args.langs or get_ud_langs(RESOURCE_DIR):
        if not treebank_files(lang):
            log.info(f"{lang}: no selected treebank files")
            continue
        stats = load_stats(lang, rebuild=args.force)
        log.info(f"{lang}: {len(stats.sentences):,} sentences with a multiword token, "
                 f"{len(stats.contracted):,} contraction types, {len(stats.files)} treebank file(s)")
