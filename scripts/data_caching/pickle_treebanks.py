import argparse
import os
from pathlib import Path
import pickle
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.config import RESOURCE_DIR
from multiblimp.script_log import script_logger
from multiblimp.treebank import Treebank
from multiblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    gblang2udlang,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle the UD treebanks (existing pickles are skipped).")
    parser.add_argument("--langs", nargs="+", default=None, help="Languages to pickle. Default: every language.")
    args = parser.parse_args()

    log = script_logger("pickle_treebanks")
    resource_dir = RESOURCE_DIR
    ud_langs = args.langs or get_ud_langs(resource_dir)

    for lang in ud_langs:
        lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        for remove_typo in [True, False]:
            if remove_typo:
                pickle_path = os.path.join(resource_dir, "ud/ud_pickles")
            else:
                pickle_path = os.path.join(resource_dir, "ud/ud_typo_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(pickle_path, f"{lang}.pickle")
            if os.path.exists(out_path):
                log.info(f"{lang}\tremove_typo={remove_typo}\texists, skipped")
                continue

            treebank = Treebank(
                lang,
                remove_diacritics=(lang in remove_diacritics_langs),
                resource_dir=resource_dir,
                remove_typo=remove_typo,
            )

            log.info(f"{lang}\tremove_typo={remove_typo}\t{len(treebank)} trees")
            if len(treebank)==0:
                log.info(f"{lang}\tremove_typo={remove_typo}\tempty, not pickled")
                continue

            with open(out_path, "wb") as f:
                pickle.dump(treebank, f)
