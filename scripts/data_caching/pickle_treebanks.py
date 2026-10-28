import argparse
import json
import os
from glob import glob
from pathlib import Path
import pickle
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.config import RESOURCE_DIR, UD_PATH
from multiblimp.script_log import script_logger
from multiblimp.treebank import Treebank
from multiblimp.validation import validation_stamp
from multiblimp.languages import (
    add_langs,
    get_ud_langs,
    remove_diacritics_langs,
    gblang2udlang,
)


STAMPS_FILE = "validation_stamps.json"


def language_stamp(lang):
    """Validation stamp (see multiblimp.validation.validation_stamp) of the treebank folders the language's pickle is built from."""
    match_on = f"UD_{lang}-*" if lang not in add_langs else add_langs[lang]
    folders = [os.path.basename(p) for p in glob(os.path.join(RESOURCE_DIR, UD_PATH, match_on))]
    return validation_stamp(folders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pickle the UD treebanks, without the sentences the validator flags for the drop tests "
                    "(validation.drop_testids in data_prep_config.json; needs validate_treebanks.py to have run). "
                    "Existing pickles are skipped when built under the current validation results; others are rebuilt."
    )
    parser.add_argument("--langs", nargs="+", default=None, help="Languages to pickle. Default: every language.")
    parser.add_argument("--redo", action="store_true", help="Rebuild pickles even if they are current.")
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
            stamps_path = os.path.join(pickle_path, STAMPS_FILE)
            stamps = json.load(open(stamps_path)) if os.path.exists(stamps_path) else {}
            stamp = language_stamp(lang)
            if os.path.exists(out_path) and not args.redo:
                if stamps.get(lang) == stamp:
                    log.info(f"{lang}\tremove_typo={remove_typo}\texists, skipped")
                    continue
                log.info(f"{lang}\tremove_typo={remove_typo}\texists but not built under the current validation results, rebuilding")

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
            stamps[lang] = stamp
            with open(stamps_path, "w") as f:
                json.dump(stamps, f, indent=1, sort_keys=True)
