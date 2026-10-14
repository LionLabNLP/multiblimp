"""LEGACY: pickles of the UniMorph and UD-derived lexicons for the old UnimorphInflector.
The live pipeline reads the parquet tables (resources/um_tables, resources/ud_tables, see
scripts/um2ud.py and scripts/ud_tables.py) and does not use these pickles."""
import os
from pathlib import Path
import pickle
import sys

sys.path.append("../../src")

from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    remove_multiples_langs,
    lang2unimorph_lang,
    lang2langcode,
    unimorph_code,
)
from multiblimp.argparse import fetch_lang_candidates
from multiblimp.unimorph import UnimorphInflector
from multiblimp.script_log import script_logger


# UniMorph text files above this are parsed in chunks (Slovak is ~1.4 GB, the rest < 100 MB)
STREAM_MIN_BYTES = 250_000_000


if __name__ == "__main__":
    log = script_logger("pickle_unimorph")
    resource_dir = RESOURCE_DIR
    ud_langs = fetch_lang_candidates(resource_dir)

    for lang in ud_langs:
        log.info(lang)
        umlang = lang2unimorph_lang.get(lang, lang)
        langcode = lang2langcode(umlang)

        um_code = unimorph_code(langcode)
        um_path = os.path.join(resource_dir, f"unimorph/{um_code}/{um_code}")
        pickle_path = os.path.join(resource_dir, "unimorph/um_pickles")
        Path(pickle_path).mkdir(parents=True, exist_ok=True)
        pickle_path = os.path.join(pickle_path, f"{um_code}.pickle")
        if (
            os.path.isfile(um_path)
            and not os.path.isfile(um_path + ".segmentations")
            and os.path.getsize(um_path) > STREAM_MIN_BYTES
        ):
            inflector = UnimorphInflector(
                langcode, None, resource_dir=resource_dir, skip_load=True,
                fill_unk_values=False,
            )
            num_rows = inflector.pickle_unimorph_streaming(
                pickle_path,
                remove_diacritics=(lang in remove_diacritics_langs),
                remove_multiples=(lang in remove_multiples_langs),
            )
            log.info(f"  UM entries: {num_rows} (streamed)")
            inflector = None
        else:
            inflector = UnimorphInflector(
                langcode,
                None,
                remove_diacritics=(lang in remove_diacritics_langs),
                remove_multiples=(lang in remove_multiples_langs),
                use_ud_inflections=False,
                resource_dir=resource_dir,
                fill_unk_values=False,
            )
            if len(inflector) > 0:
                log.info(f"  UM entries: {len(inflector.unimorph_df)}")
                inflector.pickle_unimorph_df(pickle_path)

        inflector = UnimorphInflector(
            langcode,
            None,
            remove_diacritics=(lang in remove_diacritics_langs),
            remove_multiples=(lang in remove_multiples_langs),
            use_ud_inflections=True,
            resource_dir=resource_dir,
            fill_unk_values=False,
        )
        if len(inflector) > 0:
            pickle_path = os.path.join(resource_dir, "ud_unimorph/ud_pickles")
            Path(pickle_path).mkdir(parents=True, exist_ok=True)
            pickle_path = os.path.join(pickle_path, f"{langcode}.pickle")
            log.info(f"  UD entries: {len(inflector.unimorph_df)}")
            inflector.pickle_unimorph_df(pickle_path)


