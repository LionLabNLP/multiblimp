import os
from pathlib import Path
import pickle
import sys

sys.path.append("../../src")

from multiblimp.languages import (
    get_ud_langs,
    remove_diacritics_langs,
    remove_multiples_langs,
    lang2unimorph_lang,
    lang2langcode,
)
from multiblimp.argparse import fetch_lang_candidates
from multiblimp.unimorph import UnimorphInflector


# UniMorph text files above this are parsed in chunks (Slovak is ~1.4 GB, the rest < 100 MB)
STREAM_MIN_BYTES = 250_000_000


if __name__ == "__main__":
    resource_dir = "../../resources"
    ud_langs = fetch_lang_candidates(resource_dir)

    for lang in ud_langs:
        print(lang)
        umlang = lang2unimorph_lang.get(lang, lang)
        langcode = lang2langcode(umlang)

        um_path = os.path.join(resource_dir, f"unimorph/{langcode}/{langcode}")
        pickle_path = os.path.join(resource_dir, "unimorph/um_pickles")
        Path(pickle_path).mkdir(parents=True, exist_ok=True)
        pickle_path = os.path.join(pickle_path, f"{langcode}.pickle")
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
            print("#UM entries", lang, num_rows)
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
                print("#UM entries", lang, len(inflector.unimorph_df))
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
            print("#UD entries", lang, len(inflector.unimorph_df))
            inflector.pickle_unimorph_df(pickle_path)
