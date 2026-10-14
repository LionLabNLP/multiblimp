"""Build the UD-derived lexicon tables (resources/ud_tables). Duplicate-removal thresholds
(ud_tables.dup_form_threshold / dup_feat_threshold, see
multiblimp.ud_lexicon.remove_duplicate_features) are in resources/data_prep_config.json.

Also writes the UD side of the annotation coverage statistics
(resources/annotation_coverage.json and _overview.tsv, see
multiblimp.annotation_coverage) and lists the UD feature values that have no UniMorph
tag in output/logs/ud_tables.log."""
import json
import os
import sys
from collections import Counter

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.annotation_coverage import load_coverage, write_overview
from multiblimp.config import RESOURCE_DIR
from multiblimp.ud_lexicon import create_all_ud_tables
from multiblimp.languages import get_ud_langs
from multiblimp.script_log import script_logger


if __name__ == "__main__":
    resource_dir = RESOURCE_DIR
    with open(os.path.join(RESOURCE_DIR, "data_prep_config.json")) as f:
        cfg = json.load(f)["ud_tables"]
    ud_langs = get_ud_langs(resource_dir)

    log = script_logger("ud_tables")
    log.info("lang\tlemmas\tforms\trows")
    create_all_ud_tables(
        ud_langs,
        resource_dir=resource_dir,
        load_from_pickle=True,
        dup_form_threshold=cfg["dup_form_threshold"],
        dup_feat_threshold=cfg["dup_feat_threshold"],
        log=log,
    )

    write_overview()
    unmapped, skipped = Counter(), Counter()
    for entry in load_coverage().values():
        ud = entry.get("ud", {})
        unmapped.update(ud.get("unmapped", {}))
        skipped.update(ud.get("skipped", {}))
    log.info(f"Tokens not turned into rows: {dict(skipped)}")
    log.info("UD feature values with no UniMorph tag (tokens, all languages):")
    for value, n in unmapped.most_common(60):
        log.info(f"  {value}\t{n}")
