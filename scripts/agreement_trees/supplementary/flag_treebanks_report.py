"""Compare UD's own flags and README heuristics against multiblimp.excluded_treebanks
and report anything flagged but not listed there. Nothing is filtered here.

Flag types (multiblimp.treebank.flag_treebanks): old variant, poetry only,
learner, twitter, code switching (local README/folder scan; expected in
excluded_treebanks), words removed (scraped from universaldependencies.org;
expected in excluded_treebanks) and sign language (scraped;
expected in SIGN_LANGUAGES). A failed scrape is reported, not fatal.

Usage (any directory):
    python flag_treebanks_report.py

Writes resources/flagged_treebanks.txt and output/logs/flag_treebanks_report.log; always exits 0.
"""
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_repo_root, "src"))

from multiblimp.script_log import script_logger
from multiblimp.config import RESOURCE_DIR
from multiblimp.treebank import flag_treebanks
from multiblimp.excluded_treebanks import (
    SIGN_LANGUAGES, split_treebank_folder, treebank_exclusion_reason,
)

LOCAL_TYPES = ("old variant", "poetry only", "learner", "twitter", "code switching")
SCRAPED_TYPES = ("words removed", "sign language")


def covered(flag_type: str, language: str, treebank: str) -> bool:
    """Whether (language, treebank id) is already listed for this flag type."""
    if flag_type == "sign language":
        return language in SIGN_LANGUAGES
    return treebank_exclusion_reason(language, treebank) is not None


def collect(resource_dir: str):
    rows, failed = [], []
    for flag_type in LOCAL_TYPES + SCRAPED_TYPES:
        try:
            results = flag_treebanks(flag_type, resource_dir=resource_dir)
        except OSError as e:
            failed.append((flag_type, e))
            continue
        for scraped_lang, treebanks in sorted(results.items(), key=lambda kv: str(kv[0])):
            if scraped_lang is None:
                continue
            for treebank in treebanks:
                if flag_type in LOCAL_TYPES:
                    language, treebank = split_treebank_folder(treebank)
                elif flag_type == "sign language":
                    language = scraped_lang.replace(" ", "_") + "_Sign_Language"
                else:
                    language = scraped_lang.replace(" ", "_")
                row = (flag_type, language, treebank)
                if row not in rows:
                    rows.append(row)
    return rows, failed


def main():
    resource_dir = RESOURCE_DIR
    log = script_logger("flag_treebanks_report")
    rows, failed = collect(resource_dir)

    out_path = os.path.join(resource_dir, "flagged_treebanks.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# flag_type\tlanguage\ttreebank\tlisted\n")
        for flag_type, language, treebank in rows:
            f.write(f"{flag_type}\t{language}\t{treebank}\t{covered(flag_type, language, treebank)}\n")

    log.info(f"{len(rows)} flagged treebank(s), written to {out_path}")
    for flag_type in LOCAL_TYPES + SCRAPED_TYPES:
        log.info(f"  {flag_type}: {sum(r[0] == flag_type for r in rows)}")
    for flag_type, err in failed:
        log.info(f"WARNING: could not check '{flag_type}': {err}")

    missing = [r for r in rows if not covered(*r)]
    if missing:
        log.info("\nFlagged but not listed in multiblimp.excluded_treebanks -- review:")
        for flag_type, language, treebank in missing:
            log.info(f"  {flag_type}\t{language}\t{treebank}")
    elif not failed:
        log.info("All flagged treebanks are listed.")


if __name__ == "__main__":
    main()
