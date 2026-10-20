"""Scan the treebank feature caches for agreement evidence and write the
languages worth fitting for each target.

  vp  the word_order caches (cache_treebank_features.py --vp); targets "sv:Number",
      "sp:Case", ... (family:feature)
  np  the np_instances parquets; targets "HEAD-DET_Number", ...

Run after cache_treebank_features.py. A (target, language) clears the bar when
agreement_trees.pipeline_utils.passes_agreement_bar says so; thresholds
and the modes to scan are in the "scan" block and top-level "modes" of
resources/agreement_config.json.

Writes resources/agreement_langs_{vp,np}.json: the thresholds used, each scanned cache
file with its modification time, and target -> language -> counts for the
languages that qualify. A cache that is listed as scanned but absent under a
target was found lacking, so fit_candidate.py can skip a target without
rescanning, and rescans only languages whose cache is newer than the scan (or
all of them if the thresholds changed). A partial scan (--langs, or fewer vp
families) replaces only what it scanned. Also writes
output/agreement_candidates/{vp,np}_summary.csv and {vp,np}_by_lang.csv (tables for
picking candidates; nothing reads them) and the report to
output/logs/find_agreement_candidates.log. --vp / --np scan one mode only; the two
never depend on each other.

Usage (any directory):
  python3 find_agreement_candidates.py
  python3 find_agreement_candidates.py --vp --langs German French
"""
import argparse
import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.config import (
    AGREEMENT_CONFIG_PATH, LANGS_PATH, apply_mode_flags, load_agreement_config,
)
from multiblimp.config import OUTPUT_AGREEMENT_CANDIDATES_DIR
from multiblimp.script_log import script_logger
from agreement_trees.fit.scan import (
    build_lang_config, build_lang_csv, build_summary, report_lines, save_results, scan_np, scan_thresholds, scan_vp,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=AGREEMENT_CONFIG_PATH, help="Agreement config JSON.")
    parser.add_argument("--vp", action="store_true", help="Scan only the verb-phrase caches.")
    parser.add_argument("--np", action="store_true", help="Scan only the noun-phrase caches.")
    parser.add_argument("--langs", "-l", nargs="*", default=None, help="Override the config's languages.")
    args = parser.parse_args()

    log = script_logger("find_agreement_candidates")
    config = apply_mode_flags(load_agreement_config(args.config), args.vp, args.np)
    scan_cfg = config["scan"]
    langs = (args.langs if args.langs is not None else config["langs"]) or None
    os.makedirs(OUTPUT_AGREEMENT_CANDIDATES_DIR, exist_ok=True)
    thresholds = scan_thresholds(scan_cfg)
    log.info(f"Scan settings: {json.dumps(thresholds)}")

    for mode in config["modes"]:
        if mode == "vp":
            stats, counts, units = scan_vp(langs, families=config["vp"]["families"],
                                           exclude_features=scan_cfg["exclude_features"])
            for family, (done, total) in counts.items():
                log.info(f"vp {family}: scanned {done}/{total} languages" if total
                         else f"vp {family}: no cache found, skipped")
        else:
            stats, (done, total), units = scan_np(langs, split_head_upos=scan_cfg["split_head_upos"],
                                                    exclude_features=scan_cfg["exclude_features"])
            log.info(f"np: scanned {done}/{total} languages")
        log.info(f"{mode}: {len(stats)} distinct targets found")
        if not stats:
            continue

        summary = build_summary(stats, mode)
        results = build_lang_config(stats, scan_cfg)
        summary.to_csv(os.path.join(OUTPUT_AGREEMENT_CANDIDATES_DIR, f"{mode}_summary.csv"), index=False)
        build_lang_csv(results, stats).to_csv(
            os.path.join(OUTPUT_AGREEMENT_CANDIDATES_DIR, f"{mode}_by_lang.csv"), index=False)
        save_results(mode, results, units, thresholds, set(langs or ()), set(config["vp"]["families"]), log)
        log.info(f"{mode}: {len(results)} targets with at least one usable language -> {LANGS_PATH[mode]}")
        for line in report_lines(summary, scan_cfg["min_langs"]):
            log.info(line)


if __name__ == "__main__":
    main()
