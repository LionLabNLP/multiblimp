"""Run the UD validator over the treebanks and report what it finds. Measure-only: writes one
sidecar per treebank (resources/ud/validation/, see multiblimp.validation) and summary tables
in output/diagnostics/validation/; no treebank data is filtered or changed.

Needs a checkout of UniversalDependencies/tools (default ../tools, or set UD_TOOLS_DIR) and
its dependencies (udapi, regex). Treebanks run one at a time in a memory-capped worker.

Usage (any directory):
  python3 validate_treebanks.py [--langs German Finnish] [--treebanks UD_German-GSD]
                                [--level 5] [--redo] [--report_only] [--mem_gb 4]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from glob import glob

import pandas as pd

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.pipeline_utils import run_in_pool, worker_mem_bytes
from multiblimp.config import OUTPUT_DIAGNOSTICS_DIR, RESOURCE_DIR, UD_PATH
from multiblimp.excluded_treebanks import split_treebank_folder, treebank_exclusion_reason
from multiblimp.languages import skip_langs
from multiblimp.script_log import script_logger
from multiblimp.validation import load_validation, validate_treebank

REPORT_DIR = os.path.join(OUTPUT_DIAGNOSTICS_DIR, "validation")


def list_folders(langs, treebanks):
    folders = sorted(os.path.basename(p) for p in glob(os.path.join(RESOURCE_DIR, UD_PATH, "UD_*")) if os.path.isdir(p))
    folders = [f for f in folders if split_treebank_folder(f)[0] not in skip_langs]
    if langs:
        wanted = {lang.replace(" ", "_") for lang in langs}
        folders = [f for f in folders if split_treebank_folder(f)[0] in wanted]
    if treebanks:
        folders = [f for f in folders if f in set(treebanks)]
    return folders


def build_report(folders, log):
    """Summary tables from the sidecars: per treebank, per test id, and per treebank x test id."""
    per_treebank, per_test_treebank = [], []
    for folder in folders:
        data = load_validation(folder)
        if data is None:
            continue
        lang, treebank_id = split_treebank_folder(folder)
        sentences = sum(f["sentences"] for f in data["files"].values())
        tokens = sum(f["tokens"] for f in data["files"].values())
        counts = Counter()
        sents = defaultdict(set)
        for testid, level, kind, filename, _, _, _, sentline in data["incidents"]:
            counts[testid, level, kind] += 1
            sents[testid, level, kind].add((filename, sentline))
        error_sents = {s for key, v in sents.items() if key[2] == "E" for s in v}
        warning_sents = {s for key, v in sents.items() if key[2] == "W" for s in v}
        top = Counter({key[0]: n for key, n in counts.items()}).most_common(3)
        per_treebank.append({
            "treebank": folder, "language": lang, "treebank_id": treebank_id,
            "excluded": treebank_exclusion_reason(lang, treebank_id) or "",
            "lang_code": data["lang_code"], "level_run": data["level_run"],
            "sentences": sentences, "tokens": tokens,
            "errors": sum(n for k, n in counts.items() if k[2] == "E"),
            "warnings": sum(n for k, n in counts.items() if k[2] == "W"),
            "sents_with_error": len(error_sents),
            "sents_with_warning": len(warning_sents),
            "sents_with_any": len(error_sents | warning_sents),
            "pct_sents_error": round(100 * len(error_sents) / max(sentences, 1), 2),
            "pct_sents_any": round(100 * len(error_sents | warning_sents) / max(sentences, 1), 2),
            "top_tests": "; ".join(f"{t}={n}" for t, n in top),
            "seconds": data["seconds"],
        })
        for (testid, level, kind), n in counts.items():
            per_test_treebank.append({
                "treebank": folder, "language": lang, "excluded": per_treebank[-1]["excluded"],
                "testid": testid, "level": level, "type": kind, "incidents": n,
                "sentences": len(sents[testid, level, kind]), "treebank_sentences": sentences,
                "pct_sents": round(100 * len(sents[testid, level, kind]) / max(sentences, 1), 3),
                "example": (data["examples"].get(testid) or [""])[0],
            })

    os.makedirs(REPORT_DIR, exist_ok=True)
    by_treebank = pd.DataFrame(per_treebank).sort_values("pct_sents_any", ascending=False)
    by_treebank.to_csv(os.path.join(REPORT_DIR, "by_treebank.tsv"), sep="\t", index=False)
    long = pd.DataFrame(per_test_treebank)
    long.sort_values(["testid", "pct_sents"], ascending=[True, False]).to_csv(
        os.path.join(REPORT_DIR, "by_treebank_test.tsv"), sep="\t", index=False)

    by_test = (
        long.groupby(["testid", "level", "type"])
        .agg(incidents=("incidents", "sum"), sentences=("sentences", "sum"), treebanks=("treebank", "nunique"),
             example=("example", "first"))
        .reset_index().sort_values("sentences", ascending=False)
    )
    by_test.to_csv(os.path.join(REPORT_DIR, "by_test.tsv"), sep="\t", index=False)

    total_sentences = int(by_treebank["sentences"].sum())
    used = by_treebank[by_treebank["excluded"] == ""]
    log.info(f"\n{len(by_treebank)} treebanks, {total_sentences} sentences "
             f"({len(used)} non-excluded treebanks, {int(used['sentences'].sum())} sentences)")
    log.info(f"sentences with >=1 error: {int(used['sents_with_error'].sum())}, "
             f"warning: {int(used['sents_with_warning'].sum())}, any: {int(used['sents_with_any'].sum())} (non-excluded)")
    log.info(f"treebanks with no incident: {int((by_treebank['errors'] + by_treebank['warnings'] == 0).sum())}; "
             f"validated only at level 3: {int((by_treebank['level_run'] < 5).sum())}")
    log.info("\ntop tests (sentences affected, non-excluded treebanks):")
    used_long = long[long["excluded"] == ""]
    top_tests = (used_long.groupby(["testid", "level", "type"])
                 .agg(sentences=("sentences", "sum"), treebanks=("treebank", "nunique"))
                 .reset_index().sort_values("sentences", ascending=False).head(25))
    for row in top_tests.itertuples():
        log.info(f"  {row.testid}\tL{row.level} {row.type}\t{row.sentences} sentences\t{row.treebanks} treebanks")
    log.info("\nworst treebanks (% sentences with any incident, non-excluded, >=100 sentences):")
    worst = used[used["sentences"] >= 100].head(20)
    for row in worst.itertuples():
        log.info(f"  {row.treebank}\t{row.pct_sents_any}%\t{row.top_tests}")
    log.info(f"\ntables in {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", "-l", nargs="*", default=None, help="UD language names. Default: every language.")
    parser.add_argument("--treebanks", "-t", nargs="*", default=None, help="UD folder names, e.g. UD_German-GSD.")
    parser.add_argument("--level", type=int, default=5, help="Validator level (1-5).")
    parser.add_argument("--redo", action="store_true", help="Revalidate treebanks whose sidecar is current.")
    parser.add_argument("--report_only", action="store_true", help="Only rebuild the summary tables from existing sidecars.")
    parser.add_argument("--mem_gb", type=float, default=4.0, help="RAM cap of the validation worker.")
    args = parser.parse_args()

    log = script_logger("validate_treebanks")
    folders = list_folders(args.langs, args.treebanks)

    if not args.report_only:
        log.info("treebank\tstatus\tlevel_run\tincidents\tseconds")

        def on_done(folder, result, error):
            if error is not None:
                log.info(f"{folder}\tFAILED\t\t\t{type(error).__name__}: {error}")
            elif result.get("skipped"):
                log.info(f"{folder}\tcurrent, skipped")
            else:
                log.info(f"{folder}\tdone\t{result['level_run']}\t{result['incidents']}\t{result['seconds']}")

        run_in_pool(
            validate_treebank,
            {folder: (folder, args.level, args.redo) for folder in folders},
            n_jobs=1,
            mem_bytes=worker_mem_bytes(1, args.mem_gb),
            max_tasks_per_child=1,
            on_done=on_done,
        )

    build_report(folders, log)
