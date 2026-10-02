"""Regenerate already-built NPA condition(s)' diagnostics CSV + deprel
index.html, then the cross-pipeline decision_trees/index.html overview --
purely from on-disk output/decision_trees + output/minimal_pairs artifacts,
no refitting.

Why this is cheap relative to a full pipeline run: refresh_deprel_index (the
tail of run_agreement_pipeline, extracted) only reads the decision-tree-stage
.joblib/.parquet cache and the minimal-pairs cache -- never np_instances/
{lang}.parquet (every role a language's NPs carry, not just one target_col's
two), which run_agreement_pipeline's own per-language loop reads
unconditionally before it even reaches a skip-check.

It is NOT free, though: generate_diagnostics_table re-reads every bucket
.parquet + meta.json under each condition's minimal_pairs dir, which at 88
built conditions is real I/O. By default this script now only redoes that
work for a condition whose diagnostics CSV + index.html are older than its
underlying .joblib/.parquet/meta.json inputs (see needs_refresh) -- i.e. a
condition nothing has touched since its last index build is skipped. Use
--force to bypass that check, and --target-col to limit the sweep to
specific conditions instead of every built one.

Scope: NPA only for now. sva_trees.pipeline.Pipeline/subj_aux.pipeline.
AuxPipeline have the same free-reindex property, but re-driving them
generically needs each target's own PredictionTarget/predictor_var/label
metadata, which isn't centralized anywhere yet -- left as a follow-up.

Run from scripts/: python generate_html_indexes.py
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.append("../src")

from npa.agreement import refresh_deprel_index, npa_id, FEATURE_ABBREV

INV_FEATURE_ABBREV = {v: k for k, v in FEATURE_ABBREV.items()}
from word_order.viz_overview import generate_html_overview_index
from multiblimp.config import (
    OUTPUT_DECISION_TREES_DIR, HTML_DECISION_TREES_DIR,
    OUTPUT_MINIMAL_PAIRS_DIR, OUTPUT_DIAGNOSTICS_DIR, DEBUG_MODE,
)


def _npa_identifier_to_target_col(npa_identifier: str) -> str:
    """Inverse of npa.agreement.npa_id -- "HEAD-DET_N" -> "HEAD-DET_Number".
    The on-disk directory name is the abbreviated identifier, not the real
    target_col (run_agreement_pipeline's own argument), so this has to be
    reversed rather than assumed. Relies on FEATURE_ABBREV being injective
    (true today: N/G/P/C/Def/Deg/PT/NT/Pos are all distinct)."""
    roles, abbr = npa_identifier.rsplit("_", 1)
    return f"{roles}_{INV_FEATURE_ABBREV.get(abbr, abbr)}"


def built_npa_conditions() -> dict[str, list[str]]:
    """target_col -> [languages already fit for it], read off whatever's
    already on disk under output/decision_trees/npa/*/*.joblib -- not
    npa_config.json, which lists CANDIDATE languages, most not yet run.

    The language list itself is only used below for the per-condition
    language count printed to stdout -- refresh_deprel_index doesn't take a
    langs argument; generate_html_deprel_index scans data_dir itself.
    """
    conditions = {}
    for cond_dir in sorted(glob.glob(os.path.join(OUTPUT_DECISION_TREES_DIR, "npa", "*"))):
        if not os.path.isdir(cond_dir):
            continue
        target_col = _npa_identifier_to_target_col(os.path.basename(cond_dir))
        langs = sorted(Path(p).stem for p in glob.glob(os.path.join(cond_dir, "*.joblib")))
        if langs:
            conditions[target_col] = langs
    return conditions


def _latest_mtime(*patterns: str) -> float:
    """Max mtime across every file matched by the given glob patterns, or
    -1 if none match -- so "no inputs found" always counts as "newer than
    any output", forcing a refresh rather than silently skipping one."""
    mtimes = [
        os.path.getmtime(p)
        for pattern in patterns
        for p in glob.glob(pattern)
        if os.path.isfile(p)
    ]
    return max(mtimes) if mtimes else -1.0


def needs_refresh(target_col: str) -> bool:
    """True unless target_col's diagnostics CSV + deprel index.html are
    both already newer than every .joblib/.parquet/meta.json file
    refresh_deprel_index would read to rebuild them -- i.e. unless nothing
    that feeds them has changed since the last refresh. Same default-path
    logic as refresh_deprel_index itself, so it's checking exactly what
    that call would (re)write."""
    npa_identifier = npa_id(target_col)
    save_dir = os.path.join(OUTPUT_DECISION_TREES_DIR, "npa", npa_identifier)
    html_dir = os.path.join(HTML_DECISION_TREES_DIR, "npa", npa_identifier)
    pairs_dir = os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", npa_identifier)
    diagnostics_csv = os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{npa_identifier}.csv")
    index_html = os.path.join(html_dir, "index.html")

    if not (os.path.exists(index_html) and os.path.exists(diagnostics_csv)):
        return True

    output_mtime = min(os.path.getmtime(index_html), os.path.getmtime(diagnostics_csv))
    input_mtime = _latest_mtime(
        os.path.join(save_dir, "*.joblib"),
        os.path.join(save_dir, "*.parquet"),
        os.path.join(pairs_dir, "*", "*.parquet"),
        os.path.join(pairs_dir, "*", "meta.json"),
    )
    return input_mtime > output_mtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-col", "-t", dest="target_cols", action="append",
        help="Limit the sweep to this target_col (npa.agreement's own "
             "argument, e.g. HEAD-DET_Number) -- repeatable. Default: every "
             "target_col with at least one built language on disk.",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Refresh every selected condition regardless of needs_refresh "
             "-- use after touching a condition's cache without also "
             "touching its output (e.g. hand-editing a .joblib), which the "
             "mtime check can't otherwise see.",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="Print which selected conditions would be refreshed and exit "
             "without building or writing anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    conditions = built_npa_conditions()

    if args.target_cols:
        unknown = sorted(set(args.target_cols) - set(conditions))
        if unknown:
            sys.exit(f"Not built on disk, so nothing to refresh: {unknown}")
        conditions = {tc: langs for tc, langs in conditions.items() if tc in args.target_cols}

    print(f"{len(conditions)} npa condition(s) selected: {list(conditions)}")

    to_refresh = {
        target_col: langs for target_col, langs in conditions.items()
        if args.force or needs_refresh(target_col)
    }
    skipped = [tc for tc in conditions if tc not in to_refresh]
    if skipped:
        print(f"Already up to date, skipping {len(skipped)}: {skipped}")

    if args.list:
        print(f"Would refresh {len(to_refresh)}: {list(to_refresh)}")
        sys.exit(0)

    for target_col, langs in to_refresh.items():
        print(f"\n=== {target_col} ({len(langs)} languages) ===")
        refresh_deprel_index(target_col, has_pairs=not DEBUG_MODE)

    print("\nGenerating cross-pipeline overview index")
    generate_html_overview_index(html_directory=HTML_DECISION_TREES_DIR)
