"""Rebuild every built condition's diagnostics CSV and deprel index.html (VP and NP), in parallel, then the
cross-pipeline decision_trees/index.html once, purely from the on-disk output/decision_trees and
output/minimal_pairs artifacts (no refitting, no extraction caches). fit_all.py passes --no_index to each
condition so a sweep builds no indexes inline; this script is the one place they get built (a lone
fit_candidate.py run still builds its own unless given --no_index).

A condition is skipped unless something that feeds its index is newer than the index: its .joblib / .parquet /
meta.json inputs or the index-generating code (viz_deprel, html_deprel, diagnostics); --force bypasses the
check. Stale conditions run in a process pool (-j, default min(cpu_count, 8)), each worker memory-capped with
a fresh process per condition and its log printed as one block. Conditions are discovered on disk: VP ones as
output/decision_trees/<target_id>/<target_id>_<deprel>/ (a fixed id from agreement_trees.vp.conditions or an ad-hoc
<family><Feature> candidate), NP ones as output/decision_trees/npa/<npa_id>/; unmappable directories are
reported and skipped.

First step of scripts/overview/run_all.sh, which then builds the dataset overview.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# absolute paths (a relative entry resolves against the caller's cwd); REPO_ROOT for the `resources` package
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "src"))

from agreement_trees.report.overview import refresh_indexes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", "--target-col", "-c", "-t", dest="conditions", action="append",
        help="Limit the sweep to this condition -- repeatable. Matches an SVA "
             "target_id (svNa) or target_id/deprel (svNa/nsubj), or an NPA "
             "target_col (HEAD-DET_Number) or id (HEAD-DET_N). Default: every "
             "condition with at least one fitted language on disk.",
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
    parser.add_argument(
        "--jobs", "-j", type=int, default=None,
        help="Conditions refreshed in parallel (default min(cpu_count, 8), "
             "never more than the number to refresh or than fits in memory; 1 runs inline).",
    )
    parser.add_argument(
        "--max_worker_mem_gb", type=float, default=None,
        help="Per-worker memory cap in GB (default: 80%% of available RAM "
             "split across the workers). Each worker holds every fitted "
             "language of one condition in memory at once.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each condition's full build log, not just its status line "
             "(failures always print theirs).",
    )
    parser.add_argument(
        "--skip_overview", action="store_true",
        help="Don't rebuild the cross-pipeline decision_trees/index.html.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(refresh_indexes(_parse_args()))
