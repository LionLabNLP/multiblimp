"""Fit a single SVA-style agreement condition -- either one of the 27 fixed
Na/Ga/Pa conditions (sv/sp/sa/.../ioa x Number/Gender/Person), or an ad-hoc
(family, feature) candidate from agreement_candidates.py's scan that isn't
wired to a real condition. Replaces the old one script per fixed condition
(svNa.py, svGa.py, ...) and the old separate, ad-hoc-only fit_candidate.py
-- both were thin dispatches into sva_trees.cli with no logic of their
own; which mode runs is picked by the first argument instead of by which
script you invoke. Named to match npa/fit_candidate.py (that one takes a
single flat target_col; this one takes either a fixed condition id or
--family/--feature, since sva_trees splits "known" from "candidate").

Usage (run from scripts/sva_trees):
  python3 fit_candidate.py svNa --langs English German -j 2   # fixed condition
  python3 fit_candidate.py saPa --recache
  python3 fit_candidate.py --family sv --feature Polite        # ad-hoc candidate

Everything after is passed straight to the matching sva_trees.cli parser
(see cli.py's build_parser for the full flag list shared by both modes;
run_candidate_condition adds --family/--feature/--config on top of it).
"""
import sys

sys.path.append("../../src/")

from sva_trees.cli import run_condition, run_candidate_condition
from sva_trees.conditions import CONDITIONS

if __name__ == "__main__":
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(
            f"{__doc__}\n"
            f"Fixed condition ids: {', '.join(sorted(CONDITIONS))}"
        )

    if argv[0].startswith("-"):
        # No condition id given -- candidate mode (--family/--feature are
        # required there; run_candidate_condition's own parser enforces it).
        run_candidate_condition(argv)
    else:
        condition_id, rest = argv[0], argv[1:]
        if condition_id not in CONDITIONS:
            sys.exit(
                f"Unknown condition {condition_id!r}. "
                f"Choices: {', '.join(sorted(CONDITIONS))}\n"
                "For an ad-hoc candidate instead, start with --family/--feature."
            )
        run_condition(condition_id, rest)
