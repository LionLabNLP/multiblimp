"""Fit every eligible verbal condition in turn via fit_candidate.py, then
rebuild the overview indexes. Named to match npa/fit_all.py (the npa
analogue, which similarly drives npa/fit_candidate.py over every target_col
in npa_config.json). Merges the old run_all_sva.py and run_all_candidates.py
-- both were the same "loop over a target list, subprocess into the
single-condition script, track failures" shape, and once fit_candidate.py
itself merged its own two modes, there was nothing left to tell the two
drivers apart by except which list and which mode they fed it.

By default, runs every fixed Na/Ga/Pa condition (sv/sp/sa/.../ioa x
Number/Gender/Person). With --candidates, runs every scan-surfaced
(family, feature) candidate in sva_agreement_config.json instead
(agreement_candidates.py's own output -- see fit_candidate.py's docstring
for the --family/--feature mode this subprocesses into for those).

Caution with --candidates: sva_agreement_config.json only encodes "passes
agreement_pipeline_utils.passes_agreement_bar's statistical gate" -- not
"is real agreement" (see conversation: sv's own Case and Tamil's Animacy
both cleared that same gate while being spurious -- a non-finite-verb-as-
nominal confound and a constant, non-varying tag, respectively). Check
each candidate by hand first, then use --target_cols to scope this driver
to the ones that survived that check, rather than running the full set
unscoped.

Run from scripts/sva_trees. --include_excluded is added to every fixed-
condition run (not candidate runs). Per-treebank trees are on by default
(--no_per_treebank skips them). Anything else on the command line is
passed through to every fit_candidate.py call, e.g.
  python3 fit_all.py --langs English German -j 2
  python3 fit_all.py --target_cols svNa saNa
  python3 fit_all.py --candidates
  python3 fit_all.py --candidates --target_cols sv:Polite sv:Animacy

Runs are cache-aware (languages already done are skipped, unless --recache
is passed through), but a full run is long: check what is cached first.
"""
import argparse
import json
import subprocess
import sys

sys.path.append("../../src/")

from multiblimp.condition_taxonomy import GROUP_PREFIXES
from sva_trees.conditions import CONDITIONS

CANDIDATE_CONFIG = "../../resources/sva_agreement_config.json"
FIXED_FLAGS = ["--include_excluded"]

ORDER = [f"{prefix}{suffix}" for prefix in GROUP_PREFIXES for suffix in ("Na", "Pa", "Ga")]
assert set(ORDER) == set(CONDITIONS)


def all_candidates(config_path: str) -> list[tuple[str, str]]:
    with open(config_path) as f:
        config = json.load(f)
    return sorted(
        (family, feature)
        for family, feats in config.items()
        for feature in feats
    )


def _parse_candidate_target_cols(specs: list[str]) -> list[tuple[str, str]]:
    targets = []
    for spec in specs:
        family, sep, feature = spec.partition(":")
        if not sep:
            sys.exit(f'--target_cols entries must be "family:feature" with --candidates (got {spec!r})')
        targets.append((family, feature))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", action="store_true",
                        help="Run scan-surfaced (family, feature) candidates from "
                             "sva_agreement_config.json instead of the 27 fixed conditions.")
    parser.add_argument("--target_cols", nargs="*", default=None,
                        help='Default: every fixed condition, or (with --candidates) every '
                             '"family:feature" pair in sva_agreement_config.json. With '
                             "--candidates, entries must be \"family:feature\"; without it, "
                             "a fixed condition id (e.g. svNa).")
    parser.add_argument("--skip_indexes", action="store_true",
                        help="Don't run scripts/generate_html_indexes.py afterwards")
    args, passthrough = parser.parse_known_args()

    if args.candidates:
        targets = (
            all_candidates(CANDIDATE_CONFIG) if args.target_cols is None
            else _parse_candidate_target_cols(args.target_cols)
        )
        target_argvs = [["--family", family, "--feature", feature] for family, feature in targets]
        names = [f"{family}:{feature}" for family, feature in targets]
        flags = passthrough
    else:
        conditions = args.target_cols if args.target_cols is not None else ORDER
        unknown = sorted(set(conditions) - set(CONDITIONS))
        if unknown:
            sys.exit(f"Not a known condition: {unknown}")
        target_argvs = [[cid] for cid in conditions]
        names = conditions
        flags = FIXED_FLAGS + passthrough

    failed = []
    for name, target_argv in zip(names, target_argvs):
        cmd = [sys.executable, "fit_candidate.py", *target_argv, *flags]
        print(f"\n=== {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode:
            failed.append(name)

    if not args.skip_indexes:
        print("\n=== generate_html_indexes.py", flush=True)
        subprocess.run([sys.executable, "generate_html_indexes.py"], cwd="..")

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
