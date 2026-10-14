"""Fit one agreement target: decision tree, minimal pairs and pages.

  verb phrase   sv:Number (or svNa), sp:Case, ioa:Person, ...  family:feature,
                families sv sp sa ov op oa iov iop ioa
  noun phrase   HEAD-DET_Number, DET-ADJ_Gender, ...           ROLE1-ROLE2_Feature

By default the languages are those the agreement scan found worth attempting
for the target (find_agreement_candidates.py; a live scan if it is not in the
results). A language without a feature cache is extracted first, per the
"vp"/"np" blocks of resources/agreement_config.json, which also holds the tree
settings ("fit") and the second-chance retry.

Exit status: 0 done, 1 failed, 3 skipped (no eligible language).

Usage (any directory):
  python3 fit_candidate.py sv:Number --langs English German -j 2
  python3 fit_candidate.py HEAD-DET_Number -rf
  python3 fit_candidate.py sp:Case --include_excluded
  MULTIBLIMP_DEBUG_MODE=1 python3 fit_candidate.py sv:Number --debug
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from agreement_trees.config import parse_vp_target
from agreement_trees.cli import SKIPPED_EXIT, TargetSkipped, parse_fit_args


def main():
    args, config = parse_fit_args()
    try:
        if parse_vp_target(args.target) is not None:
            from agreement_trees.cli import run_vp
            run_vp(args.target, args, config)
        else:
            from agreement_trees.cli import run_np
            run_np(args.target, args, config)
    except TargetSkipped as e:
        print(f"Skipping {args.target}: {e}")
        sys.exit(SKIPPED_EXIT)


if __name__ == "__main__":
    main()
