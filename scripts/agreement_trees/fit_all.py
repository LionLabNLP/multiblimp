"""Fit every agreement target in turn via fit_candidate.py, then rebuild the
overview indexes.

Targets: "fit.targets" in resources/agreement_config.json if non-empty,
otherwise every target in the saved agreement scan results (agreement_langs_vp /
_np.json) for the enabled "modes". Use --targets to scope a run. The scan only
says a target passes the statistical bar, not that it is real agreement (e.g.
sv:Case or a lexical feature can pass while spurious): list vetted targets in
"fit.targets" to keep a full sweep to those.

--include_excluded is passed to every fit. Anything else on the command line is
passed through to every fit_candidate.py call, e.g.
  python3 fit_all.py --langs English German -j 2
  python3 fit_all.py --targets sv:Number HEAD-DET_Number
  python3 fit_all.py -rf

Runs are cache-aware (finished languages are skipped), but a full run is long.
"""
import argparse
import os
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(_here)), "src"))

from agreement_trees.cli import SKIPPED_EXIT
from agreement_trees.config import (
    apply_mode_flags, load_agreement_config, load_scan_results, parse_vp_target,
)

DEFAULT_FLAGS = ["--include_excluded"]
INDEX_SCRIPT = os.path.join(_here, "..", "overview", "generate_html_indexes.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vp", action="store_true", help="Only verb-phrase targets.")
    parser.add_argument("--np", action="store_true", help="Only noun-phrase targets.")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Targets to fit (default: see above).")
    parser.add_argument("--skip_indexes", action="store_true",
                        help="Build no indexes: every fit gets --no_index and generate_html_indexes.py "
                             "is not run afterwards.")
    parser.add_argument("--index_jobs", type=int, default=None,
                        help="Workers for the final index build (default: generate_html_indexes.py's own).")
    parser.add_argument("--config", default=None, help="Agreement config JSON.")
    args, passthrough = parser.parse_known_args()
    if args.config:
        passthrough += ["--config", args.config]

    config = apply_mode_flags(load_agreement_config(args.config), args.vp, args.np)
    targets = args.targets or config["fit"]["targets"]
    if targets:
        targets = [t for t in targets if ("vp" if parse_vp_target(t) else "np") in config["modes"]]
    else:
        targets = [t for mode in config["modes"] for t in sorted(load_scan_results(mode).get("targets", {}))]
    if not targets:
        sys.exit("No targets: run find_agreement_candidates.py first, or list targets in fit.targets.")

    failed, skipped = [], []
    for target in targets:
        cmd = [sys.executable, os.path.join(_here, "fit_candidate.py"), target,
               *DEFAULT_FLAGS, "--no_index", *passthrough]
        print(f"\n=== {' '.join(cmd)}", flush=True)
        returncode = subprocess.run(cmd).returncode
        if returncode == SKIPPED_EXIT:
            skipped.append(target)
        elif returncode:
            failed.append(target)

    if not args.skip_indexes:
        cmd = [sys.executable, INDEX_SCRIPT]
        if args.index_jobs:
            cmd += ["-j", str(args.index_jobs)]
        print(f"\n=== {' '.join(cmd)}", flush=True)
        subprocess.run(cmd)

    if skipped:
        print(f"\nSkipped (no eligible language): {', '.join(skipped)}")
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
