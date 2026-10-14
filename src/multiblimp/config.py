UD_PATH = "ud/ud-treebanks-v2.18/"

import os

# Seed for the decision-tree fit and the leaf sampling.
SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Input data (UD/UniMorph checkouts and pickles, scan configs); never redirected by MULTIBLIMP_RUN_DIR.
RESOURCE_DIR = os.path.join(REPO_ROOT, "resources")

# MULTIBLIMP_RUN_DIR redirects every generated artifact (output/, html/,
# treebank_features/) into that directory, e.g. for a single-language run
# that must not touch the main caches. Resources stay in resources/.
RUN_DIR = os.environ.get("MULTIBLIMP_RUN_DIR") or REPO_ROOT

# MULTIBLIMP_DEBUG_MODE switches the pipeline into its data-debugging mode
# (fit_candidate.py --debug): no minimal pairs, no annotation filling (fetch_all
# off), but pooled/per-treebank/incl.-unk trees. Must be set in the environment
# before this module is first imported, since these paths are fixed at import
# time. Output, html and the feature caches get a "_debug" suffix: the caches
# differ by fetch_all, so a debug run must never share them with the main mode.
DEBUG_MODE = os.environ.get("MULTIBLIMP_DEBUG_MODE", "") not in ("", "0", "false", "False")
_debug_suffix = "_debug" if DEBUG_MODE else ""

OUTPUT_DIR = os.path.join(RUN_DIR, f"output{_debug_suffix}")
OUTPUT_DECISION_TREES_DIR = os.path.join(OUTPUT_DIR, "decision_trees")
OUTPUT_MINIMAL_PAIRS_DIR = os.path.join(OUTPUT_DIR, "minimal_pairs")
OUTPUT_DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
OUTPUT_OVERVIEW_DIR = os.path.join(OUTPUT_DIR, "overview")

TREEBANK_FEATURES_DIR = os.path.join(RUN_DIR, f"treebank_features{_debug_suffix}")
NP_INSTANCES_DIR = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")
NP_REPORT_DIR = os.path.join(RUN_DIR, f"report{_debug_suffix}", "np_types")

HTML_DIR = os.path.join(RUN_DIR, f"html{_debug_suffix}")
HTML_DECISION_TREES_DIR = os.path.join(HTML_DIR, "decision_trees")

# One overwritten-per-run log per script, shared by both modes.
OUTPUT_LOGS_DIR = os.path.join(RUN_DIR, "output", "logs")

# Scan results (target -> languages) live next to the settings, but follow MULTIBLIMP_RUN_DIR
# so an isolated run never overwrites the main ones.
SCAN_RESULTS_DIR = RESOURCE_DIR if RUN_DIR == REPO_ROOT else os.path.join(RUN_DIR, "resources")
DEBUG_SUFFIX = _debug_suffix

# Candidate-scan tables (unsuffixed like logs: shared by both modes).
OUTPUT_AGREEMENT_CANDIDATES_DIR = os.path.join(OUTPUT_DIR, "agreement_candidates")
