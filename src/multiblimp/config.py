UD_PATH = "ud/ud-treebanks-v2.18/"

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MULTIBLIMP_RUN_DIR redirects every generated artifact (output/, html/,
# treebank_features/) into that directory, e.g. for a single-language run
# that must not touch the main caches. Resources stay in resources/.
RUN_DIR = os.environ.get("MULTIBLIMP_RUN_DIR") or REPO_ROOT

# MULTIBLIMP_DEBUG_MODE switches the pipeline into its data-debugging mode
# (see sva_trees.cli's --debug): no minimal pairs, expand_anno disabled, but
# pooled/per-treebank/incl.-unk trees. Must be set in the environment before
# this module is first imported, since these paths are fixed at import time.
# Output/html get a "_debug" suffix so a debug run can never share (and
# silently corrupt) the main mode's decision trees, pairs or pages;
# treebank_features/ stays unsuffixed -- both modes still read/extract UD
# treebanks through the same resource-level machinery.
DEBUG_MODE = os.environ.get("MULTIBLIMP_DEBUG_MODE", "") not in ("", "0", "false", "False")
_debug_suffix = "_debug" if DEBUG_MODE else ""

OUTPUT_DIR = os.path.join(RUN_DIR, f"output{_debug_suffix}")
OUTPUT_DECISION_TREES_DIR = os.path.join(OUTPUT_DIR, "decision_trees")
OUTPUT_MINIMAL_PAIRS_DIR = os.path.join(OUTPUT_DIR, "minimal_pairs")
OUTPUT_DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
OUTPUT_OVERVIEW_DIR = os.path.join(OUTPUT_DIR, "overview")

TREEBANK_FEATURES_DIR = os.path.join(RUN_DIR, "treebank_features")

HTML_DIR = os.path.join(RUN_DIR, f"html{_debug_suffix}")
HTML_DECISION_TREES_DIR = os.path.join(HTML_DIR, "decision_trees")
