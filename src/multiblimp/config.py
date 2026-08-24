UD_PATH = "ud/ud-treebanks-v2.18/"

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
OUTPUT_DECISION_TREES_DIR = os.path.join(OUTPUT_DIR, "decision_trees")
OUTPUT_MINIMAL_PAIRS_DIR = os.path.join(OUTPUT_DIR, "minimal_pairs")
OUTPUT_DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
OUTPUT_OVERVIEW_DIR = os.path.join(OUTPUT_DIR, "overview")

TREEBANK_FEATURES_DIR = os.path.join(REPO_ROOT, "treebank_features")

HTML_DIR = os.path.join(REPO_ROOT, "html")
HTML_DECISION_TREES_DIR = os.path.join(HTML_DIR, "decision_trees")
