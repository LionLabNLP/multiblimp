UD_PATH = "ud/ud-treebanks-v2.18/"

import os

# Seed for the decision-tree fit and the leaf sampling.
SEED = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Input data (UD/UniMorph checkouts and pickles, scan configs); never redirected by MULTIBLIMP_RUN_DIR.
RESOURCE_DIR = os.path.join(REPO_ROOT, "resources")

# Checkout of UniversalDependencies/tools (validate.py + udtools); override with UD_TOOLS_DIR.
UD_TOOLS_DIR = os.environ.get("UD_TOOLS_DIR") or os.path.join(os.path.dirname(REPO_ROOT), "tools")
# Per-treebank validator results (see multiblimp.validation); input-side data like the pickles.
VALIDATION_DIR = os.path.join(RESOURCE_DIR, "ud", "validation")

# MULTIBLIMP_RUN_DIR redirects every generated artifact (output/, html/,
# treebank_features/) into that directory, e.g. for a single-language run
# that must not touch the main caches. Resources stay in resources/.
RUN_DIR = os.environ.get("MULTIBLIMP_RUN_DIR") or REPO_ROOT

OUTPUT_DIR = os.path.join(RUN_DIR, "output")
OUTPUT_DECISION_TREES_DIR = os.path.join(OUTPUT_DIR, "decision_trees")
OUTPUT_MINIMAL_PAIRS_DIR = os.path.join(OUTPUT_DIR, "minimal_pairs")
OUTPUT_DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
OUTPUT_OVERVIEW_DIR = os.path.join(OUTPUT_DIR, "overview")
OUTPUT_BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
OUTPUT_MODEL_RESULTS_DIR = os.path.join(OUTPUT_DIR, "model_results")

TREEBANK_FEATURES_DIR = os.path.join(RUN_DIR, "treebank_features")
NP_INSTANCES_DIR = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")
NP_REPORT_DIR = os.path.join(RUN_DIR, "report", "np_types")

HTML_DIR = os.path.join(RUN_DIR, "html")
HTML_DECISION_TREES_DIR = os.path.join(HTML_DIR, "decision_trees")

# One overwritten-per-run log per script.
OUTPUT_LOGS_DIR = os.path.join(RUN_DIR, "output", "logs")

# Scan results (target -> languages) live next to the settings, but follow MULTIBLIMP_RUN_DIR
# so an isolated run never overwrites the main ones.
SCAN_RESULTS_DIR = RESOURCE_DIR if RUN_DIR == REPO_ROOT else os.path.join(RUN_DIR, "resources")

# Candidate-scan tables.
OUTPUT_AGREEMENT_CANDIDATES_DIR = os.path.join(OUTPUT_DIR, "agreement_candidates")
