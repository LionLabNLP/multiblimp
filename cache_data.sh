#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Data caching: extraction-only sweeps (settings in resources/agreement_config.json)
(cd scripts/agreement_trees && python3 cache_treebank_features.py)

# Agreement condition scans (read the caches above, write the agreement configs)
(cd scripts/agreement_trees && python3 find_agreement_candidates.py)
