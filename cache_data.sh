#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Data caching: extraction-only sweeps (parse treebanks into the feature caches)
N_JOBS="${N_JOBS:-1}"
(cd scripts/sva_trees && python3 vp_types.py --all --n_jobs "$N_JOBS")
(cd scripts/npa && python3 np_types.py --streaming --max_treebank_len 10000 --n_jobs "$N_JOBS")

# Agreement condition scans (read the caches above, write the agreement configs)
(cd scripts/sva_trees && python3 agreement_candidates.py)
(cd scripts/npa && python3 agreement_candidates.py)
