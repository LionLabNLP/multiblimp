#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements.txt

# UD
git clone https://github.com/LionLabNLP/um2ud_annotation.git resources/um2ud_annotation
(
  cd resources/ud
  bash download_all.sh
  bash download_afrisud.sh
  bash changelog.sh
)

# Compare UD flags (scraped + README) against the exclusion lists; review any output
python3 scripts/agreement_trees/supplementary/flag_treebanks_report.py

# UD validator results (needs a UniversalDependencies/tools checkout, see UD_TOOLS_DIR), then the UD pickles
# without the sentences flagged by validation.drop_testids (resources/data_prep_config.json), then UD-derived tables
python3 scripts/data_caching/validate_treebanks.py
python3 scripts/data_caching/pickle_treebanks.py
python3 scripts/data_caching/ud_tables.py

# multiword-token statistics from the raw UD files (surface forms of the minimal pairs)
python3 scripts/data_caching/mwt_stats.py

# UniMorph (needs a GitHub SSH key), then its UD-notation tables (needs the UD-derived tables above)
(cd resources/unimorph && bash download_all.sh && bash changelog.sh)
python3 scripts/data_caching/um2ud.py
