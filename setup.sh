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
  cd ud-treebanks-v2.18/UD_Polish-LFG
  for f in train dev test; do
    python3 ../../fix_polish_subgender_animacy.py pl_lfg-ud-$f.conllu
  done
)

# UD pickles, then UD-derived UniMorph
(cd scripts/pickle && python3 pickle_treebanks.py)
python3 scripts/ud2um.py

# UniMorph (needs a GitHub SSH key)
(cd resources/unimorph && bash download_all.sh && bash changelog.sh)
(cd scripts/pickle && python3 pickle_unimorph.py)