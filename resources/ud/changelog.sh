#!/usr/bin/env bash
# One-time, in-place edits to the downloaded data; skipped once the marker exists.
set -eu
cd "$(dirname "$0")"
MARKER="ud-treebanks-v2.18/.changelog_applied"
if [ -e "$MARKER" ]; then
  echo "changelog already applied ($MARKER), skipping"
  exit 0
fi

# Rename UD_Norwegian-Bokmaal -> UD_Norwegian_Bokmål-NDT 
mv ud-treebanks-v2.18/UD_Norwegian-Bokmaal ud-treebanks-v2.18/UD_Norwegian_Bokmaal-NDT 

# Rename UD_Norwegian-Nynorsk -> UD_Norwegian_Nynorsk-NDT
mv ud-treebanks-v2.18/UD_Norwegian-Nynorsk ud-treebanks-v2.18/UD_Norwegian_Nynorsk-NDT 

# Remove colons from Gheg
sed -r "s/([a-z]*):([a-z]*)/\1\2/g" ud-treebanks-v2.18/UD_Gheg-GPS/aln_gps-ud-test.conllu > tmp.conllu; mv tmp.conllu ud-treebanks-v2.18/UD_Gheg-GPS/aln_gps-ud-test.conllu

# UD_Polish-LFG marks the virile/non-virile masculine split via SubGender=
# Masc{1,2,3}; every other Polish treebank (MPDT/PDB/PUD) marks the same
# split via Animacy=Hum/Nhum/Inan instead, so pooled Polish data lost this
# signal for every LFG-sourced token. Add the equivalent Animacy feature
# (fix_polish_subgender_animacy.py: additive only, keeps SubGender, see its
# own docstring) -- suggested upstream to UD as well, not just applied here.
for split in train dev test; do
    python3 fix_polish_subgender_animacy.py \
        ud-treebanks-v2.18/UD_Polish-LFG/pl_lfg-ud-$split.conllu tmp.conllu
    mv tmp.conllu ud-treebanks-v2.18/UD_Polish-LFG/pl_lfg-ud-$split.conllu
done

# Split the Old Norse part of UD_Icelandic-IcePaHC (texts dated up to and including 1350, 
# the year that starts each sent_id) into UD_Old_Norse-IcePaHC
mkdir -p ud-treebanks-v2.18/UD_Old_Norse-IcePaHC
cp ud-treebanks-v2.18/UD_Icelandic-IcePaHC/README.md ud-treebanks-v2.18/UD_Icelandic-IcePaHC/LICENSE.txt ud-treebanks-v2.18/UD_Old_Norse-IcePaHC/
for split in train dev test; do 
    awk 'BEGIN { RS = ""; ORS = "\n\n" } match($0, /# sent_id = [0-9]+/) && substr($0, RSTART + 12, RLENGTH - 12) + 0 <= 1350' ud-treebanks-v2.18/UD_Icelandic-IcePaHC/is_icepahc-ud-$split.conllu > ud-treebanks-v2.18/UD_Old_Norse-IcePaHC/non_icepahc-ud-$split.conllu;
done

touch "$MARKER"
