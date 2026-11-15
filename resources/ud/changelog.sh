#!/usr/bin/env bash
# One-time, in-place edits to the downloaded data; skipped once the marker exists.
# Usage: changelog.sh [path/to/ud-treebanks-vX.Y]  (default: the release named in UD_VERSION)
set -eu
cd "$(dirname "$0")"
UD_DIR="${1:-ud-treebanks-v$(tr -d '[:space:]' < UD_VERSION)}"
UD_DIR="${UD_DIR%/}"
MARKER="$UD_DIR/.changelog_applied"
if [ -e "$MARKER" ]; then
  echo "changelog already applied ($MARKER), skipping"
  exit 0
fi

# rename_treebank OLD NEW: rename a folder in UD_DIR (skipped if OLD is missing or NEW exists)
rename_treebank() {
    if [ -d "$UD_DIR/$1" ] && [ ! -e "$UD_DIR/$2" ]; then
        mv "$UD_DIR/$1" "$UD_DIR/$2"
    else
        echo "Skipping rename $1 -> $2 (source missing or target exists)" >&2
    fi
}

# Rename UD_Norwegian-Bokmaal -> UD_Norwegian_Bokmaal-NDT
rename_treebank UD_Norwegian-Bokmaal UD_Norwegian_Bokmaal-NDT

# Rename UD_Norwegian-Nynorsk -> UD_Norwegian_Nynorsk-NDT
rename_treebank UD_Norwegian-Nynorsk UD_Norwegian_Nynorsk-NDT

# UD_Romanian-ArT is Aromanian, which we treat as its own language
rename_treebank UD_Romanian-ArT UD_Aromanian-ArT

# Remove colons from Gheg
sed -r "s/([a-z]*):([a-z]*)/\1\2/g" "$UD_DIR"/UD_Gheg-GPS/aln_gps-ud-test.conllu > tmp.conllu; mv tmp.conllu "$UD_DIR"/UD_Gheg-GPS/aln_gps-ud-test.conllu

# UD_Polish-LFG marks the virile/non-virile masculine split via SubGender=
# Masc{1,2,3}; every other Polish treebank (MPDT/PDB/PUD) marks the same
# split via Animacy=Hum/Nhum/Inan instead, so pooled Polish data lost this
# signal for every LFG-sourced token. Add the equivalent Animacy feature
# (fix_polish_subgender_animacy.py: additive only, keeps SubGender, see its
# own docstring) -- suggested upstream to UD as well, not just applied here.
for split in train dev test; do
    python3 fix_polish_subgender_animacy.py \
        "$UD_DIR"/UD_Polish-LFG/pl_lfg-ud-$split.conllu tmp.conllu
    mv tmp.conllu "$UD_DIR"/UD_Polish-LFG/pl_lfg-ud-$split.conllu
done

# Split the Old Norse part of UD_Icelandic-IcePaHC (texts dated up to and including 1350, 
# the year that starts each sent_id) into UD_Old_Norse-IcePaHC
mkdir -p "$UD_DIR"/UD_Old_Norse-IcePaHC
cp "$UD_DIR"/UD_Icelandic-IcePaHC/README.md "$UD_DIR"/UD_Icelandic-IcePaHC/LICENSE.txt "$UD_DIR"/UD_Old_Norse-IcePaHC/
for split in train dev test; do 
    awk 'BEGIN { RS = ""; ORS = "\n\n" } match($0, /# sent_id = [0-9]+/) && substr($0, RSTART + 12, RLENGTH - 12) + 0 <= 1350' "$UD_DIR"/UD_Icelandic-IcePaHC/is_icepahc-ud-$split.conllu > "$UD_DIR"/UD_Old_Norse-IcePaHC/non_icepahc-ud-$split.conllu;
done

touch "$MARKER"
