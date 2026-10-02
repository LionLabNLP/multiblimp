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
