import sys

sys.path.append("../")

from word_order.prediction_target import PredictionTarget

# --- < NPA (Noun Phrase Agreement) > ---
# Mirrors sva_trees' "< SVA >" targets, but for NP-internal concord: a
# NOUN/PROPN/PRON head with a single dependent type per target (det/nummod/
# amod/case/nmod:poss), rather than subject-verb agreement across a clause.
#
# Deliberately separate PredictionTarget instances from word_order.
# prediction_target's amod_target/case_target (even though those are
# equivalent today): scripts here build/mutate several targets in the same
# process to compare extraction sweeps, and PredictionTarget fields (e.g.
# head_feats, swap_feat) get mutated in place by callers (see sva_trees
# scripts) -- sharing instances would let one target's mutation leak into
# another's run.
#
# Previously-open questions, now resolved: heads include PROPN and PRON
# alongside NOUN (PROPN rarely takes det/amod but commonly takes case, e.g.
# "to Berlin", and can itself be an nmod:poss dependent, e.g. "mayor of
# Berlin's office"; PRON heads, e.g. "she"/"mine", rarely take det/amod
# either but can take nmod:poss, e.g. "a friend of mine"). Children now also
# include nummod (numeral concord, e.g. "these three dogs") and nmod:poss
# (possessive-pronoun modifier, e.g. "a friend of mine" -- specifically the
# "nmod:poss" deprel subtype, restricted to PRON children, not plain "nmod"
# generally: an empirical sweep comparing candidate selection criteria
# showed neither a upos+Poss=Yes filter nor a upos=PRON+Case=Gen filter
# reliably substitutes for UD's own "nmod:poss" tag across languages, and
# genitive-noun possessors, e.g. Turkish izafet or English "John's", aren't
# pronouns, hence PRON-only). nummod/nmod:poss deliberately still get their
# own single-deprel targets here (like det/amod/case), not word_order.
# prediction_target.dnan_target's require-all-three-at-once grouping -- see
# the module docstring above for why NPA keeps one deprel per target rather
# than requiring co-occurrence.

det_target = PredictionTarget(
    child_deprels=["det"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos=["DET"],
)

nummod_target = PredictionTarget(
    child_deprels=["nummod"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos=["NUM"],
)

amod_target = PredictionTarget(
    child_deprels=["amod"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos=["ADJ"],
)

case_target = PredictionTarget(
    child_deprels=["case"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos=["ADP"],
)

nmod_target = PredictionTarget(
    child_deprels=["nmod:poss"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos=["PRON"],
)

npa_targets = {
    "det": det_target,
    "nummod": nummod_target,
    "amod": amod_target,
    "case": case_target,
    "nmod": nmod_target,
}

np_target = PredictionTarget(
    child_deprels=["det", "nummod", "amod", "case", "nmod:poss"],
    head_pos=["NOUN", "PROPN", "PRON"],
    child_pos={
        "det": ["DET"],
        "nummod": ["NUM"],
        "amod": ["ADJ"],
        "case": ["ADP"],
        "nmod:poss": ["PRON"],
    },
)