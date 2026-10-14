
from word_order.prediction_target import PredictionTarget

# LEGACY (no importers; np_types.QUALIFYING_DEPS carries the same deprel/UPOS pairs).
# NPA targets: a NOUN/PROPN/PRON head with one dependent type per target (det, nummod, amod, case,
# nmod:poss on PRON children), separate PredictionTarget instances from word_order.prediction_target's
# because callers mutate target fields in place.

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