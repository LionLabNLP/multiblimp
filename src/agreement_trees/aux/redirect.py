from collections import defaultdict
from functools import partial

from agreement_trees.extract.prediction_target import PredictionTarget, child_slot, filter_head_feats

# Synthetic feature written into the chosen aux token's feats by redirect_to_aux ("1" or "2+" auxiliaries
# on the predicate); extract_node_features picks it up like a real UD feature.
AUX_COUNT_FEAT = "SubjAuxCount"

# Fresh PredictionTarget instances (not agreement_trees.extract.prediction_target's singletons, which callers mutate in place).
nsubj_aux_target = PredictionTarget(
    child_deprels=["nsubj"],
    head_pos=["AUX"],
    child_pos=["NOUN", "PROPN", "PRON"],
    head_feats={"VerbForm": partial(filter_head_feats)},
)


# object / indirect-object counterparts (e.g. Basque synthetic auxiliaries)
obj_aux_target = PredictionTarget(
    child_deprels=["obj"],
    head_pos=["AUX"],
    child_pos=["NOUN", "PROPN", "PRON"],
    head_feats={"VerbForm": partial(filter_head_feats)},
)

iobj_aux_target = PredictionTarget(
    child_deprels=["iobj"],
    head_pos=["AUX"],
    child_pos=["NOUN", "PROPN", "PRON"],
    head_feats={"VerbForm": partial(filter_head_feats)},
)


def redirect_to_aux(treebank, deprels=("nsubj",)) -> None:
    """Re-point every `deprels` dependent (nsubj by default; obj/iobj work the same) of a predicate that has an
    aux, aux:pass or cop child at that auxiliary, so the ordinary head-vs-child extraction compares the
    argument with the token that carries the agreement (not a non-finite main verb or a non-verbal predicate).
    A cop counts only if it is tagged AUX. Clauses without aux/cop are untouched: their head stays a VERB or
    non-verbal predicate and the target's head_pos=["AUX"] filters them out.

    The chosen token's feats get AUX_COUNT_FEAT = "1" (one aux/aux:pass/cop child) or "2+" (stacked, or
    aux + cop); AuxPipeline selects on it. Choice among several: the child marked VerbForm=Fin (leftmost if
    several), else the leftmost. Leftmost alone was wrong for 16/159 stacked German instances, where the finite
    auxiliary can come last ("gelesen haben wird").

    Mutates tokens in place: use a freshly loaded Treebank. One pass per tree collects the dependents and a
    head -> auxiliaries index (a separate "any aux?" pre-pass measured slower at corpus scale).
    """
    for tree in treebank:
        nsubj_tokens = []
        aux_by_head = defaultdict(list)
        for t in tree:
            deprel = t["deprel"]
            if child_slot(deprel) in deprels:
                nsubj_tokens.append(t)
            elif deprel == "aux" or deprel == "aux:pass" or (deprel == "cop" and t["upos"] == "AUX"):
                aux_by_head[t["head"]].append(t)

        if not nsubj_tokens or not aux_by_head:
            continue

        for token in nsubj_tokens:
            aux_children = aux_by_head.get(token["head"])
            if not aux_children:
                continue
            aux_children = sorted(aux_children, key=lambda t: t["id"])

            fin_children = [
                t for t in aux_children if (t.get("feats") or {}).get("VerbForm") == "Fin"
            ]
            chosen = fin_children[0] if fin_children else aux_children[0]
            token["head"] = chosen["id"]

            if type(chosen.get("feats")) != dict:
                chosen["feats"] = dict()
            chosen["feats"][AUX_COUNT_FEAT] = "1" if len(aux_children) == 1 else "2+"
