import sys
from collections import defaultdict

sys.path.append("../")

from word_order.prediction_target import PredictionTarget

# Synthetic feature key injected into the chosen aux token's own `feats`
# dict by redirect_nsubj_to_aux, picked up by word_order.process_treebank's
# existing feature-extraction machinery exactly like a real UD feature (see
# extract_node_features's `all_feats.update(set(node.get("feats",
# {}).keys()))` -- newly-seen keys get folded in on the fly, no frequency
# threshold or other gate to worry about). Namespaced/spelled unlike any
# real UD feature name to avoid collision.
AUX_COUNT_FEAT = "SubjAuxCount"

# A fresh PredictionTarget rather than mutating/reusing word_order.
# prediction_target.nsubj_target: that singleton gets mutated in place by
# sva_trees scripts (svNa.py etc. do `target.head_feats = ...`), so building
# our own avoids any chance of inheriting stray mutated state -- same
# reasoning npa.targets uses for not sharing instances with word_order.
# prediction_target's targets.
nsubj_aux_target = PredictionTarget(
    child_deprels=["nsubj"],
    head_pos=["AUX"],
    child_pos=["NOUN", "PROPN", "PRON"],
)


def redirect_nsubj_to_aux(treebank) -> None:
    """For every nsubj token whose governing predicate has an aux/aux:pass/cop
    child, rewrite the nsubj's head to point at that aux (or copula) instead
    of the predicate itself -- so nsubj_aux_target's ordinary head-vs-child
    extraction (word_order.process_treebank.extract_instances, unmodified)
    compares the subject against the token that actually carries the
    agreement morphology, not a (possibly non-finite) main verb or a
    non-verbal predicate that carries no agreement morphology of its own at
    all (e.g. an ADJ/NOUN predicate in a copular clause).

    cop is included alongside aux/aux:pass, upos=="AUX"-gated (cop is
    overwhelmingly AUX-tagged by UD convention, but not universally, e.g.
    some languages' pronominal-copula analyses use a different upos --
    those are deliberately left unredirected rather than risking a
    non-auxiliary token feeding into an "aux" agreement target). This picks
    up copular clauses ("She is happy": nsubj(happy, She), cop(happy, is))
    that plain aux/aux:pass would otherwise miss entirely -- cop attaches to
    the predicate exactly where aux/aux:pass would, so no extra lookup
    structure is needed, just widening which deprels feed the same
    head_id -> [children] index. A clause can have both at once ("He would
    be happy": aux(happy, would) + cop(happy, be)) -- treated as a 2-strong
    stacked group like any other, so the Fin tie-break below picks the
    actually-finite element regardless of whether it's technically the aux
    or the cop.

    Clauses with no aux/cop are left untouched: their nsubj still points at
    the main verb (upos VERB) or non-verbal predicate, so
    nsubj_aux_target's head_pos=["AUX"] filters them out downstream for
    free -- no separate "has an aux" check needed.

    Also stamps the chosen token's own feats with AUX_COUNT_FEAT = "1" or
    "2+", recording whether the predicate had exactly one aux/aux:pass/cop
    child or more than one (stacked auxiliaries, e.g. "will have been
    eating", or an aux+cop mix like "would be happy") -- callers split the
    extracted dataframe on the resulting "head_{AUX_COUNT_FEAT}" column
    (see pipeline.split_by_aux_count).

    Tie-break for stacked auxiliaries: prefers whichever aux/aux:pass/cop
    child is explicitly marked VerbForm=Fin (leftmost among ties, if
    somehow more than one is marked Fin); falls back to leftmost (lowest
    token id) only when none are marked Fin at all. Started out as a pure
    leftmost heuristic, but checking it against real German stacked-aux
    instances found leftmost wrong ~10% of the time (16/159): German
    subordinate-clause verb clusters can put the finite auxiliary *last*,
    not first ("...gelesen haben wird" -- Inf, Inf, Fin), the classic
    Oberfeld/Unterfeld verb-cluster-ordering flexibility. French's few
    stacked-aux instances were all leftmost-Fin, but there were too few
    (2) to be informative either way -- VerbForm=Fin is the more reliable
    signal generally, position was only ever an approximation of it.

    Mutates tokens in place. Callers should load a fresh Treebank per call
    (word_order.process_treebank.load_treebank does this) rather than
    reusing one already passed through this function for another target.

    Performance: one single pass per tree collects both the nsubj tokens
    and a head_id -> [aux/aux:pass/cop children] index simultaneously,
    turning each nsubj's lookup into an O(1) dict access instead of the
    naive approach's O(n) rescan of the whole tree per nsubj (O(n*k) total
    for k nsubj tokens in one sentence). Deliberately a single combined
    pass, not a separate "does this tree have any aux at all" pre-pass
    followed by a second pass to collect nsubj tokens: measured that
    two-pass version as *slower* than the naive approach at full-corpus
    scale, since it paid for building the aux index even on the (typically
    more common) sentences with no nsubj at all, which the naive approach
    barely touches (one cheap deprel check, nothing else). This version
    never does more per-tree work than one pass regardless of what the tree
    contains. Verified bit-identical output (head reassignment +
    AUX_COUNT_FEAT) to the naive version on the full German UD corpus (182k
    sentences), ~1.3x faster there (this timing predates the cop addition,
    which only widens an existing elif and doesn't change the algorithm's
    shape).
    """
    for tree in treebank:
        nsubj_tokens = []
        aux_by_head = defaultdict(list)
        for t in tree:
            deprel = t["deprel"]
            if deprel == "nsubj":
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
