from dataclasses import dataclass
from typing import Callable


def filter_head_feats(value, exclude=None, require=None):
    """Predicate for PredictionTarget.head_feats (e.g. {"VerbForm": ...}); picklable,
    use via functools.partial:
        target.head_feats = {"VerbForm": partial(filter_head_feats, require="Fin")}

    Prefix match (startswith), so finer-grained participle values such as PartRes,
    PartFut or PartPast count as Part. A multi-valued value ("Fin,Inf") never passes;
    a missing value fails require= and passes exclude=. For VerbForm, a missing value
    has already been restored from the lexicon or assumed Fin by
    process_treebank.extract_node_features.
    """
    if isinstance(value, str) and "," in value:
        return False
    matches = isinstance(value, str) and value.startswith(exclude if exclude is not None else (require or ""))
    if exclude is not None:
        return not matches
    if require is not None:
        return matches
    return True


@dataclass
class PredictionTarget:
    """What to predict: the child deprel(s) of a head, optionally restricted by the head's deprel, POS or
    feature predicates and by the children's POS. Targets are module-level singletons that callers copy
    (dataclasses.replace) before changing swap_feat or head_feats.

    Example: PredictionTarget(child_deprels=["nsubj"], head_pos=["VERB"], child_pos=["NOUN", "PROPN", "PRON"])
    """

    child_deprels: list[str]

    # head's deprel (e.g. only root verbs)
    head_deprel: str | None = None

    # head's POS
    head_pos: list[str] | None = None

    # child's POS: one list for all children or {deprel: list}
    child_pos: list[str] | dict[str, list[str]] | None = None

    # {feature: predicate on the head's value}, e.g. {"VerbForm": partial(filter_head_feats, require="Inf")}
    head_feats: dict[str, Callable] | None = None

    # feature to swap inflection for
    swap_feat: str | None = None

    def __post_init__(self):
        if not self.child_deprels:
            raise ValueError("child_deprels must be non-empty")


# Child deprels that fill another deprel's slot in a target (a passive subject is a subject).
CHILD_DEPREL_ALIASES = {"nsubj:pass": "nsubj"}


def child_slot(deprel: str) -> str:
    return CHILD_DEPREL_ALIASES.get(deprel, deprel)


# --- < SVA > ---
nsubj_target = PredictionTarget(
    child_deprels=["nsubj"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"]
)

# Object-verb / indirect-object-verb agreement: same shape as nsubj_target (no head_deprel="root", so
# subordinate clauses count). Polypersonal agreement is rare in UD (Basque, Georgian, Abkhaz, Swahili,
# Hungarian's definite conjugation, ...) and mostly annotated with layered features (Number[obj],
# Person[erg]); see multiblimp.swap_features' swap_number_obj_any and friends.
obj_agr_target = PredictionTarget(
    child_deprels=["obj"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"]
)

iobj_agr_target = PredictionTarget(
    child_deprels=["iobj"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"]
)
