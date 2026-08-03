from dataclasses import dataclass


def filter_head_feats(value, exclude=None, require=None):
    """Predicate for PredictionTarget.head_feats (e.g. {"VerbForm": ...}).

    Picklable replacement for the ad-hoc `lambda x: x != "Part"` /
    `lambda x: x == "Part"` the scripts/sva_trees/*.py runners used to set
    head_feats with — plain lambdas can't be pickled, which breaks
    sva_trees.pipeline.Pipeline's n_jobs > 1 (ProcessPoolExecutor pickles the
    whole Pipeline, including target.head_feats, to send to each worker).

    Use via functools.partial, e.g.:
        target.head_feats = {"VerbForm": partial(filter_head_feats, exclude="Part")}
        target.head_feats = {"VerbForm": partial(filter_head_feats, require="Part")}
    """
    if exclude is not None:
        return value != exclude
    if require is not None:
        return value == require
    return True


@dataclass
class PredictionTarget:
    """
    Defines what word order pattern to predict.

    Examples:
        PredictionTarget(child_deprels=["amod"])
        PredictionTarget(child_deprels=["nsubj", "obj"], head_pos="VERB", head_deprel="root")
    """

    child_deprels: list[str]

    # Optional: filter by head's deprel (e.g., only root verbs for SVO)
    head_deprel: str | None = None

    # Optional: filter by head's POS
    head_pos: list[str] | None = None

    # Optional: filter by child's POS (either all children same POS or mapping deprel->POS)
    child_pos: list[str] | dict[str, list[str]] | None = None

    # Optional: filter by head's specific feature annotation
    head_feats: dict[str: str] | None = None

    # Feature to swap inflection for
    swap_feat: str | None = None

    def __post_init__(self):
        if not self.child_deprels:
            raise ValueError("child_deprels must be non-empty")


amod_target = PredictionTarget(
    child_deprels=["amod"],
    head_pos=["NOUN"],
    child_pos=["ADJ"],
)

core_arg_target = PredictionTarget(
    child_deprels=["nsubj", "obj"],
    head_pos=["VERB"],
    head_deprel="root",
)

iobj_target = PredictionTarget(
    child_deprels=["iobj"],
    head_pos=["VERB"],
    head_deprel="root",
)

dnan_target = PredictionTarget(
    child_deprels=["det", "nummod", "amod"],
    head_pos=["NOUN"],
    child_pos={"det": ["DET"], "nummod": ["NUM"], "amod": ["ADJ"]},
)

acl_relcl_target = PredictionTarget(
    child_deprels=["acl:relcl"],
    head_pos=["NOUN"],
    child_pos=["VERB"],
)

obl_target = PredictionTarget(
    child_deprels=["obl"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"],
)

obl_noun_target = PredictionTarget(
    child_deprels=["obl"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"],
)

case_target = PredictionTarget(
    child_deprels=["case"],
    head_pos=["NOUN"],
    child_pos=["ADP"],
)

advmod_target = PredictionTarget(
    child_deprels=["advmod"],
    head_pos=["VERB"],
    child_pos=["ADV"],
)

nmod_noun_target = PredictionTarget(
    child_deprels=["nmod"],
    head_pos=["NOUN"],
    child_pos=["NOUN"],
)

advcl_noun_verb_target = PredictionTarget(
    child_deprels=["advcl"],
    head_pos=["VERB"],
    child_pos=["NOUN"],
)

# --- < SVA > ---
nsubj_target = PredictionTarget(
    child_deprels=["nsubj"],
    head_pos=["VERB"],
    child_pos=["NOUN", "PROPN", "PRON"]
)
