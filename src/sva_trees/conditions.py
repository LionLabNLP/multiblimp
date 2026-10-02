"""Every SVA-style agreement condition (sv/sp/sa, ov/op/oa, iov/iop/ioa x Na/Ga/Pa) as data.
scripts/sva_trees/fit_candidate.py <id> just names its condition;
run_condition (cli.py) builds the pipeline from this table."""

import dataclasses
import os
from dataclasses import dataclass
from functools import partial

from multiblimp.config import TREEBANK_FEATURES_DIR
from multiblimp.swap_features import (
    swap_number_subj_any, swap_gender_any, swap_any_person,
    swap_number_obj_any, swap_gender_obj_any, swap_person_obj_any,
    swap_number_iobj_any, swap_gender_iobj_any, swap_person_iobj_any,
)
from word_order.prediction_target import (
    PredictionTarget, nsubj_target, obj_agr_target, iobj_agr_target, filter_head_feats,
)
from subj_aux.redirect import nsubj_aux_target, obj_aux_target, iobj_aux_target

FEATURES = {"Na": "Number", "Ga": "Gender", "Pa": "Person"}

# (relation, feature suffix) -> swap function.
INFLECTION_MAPS = {
    ("nsubj", "Na"): swap_number_subj_any,
    ("nsubj", "Ga"): swap_gender_any,
    ("nsubj", "Pa"): swap_any_person,
    ("obj", "Na"): swap_number_obj_any,
    ("obj", "Ga"): swap_gender_obj_any,
    ("obj", "Pa"): swap_person_obj_any,
    ("iobj", "Na"): swap_number_iobj_any,
    ("iobj", "Ga"): swap_gender_iobj_any,
    ("iobj", "Pa"): swap_person_iobj_any,
}

VERB_UNIMORPH_ARGS = {
    "filter_entries": {"upos": ["V"]},
    "combine_um_ud": True,
    "remove_multiword_forms": True,
}
AUX_UNIMORPH_ARGS = {
    "filter_entries": {"upos": ["V", "AUX"]},
    "combine_um_ud": True,
    "remove_multiword_forms": True,
}


@dataclass(frozen=True)
class Condition:
    id: str
    base_target: PredictionTarget
    feature_suffix: str
    # "fin" (non-participles), "part" (participles) or None (no VerbForm filter)
    verb_form: str | None
    unimorph_args: dict
    word_order_subdir: str
    rm_columns: tuple
    # extraction target of an auxiliary condition (subj_aux/redirect.py), else None
    aux_target: PredictionTarget | None = None

    @property
    def is_aux(self):
        return self.aux_target is not None

    @property
    def head_label(self):
        if self.is_aux:
            return "Aux"
        return "Participle" if self.verb_form == "part" else "Verb"

    @property
    def swap_feat(self):
        return FEATURES[self.feature_suffix]

    @property
    def deprel(self):
        return self.base_target.child_deprels[0]

    @property
    def inflection_map(self):
        return INFLECTION_MAPS[(self.deprel, self.feature_suffix)]

    @property
    def predictor_var(self):
        return f"head_{self.deprel}_{self.swap_feat}_agreement"

    def make_target(self) -> PredictionTarget:
        """A fresh copy: the base targets are module-level singletons."""
        target = dataclasses.replace(self.base_target, swap_feat=self.swap_feat)
        if self.is_aux:
            return target
        # "fin" means strictly finite (VerbForm=Fin, prefix-matched same as
        # "part"/Part -- see filter_head_feats's own docstring for why
        # prefix, not equality), not merely "not a participle": an
        # exclude="Part" filter here let non-finite, non-participle forms
        # (VerbForm=Inf, Vnoun, Conv, ...) straight through as long as they
        # were UPOS=VERB, which is how e.g. Hindi/Sanskrit infinitives and
        # Turkish verbal nouns (VerbForm=Vnoun) ended up counted as "sv"
        # instances -- caught via sva_trees/agreement_candidates.py
        # surfacing a spurious "Case agreement" candidate that turned out to
        # be infinitival/nominalized-clause case-marking (both the
        # nominalized head and its subject get their own, usually
        # DIFFERENT, case), not real subject-verb concord.
        # missing_matches=True only for the "fin" branch: an UNTAGGED verb
        # is assumed finite (UD convention -- Fin is the unmarked default;
        # see filter_head_feats' own docstring and process_treebank.
        # expand_anno's identical assumption), but a missing tag must never
        # count as "part" -- that's the inverse mistake (a treebank that
        # just doesn't annotate VerbForm would then wrongly feed sv*'s
        # non-finite rows into sp* too).
        target.head_feats = {"VerbForm": partial(
            filter_head_feats,
            **({"require": "Part"} if self.verb_form == "part"
               else {"require": "Fin", "missing_matches": True}))}
        return target

    def word_order_dir(self):
        return os.path.join(TREEBANK_FEATURES_DIR, self.word_order_subdir)


def _build_conditions():
    conditions = {}
    # (prefix, relation target, verb form / aux extraction target)
    families = [
        ("sv", nsubj_target, "fin", None),
        ("sp", nsubj_target, "part", None),
        ("sa", nsubj_target, None, nsubj_aux_target),
        ("ov", obj_agr_target, "fin", None),
        ("op", obj_agr_target, "part", None),
        ("oa", obj_agr_target, None, obj_aux_target),
        ("iov", iobj_agr_target, "fin", None),
        ("iop", iobj_agr_target, "part", None),
        ("ioa", iobj_agr_target, None, iobj_aux_target),
    ]
    for suffix in FEATURES:
        for prefix, target, verb_form, aux_target in families:
            deprel = target.child_deprels[0]
            cid = f"{prefix}{suffix}"
            conditions[cid] = Condition(
                id=cid, base_target=aux_target or target, feature_suffix=suffix, verb_form=verb_form,
                unimorph_args=AUX_UNIMORPH_ARGS if aux_target else VERB_UNIMORPH_ARGS,
                # sa keeps its original cache dir name
                word_order_subdir=(("subj_aux" if deprel == "nsubj" else f"{deprel}_aux")
                                   if aux_target else f"{deprel}_{verb_form}"),
                rm_columns=(f"{deprel}_child-deprel_conj",),
                aux_target=aux_target,
            )
    return conditions


CONDITIONS = _build_conditions()
