"""Every SVA-style agreement condition (sv/sp/sa, ov/op/oa, iov/iop/ioa x Na/Ga/Pa) as data.
scripts/agreement_trees/fit_candidate.py <id> just names its condition;
run_vp (cli.py) builds the pipeline from this table."""

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
from agreement_trees.extract.prediction_target import (
    PredictionTarget, nsubj_target, obj_agr_target, iobj_agr_target, filter_head_feats, filter_other_verb_form,
)
from agreement_trees.aux.redirect import nsubj_aux_target, obj_aux_target, iobj_aux_target

FEATURES = {"Na": "Number", "Ga": "Gender", "Pa": "Person"}

# (relation, feature suffix) -> inflection map (swap feature candidates, value map or None).
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
    # "fin" (finite), "part" (participles), "other" (anything else, incl. no VerbForm) or None (no VerbForm filter)
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
        # "fin" means strictly finite (VerbForm=Fin), not merely "not a participle"
        if self.verb_form == "other":
            target.head_feats = {"VerbForm": filter_other_verb_form}
        else:
            target.head_feats = {"VerbForm": partial(
                filter_head_feats, require="Part" if self.verb_form == "part" else "Fin")}
        return target

    def word_order_dir(self):
        return os.path.join(TREEBANK_FEATURES_DIR, self.word_order_subdir)


def _build_conditions():
    conditions = {}
    # (prefix, relation target, verb form / aux extraction target)
    families = [
        ("sv", nsubj_target, "fin", None),
        ("sp", nsubj_target, "part", None),
        ("sx", nsubj_target, "other", None),
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
