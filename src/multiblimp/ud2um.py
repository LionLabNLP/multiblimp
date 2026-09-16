import os
import sys
import pandas as pd
from collections import defaultdict, Counter
from typing import *

from tqdm import tqdm

from .languages import lang2langcode, skip_langs
from .treebank import Treebank, has_typo
from .unimorph import UD2UM, LAYERED_FEAT_SEP

sys.path.append("../../")
# The plain (non-bracketed) UD-feats-dict -> UM-tag-list conversion lives in
# the vendored package now (resources.um2ud_annotation.UD2UM_mapper): it
# already handles upos lookup, comma-valued disjunctions, and deciding
# whether a compound tag "fuses" upos (see that module's ud_feats_to_um_tags
# docstring). This module only adds the layered/argument-marking bracket
# handling below (_layered_feat_tag, LAYERED_FEAT_SEP) on top of it --
# that's this project's own round-trip encoding, not a UM or UD tagset
# convention, so it doesn't belong in the package. See ud_feats_to_um_tags
# below for how the two are combined.
from resources.um2ud_annotation.UD2UM_mapper import (
    ud_feats_to_um_tags as _plain_ud_feats_to_um_tags,
)


def _layered_feat_tag(base_feat: str, suffix: str, val: str) -> Optional[str]:
    """UD2UM-encode one value of a layered/argument-marking UD feature (e.g.
    Number[obj]=Plur -> "PL$obj") via multiblimp.unimorph.LAYERED_FEAT_SEP's
    round-trip convention -- see that constant's docstring. `suffix` is kept
    verbatim from the source UD feature name ("obj", "erg", "abs", "subj",
    "io", ...): treebanks disagree on whether an argument-marking bracket
    names a grammatical relation (Georgian's [subj]/[obj]/[io], Madi's
    [subj]/[obj]) or a case (Basque's [erg]/[abs]/[dat]), and this encoding
    doesn't need to care which -- it's this module's own lossless internal
    round-trip for UD-derived data, not a real UniMorph tag (genuine
    UniMorph argument-marking/possessor paradigm data uses ARG*/PSS* tags
    instead, handled separately by UnimorphInflector.ufeats2dict). None if
    `val` has no UD2UM entry for `base_feat` at all (same as the plain-
    feature case below).
    """
    um_code = UD2UM.get((base_feat, val))
    return f"{um_code}{LAYERED_FEAT_SEP}{suffix}" if um_code is not None else None


def ud_feats_to_um_tags(upos: str, feats: Optional[Dict[str, str]]) -> Optional[List[str]]:
    """Convert a UD token's upos + feats into real UniMorph tag strings
    (e.g. ["V", "PST", "3", "SG"]). Plain features are delegated to
    resources.um2ud_annotation.UD2UM_mapper.ud_feats_to_um_tags (handles the
    upos tag, comma-valued disjunctions, and skipping values with no UD2UM
    entry -- see that function's docstring). Layered/argument-marking
    features (Number[obj], Person[erg], Number[psor], ...) are this
    project's own addition on top, via _layered_feat_tag -- e.g. Basque/
    Georgian verbal object/indirect-object agreement is annotated
    exclusively this way, with no plain-feature fallback. Returns None if
    upos itself has no UM counterpart (PUNCT, SYM, X, ...).
    """
    plain_feats = {}
    layered_feats = []
    for feat, raw_val in (feats or {}).items():
        base_feat, bracket, suffix = feat.partition("[")
        if bracket:
            layered_feats.append((base_feat, suffix.rstrip("]"), raw_val))
        else:
            plain_feats[feat] = raw_val

    tags = _plain_ud_feats_to_um_tags(upos, plain_feats)
    if tags is None:
        return None

    for base_feat, suffix, raw_val in layered_feats:
        sub_tags = [
            tag for val in raw_val.split(",")
            if (tag := _layered_feat_tag(base_feat, suffix, val)) is not None
        ]
        if sub_tags:
            # dict.fromkeys: dedupe while preserving first-seen order
            tags.append("/".join(dict.fromkeys(sub_tags)))

    return tags


def create_all_unimorph_from_ud(
    ud_langs: List[str],
    verbose=False,
    resource_dir=".",
    dup_form_threshold=100.0,
    dup_feat_threshold: float = 100.0,
    load_from_pickle=False,
    save_ud2um_stats=False
):
    ud_unimorph_dir = os.path.join(resource_dir, "ud_unimorph")
    if not os.path.exists(ud_unimorph_dir):
        os.makedirs(ud_unimorph_dir)

    for lang in tqdm(sorted(ud_langs)):
        if lang in skip_langs:
            continue

        isolang = lang2langcode(lang)
        um_file = os.path.join(ud_unimorph_dir, isolang)

        ud_lang = lang.replace(" ", "_")
        treebank = Treebank(
            ud_lang,
            remove_diacritics=False,
            load_from_pickle=load_from_pickle,
            resource_dir=resource_dir,
            remove_typo=False,
            pickle_path="ud/ud_typo_pickles",
        )

        df = create_unimorph_from_ud(
            treebank,
            file=um_file,
            skip_no_lemma=True,
            skip_prep_lemma=True,
            verbose=verbose,
            dup_form_threshold=dup_form_threshold,
            dup_feat_threshold=dup_feat_threshold,
        )
        if save_ud2um_stats:
            with open(os.path.join(ud_unimorph_dir, "UM_from_UD.txt"), "a") as f:
                print(lang, len(df.lemma.unique()), len(df.form.unique()), len(df), sep="\t", file=f)
        else:
            print(lang, len(df.lemma.unique()), len(df.form.unique()), len(df), sep="\t")


def create_unimorph_from_ud(
    treebank: Treebank,
    file: Optional[str] = None,
    skip_no_lemma: bool = False,
    skip_prep_lemma: bool = False,
    verbose: bool = False,
    dup_form_threshold: float = 100.0,
    dup_feat_threshold: float = 100.0,
):
    rows = []

    iterator = tqdm(treebank) if verbose else treebank

    for tree in iterator:
        for token in tree:
            form = token["form"]
            upos = token["upos"]
            lemma = token["lemma"]

            if skip_no_lemma and lemma == "_":
                continue

            if skip_prep_lemma and "_" in lemma:
                continue

            if has_typo(token):
                continue

            token_feats = token["feats"] or {}
            if len(token_feats) == 0:
                continue

            um_tags = ud_feats_to_um_tags(upos, token_feats)
            if um_tags is None:
                continue

            row = (lemma.lower(), form.lower(), ";".join(um_tags))
            rows.append(row)

    row_freqs = [(*row, freq) for row, freq in Counter(rows).items()]

    row_freqs = remove_duplicate_features(
        row_freqs, dup_form_threshold, dup_feat_threshold
    )

    df = pd.DataFrame(
        sorted(row_freqs), columns=["lemma", "form", "ufeats", "frequency"]
    )

    if (file is not None) and (len(df) > 0):
        df.to_csv(file, sep="\t", index=False)

    return df


def remove_duplicate_features(
    row_freqs: List[Tuple[str, str, str, int]],
    dup_form_threshold: float,
    dup_feat_threshold: float,
) -> List[Tuple[str, str, str, int]]:
    """
    Form mismatch might indicate annotation error, since we can expect
    a lemma+feat combination to usually map to a single form
    (lemma1, form1, feat1)  <> (lemma1, form2, feat1)

    Feature mismatch might indicate feature annotation error, but here
    we only do this if the less frequent triple occurs once. Different
    lemma+form combinations can be instantiations of different feature
    sets, e.g. I saw_PRS / I saw_PST / you saw_PST
    (lemma1, form1, feat1)  <> (lemma1, form1, feat2)
    """
    lemma_feat2form = defaultdict(list)
    lemma_form2feat = defaultdict(list)

    for lemma, form, features, freq in row_freqs:
        lemma_feat2form[lemma, features].append((form, freq))
        lemma_form2feat[lemma, form].append((features, freq))

    dedup_forms = []
    for (lemma, features), forms in lemma_feat2form.items():
        if len(forms) == 1:
            form, freq = forms[0]
            dedup_forms.append((lemma, form, features, freq))
        else:
            sorted_forms: List[str, int] = sorted(
                forms, key=lambda x: x[1], reverse=True
            )
            most_common_form = sorted_forms[0]
            # If the most frequent form is _n_ times more frequent we remove that item
            dedup_forms.append(
                (lemma, most_common_form[0], features, most_common_form[1])
            )
            for next_common_form in sorted_forms[1:]:
                if (most_common_form[1] / next_common_form[1]) <= dup_form_threshold:
                    dedup_forms.append(
                        (lemma, next_common_form[0], features, next_common_form[1])
                    )

    dedup_feats = []
    for (lemma, form), all_features in lemma_form2feat.items():
        if len(all_features) == 1:
            features, freq = all_features[0]
            dedup_feats.append((lemma, form, features, freq))
        else:
            sorted_feats: List[str, int] = sorted(
                all_features, key=lambda x: x[1], reverse=True
            )
            most_common_feat = sorted_feats[0]
            # If the most frequent feature is _n_ times more frequent we remove that item
            dedup_feats.append((lemma, form, most_common_feat[0], most_common_feat[1]))
            for next_common_feat in sorted_feats[1:]:
                most_common_feat_set = set(most_common_feat[0].split(";"))
                next_common_feat_set = set(next_common_feat[0].split(";"))
                mismatching_feats = set(next_common_feat_set).symmetric_difference(
                    most_common_feat_set
                )

                if (
                    ((most_common_feat[1] / next_common_feat[1]) <= dup_feat_threshold)
                    or (len(mismatching_feats) > 2)
                    or (next_common_feat[1] > 1)
                    or (most_common_feat[0][0] != next_common_feat[0][0])
                ):
                    dedup_feats.append(
                        (lemma, form, next_common_feat[0], next_common_feat[1])
                    )

    dedup_rows = list(set(dedup_forms) & set(dedup_feats))

    return dedup_rows
