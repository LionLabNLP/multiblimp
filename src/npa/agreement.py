import json
import os
import re
import sys
from collections import Counter, defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

sys.path.append("../")

from word_order.decision_tree import fit_pipeline, set_dt_features_in_df
from word_order.process_treebank import META_FEATURES
from word_order.viz_tree import tree2html
from word_order.viz_deprel import generate_html_deprel_index
from multiblimp.languages import gblang2udlang, lang2langcode
from multiblimp.unimorph import load_inflector
from sva_trees.pipeline import get_impurity
from sva_trees.create_pairs import bucket_examples_html
from sva_trees.diagnostics import (
    generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json,
)
from npa.np_types import ROLE_PRIORITY
from multiblimp.config import (
    HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, OUTPUT_MINIMAL_PAIRS_DIR,
    OUTPUT_DIAGNOSTICS_DIR,
)

UNDEFINED = "UNDEFINED"

# Direct dependents that count as an NP's own UPOS-type projection (see
# npa.np_types.QUALIFYING_DEPS) filtered to the upos this role's own
# inflector should cover -- used to build a role-specific inflector (see
# build_role_inflector) via multiblimp.unimorph.load_inflector's
# unimorph_args["filter_entries"]["upos"]. "HEAD" spans all three head
# UPOS tags np_types.py's build_np_data scans for (NOUN/PROPN/PRON), since
# an NPA target_col's HEAD role is never restricted to just one of them.
ROLE_UPOS = {
    "HEAD": ["NOUN", "PROPN", "PRON"],
    "DET": ["DET"],
    "NUM": ["NUM"],
    "ADJ": ["ADJ"],
    "ADP": ["ADP"],
    "PRON": ["PRON"],
}

# Short, on-disk identifier for each UD feature NPA target_cols can be built
# over (see npa_id) -- keys are exactly the feature names np_instance()
# (npa.np_types) can produce a "{Role1}-{Role2}_{Feature}" pairwise column
# for, i.e. any morphological feature attested on at least one NPA role.
# word_order.viz_overview._NPA_FEATURE_NAMES keeps its own reverse copy of
# this table (word_order/ doesn't import npa/, same boundary
# word_order.viz_overview's sv/sp/sa handling already keeps from sva_trees/)
# -- keep the two in sync by hand if this table changes.
FEATURE_ABBREV = {
    "Number": "N",
    "Gender": "G",
    "Person": "P",
    "Case": "C",
    "Definite": "Def",
    "Degree": "Deg",
    "PronType": "PT",
    "NumType": "NT",
    "Poss": "Pos",
}


def npa_id(target_col: str) -> str:
    """"HEAD-DET_Number" -> "HEAD-DET_N" -- the on-disk/URL identifier for
    one NPA target_col, feature abbreviated via FEATURE_ABBREV (falling
    back to the bare feature name for one this table doesn't cover, so an
    unanticipated feature still gets a usable, if longer, identifier
    instead of an error). Used to build decision_trees/minimal_pairs/
    diagnostics paths (see run_agreement_pipeline) -- all nested one level
    under a shared "npa/" folder, e.g. "decision_trees/npa/HEAD-DET_N",
    rather than SVA's one-top-level-folder-per-target convention, since NPA
    has many more target_cols (every role-pair x feature combination) than
    SVA has deprels.
    """
    role1, role2, feat = _split_target_col(target_col)
    return f"{role1}-{role2}_{FEATURE_ABBREV.get(feat, feat)}"


def fit_npa_tree(df: pd.DataFrame, target_col: str, drop_unk: bool = True,
                  max_depth: int = 12, min_samples_leaf: int = 25,
                  min_impurity_decrease: float = 0.005, test_size: float = 0.1,
                  leaf_threshold: float = 0.1, omit_extra: set | None = None):
    """Fit a decision tree predicting one NPA pairwise agreement column (e.g.
    "HEAD-DET_Number", values "yes"/"no"/"unk") from every other column in
    `df`. Adapts word_order.decision_tree.fit_dt for NPA's role-based naming
    ("HEAD_*"/"DET_*") and yes/no/unk labels rather than SVA's deprel-based
    naming and Yes/No/+-/-- labels -- reuses fit_pipeline/set_dt_features_in_df
    directly (both naming-agnostic) rather than fit_dt itself, which assumes
    SVA's "head_{deprel}_{feat}_agreement" column pattern and a
    PredictionTarget.

    Omits from the predictor set: META_FEATURES (sen/no_space_after/treebank/
    sent_id/tree_idx/treebank_link/sen_str -- "sen"/"no_space_after" are
    list-valued, which nunique()/OneHotEncoder can't handle, so these must
    stay excluded once np_instance's meta dict carries them), any *_idx or
    *_dir column (matches fit_dt's own omissions), np_type, every OTHER
    pairwise agreement column (anything else containing "-") -- those are
    similarly-derived signal for a different role pair, not organic
    predictors -- and, critically, every raw column ending in "_{feat}"
    (e.g. HEAD_Case, DET_Case, ADJ_Case, and their sibling-feat/child-feat
    variants) for target_col's own feature. Without this a tree predicting
    e.g. "HEAD-DET_Case" would simply split on HEAD_Case and DET_Case
    directly -- the two raw values the "yes"/"no" label is itself computed
    from (see np_types.pairwise_agreement) -- collapsing every leaf to
    trivial 100%-confidence rules that recover nothing about *why* the two
    agree, which is exactly the failure mode observed empirically (a real
    fitted HEAD-DET_Case tree whose first two splits were literally
    "DET.Case = Nom" / "HEAD.Case = Nom"). This mirrors
    sva_trees.pipeline.Pipeline/subj_aux.pipeline.SubjAuxPipeline's own
    fit_dt calls exactly (both pass `omit_feats=set(col for col in full_df
    if col.endswith(f"_{{feat}}") and not col.startswith("swap_"))`) --
    fit_npa_tree previously lacked this parallel, which is what this
    docstring used to (incorrectly) describe as "matches fit_dt's existing
    convention of leaving those in": that was only ever true of fit_dt
    itself in isolation, not of how every actual caller invokes it.

    The dropped columns aren't lost, though -- set_dt_features_in_df's
    additional_vars/omit_feats re-attach every omitted column onto the
    returned dt_df afterward (same as SVA), so HEAD_Case/DET_Case are still
    present for create_npa_pairs, tree2html's per-node sample tables, and
    the unk-split diagnostics -- they just never get to influence which
    leaf a row lands in.

    Returns (model, dt_df, y_train, sub_df) -- dt_df is X_train enriched
    with leaf_id/leaf_top1_entropy/leaf_decision/keep etc (see
    set_dt_features_in_df), ready for create_npa_pairs; sub_df is the full
    filtered-but-unsplit dataset (all "yes"/"no" rows, not just the train
    split), for callers that want it separately (e.g.
    word_order.viz_tree.tree2html's full_df, which reports stats over the
    complete dataset rather than just the train split dt_df covers).
    (None, None, None, sub_df) if there's not enough data to fit -- sub_df
    is still returned (not None) so a caller like run_agreement_pipeline can
    fall back to it as tree2html's dt_df/full_df, matching sva_trees.
    pipeline.Pipeline's own "dt_df = full_df" fallback when nothing gets fit.
    """
    sub_df = df[df[target_col].notna()].copy()
    if drop_unk:
        sub_df = sub_df[sub_df[target_col] != "unk"]
    if len(sub_df) < 10 or sub_df[target_col].nunique() < 2:
        return None, None, None, sub_df

    _, _, feat = _split_target_col(target_col)
    omit_feats = set(META_FEATURES) | {"np_type"}
    omit_feats.update(c for c in sub_df.columns if "idx" in c)
    omit_feats.update(c for c in sub_df.columns if c.endswith("_dir"))
    omit_feats.update(c for c in sub_df.columns if "-" in c and c != target_col)
    omit_feats.update(
        c for c in sub_df.columns
        if c.endswith(f"_{feat}") and not c.startswith("swap_")
    )
    if omit_extra:
        omit_feats.update(omit_extra)

    predictor_cols = [c for c in sub_df.columns if c not in omit_feats and c != target_col]
    X = sub_df[predictor_cols].copy()
    X = X.loc[:, X.nunique() > 1].copy()
    y = sub_df[target_col].copy()

    model, X_train, X_test, y_train, y_test = fit_pipeline(
        X, y, model_type="decision_tree", max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease, test_size=test_size,
    )

    dt_df = set_dt_features_in_df(
        model, X_train, sub_df, target=None, predictor_var=target_col,
        additional_vars=omit_feats, threshold=leaf_threshold, omit_feats=omit_feats,
    )

    print("Train acc", model.score(X_train, y_train))
    if X_test is not None:
        print("Test acc ", model.score(X_test, y_test))

    return model, dt_df, y_train, sub_df


def _split_target_col(target_col: str) -> tuple[str, str, str]:
    """"HEAD-DET_Number" -> ("HEAD", "DET", "Number")."""
    role1, rest = target_col.split("-", 1)
    role2, feat = rest.split("_", 1)
    return role1, role2, feat


def _drop_irrelevant_roles(df: pd.DataFrame, keep_roles) -> pd.DataFrame:
    """Drop single-role "{role}_*" columns for every NPA role not in
    `keep_roles` -- e.g. for target_col "HEAD-DET_Number", drops ADJ_*/
    ADP_*/NUM_*/PRON_* entirely. A NPA instance's df carries columns for
    every role that occurs *anywhere* in the dataset (see np_instance),
    not just the two roles a given target_col cares about, so without this
    tree2html's node-click sample table would show every role present in
    any NP, not just the pair actually being predicted.

    Pairwise columns ("{role1}-{role2}_*") are untouched: a role name is
    never a string-prefix of a pairwise column, since the character right
    after the role name there is "-", not "_" (e.g. "ADJ-ADP_Case" doesn't
    start with "ADJ_").
    """
    drop_roles = set(ROLE_PRIORITY) - set(keep_roles)
    drop_cols = [
        col for col in df.columns
        if any(col.startswith(f"{role}_") for role in drop_roles)
    ]
    return df.drop(columns=drop_cols)


def create_npa_pairs(dt_df: pd.DataFrame, target_col: str, inflector,
                      swap_role: str | None = None, context_inflector=None,
                      leaf_threshold: float = 0.1,
                      verbose: bool = True) -> tuple[dict[str, pd.DataFrame], int]:
    """Minimal re-inflected pairs from a fit_npa_tree dt_df, for rows where
    the tree confidently predicts target_col == "yes" (leaf_top1_entropy <
    leaf_threshold and leaf_decision True -- same convention as sva_trees.
    create_pairs).

    Reinflects `swap_role`'s own token via inflector.inflect on that role's
    own feature columns ("{swap_role}_{Feat}"), keeping the OTHER role
    (`fixed_role`, the other side of target_col) fixed. Default swap_role is
    the second role in target_col (e.g. "DET" for "HEAD-DET_Number") --
    matching sva_trees.create_pairs' controller/target split: the fixed role
    is the agreement controller (like nsubj/subject), the swapped role is
    the target that must covary (like the verb head). A successful
    reinflection to a form with disjoint features from the original (e.g.
    Number Sing -> Plur) demonstrates the concord is real: the original
    fixed-role form no longer matches the reinflected one.

    context_inflector, if given, guards against a false "correct_swaps":
    after a candidate swap, it checks whether the FIXED role's own form
    could *also* be read with the swapped feature value (e.g. German "die"
    is syncretic between singular-feminine and plural, so "die Möglichkeit"
    is valid German even after swapping the determiner from plural to
    singular-looking -- the fixed noun's own ambiguity, not the swap,
    would be masking a real disagreement). Rows where that happens go to
    "ambiguous_subjects" instead of "correct_swaps" -- named to match
    sva_trees.create_pairs' own bucket (its "the subject's own form is
    ambiguously compatible with the swap" concept, generalized here from
    "subject" to "whichever role is fixed"), not because NPA's fixed role
    is ever literally a subject; see create_npa_pairs_for_target_col, which
    relies on this exact vocabulary to reuse sva_trees.diagnostics/
    word_order.html.html_deprel's diagnostics-enabled page unmodified.
    Should be an inflector covering the fixed role's POS (e.g. a
    noun-configured inflector when swap_role="DET" and fixed_role="HEAD").
    None (default) skips the check.

    A deliberately simplified sibling of sva_trees.create_pairs.create_pairs:
    same "correct_swaps"/"same_forms"/"same_features"/"undefined_features"/
    "no_inflections"/"no_candidates"/"ambiguous_subjects" bucket vocabulary
    and swap logic, but no HTML example rendering, no max_num_of_pairs
    balancing (added around it, not inside it, by
    create_npa_pairs_for_target_col).

    Returns (dict[bucket_name, pd.DataFrame], items_seen) -- items_seen is
    every attempted swap_form across all rows (see create_npa_pairs_for_target_col,
    which needs the exact count -- not derivable from the bucket sizes alone
    since a single row can yield more than one swap_form/item).
    """
    role1, role2, _feat = _split_target_col(target_col)
    if swap_role is None:
        swap_role = role2
    fixed_role = role2 if swap_role == role1 else role1

    if "leaf_top1_entropy" in dt_df.columns and "leaf_decision" in dt_df.columns:
        keep = (dt_df["leaf_top1_entropy"] < leaf_threshold) & dt_df["leaf_decision"]
    else:
        keep = pd.Series(True, index=dt_df.index)
    swap_df = dt_df[keep & (dt_df[target_col] == "yes")]

    ufeat, _ = inflector.inflection_map
    feat_cols = [
        (col, m.group(1)) for col in swap_df.columns
        if (m := re.match(rf"^{re.escape(swap_role)}_([A-Z][a-zA-Z]*)$", col))
    ]
    fixed_feat_cols = [
        (col, m.group(1)) for col in swap_df.columns
        if (m := re.match(rf"^{re.escape(fixed_role)}_([A-Z][a-zA-Z]*)$", col))
    ]

    diagnostics = {
        "correct_swaps": [], "same_forms": [], "same_features": [],
        "ambiguous_subjects": [], "undefined_features": [],
        "no_inflections": [], "no_candidates": [],
    }
    feature_distribution = Counter()
    items_seen = 0

    # itertuples(index=False) defaults to name="Pandas" -- a namedtuple,
    # which silently renames any column that isn't a valid Python
    # identifier (e.g. "HEAD_under_obl:agent", "HEAD_Number[psor]" -- most
    # of this df's columns) to a positional placeholder ("_12", "_13", ...)
    # in ._asdict(). name=None + zip(columns, row_tuple) (same pattern
    # sva_trees.create_pairs.create_pairs uses) keeps the real column names
    # regardless of identifier-safety.
    columns = list(swap_df.columns)
    row_iter = swap_df.itertuples(index=False, name=None)
    for row_tuple in (tqdm(row_iter, total=len(swap_df)) if verbose else row_iter):
        row_dict = dict(zip(columns, row_tuple))
        og_feats = {feat: row_dict.get(col) for col, feat in feat_cols}
        form = row_dict.get(f"{swap_role}_form")
        fixed_form = row_dict.get(f"{fixed_role}_form")
        if form is None or pd.isna(form):
            continue

        swap_forms, feature_vals, swap_feats = inflector.inflect(
            form, og_feats, return_swap_feats=True
        )
        items_seen += 1

        if swap_forms is None:
            diagnostics["no_candidates"].append(row_dict)
            continue
        if len(swap_forms) == 0:
            diagnostics["no_inflections"].append(row_dict)
            continue

        for swap_form in swap_forms:
            item = dict(row_dict)
            item[f"swap_{swap_role}"] = swap_form
            for feat, vals in swap_feats.get(swap_form, {}).items():
                item[f"after_{swap_role}_{feat}"] = "/".join(sorted(vals))

            if swap_form == form:
                diagnostics["same_forms"].append(item)
                continue

            swap_feature_vals = inflector.get_form_features(swap_form, og_feats, ufeat)
            feature_key = (
                "|".join(sorted(feature_vals)) + " -> " + "|".join(sorted(swap_feature_vals))
            )
            item["feature_vals"] = feature_key

            if len(feature_vals & swap_feature_vals) == 0 and UNDEFINED not in swap_feature_vals:
                if len(swap_feature_vals) == 0:
                    diagnostics["undefined_features"].append(item)
                    continue

                if context_inflector is None or fixed_form is None or pd.isna(fixed_form):
                    fixed_features = set()
                else:
                    fixed_og_feats = {
                        feat: row_dict.get(col) for col, feat in fixed_feat_cols
                    }
                    fixed_features = context_inflector.get_form_features(
                        fixed_form, fixed_og_feats, ufeat, only_try_ud_if_no_um=True,
                    )
                if len(fixed_features & swap_feature_vals) > 0:
                    diagnostics["ambiguous_subjects"].append(item)
                else:
                    feature_distribution[feature_key] += 1
                    diagnostics["correct_swaps"].append(item)
            else:
                diagnostics["same_features"].append(item)

    if verbose:
        coverage = len(diagnostics["correct_swaps"]) / max(items_seen, 1) * 100
        print(f"swap_role={swap_role} fixed_role={fixed_role} items_seen={items_seen} "
              f"correct_swaps={len(diagnostics['correct_swaps'])} coverage={coverage:.1f}%")
        print("feature_vals distribution:", dict(feature_distribution))

    return {k: pd.DataFrame(v) for k, v in diagnostics.items()}, items_seen


def swap_roles_for_target_col(target_col: str) -> list[str]:
    """Which role(s) of `target_col` (e.g. "HEAD-DET_Number" or
    "DET-ADJ_Gender") create_npa_pairs_for_target_col should reinflect.

    ROLE_PRIORITY (npa.np_types) always sorts HEAD first when it's one of
    the pair -- np_instance() builds pairwise columns via
    itertools.combinations() over np_roles()' ROLE_PRIORITY-ordered role
    list, so "role1 == 'HEAD'" is a complete, order-independent test for
    "this pair involves the NP head". When it does, only the OTHER
    (non-head) role is ever reinflected: the head is the agreement
    controller a determiner/adjective/etc. must covary WITH, mirroring
    SVA's subject-controls-the-verb asymmetry (create_pairs there always
    keeps the "subject" role fixed and reinflects the other one) -- just
    with head/dependent swapped relative to SVA's own head=verb,
    child=subject roles: NPA's head is the fixed controller, its
    modifier is the covarying dependent that gets reinflected.

    When neither role is HEAD (two co-dependents of the same head, e.g.
    "DET-ADJ_Gender", with no inherent controller/target asymmetry between
    them), both roles are returned -- create_npa_pairs_for_target_col
    reinflects each one in turn, holding the other fixed, since there's no
    principled reason to prefer one direction over the other.
    """
    role1, role2, _feat = _split_target_col(target_col)
    return [role2] if role1 == "HEAD" else [role1, role2]


def build_role_inflector(role: str, feature: str, lang: str, resource_dir: str,
                          langcode: str | None = None, unimorph_args: dict | None = None):
    """A load_inflector() instance for `role` (e.g. "DET", "HEAD"), filtered
    to that role's own UPOS tag(s) (ROLE_UPOS) and configured to swap
    `feature` to any other observed value (inflection_map=(feature, None) --
    same "_any" shape as multiblimp.swap_features.swap_gender_any/
    swap_case_any: NPA fits one tree per target_col, whatever feature that
    target_col names, rather than one script per fixed feature the way
    svNa.py/svGa.py/svPa.py do, so there's no single fixed swap_* constant
    to import here).

    Returns (inflector, num_lemma, num_form) -- matches load_inflector's own
    return shape (minus skip_lang, which every caller here ignores, same as
    sva_trees.pipeline.Pipeline._process_language_impl).
    """
    args = dict(unimorph_args) if unimorph_args else {
        "combine_um_ud": True,
        "remove_multiword_forms": True,
    }
    upos = ROLE_UPOS.get(role)
    if upos:
        args = dict(args)
        args["filter_entries"] = {**args.get("filter_entries", {}), "upos": upos}

    inflector, _skip_lang, num_lemma, num_form = load_inflector(
        lang=lang,
        langcode=langcode or lang2langcode(lang),
        unimorph_args=args,
        inflection_map=(feature, None),
        resource_dir=resource_dir,
    )
    return inflector, num_lemma, num_form


def _coverage_stats_npa(df: pd.DataFrame, role1: str, role2: str, inflector) -> tuple[int, int, int]:
    """(# forms of interest, # covered by UM alone, # covered by UM+UD
    combined) for one NPA pair -- the NPA analogue of sva_trees.create_pairs.
    _coverage_stats, generalized from that function's hardcoded "head_form"/
    "{child_deprel}_form" column names to any two role names.

    Forms of interest: every distinct role1_form/role2_form value across ALL
    of `df` (not just the "yes"-labelled swap candidates), i.e. every word
    that plays either role in this pair, regardless of whether it ends up
    eligible for a swap. Checked against a single `inflector`'s UniMorph
    tables -- matching sva_trees.create_pairs' own precedent of using just
    the (one) inflector passed to create_pairs for both sides' coverage,
    rather than mixing in a second, differently-POS-filtered inflector.
    """
    form_cols = [c for c in (f"{role1}_form", f"{role2}_form") if c in df.columns]
    forms_of_interest = set()
    for col in form_cols:
        forms_of_interest.update(df[col].dropna().unique())

    um_forms = (
        set(inflector.unimorph_df["form"].unique())
        if inflector.unimorph_df is not None and len(inflector.unimorph_df) > 0
        else set()
    )
    um_ud_forms = inflector.unique_forms

    num_covered_um = sum(1 for f in forms_of_interest if f in um_forms)
    num_covered_um_ud = sum(1 for f in forms_of_interest if f in um_ud_forms)

    return len(forms_of_interest), num_covered_um, num_covered_um_ud


def _unk_split_counts(full_df: pd.DataFrame, target_col: str, feat: str,
                       role_a: str, role_b: str) -> dict[str, int]:
    """Rows whose target_col agreement label is "unk" (i.e. one side's raw
    feature value was missing at pairwise_agreement() time -- see
    npa.np_types.pairwise_agreement), split by which side: role_a's own
    feature missing (role_b's present), role_b's missing (role_a's
    present), or both. Keyed "head_unk"/"nsubj_unk"/"both_unk" to match
    exactly what sva_trees.diagnostics.language_diagnostics_row reads out of
    meta.json (see create_npa_pairs_for_target_col, which always passes
    role_a=the reinflected role and role_b=the fixed one, so head_unk lines
    up with head_role_label and nsubj_unk with subject_label/nsubj_label in
    the HTML -- same convention SVA's own head_unk="head/verb missing",
    nsubj_unk="child/subject missing" split uses).

    {"head_unk": 0, "nsubj_unk": 0, "both_unk": 0} if either role's raw
    feature column isn't present at all (e.g. a role that never got its own
    single-feature column because it had only one distinct value and was
    dropped -- np_instance still always emits pairwise agreement columns
    regardless, so target_col itself is unaffected).
    """
    col_a, col_b = f"{role_a}_{feat}", f"{role_b}_{feat}"
    if col_a not in full_df.columns or col_b not in full_df.columns:
        return {"head_unk": 0, "nsubj_unk": 0, "both_unk": 0}

    is_unk = full_df[target_col].astype(str) == "unk"
    a_missing = full_df[col_a].isna()
    b_missing = full_df[col_b].isna()
    return {
        "head_unk": int((is_unk & a_missing & ~b_missing).sum()),
        "nsubj_unk": int((is_unk & ~a_missing & b_missing).sum()),
        "both_unk": int((is_unk & a_missing & b_missing).sum()),
    }


_NPA_BUCKET_NAMES = [
    "correct_swaps", "same_forms", "same_features", "ambiguous_subjects",
    "undefined_features", "no_inflections", "no_candidates",
]


def create_npa_pairs_for_target_col(
    dt_df: pd.DataFrame, target_col: str, inflectors: dict,
    leaf_threshold: float = 0.1, save_to: str | None = None,
    full_df: pd.DataFrame | None = None, label_distribution: dict | None = None,
    num_lemma=None, num_form=None, max_examples: int = 5, verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """The production entry point for NPA minimal pairs: wraps
    create_npa_pairs with the head-vs-non-head swap-direction rule
    (swap_roles_for_target_col) and assembles output in the exact shape
    sva_trees.diagnostics/word_order.html.html_deprel's diagnostics-enabled
    page expects (bucket "<name>.parquet" files, a meta.json with
    coverage/unk-drop/label-distribution/examples), so a target_col's
    per-language pairs_dir can be read by sva_trees.diagnostics.
    generate_diagnostics_table completely unmodified -- see
    run_agreement_pipeline, which drives this per language and then does
    exactly that.

    inflectors: dict[role, inflector] with an entry for every role
    swap_roles_for_target_col(target_col) returns AND for that role's fixed
    counterpart (build_role_inflector covers both; run_agreement_pipeline
    builds one for every role in the pair upfront, since which one ends up
    "fixed" vs "swapped" depends on the direction).

    When target_col involves HEAD, this makes exactly one create_npa_pairs
    call (swap the non-head role, HEAD fixed). Otherwise it makes two --
    once per direction -- and concatenates each bucket across both calls,
    so e.g. "DET-ADJ_Gender"'s correct_swaps holds both "DET reinflected,
    ADJ fixed" and "ADJ reinflected, DET fixed" rows side by side
    (distinguishable via which of swap_DET/swap_ADJ is set on a given row).
    items_seen and the "keep" pool both sum across directions accordingly,
    so per-bucket percentages stay correctly scoped to "every attempt made"
    rather than silently only covering the first direction.

    The head_unk/nsubj_unk/both_unk (and, correspondingly, the
    head_role_label/subject_label/nsubj_label wording a caller should use
    when rendering this target_col's page) are computed against ONE
    (swap_role, fixed_role) split: swap_roles_for_target_col(target_col)[0]
    as the reinflected role and its counterpart as fixed -- the only
    meaningful split when there's one direction, and an arbitrary-but-
    consistent pick between the two symmetric directions when there are two
    (there's no single "the" fixed role in that case either way).

    Returns (diagnostic_dfs, meta) -- meta additionally carries
    "swap_role"/"fixed_role" (the pick described above, for a caller to
    build head_role_label/subject_label/nsubj_label from) beyond the keys
    written into meta.json.
    """
    if full_df is None:
        full_df = dt_df
    role1, role2, feat = _split_target_col(target_col)
    swap_roles = swap_roles_for_target_col(target_col)

    collected = defaultdict(list)
    total_items_seen = 0
    total_n_keep = 0
    if "leaf_top1_entropy" in dt_df.columns and "leaf_decision" in dt_df.columns:
        keep = (dt_df["leaf_top1_entropy"] < leaf_threshold) & dt_df["leaf_decision"]
    else:
        keep = pd.Series(True, index=dt_df.index)
    n_raw = int((dt_df[target_col] == "yes").sum())

    for swap_role in swap_roles:
        fixed_role = role2 if swap_role == role1 else role1
        buckets, items_seen = create_npa_pairs(
            dt_df, target_col, inflector=inflectors[swap_role],
            swap_role=swap_role, context_inflector=inflectors.get(fixed_role),
            leaf_threshold=leaf_threshold, verbose=verbose,
        )
        total_items_seen += items_seen
        total_n_keep += int((keep & (dt_df[target_col] == "yes")).sum())
        for name, bdf in buckets.items():
            if len(bdf):
                collected[name].append(bdf)

    diagnostic_dfs = {
        name: (pd.concat(collected[name], ignore_index=True, sort=False)
               if collected.get(name) else pd.DataFrame())
        for name in _NPA_BUCKET_NAMES
    }

    swap_role = swap_roles[0]
    fixed_role = role2 if swap_role == role1 else role1
    unk_counts = _unk_split_counts(full_df, target_col, feat, swap_role, fixed_role)
    num_forms_of_interest, num_covered_um, num_covered_um_ud = _coverage_stats_npa(
        full_df, role1, role2, inflectors[swap_role]
    )

    meta = {
        "leaf_threshold": leaf_threshold,
        "num_ud_candidates_raw": n_raw,
        "num_ud_candidates_keep": total_n_keep,
        "items_seen": total_items_seen,
        "num_lemma": num_lemma,
        "num_form": num_form,
        "num_forms_of_interest": num_forms_of_interest,
        "num_covered_um": num_covered_um,
        "num_covered_um_ud": num_covered_um_ud,
        **unk_counts,
    }
    if label_distribution is not None:
        meta["label_distribution"] = label_distribution

    # Trimmed to just (role1, role2) for the EXAMPLES rendering only -- the
    # reinflection logic above (create_npa_pairs) already ignored every
    # other role's columns regardless, since it only ever reads role1_*/
    # role2_* by name, but a displayed sample row showing e.g. an unrelated
    # ADP_form value (present just because that NP happened to also have a
    # preposition, not because it's part of THIS agreement condition) would
    # misleadingly suggest ADP is relevant here. diagnostic_dfs itself
    # (saved to parquet below, and returned as-is) keeps every column --
    # those "irrelevant" roles' features are still fair game as decision-
    # tree predictors (see fit_npa_tree, which is never handed a trimmed
    # df), just not part of what a reinflection *example* should show.
    display_dfs = {
        name: _drop_irrelevant_roles(bdf, (role1, role2))
        for name, bdf in diagnostic_dfs.items()
    }
    meta["examples"] = bucket_examples_html(
        display_dfs, list(display_dfs), max_examples=max_examples,
        child_deprel=fixed_role, default_kind=swap_role, swap_feature=feat,
    )

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        for name, bdf in diagnostic_dfs.items():
            bdf.to_parquet(os.path.join(save_to, f"{name}.parquet"))
        with open(os.path.join(save_to, "meta.json"), "w") as f:
            json.dump(meta, f)

    return diagnostic_dfs, {**meta, "swap_role": swap_role, "fixed_role": fixed_role}


def refresh_deprel_index(target_col: str,
                          save_dir: str | None = None,
                          html_dir: str | None = None,
                          decision_trees_root: str = OUTPUT_DECISION_TREES_DIR,
                          html_decision_trees_root: str = HTML_DECISION_TREES_DIR,
                          pairs_dir: str | None = None,
                          diagnostics_csv: str | None = None,
                          leaf_threshold: float = 0.1) -> None:
    """Rebuild one NPA target_col's diagnostics CSV + deprel index.html,
    purely from on-disk output/decision_trees + output/minimal_pairs
    artifacts -- the tail of run_agreement_pipeline (everything from
    "Generating diagnostics table" on), extracted so a caller that only
    wants to refresh an already-fully-built condition's index (see
    scripts/generate_html_indexes.py) doesn't have to pay for
    run_agreement_pipeline's per-language loop first.

    That loop unconditionally reads each language's full np_instances/
    {lang}.parquet (every role the language's NPs ever carry, not just this
    target_col's two) before it even reaches a per-language skip-check --
    tens of MB per language, repeated once per target_col swept, for zero
    benefit when nothing about that language actually needs rebuilding.
    Nothing in this function touches np_instances at all: diagnostics come
    from pairs_dir, and generate_html_deprel_index's own data_dir scan reads
    only the (much smaller) decision-tree-stage .joblib/.parquet cache.

    save_dir/html_dir/pairs_dir/diagnostics_csv default the same way
    run_agreement_pipeline's own do (npa_id(target_col)-derived paths under
    decision_trees_root's/html_decision_trees_root's own "npa/" subtree).
    """
    npa_identifier = npa_id(target_col)
    save_dir = save_dir or os.path.join(decision_trees_root, "npa", npa_identifier)
    html_dir = html_dir or os.path.join(html_decision_trees_root, "npa", npa_identifier)
    pairs_dir = pairs_dir or os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", npa_identifier)
    diagnostics_csv = diagnostics_csv or os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{npa_identifier}.csv")
    role1, role2, feat = _split_target_col(target_col)
    swap_roles = swap_roles_for_target_col(target_col)
    swap_role = swap_roles[0]
    fixed_role = role2 if swap_role == role1 else role1

    print("Generating diagnostics table")
    diagnostics_df = generate_diagnostics_table(pairs_dir)
    write_diagnostics_csv(diagnostics_df, diagnostics_csv)
    diagnostics_by_lang = {
        row["Language"]: diagnostics_row_to_json(
            row, lang_dir=os.path.join(pairs_dir, row["Language"])
        )
        for _, row in diagnostics_df.iterrows()
    }

    print("Generating deprel index")
    generate_html_deprel_index(
        data_dir=save_dir,
        html_directory=html_dir,
        target_col=target_col,
        leaf_threshold=leaf_threshold,
        pairs_dir=pairs_dir,
        diagnostics_by_lang=diagnostics_by_lang,
        # A language with no fitted tree (100% one label, or too little
        # variance to fit at all) still gets real pairs/diagnostics
        # computed -- see run_agreement_pipeline's unconditional
        # create_npa_pairs_for_target_col call -- so it's worth showing
        # rather than only naming in the omitted-languages note, as long as
        # its "yes" count is more than noise. Entropy/accuracy render blank
        # for these rows (see generate_html_deprel_index's own NaN
        # handling) since there's no tree to report them from.
        include_trivial_labels={"yes"},
        include_trivial_min_count=10,
        agreement_label=f"NP {role1}–{role2} ({feat})",
        head_role_label=swap_role,
        subject_label=fixed_role,
        nsubj_label=fixed_role,
    )


def run_agreement_pipeline(target_col: str, langs: list[str], instances_dir: str,
                            save_dir: str | None = None,
                            html_dir: str | None = None,
                            decision_trees_root: str = OUTPUT_DECISION_TREES_DIR,
                            html_decision_trees_root: str = HTML_DECISION_TREES_DIR,
                            resource_dir: str = "../../resources",
                            pairs_dir: str | None = None,
                            diagnostics_csv: str | None = None,
                            never_skip: bool = False,
                            drop_unk: bool = True, max_depth: int = 12,
                            min_samples_leaf: int = 10, test_size: float = 0.1,
                            leaf_threshold: float = 0.1,
                            palette_map: dict | None = None,
                            unimorph_args: dict | None = None,
                            build_pairs: bool = True, verbose: bool = True) -> None:
    """Per-language: fit_npa_tree + word_order.viz_tree.tree2html, then (when
    build_pairs, the default) create_npa_pairs_for_target_col, for one NPA
    pairwise agreement column, e.g. "HEAD-DET_Number" -- then, once every
    language is done, a diagnostics table + the SAME diagnostics-enabled
    deprel index page sva_trees.pipeline.Pipeline/subj_aux.pipeline.
    SubjAuxPipeline produce (word_order.viz_deprel.generate_html_deprel_index
    with diagnostics_by_lang populated -- never the plain "classic" table
    page). Does NOT rebuild the cross-target_col overview index itself
    (word_order.viz_overview.generate_html_overview_index, which groups
    every NPA target_col into one "Noun Phrase" section, subdivided by role
    pair -- see word_order.viz_overview._classify_deprel) -- that's rescan-
    everything-recursively work best done once after a whole sweep, not
    once per target_col; see scripts/generate_html_indexes.py.

    The NPA analog of sva_trees.pipeline.Pipeline._process_language_impl
    plus its post-loop indexing stage, both folded into one function since
    (unlike Pipeline, which fits several deprels per language in one pass)
    NPA fits one target_col across every language per call. Mirrors
    Pipeline's caching structure throughout (skip a fit if {lang}.joblib
    already exists; skip tree2html separately if {lang}.html exists; skip
    create_npa_pairs_for_target_col if its pairs_dir/{lang} already exists;
    "dt_df = full_df" when no tree gets fit) but does NOT build
    treebank_features/npa/np_instances/{lang}.parquet itself -- that's
    scripts/npa/np_types.py's job (a separate, expensive treebank-parsing
    sweep); a language missing its parquet here is simply skipped, not
    built on the fly.

    get_impurity and tree2html are imported from sva_trees.pipeline /
    word_order.viz_tree respectively and called directly (not reimplemented,
    not wrapped) -- tree2html's node-click sample rendering (per-token
    feature lists, swap-feature bolding, sentence highlighting) natively
    recognizes NPA's "{Role1}-{Role2}_{Feature}" predictor_var convention
    (see word_order.viz_tree._agreement_context) and its uppercase role
    prefixes (word_order.viz_tree._build_feat_columns), passed target=None
    same as every other call in this module.

    save_dir (model/tree-data cache: .joblib/.parquet) and html_dir (the
    actual served .html page) are two separate directories -- unlike SVA/
    subj_aux, which keep those in genuinely different parent trees from the
    start (output/decision_trees/... vs. html/decision_trees/...), this
    function used to colocate both in one dir; now split the same way, each
    defaulting to "{decision_trees_root}/npa/{npa_id(target_col)}" /
    "{html_decision_trees_root}/npa/{npa_id(target_col)}" (e.g.
    "output/decision_trees/npa/HEAD-DET_N" /
    "html/decision_trees/npa/HEAD-DET_N") -- every NPA target_col nested
    under one shared "npa/" folder (unlike SVA's one-top-level-folder-per-
    target convention: NPA has many more target_cols, every role-pair x
    feature combination, than SVA has deprels, so grouping them under a
    shared parent keeps decision_trees/'s own top level from being
    dominated by NPA entries). generate_html_overview_index's index.html
    glob is recursive specifically to still find pages nested this way (see
    that function). pairs_dir/diagnostics_csv
    default to the same "npa/{npa_id(target_col)}" convention under
    output/minimal_pairs/ and output/diagnostics/ respectively (flat there
    -- those two don't feed the overview index, so they don't need
    decision_trees_root's own top-level-folder-count concern).

    unimorph_args: passed to build_role_inflector for every role in the
    pair, minus "filter_entries" (that part is always role-specific, set by
    build_role_inflector itself via ROLE_UPOS). None (default) uses
    build_role_inflector's own default ({"combine_um_ud": True,
    "remove_multiword_forms": True}).

    Also unlike sva_trees.pipeline.Pipeline: no ProcessPoolExecutor/memory
    capping. Reading an already-built np_instances parquet and fitting one
    decision tree per language is far cheaper than SVA's treebank-parsing +
    extract_node_features extraction step, which is what that machinery
    exists to protect against.
    """
    npa_identifier = npa_id(target_col)
    save_dir = save_dir or os.path.join(decision_trees_root, "npa", npa_identifier)
    html_dir = html_dir or os.path.join(html_decision_trees_root, "npa", npa_identifier)
    pairs_dir = pairs_dir or os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", npa_identifier)
    diagnostics_csv = diagnostics_csv or os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{npa_identifier}.csv")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    role1, role2, feat = _split_target_col(target_col)
    swap_roles = swap_roles_for_target_col(target_col)
    swap_role = swap_roles[0]
    fixed_role = role2 if swap_role == role1 else role1

    for lang in sorted(langs):
        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        instances_path = os.path.join(instances_dir, f"{cached_lang}.parquet")
        if not os.path.exists(instances_path):
            print(f"{lang}: skip, no np_instances parquet at {instances_path}")
            continue

        raw_df = pd.read_parquet(instances_path)
        if target_col not in raw_df.columns:
            print(f"{lang}: skip, missing column {target_col}")
            continue

        full_df = raw_df[raw_df[target_col].notna()].copy()
        if len(full_df) == 0:
            print(f"{lang}: skip, no rows with {target_col} set")
            continue

        # full_df/dt_df deliberately keep every role's columns here (not
        # just role1/role2) -- np_instances carries every role attested
        # anywhere in a language's NPs at once (see npa.np_types.
        # np_instance), and an "irrelevant" role's own features (e.g.
        # whether this NP also has an ADP dependent) are still fair game as
        # decision-tree PREDICTORS for target_col. Only the OUTPUT-facing
        # views get trimmed to (role1, role2): tree2html's own full_df copy
        # right below, and create_npa_pairs_for_target_col's examples
        # tables (see its own _drop_irrelevant_roles call) -- reinflection
        # itself only ever touches role1_*/role2_* columns regardless, but
        # a *displayed* example (or tree2html's per-node feature list)
        # showing an unrelated ADP_form value would misleadingly suggest
        # it's part of this agreement condition.
        print(lang)

        joblib_path = os.path.join(save_dir, f"{cached_lang}.joblib")
        dt_parquet_path = os.path.join(save_dir, f"{cached_lang}.parquet")
        html_path = os.path.join(html_dir, f"{cached_lang}.html")

        pred_values = set(full_df[target_col].values) - ({"unk"} if drop_unk else set())
        model, dt_df, learn_dt = None, None, False

        if never_skip or not os.path.exists(joblib_path):
            if len(pred_values) > 1:
                model, dt_df, y_train, _ = fit_npa_tree(
                    full_df, target_col, drop_unk=drop_unk, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_impurity_decrease=get_impurity(len(full_df)),
                    test_size=test_size, leaf_threshold=leaf_threshold,
                )
                learn_dt = model is not None
            if learn_dt:
                joblib.dump(model, joblib_path)
                dt_df.to_parquet(dt_parquet_path)
            else:
                model, dt_df = None, full_df
        else:
            model = joblib.load(joblib_path)
            dt_df = pd.read_parquet(dt_parquet_path)
            learn_dt = True

        # Run before tree2html (not after, as before) so the v2 page's
        # generated-pairs section can join this same run's own
        # correct_swaps rows in by leaf_id -- create_npa_pairs_for_target_col
        # doesn't depend on anything tree2html produces, so this reorder is
        # safe (mirrors sva_trees.pipeline.Pipeline's own reorder for the
        # same reason).
        lang_pairs_dir = os.path.join(pairs_dir, cached_lang)
        correct_swaps_df = None
        if build_pairs:
            if never_skip or not os.path.isdir(lang_pairs_dir):
                inflectors = {}
                for role in {role1, role2}:
                    inflector, num_lemma, num_form = build_role_inflector(
                        role, feat, lang, resource_dir, unimorph_args=unimorph_args,
                    )
                    inflectors[role] = inflector
                    if role == swap_role:
                        swap_num_lemma, swap_num_form = num_lemma, num_form

                label_distribution = {
                    str(k): int(v) for k, v in full_df[target_col].value_counts().items()
                }
                diagnostic_dfs, _ = create_npa_pairs_for_target_col(
                    dt_df, target_col, inflectors, leaf_threshold=leaf_threshold,
                    save_to=lang_pairs_dir, full_df=full_df,
                    label_distribution=label_distribution,
                    num_lemma=swap_num_lemma, num_form=swap_num_form, verbose=verbose,
                )
                correct_swaps_df = diagnostic_dfs.get("correct_swaps")
            else:
                # Pairs already built by an earlier run -- read the cached
                # correct_swaps bucket straight off disk (re-running
                # create_npa_pairs_for_target_col here would defeat this
                # skip's whole point) so tree2html can still join it in
                # below even when only the HTML itself needs regenerating.
                correct_swaps_path = os.path.join(lang_pairs_dir, "correct_swaps.parquet")
                if os.path.exists(correct_swaps_path):
                    correct_swaps_df = pd.read_parquet(correct_swaps_path)

        if never_skip or not os.path.exists(html_path):
            tree2html(
                pipeline_model=model,
                dt_df=dt_df,
                full_df=_drop_irrelevant_roles(full_df, (role1, role2)),
                predictor_var=target_col,
                target=None,
                out_file=html_path,
                max_rows=15,
                meta={"Language": lang},
                only_show_real_orders=True,
                correlate_features=True,
                show_features=True,
                full_tree_html=learn_dt,
                palette_map=palette_map or {"yes": "#31cb9f", "no": "#f16393", "unk": "#b893de"},
                leaf_threshold=leaf_threshold,
                head_label="NP head",
                # Same trim as full_df above, and for the same reason:
                # correct_swaps_df still carries every role np_instances
                # ever saw (ADJ/ADP/NUM/PRON/...), not just role1/role2.
                # Left untrimmed, html_tree.py's _v2_raw_pair_role_prefixes
                # (which only sees this dataframe's own columns, not
                # target_col) picks up a bystander role instead of the real
                # swap role -- its swap_{role} column is then always empty,
                # so every row silently fails _v2_pair_swap_role and the
                # pairs section renders 0 pairs despite a real, nonzero
                # leaf count (which comes from a separate, already-correct
                # tally).
                correct_swaps_df=(
                    _drop_irrelevant_roles(correct_swaps_df, (role1, role2))
                    if correct_swaps_df is not None else None
                ),
            )

    if not build_pairs:
        return

    refresh_deprel_index(
        target_col,
        save_dir=save_dir,
        html_dir=html_dir,
        pairs_dir=pairs_dir,
        diagnostics_csv=diagnostics_csv,
        leaf_threshold=leaf_threshold,
    )

    # Cross-pipeline overview index is no longer rebuilt here -- see
    # sva_trees.pipeline.Pipeline.run's identical comment;
    # scripts/generate_html_indexes.py now does this once, after all
    # conditions are (re)built.
