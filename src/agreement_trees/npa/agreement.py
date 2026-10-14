import functools
import json
import os
import re
import sys
from collections import Counter, defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

from agreement_trees.fit.decision_tree import MISSING, MIN_DF_LEN, fit_pipeline, set_dt_features_in_df, fit_excluded_for_display, split_unk_reasons
from agreement_trees.extract.process_treebank import META_FEATURES
from agreement_trees.report.viz_tree import tree2html
from agreement_trees.extract.per_treebank import (
    fit_treebank_trees, render_treebank_pages, build_nav, split_excluded, restrict_treebanks, pooled_page_stem,
    treebanks_done, write_treebanks_marker,
)
from agreement_trees.report.viz_deprel import generate_html_deprel_index
from multiblimp.excluded_treebanks import treebank_row_exclusion_reason
from multiblimp.languages import gblang2udlang, lang2langcode
from multiblimp.ud_table import clean_lemma
from multiblimp.ud_inflector import load_ud_inflector
from agreement_trees.taxonomy import NPA_FEATURE_NAMES
from agreement_trees.pipeline_utils import (
    find_other_running_instances, match_casing, run_in_pool, worker_mem_bytes, write_label_distribution,
)
from agreement_trees.npa.np_cache import cache_np_language, np_cache_status
from agreement_trees.vp.pipeline import get_impurity
from agreement_trees.pairs.create_pairs import (
    ambiguous_readings, bucket_examples_html, bucket_frame, coverage_stats, leaf_keep, process_item,
)
from agreement_trees.pairs.second_chance import second_chance_decision
from agreement_trees.pairs.diagnostics import (
    generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json,
)
from agreement_trees.npa.np_types import ROLE_PRIORITY
from multiblimp.config import (
    HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, OUTPUT_MINIMAL_PAIRS_DIR,
    OUTPUT_DIAGNOSTICS_DIR, RESOURCE_DIR, NP_INSTANCES_DIR,
)

UNDEFINED = "UNDEFINED"   # LEGACY: the old inflector's placeholder, UDInflector never returns it

# UPOS each role's inflector covers (build_role_inflector); HEAD spans all three head UPOS of np_types.
ROLE_UPOS = {
    "HEAD": ["NOUN", "PROPN", "PRON"],
    "DET": ["DET"],
    "NUM": ["NUM"],
    "ADJ": ["ADJ"],
    "ADP": ["ADP"],
    "PRON": ["PRON"],
}

# Short on-disk identifier per feature a "{Role1}-{Role2}_{Feature}" target_col can name (see npa_id);
# the inverse of condition_taxonomy.NPA_FEATURE_NAMES, which the overview pages use.
FEATURE_ABBREV = {name: abbrev for abbrev, name in NPA_FEATURE_NAMES.items()}


def npa_id(target_col: str) -> str:
    """"HEAD-DET_Number" -> "HEAD-DET_N", the on-disk/URL identifier of a target_col (unknown features keep
    their full name). Paths nest under a shared "npa/" folder, e.g. "decision_trees/npa/HEAD-DET_N"."""
    role1, role2, feat = _split_target_col(target_col)
    return f"{role1}-{role2}_{FEATURE_ABBREV.get(feat, feat)}"


_PAIRWISE_COL_RE = re.compile(r"[A-Za-z]+-[A-Za-z]+_[A-Za-z]+(?:\[[^\]]+\])?")


def npa_fit_omit_feats(columns, target_col: str, omit_extra: set | None = None) -> set:
    """Columns kept out of an NPA tree fit for target_col (see fit_npa_tree's
    docstring for the reasoning behind each rule)."""
    _, _, feat = _split_target_col(target_col)
    omit_feats = set(META_FEATURES) | {"np_type"}
    omit_feats.update(c for c in columns if "idx" in c)
    omit_feats.update(c for c in columns if c.endswith("_dir"))
    # every other pairwise agreement column (derived from the same kind of values as the label);
    # hyphenated child/sibling columns are ordinary predictors
    omit_feats.update(c for c in columns if _PAIRWISE_COL_RE.fullmatch(c) and c != target_col)
    omit_feats.update(
        c for c in columns
        if c.endswith(f"_{feat}") and not c.startswith("swap_")
    )
    # xpos of every role is blocked for now: in some tagsets it spells out the
    # agreement feature the label is computed from (e.g. Czech positional tags).
    omit_feats.update(c for c in columns if c.endswith("_xpos"))
    if omit_extra:
        omit_feats.update(omit_extra)
    return omit_feats


def fit_npa_tree(df: pd.DataFrame, target_col: str, drop_unk: bool = True,
                  max_depth: int = 12, min_samples_leaf: int = 25,
                  min_impurity_decrease: float = 0.005, test_size: float = 0.1,
                  leaf_threshold: float = 0.95, omit_extra: set | None = None):
    """Fit a decision tree predicting one pairwise agreement column (e.g. "HEAD-DET_Number", values
    yes/no/unk) from every other column of `df`; fit_dt's counterpart for role-based naming and
    yes/no/unk labels (reuses fit_pipeline/set_dt_features_in_df).

    Omitted from the predictors (npa_fit_omit_feats): META_FEATURES, *_idx and *_dir columns, np_type, every
    other pairwise agreement column, xpos, and every raw column ending in "_{feat}" of the target's own feature
    (HEAD_Case, DET_Case, ... ): the label is computed from exactly those, so the tree would just split on
    them. The omitted columns are re-attached to the returned dt_df (set_dt_features_in_df), so pairs, the
    page's sample tables and the unk split still have them.

    Returns (model, dt_df, y_train, sub_df): dt_df is X_train with leaf_id/leaf_top1_acc/leaf_decision etc.,
    sub_df the full filtered, unsplit dataset (tree2html's full_df). (None, None, None, sub_df) if there is
    too little data to fit.
    """
    sub_df = df[df[target_col].notna()].copy()
    if drop_unk:
        sub_df = sub_df[sub_df[target_col] != "unk"]
    if len(sub_df) < MIN_DF_LEN or sub_df[target_col].nunique() < 2:
        return None, None, None, sub_df

    omit_feats = npa_fit_omit_feats(sub_df.columns, target_col, omit_extra)

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


def canonical_target_col(target_col: str) -> str:
    """Flip a target_col naming its role pair backwards to ROLE_PRIORITY order, the only order np_instance
    writes ("DET-ADJ_Gender", never "ADJ-DET_Gender"); unchanged if canonical, unknown or unparseable."""
    try:
        role1, role2, feat = _split_target_col(target_col)
    except ValueError:
        return target_col
    if role1 in ROLE_PRIORITY and role2 in ROLE_PRIORITY \
            and ROLE_PRIORITY[role1] > ROLE_PRIORITY[role2]:
        return f"{role2}-{role1}_{feat}"
    return target_col


def _drop_irrelevant_roles(df: pd.DataFrame, keep_roles) -> pd.DataFrame:
    """Drop the single-role "{role}_*" columns of every role not in `keep_roles`, so tree2html's sample table
    shows only the pair being predicted (np_instances carries every role seen anywhere). Pairwise columns
    ("ADJ-ADP_Case") are untouched: after a role name they continue with "-", not "_"."""
    drop_roles = set(ROLE_PRIORITY) - set(keep_roles)
    drop_cols = [
        col for col in df.columns
        if any(col.startswith(f"{role}_") for role in drop_roles)
    ]
    return df.drop(columns=drop_cols)


def create_npa_pairs(dt_df: pd.DataFrame, target_col: str, inflector,
                      swap_role: str | None = None, context_inflector=None,
                      leaf_threshold: float | dict = 0.95,
                      second_chance_threshold: float | None = None,
                      verbose: bool = True) -> tuple[dict[str, pd.DataFrame], int]:
    """Minimal re-inflected pairs from a fit_npa_tree dt_df, for rows predicted "yes" in a kept leaf
    (leaf_top1_acc > leaf_threshold and leaf_decision; same convention as agreement_trees.pairs.create_pairs).
    leaf_threshold may be a {leaf_id: cutoff} map (second chance), with second_chance_threshold the strict
    bar, passed on the retry call only (rows then get "second_chance"/"second_chance_lax_threshold").

    Reinflects `swap_role`'s token via inflector.inflect on that role's feature columns, keeping the other
    role fixed. Default swap_role is the second role of target_col ("DET" for "HEAD-DET_Number"): the fixed
    role is the controller, the swapped one the target that must covary. A swap to a form with disjoint
    features shows the concord is real.

    context_inflector (covering the fixed role's POS) guards against a false "correct_swaps": if the fixed
    role's own form could also be read with the swapped value (German "die" is singular-feminine and plural),
    the row goes to "ambiguous_subjects" (the bucket name is shared with sva_trees; the fixed role is not
    literally a subject). None skips the check.

    Every swap form goes through agreement_trees.pairs.create_pairs.process_item, as in VP: same buckets and checks
    (conflicting features, extra argument slots or dimensions in the candidate); only the context check differs
    (the fixed role instead of the subject).

    Returns (dict[bucket, DataFrame], items_seen); items_seen counts every attempted swap form, which the
    bucket sizes alone cannot give (one row can yield several).
    """
    role1, role2, _feat = _split_target_col(target_col)
    if swap_role is None:
        swap_role = role2
    fixed_role = role2 if swap_role == role1 else role1

    row_threshold, keep = leaf_keep(dt_df, leaf_threshold)
    swap_df = dt_df[keep & (dt_df[target_col] == "yes")]
    if second_chance_threshold is not None:
        swap_df = swap_df.copy()
        swap_df["second_chance"] = swap_df["leaf_top1_acc"] <= second_chance_threshold
        swap_df["second_chance_lax_threshold"] = (
            row_threshold.loc[swap_df.index] if isinstance(row_threshold, pd.Series) else row_threshold
        )

    ufeat, _ = inflector.inflection_map
    # plain and bracketed ("Number[psor]") feature columns, as in agreement_trees.pairs.create_pairs
    def feature_columns(role):
        return [
            (col, m.group(1)) for col in swap_df.columns
            if (m := re.match(rf"^{re.escape(role)}_([A-Z][a-zA-Z]*(?:\[[^\]]+\])?)$", col))
        ]
    feat_cols = feature_columns(swap_role)
    fixed_feat_cols = feature_columns(fixed_role)

    diagnostics = {
        "correct_swaps": [], "same_forms": [], "same_features": [],
        "ambiguous_subjects": [], "conflicting_features": [], "undefined_features": [],
        "no_inflections": [], "no_candidates": [], "ambiguous_readings": [],
    }
    feature_distribution = Counter()
    items_seen = 0

    # the loop reads only the swap/fixed roles' feature and form columns (the frame is thousands of columns
    # wide); bucket frames are built columnar from swap_df afterwards (bucket_frame). itertuples(name=None)
    # keeps real column names (a namedtuple renames non-identifiers like "HEAD_Number[psor]").
    needed = [col for col, _ in feat_cols] + [col for col, _ in fixed_feat_cols]
    needed += [c for c in (f"{swap_role}_form", f"{fixed_role}_form", f"{swap_role}_form_orig", f"{swap_role}_lemma") if c in swap_df.columns]
    columns = list(dict.fromkeys(needed))
    loop_col_set = set(columns)

    def added_cols(item):
        return {k: v for k, v in item.items() if k not in loop_col_set}

    row_iter = enumerate(swap_df[columns].itertuples(index=False, name=None))
    for pos, row_tuple in (tqdm(row_iter, total=len(swap_df)) if verbose else row_iter):
        row_dict = dict(zip(columns, row_tuple))
        og_feats = {feat: row_dict.get(col) for col, feat in feat_cols if row_dict.get(col) != MISSING}
        form = row_dict.get(f"{swap_role}_form")
        fixed_form = row_dict.get(f"{fixed_role}_form")
        if form is None or pd.isna(form):
            continue

        lemma = clean_lemma(row_dict.get(f"{swap_role}_lemma"), form)
        swap_forms, feature_vals, swap_feats, readings = inflector.inflect(
            form, og_feats, return_swap_feats=True, return_readings=True, lemma=lemma
        )
        items_seen += 1

        if swap_forms is None:
            diagnostics["no_candidates"].append((pos, {}))
            continue
        if len(swap_forms) == 0:
            diagnostics["no_inflections"].append((pos, {}))
            continue

        context_features = set()
        if context_inflector is not None and fixed_form is not None and not pd.isna(fixed_form):
            context_features = context_inflector.get_form_features(
                fixed_form, {feat: row_dict.get(col) for col, feat in fixed_feat_cols if row_dict.get(col) != MISSING}, ufeat,
                only_try_ud_if_no_um=True,
            )
        results = []
        for swap_form in swap_forms:
            item_type, item = process_item(
                dict(row_dict), form, swap_form, og_feats, ufeat, feature_vals, feature_distribution,
                inflector, context_features, take_features_from=swap_role,
                swap_bundle=swap_feats.get(swap_form, {}),
            )
            results.append((item_type, swap_form, item))

        # accepted swap forms from different readings: none of them is kept
        correct = {sf for bucket, sf, _ in results if bucket == "correct_swaps"}
        ambiguous = ambiguous_readings(correct, readings)
        for bucket, _, item in results:
            if ambiguous and bucket == "correct_swaps":
                item["alternatives"] = sorted(correct)
                bucket = "ambiguous_readings"
            diagnostics[bucket].append((pos, added_cols(item)))

    if verbose:
        coverage = len(diagnostics["correct_swaps"]) / max(items_seen, 1) * 100
        print(f"swap_role={swap_role} fixed_role={fixed_role} items_seen={items_seen} "
              f"correct_swaps={len(diagnostics['correct_swaps'])} coverage={coverage:.1f}%")
        print("feature_vals distribution:", dict(feature_distribution))

    return {k: bucket_frame(swap_df, v) for k, v in diagnostics.items()}, items_seen


def swap_roles_for_target_col(target_col: str) -> list[str]:
    """The role(s) of `target_col` that create_npa_pairs_for_target_col reinflects. ROLE_PRIORITY puts HEAD
    first, so role1 == "HEAD" means the pair involves the head: only the other role is reinflected (the head is
    the controller). For two co-dependents ("DET-ADJ_Gender") both roles are returned, each reinflected in
    turn with the other fixed."""
    role1, role2, _feat = _split_target_col(target_col)
    return [role2] if role1 == "HEAD" else [role1, role2]


def build_role_inflector(role: str, feature: str, lang: str, resource_dir: str,
                          langcode: str | None = None, unimorph_args: dict | None = None):
    """A load_ud_inflector() for `role`, filtered to its UPOS (ROLE_UPOS) and swapping `feature` to any other
    observed value (inflection_map=(feature, None), the "_any" shape: one tree per target_col, whatever its
    feature). Returns (inflector, num_lemma, num_form)."""
    args = dict(unimorph_args) if unimorph_args else {
        "combine_um_ud": True,
        "remove_multiword_forms": True,
    }
    upos = ROLE_UPOS.get(role)
    if upos:
        args["filter_entries"] = {**args.get("filter_entries", {}), "upos": upos}

    inflector, num_lemma, num_form = load_ud_inflector(
        lang=lang,
        langcode=langcode or lang2langcode(lang),
        unimorph_args=args,
        inflection_map=(feature, None),
        resource_dir=resource_dir,
    )
    return inflector, num_lemma, num_form


def _unk_split_counts(full_df: pd.DataFrame, target_col: str, feat: str,
                       role_a: str, role_b: str) -> dict[str, int]:
    """Rows labelled "unk", split by which side's raw feature is missing: role_a's, role_b's or both. Keyed
    head_unk / nsubj_unk / both_unk like agreement_trees.pairs.diagnostics reads them (role_a is the reinflected role). All
    zero if either feature column is absent."""
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
    "correct_swaps", "same_forms", "same_features", "ambiguous_subjects", "conflicting_features",
    "undefined_features", "no_inflections", "no_candidates", "ambiguous_readings",
]


def create_npa_pairs_for_target_col(
    dt_df: pd.DataFrame, target_col: str, inflectors: dict,
    leaf_threshold: float | dict = 0.95, save_to: str | None = None,
    full_df: pd.DataFrame | None = None, label_distribution: dict | None = None,
    num_lemma=None, num_form=None, max_examples: int = 5, verbose: bool = True,
    extra_meta: dict | None = None, second_chance_threshold: float | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """The NPA minimal-pairs entry point: create_npa_pairs with the swap-direction rule
    (swap_roles_for_target_col), output shaped for agreement_trees.pairs.diagnostics and the diagnostics page (bucket
    parquets plus meta.json with coverage, unk counts, label distribution, examples).

    inflectors: {role: inflector} covering both roles of the pair (either can be the fixed one).
    With HEAD in the pair this makes one create_npa_pairs call; otherwise one per direction, the buckets are
    concatenated and items_seen / the keep pool summed (distinguishable via swap_DET / swap_ADJ). The unk split
    and labels use swap_roles[0] as the reinflected role, an arbitrary but consistent pick when there are two.

    leaf_threshold / second_chance_threshold / extra_meta: as in agreement_trees.pairs.create_pairs.create_pairs.
    Returns (diagnostic_dfs, meta); meta also carries "swap_role"/"fixed_role" for the page labels.
    """
    if full_df is None:
        full_df = dt_df
    role1, role2, feat = _split_target_col(target_col)
    swap_roles = swap_roles_for_target_col(target_col)

    collected = defaultdict(list)
    total_items_seen = 0
    total_n_keep = 0
    total_n_keep_second = 0
    row_threshold, keep = leaf_keep(dt_df, leaf_threshold)
    n_raw = int((dt_df[target_col] == "yes").sum())

    for swap_role in swap_roles:
        fixed_role = role2 if swap_role == role1 else role1
        buckets, items_seen = create_npa_pairs(
            dt_df, target_col, inflector=inflectors[swap_role],
            swap_role=swap_role, context_inflector=inflectors.get(fixed_role),
            leaf_threshold=leaf_threshold,
            second_chance_threshold=second_chance_threshold, verbose=verbose,
        )
        total_items_seen += items_seen
        total_n_keep += int((keep & (dt_df[target_col] == "yes")).sum())
        if second_chance_threshold is not None:
            total_n_keep_second += int(
                (keep & (dt_df[target_col] == "yes")
                 & (dt_df["leaf_top1_acc"] <= second_chance_threshold)).sum())
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
    num_forms_of_interest, num_covered_um, num_covered_um_ud = coverage_stats(
        full_df, [f"{role1}_form", f"{role2}_form"], inflectors[swap_role]
    )

    meta = {
        "leaf_min_acc": leaf_threshold,
        "num_ud_candidates_raw": n_raw,
        "num_ud_candidates_keep": total_n_keep,
        "num_ud_candidates_keep_second_chance": total_n_keep_second,
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
    if extra_meta:
        meta.update(extra_meta)

    # example tables show only (role1, role2); the saved frames keep every column (other roles stay valid
    # tree predictors)
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
                          leaf_threshold: float = 0.95,
                          has_pairs: bool = True) -> None:
    """Rebuild one target_col's diagnostics CSV and deprel index.html from the on-disk decision-tree and
    minimal-pair artifacts only (the tail of run_agreement_pipeline), so generate_html_indexes.py can refresh a
    built condition without the per-language loop reading np_instances. Paths default as in
    run_agreement_pipeline. has_pairs=False (data-debugging mode) skips the diagnostics table and renders the
    plain accuracy-scatter index.
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

    diagnostics_by_lang = None
    if has_pairs:
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
        pairs_dir=pairs_dir if has_pairs else None,
        diagnostics_by_lang=diagnostics_by_lang,
        agreement_label=f"NP {role1}–{role2} ({feat})",
        head_role_label=swap_role,
        subject_label=fixed_role,
        nsubj_label=fixed_role,
        debug_view=not has_pairs,
    )


def _npa_unk_split(role1: str, role2: str, feat: str) -> dict:
    """unk_split config: an incl.-unk tree predicts *why* ("{role}'s {feature} unknown" per
    role, "{feature} unknown on both"), not one merged "unk"."""
    def shown(role):
        return "NP head" if role == "HEAD" else role
    label_a, label_b = f"{shown(role1)}'s {feat} unknown", f"{shown(role2)}'s {feat} unknown"
    both_label = f"{feat} unknown on both"
    grey = "#94a1b2"
    # "unknown (other)" is left out of the palette so its always-empty swatch does not
    # appear in every legend (it still falls back to the default unk color)
    return {
        "col_a": f"{role1}_{feat}", "label_a": label_a,
        "col_b": f"{role2}_{feat}", "label_b": label_b,
        "both_label": both_label, "other_label": "unknown (other)",
        "palette": {
            "yes": "#31cb9f", "no": "#f16393",
            label_a: "#f6c453", label_b: "#5aa9e6",
            both_label: grey,
        },
    }


def _fit_language(lang: str, ctx: dict) -> None:
    """One language: extract its np_instances if missing, fit the tree, build
    pairs and render the pages. Module-level so it can run in a worker process."""
    target_col = ctx["target_col"]
    instances_dir = ctx["instances_dir"]
    save_dir = ctx["save_dir"]
    html_dir = ctx["html_dir"]
    resource_dir = ctx["resource_dir"]
    pairs_dir = ctx["pairs_dir"]
    never_skip = ctx["never_skip"]
    drop_unk = ctx["drop_unk"]
    max_depth = ctx["max_depth"]
    min_samples_leaf = ctx["min_samples_leaf"]
    test_size = ctx["test_size"]
    leaf_threshold = ctx["leaf_threshold"]
    palette_map = ctx["palette_map"]
    unimorph_args = ctx["unimorph_args"]
    build_pairs = ctx["build_pairs"]
    verbose = ctx["verbose"]
    per_treebank = ctx["per_treebank"]
    include_excluded = ctx["include_excluded"]
    detailed_unk = ctx["detailed_unk"]
    incl_unk = ctx["incl_unk"]
    second_chance = ctx["second_chance"]
    role1 = ctx["role1"]
    role2 = ctx["role2"]
    feat = ctx["feat"]
    swap_role = ctx["swap_role"]
    impurity_fn = ctx["impurity_fn"]
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    if ctx["extract"] is not None and np_cache_status(lang, ctx["recache_data"]) is None:
        cache_np_language(lang, None, **ctx["extract"])
    instances_path = os.path.join(instances_dir, f"{cached_lang}.parquet")
    if not os.path.exists(instances_path):
        print(f"{lang}: skip, no np_instances parquet at {instances_path}")
        return

    raw_df, excluded_raw_df = split_excluded(pd.read_parquet(instances_path))
    if ctx["only_treebanks"]:
        raw_df, excluded_raw_df = restrict_treebanks(raw_df, ctx["only_treebanks"]), None
    if target_col not in raw_df.columns:
        print(f"{lang}: skip, missing column {target_col}")
        return

    full_df = raw_df[raw_df[target_col].notna()].copy()
    if len(full_df) == 0:
        print(f"{lang}: skip, no rows with {target_col} set")
        return

    # full_df/dt_df keep every role's columns (other roles' features remain valid tree predictors); only the
    # output views (tree2html's full_df, the example tables) are trimmed to role1/role2
    print(lang)

    unk_split_cfg = _npa_unk_split(ctx["role1"], ctx["role2"], ctx["feat"]) if ctx["detailed_unk"] else None
    shown_labels = full_df[target_col]
    if unk_split_cfg:
        shown_labels = split_unk_reasons(
            full_df, target_col,
            **{k: unk_split_cfg[k] for k in ("col_a", "label_a", "col_b", "label_b", "both_label", "other_label")})
    write_label_distribution(
        save_dir, cached_lang, {str(k): int(v) for k, v in shown_labels.value_counts().items()},
    )

    joblib_path = os.path.join(save_dir, f"{cached_lang}.joblib")
    dt_parquet_path = os.path.join(save_dir, f"{cached_lang}.parquet")
    html_path = os.path.join(html_dir, f"{cached_lang}.html")

    pred_values = set(full_df[target_col].values) - ({"unk"} if drop_unk else set())
    model, dt_df, learn_dt = None, None, False

    if never_skip or not os.path.exists(joblib_path):
        if len(pred_values) > 1:
            model, dt_df, _, _ = fit_npa_tree(
                full_df, target_col, drop_unk=drop_unk, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=impurity_fn(len(full_df)),
                test_size=test_size, leaf_threshold=leaf_threshold,
            )
            learn_dt = model is not None
        if learn_dt:
            joblib.dump(model, joblib_path)
            dt_df.to_parquet(dt_parquet_path)
        else:
            for stale in (joblib_path, dt_parquet_path):
                if os.path.exists(stale):
                    os.remove(stale)
            model = None
            dt_df = full_df.assign(**{target_col: shown_labels}) if unk_split_cfg else full_df
    else:
        model = joblib.load(joblib_path)
        dt_df = pd.read_parquet(dt_parquet_path)
        learn_dt = True

    # pairs are built before tree2html so the page's pairs section can join this run's correct_swaps by leaf_id
    lang_pairs_dir = os.path.join(pairs_dir, cached_lang)
    correct_swaps_df = None
    lang_leaf_threshold = leaf_threshold
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
            pairs_kwargs = dict(
                save_to=lang_pairs_dir, full_df=full_df,
                label_distribution=label_distribution,
                num_lemma=swap_num_lemma, num_form=swap_num_form, verbose=verbose,
            )
            diagnostic_dfs, _ = create_npa_pairs_for_target_col(
                dt_df, target_col, inflectors, leaf_threshold=leaf_threshold,
                **pairs_kwargs,
            )
            if second_chance is not None:
                retry, lax, info = second_chance_decision(
                    dt_df, target_col, label_distribution,
                    strict_threshold=leaf_threshold,
                    min_samples_leaf=min_samples_leaf, cfg=second_chance)
                print(f"{lang} {target_col}: second chance: {info['reason']}")
                if retry:
                    # adopted even with zero correct_swaps: the lax pass's numbers are the ones to record
                    diagnostic_dfs, _ = create_npa_pairs_for_target_col(
                        dt_df, target_col, inflectors, leaf_threshold=lax,
                        extra_meta={"second_pass": True, "second_chance_info": info},
                        second_chance_threshold=leaf_threshold,
                        **pairs_kwargs,
                    )
                    lang_leaf_threshold = lax
                    if not len(diagnostic_dfs["correct_swaps"]):
                        print(f"{lang} {target_col}: retry attempted, yielded no pairs")
            correct_swaps_df = diagnostic_dfs.get("correct_swaps")
        else:
            # pairs built by an earlier run: read the cached correct_swaps bucket for the page
            correct_swaps_path = os.path.join(lang_pairs_dir, "correct_swaps.parquet")
            if os.path.exists(correct_swaps_path):
                correct_swaps_df = pd.read_parquet(correct_swaps_path)
            # a retried language's tree page must show the (lax) cutoffs
            # its pairs were actually built with
            meta_path = os.path.join(lang_pairs_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    cached_threshold = json.load(f).get("leaf_min_acc")
                if isinstance(cached_threshold, dict):
                    lang_leaf_threshold = {int(k): v for k, v in cached_threshold.items()}

    shown_df = _drop_irrelevant_roles(full_df, (role1, role2))
    tree_html_kwargs = dict(
        discounted_classes=[c for c in shown_labels.unique() if c not in ("yes", "no")] if drop_unk else None,
        predictor_var=target_col,
        target=None,
        max_rows=15,
        only_show_real_orders=True,
        correlate_features=True,
        show_features=True,
        full_tree_html=learn_dt,
        palette_map=palette_map or (unk_split_cfg["palette"] if unk_split_cfg
                                    else {"yes": "#31cb9f", "no": "#f16393", "unk": "#b893de"}),
        leaf_threshold=lang_leaf_threshold,
        strict_leaf_threshold=leaf_threshold,
        head_label="NP head",
        fit_excluded=fit_excluded_for_display(npa_fit_omit_feats(full_df.columns, target_col)),
    )

    treebank_nav, treebanks_fresh = None, False
    if per_treebank and (learn_dt or incl_unk) and "treebank" in full_df.columns:
        extra_df = None
        if include_excluded and excluded_raw_df is not None and target_col in excluded_raw_df.columns:
            extra_df = excluded_raw_df[excluded_raw_df[target_col].notna()].copy()
            if not len(extra_df):
                extra_df = None
        if full_df["treebank"].nunique() > 1 or extra_df is not None or incl_unk:
            tb_cache_dir = os.path.join(save_dir, "treebanks")
            unk_split = _npa_unk_split(role1, role2, feat)
            effective_unk_split = unk_split if detailed_unk else None
            tb_summary, treebanks_fresh = fit_treebank_trees(
                full_df, cached_lang, model, target_col, None, tb_cache_dir,
                fit_kwargs=dict(min_samples_leaf=min_samples_leaf, max_depth=max_depth, test_size=test_size),
                omit_feats_fn=lambda d: npa_fit_omit_feats(d.columns, target_col),
                impurity_fn=impurity_fn, pooled_impurity=impurity_fn(len(full_df)),
                drop_unk=drop_unk, never_skip=never_skip,
                extra_df=extra_df, include_excluded=include_excluded,
                reason_fn=treebank_row_exclusion_reason,
                unk_split=effective_unk_split, incl_unk=incl_unk)
            if treebanks_fresh or never_skip or not os.path.isdir(os.path.join(html_dir, cached_lang)):
                render_treebank_pages(
                    tb_summary, cached_lang, shown_df, tb_cache_dir, html_dir, tree_html_kwargs,
                    extra_df=(_drop_irrelevant_roles(extra_df, (role1, role2))
                              if extra_df is not None else None),
                    unk_split=effective_unk_split)
            treebank_nav = build_nav(tb_summary, cached_lang)
            html_path = os.path.join(html_dir, f"{pooled_page_stem(cached_lang, tb_summary)}.html")

    if per_treebank and not os.path.exists(os.path.join(save_dir, "treebanks", f"{cached_lang}.json")):
        write_treebanks_marker(os.path.join(save_dir, "treebanks", f"{cached_lang}.json"),
                               include_excluded, incl_unk, detailed_unk)

    if never_skip or treebanks_fresh or not os.path.exists(html_path):
        tree2html(
            pipeline_model=model,
            dt_df=dt_df,
            full_df=shown_df,
            out_file=html_path,
            meta={"Language": lang},
            treebank_nav=treebank_nav,
            # same role trim as full_df, or html_tree picks a bystander role as the swap role
            correct_swaps_df=(
                _drop_irrelevant_roles(correct_swaps_df, (role1, role2))
                if correct_swaps_df is not None else None
            ),
            **tree_html_kwargs,
        )


def _is_done(lang: str, ctx: dict) -> bool:
    """Pages, pairs and per-treebank summary all exist for the current settings."""
    if ctx["never_skip"]:
        return False
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    return (os.path.exists(os.path.join(ctx["html_dir"], f"{cached_lang}.html"))
            and (not ctx["build_pairs"] or os.path.isdir(os.path.join(ctx["pairs_dir"], cached_lang)))
            and (not ctx["per_treebank"] or treebanks_done(
                os.path.join(ctx["save_dir"], "treebanks", f"{cached_lang}.json"),
                ctx["include_excluded"], ctx["incl_unk"], ctx["detailed_unk"])))


def run_agreement_pipeline(target_col: str, langs: list[str], instances_dir: str = NP_INSTANCES_DIR,
                            save_dir: str | None = None,
                            html_dir: str | None = None,
                            decision_trees_root: str = OUTPUT_DECISION_TREES_DIR,
                            html_decision_trees_root: str = HTML_DECISION_TREES_DIR,
                            resource_dir: str = RESOURCE_DIR,
                            pairs_dir: str | None = None,
                            diagnostics_csv: str | None = None,
                            never_skip: bool = False,
                            recache_data: bool = False,
                            extract: dict | None = None,
                            drop_unk: bool = True, max_depth: int = 12,
                            min_samples_leaf: int = 10, test_size: float = 0.1,
                            leaf_threshold: float = 0.95,
                            impurity: dict | None = None,
                            palette_map: dict | None = None,
                            unimorph_args: dict | None = None,
                            build_pairs: bool = True, verbose: bool = True,
                            per_treebank: bool = True,
                            include_excluded: bool = False,
                            detailed_unk: bool = False,
                            incl_unk: bool = False,
                            second_chance=None,
                            build_index: bool = True,
                            n_jobs: int = 1,
                            max_worker_mem_gb: float | None = None,
                            mem_headroom: float = 0.8,
                            max_tasks_per_child: int = 1,
                            force: bool = False,
                            only_treebanks: list[str] | None = None) -> None:
    """Fit one NP pairwise agreement column (e.g. "HEAD-DET_Number") for every language (tree, page and, with
    build_pairs, minimal pairs, in worker processes like agreement_trees.vp.pipeline.Pipeline), then the diagnostics
    table and deprel index (build_index).

    A language without an np_instances parquet is extracted first (`extract` = max_treebank_len/streaming/
    chunk_size/fetch_all for agreement_trees.npa.np_cache; None never extracts). never_skip redoes fit, pairs and pages;
    recache_data also re-extracts; a language whose pages, pairs and per-treebank summary exist is skipped.

    Paths default to {decision_trees_root}/npa/{npa_id(target_col)} and the html/pairs/diagnostics equivalents.
    unimorph_args goes to build_role_inflector. include_excluded adds trees for excluded treebanks (never the
    pooled tree or pairs); incl_unk also fits {lang}__unk; detailed_unk splits its unk class by cause;
    second_chance is a SecondChanceConfig (needs build_pairs); impurity holds get_impurity's parameters.
    """
    if not force:
        other_pids = find_other_running_instances(sys.argv[0])
        if other_pids:
            raise RuntimeError(
                f"Another instance of {os.path.basename(sys.argv[0])} appears to already be running "
                f"(PID(s): {other_pids}). If they're stale, kill them first: "
                f"kill {' '.join(map(str, other_pids))}. Else, pass force=True.")

    npa_identifier = npa_id(target_col)
    save_dir = save_dir or os.path.join(decision_trees_root, "npa", npa_identifier)
    html_dir = html_dir or os.path.join(html_decision_trees_root, "npa", npa_identifier)
    pairs_dir = pairs_dir or os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", npa_identifier)
    diagnostics_csv = diagnostics_csv or os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{npa_identifier}.csv")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    role1, role2, feat = _split_target_col(target_col)
    swap_role = swap_roles_for_target_col(target_col)[0]

    ctx = dict(
        target_col=target_col, instances_dir=instances_dir, save_dir=save_dir, html_dir=html_dir,
        resource_dir=resource_dir, pairs_dir=pairs_dir, never_skip=never_skip, recache_data=recache_data,
        extract=extract, drop_unk=drop_unk, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        test_size=test_size, leaf_threshold=leaf_threshold,
        impurity_fn=functools.partial(get_impurity, **(impurity or {})),
        palette_map=palette_map, unimorph_args=unimorph_args, build_pairs=build_pairs, verbose=verbose,
        per_treebank=per_treebank, include_excluded=include_excluded, detailed_unk=detailed_unk,
        incl_unk=incl_unk, only_treebanks=only_treebanks or [], second_chance=second_chance, role1=role1, role2=role2, feat=feat,
        swap_role=swap_role, fixed_role=role2 if swap_role == role1 else role1,
    )

    tasks = {}
    for lang in sorted(langs):
        if _is_done(lang, ctx):
            print(f"{lang}: Skip, already done")
        else:
            tasks[lang] = (lang, ctx)

    mem_bytes = worker_mem_bytes(n_jobs, max_worker_mem_gb, mem_headroom)
    if mem_bytes is None:
        print("Could not detect system RAM; running without a memory cap")
    else:
        print(f"Capping each of {n_jobs} worker(s) to {mem_bytes / 1024**3:.1f} GB RAM")

    def report_failure(lang, _result, error):
        if error is not None:
            print(f"  FAILED: {lang}: {error}")

    run_in_pool(_fit_language, tasks, n_jobs, mem_bytes, max_tasks_per_child, on_done=report_failure)

    if build_index:
        refresh_deprel_index(
            target_col,
            save_dir=save_dir,
            html_dir=html_dir,
            pairs_dir=pairs_dir,
            diagnostics_csv=diagnostics_csv,
            leaf_threshold=leaf_threshold,
            has_pairs=build_pairs,
        )
