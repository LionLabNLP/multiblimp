import sys
import os
import re
import json
import html as html_lib

import pandas as pd
from collections import Counter
from typing import *
from tqdm import tqdm

sys.path.append("../")
from multiblimp.swap_features import *
from word_order.prediction_target import PredictionTarget, nsubj_target
from word_order.utils import build_grew_link
from multiblimp.unimorph import load_inflector


UNDEFINED="UNDEFINED"


def _coverage_stats(df, child_deprel, inflector):
    """(# forms of interest, # covered by UM alone, # covered by UM+UD combined).

    Forms of interest: every distinct head_form / <child_deprel>_form value across
    ALL of df (not just the "Yes"-labeled swap candidates) -- i.e. every word that
    plays the head or child role for this prediction target, regardless of whether
    it ends up eligible for a swap. "Covered" means the form string appears at all
    in the relevant UniMorph data, independent of whether it carries the right
    features for a swap.
    """
    form_cols = [c for c in (f"head_form", f"{child_deprel}_form") if c in df.columns]
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


# ── Example-row rendering ─────────────────────────────────────────────────────
# For any diagnostics bucket (no_candidates, same_features, correct_swaps, ...),
# renders an HTML table of sample rows -- reconstructed before/after sentences
# with the agreement-relevant tokens highlighted, plus the key columns and a
# grew.fr treebank link -- via bucket_examples_html(). create_pairs() below
# folds this output into the "examples" key of each language's meta.json,
# which word_order.html.html_deprel's report links to inline, per bucket.
# Formerly its own module (sva_trees.flowchart); folded in here since
# create_pairs() was its only caller.

# ── Sentence reconstruction ──────────────────────────────────────────────────
# Mirrors word_order/create_pairs.py's get_sen_str/get_swapped_sen_str: correct
# spacing needs the live tree's SpaceAfter=No misc annotations, which aren't
# carried in the saved "sen" column (just bare form strings). When a treebank
# is available we look the tree up by tree_idx and use its real SpaceAfter
# data; otherwise we fall back to a punctuation heuristic.

def _no_space_afters(tree) -> list:
    return [(tok["misc"] or {}).get("SpaceAfter") == "No" for tok in tree]


def _render_token(tok, highlighted: bool) -> str:
    """Escape a token and wrap it in <strong> if it's an agreement-relevant token
    (the head/child whose feature is being checked or was swapped)."""
    text = html_lib.escape(str(tok))
    return f"<strong>{text}</strong>" if highlighted else text


def _join_with_spacing(tokens, no_space_afters, highlight_idx=frozenset()) -> str:
    parts = [
        _render_token(tok, i in highlight_idx) + ("" if no_space else " ")
        for i, (tok, no_space) in enumerate(zip(tokens, no_space_afters))
    ]
    return "".join(parts).strip()


_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'", "’", "%"}


def _fallback_join(tokens, highlight_idx=frozenset()) -> str:
    """Heuristic spacing used when no treebank is available: no space before
    closing punctuation. Less accurate than real SpaceAfter data."""
    parts = []
    for i, tok in enumerate(tokens):
        if i > 0 and str(tok) not in _NO_SPACE_BEFORE:
            parts.append(" ")
        parts.append(_render_token(tok, i in highlight_idx))
    return "".join(parts)


def _detect_swap_col(row):
    """Find the swap_<kind> column process_item set on this row, if any.

    Only matches scalar string values, since the source df already has unrelated
    columns sharing the "swap_" prefix (e.g. swap_order_candidates, a list)."""
    for col in row.index:
        if col.startswith("swap_") and isinstance(row.get(col), str) and row[col]:
            return col, col[len("swap_"):]
    return None, None


def _resolve_kind(row, default_kind="head"):
    _, kind = _detect_swap_col(row)
    return kind or default_kind


# ── Metadata: feats + treebank link ──────────────────────────────────────────

def _feats_summary(row, prefix, highlight_feat=None) -> str:
    """Collect this node's set morphological features (e.g. head_Number, head_Case)
    into a collapsed <details> list, one Feat=Val per line — same pattern as the
    "_features" columns in word_order/html/html_tree.py's example table, so a long
    feature set never overflows into neighboring cells. Column naming matches
    extract_node_features's f"{prefix}_{feat}" convention; excludes the many other
    {prefix}_-prefixed columns (form, idx, deprel, sibling-*, child-*, ...) since
    those never look like a bare CamelCase feature name.

    highlight_feat: the feature actually being swapped/compared (e.g. "Number"),
    bolded in the list so it's easy to spot among the node's other features."""
    pat = re.compile(rf"^{re.escape(prefix)}_([A-Z][a-zA-Z]*)$")
    pairs = []
    highlight_val = None
    for col in row.index:
        match = pat.match(col)
        if not match:
            continue
        val = row[col]
        if pd.isna(val) or val in (None, "None", "", "_missing"):
            continue
        feat = match.group(1)
        val_str = html_lib.escape(str(val))
        line = f"{feat}={val_str}"
        if highlight_feat and feat == highlight_feat:
            line = f'<strong class="swap-feat">{line}</strong>'
            highlight_val = val_str
        pairs.append((feat, line))

    if not pairs:
        return "&mdash;"

    pairs.sort(key=lambda p: p[0])
    items = "<br>".join(line for _, line in pairs)
    badge = (
        f' &middot; <span class="swap-feat-badge">{html_lib.escape(highlight_feat)}={highlight_val}</span>'
        if highlight_val is not None else ""
    )
    return f'<details><summary>{len(pairs)} feats{badge}</summary>{items}</details>'


def _treebank_link(row) -> str:
    """grew.fr query-link for a single example row — shared with word_order/
    viz_tree.py's build_treebank_links via word_order.utils.build_grew_link."""
    form_cols = [c for c in row.index if c.endswith("_form") and pd.notna(row.get(c))]
    form_values = [row[c] for c in form_cols]
    link = build_grew_link(row.get("treebank"), row.get("sent_id"), form_values)
    return link if link is not None else "&mdash;"


def _highlight_indices(row, kind, child_deprel=None) -> set:
    """0-based positions in `sen` of the agreement-relevant tokens: the swapped
    node (e.g. head) and, when known, its agreement partner (e.g. nsubj)."""
    idx = set()
    idx_col = f"{kind}_idx"
    if idx_col in row.index and pd.notna(row.get(idx_col)):
        idx.add(int(row[idx_col]) - 1)
    if child_deprel:
        child_idx_col = f"{child_deprel}_idx"
        if child_idx_col in row.index and pd.notna(row.get(child_idx_col)):
            idx.add(int(row[child_idx_col]) - 1)
    return idx


def _sentence_pair(row, treebank=None, child_deprel=None, default_kind="head"):
    """Return (before, after) HTML sentence strings for one example row, with
    agreement-relevant tokens wrapped in <strong>. `after` is None when the
    bucket has no swap_<kind> value (e.g. no_candidates).

    Spacing precedence: the row's own "no_space_after" column (saved alongside
    "sen" at extraction time, process_treebank.py's tree_no_space_after) if
    present, else a live treebank lookup by tree_idx (for data saved before
    "no_space_after" existed), else the punctuation heuristic."""
    sen = row.get("sen")
    if sen is None:
        return None, None
    sen = list(sen)

    swap_col, kind = _detect_swap_col(row)
    kind = kind or default_kind
    idx_col = f"{kind}_idx"
    highlight_idx = _highlight_indices(row, kind, child_deprel)

    swapped_tokens = None
    if swap_col is not None and idx_col in row.index and pd.notna(row.get(idx_col)):
        swapped_tokens = list(sen)
        swapped_tokens[int(row[idx_col]) - 1] = row[swap_col]

    nsa = row.get("no_space_after")
    if nsa is not None and len(nsa) == len(sen):
        before = _join_with_spacing(sen, nsa, highlight_idx)
        after = _join_with_spacing(swapped_tokens, nsa, highlight_idx) if swapped_tokens else None
        return before, after

    if treebank is not None and "tree_idx" in row.index and pd.notna(row.get("tree_idx")):
        try:
            tree = treebank[int(row["tree_idx"])]
            nsa = _no_space_afters(tree)
            if len(nsa) == len(sen):
                before = _join_with_spacing(sen, nsa, highlight_idx)
                after = _join_with_spacing(swapped_tokens, nsa, highlight_idx) if swapped_tokens else None
                return before, after
        except (IndexError, KeyError, TypeError):
            pass  # fall through to the heuristic below

    before = _fallback_join(sen, highlight_idx)
    after = _fallback_join(swapped_tokens, highlight_idx) if swapped_tokens else None
    return before, after


# ── Table rendering ───────────────────────────────────────────────────────────

def _fmt_cell(val, max_len=40):
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple, set)):
        val = ", ".join(map(str, val))
    text = str(val)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return html_lib.escape(text)


def _fmt_sentence(html_text):
    """`html_text` comes pre-escaped (with <strong> highlights) from _sentence_pair;
    do not re-escape or truncate by character, which would corrupt the tags."""
    return html_text if html_text is not None else "&mdash;"


def _diverse_sample(item_df: pd.DataFrame, max_examples: int, by: str = "feature_vals") -> pd.DataFrame:
    """Sample up to max_examples rows, covering as many distinct `by` values as
    possible first (e.g. both "SG -> PL" and "PL -> SG"), rather than just the
    first N rows which could all be a single, over-represented category."""
    if by not in item_df.columns or item_df[by].nunique(dropna=True) <= 1:
        return item_df.head(max_examples)

    groups = [g for _, g in item_df.groupby(by, sort=False)]
    picks = []
    round_idx = 0
    while len(picks) < max_examples and any(len(g) > round_idx for g in groups):
        for g in groups:
            if len(picks) >= max_examples:
                break
            if len(g) > round_idx:
                picks.append(g.iloc[round_idx])
        round_idx += 1

    return pd.DataFrame(picks) if picks else item_df.head(max_examples)


def _examples_table_html(item_df: pd.DataFrame, max_examples: int, treebank=None,
                          child_deprel=None, default_kind="head", swap_feature=None) -> str:
    """Small HTML table with up to max_examples sample rows from one diagnostics bucket,
    covering as many distinct feature_vals categories (e.g. SG -> PL vs PL -> SG) as
    the example budget allows."""
    if item_df is None or len(item_df) == 0:
        return '<p class="empty">no examples</p>'

    preferred = [c for c in item_df.columns if c.endswith("_form") or c in ("feature_vals", "alternatives")]
    forms = sorted(c for c in preferred if c.endswith("_form"))
    rest = [c for c in preferred if c not in forms]
    cols = (forms + rest)[:6] or list(item_df.columns[:6])

    has_sentence = "sen" in item_df.columns
    has_treebank = "treebank" in item_df.columns and "sent_id" in item_df.columns
    sample = _diverse_sample(item_df, max_examples)

    meta_cols = []
    if default_kind:
        meta_cols.append(f"{default_kind}_feats")
        meta_cols.append(f"{default_kind}_feats_after")
    if child_deprel:
        meta_cols.append(f"{child_deprel}_feats")
    if has_treebank:
        meta_cols.append("treebank")

    header_cols = (["before", "after"] if has_sentence else []) + cols + meta_cols
    header = "".join(f"<th>{html_lib.escape(c)}</th>" for c in header_cols)
    body_rows = ""
    for _, row in sample.iterrows():
        cells = ""
        if has_sentence:
            before, after = _sentence_pair(row, treebank=treebank, child_deprel=child_deprel,
                                            default_kind=default_kind)
            cells += f'<td class="sentence">{_fmt_sentence(before)}</td>'
            cells += f'<td class="sentence">{_fmt_sentence(after)}</td>'
        cells += "".join(f"<td>{_fmt_cell(row[c])}</td>" for c in cols)
        if default_kind:
            kind = _resolve_kind(row, default_kind)
            cells += f'<td class="feats">{_feats_summary(row, kind, highlight_feat=swap_feature)}</td>'
            cells += f'<td class="feats">{_feats_summary(row, f"after_{kind}", highlight_feat=swap_feature)}</td>'
        if child_deprel:
            cells += f'<td class="feats">{_feats_summary(row, child_deprel, highlight_feat=swap_feature)}</td>'
        if has_treebank:
            cells += f'<td class="treebank">{_treebank_link(row)}</td>'
        body_rows += f"<tr>{cells}</tr>"

    table = f'<table class="examples"><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>'
    return f'<div class="examples-scroll">{table}</div>'


def bucket_examples_html(diagnostic_dfs: dict, item_types, max_examples: int = 5, treebank=None,
                          child_deprel=None, default_kind="head", swap_feature=None) -> dict:
    """_examples_table_html for every requested bucket at once, keyed by
    item_type -- what create_pairs() folds into the "examples" key of each
    language's meta.json, which word_order.html.html_deprel's report links to
    inline. Missing/empty buckets still get a "no examples" fragment rather
    than a missing key, since callers key off the same items every time.
    """
    return {
        item_type: _examples_table_html(
            diagnostic_dfs.get(item_type, pd.DataFrame()), max_examples,
            treebank=treebank, child_deprel=child_deprel,
            default_kind=default_kind, swap_feature=swap_feature,
        )
        for item_type in item_types
    }


def process_item(
        item,
        form,
        swap_form,
        form_features,
        ufeat,
        feature_vals,
        feature_distribution,
        inflector,
        context_inflector,
        take_features_from,
        max_num_of_pairs=None,
        swap_bundle=None,
    ) -> Tuple[str, Dict[str, str]]:
        item[f"swap_{take_features_from}"] = swap_form

        if swap_form == form:
            return "same_forms", item
        else:
            swap_feature_vals = inflector.get_form_features(
                swap_form, form_features, ufeat
            )
            feature_key = (
                "|".join(sorted(feature_vals))
                + " -> "
                + "|".join(sorted(swap_feature_vals))
            )

            # The reinflected form's OTHER features, acc to reinflection source
            for feat, vals in (swap_bundle or {}).items():
                item[f"after_{take_features_from}_{feat}"] = "/".join(sorted(vals))

            add_item = False
            for feat1 in feature_vals:
                for feat2 in swap_feature_vals:
                    if (
                        max_num_of_pairs is None
                        or feature_distribution[f"{feat1} -> {feat2}"]
                        < max_num_of_pairs
                    ):
                        add_item = True
                    else:
                        add_item = None

            if (len(feature_vals & swap_feature_vals) == 0) and (
                UNDEFINED not in swap_feature_vals
            ):
                if (len(swap_feature_vals) > 0) and add_item:
                    if context_inflector is None:
                        child_features = set()
                    else:
                        child_features = context_inflector.get_form_features(
                            item["child"],
                            {"Case": "Nom"},  # ergative?
                            ufeat,
                            only_try_ud_if_no_um=True,
                        )
                    if len(child_features & swap_feature_vals) > 0:
                        return "ambiguous_subjects", item
                    else:
                        item["feature_vals"] = feature_key
                        num_combinations = len(feature_vals) * len(swap_feature_vals)
                        for feat1 in feature_vals:
                            for feat2 in swap_feature_vals:
                                feature_distribution[f"{feat1} -> {feat2}"] += (
                                    1 / num_combinations
                                )
                        if num_combinations > 1:
                            feature_distribution[feature_key] += 1
                        return "correct_swaps", item
                elif add_item is None:
                    return "extra_pairs", item
                else:
                    return "undefined_features", item
            elif len(feature_vals & swap_feature_vals) > 0:
                wrong_item = dict(item)
                wrong_item[f"swap_{take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "same_features", wrong_item
            else:
                wrong_item = dict(item)
                wrong_item[f"swap_{take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "undefined_features", wrong_item


def create_pairs(df, swap_feat, inflector, target: PredictionTarget = nsubj_target,
                  swap_target=["head",], context_inflector=None, max_num_of_pairs=None,
                  leaf_threshold=0.1, save_to=None,
                  max_examples=5, num_lemma=None, num_form=None, full_df=None,
                  unk_counts=None, label_distribution=None):
    """
    Create re-inflected minimal sentence pairs for each row in the decision tree dataframe.

    Args:
        dt_df (pd.DataFrame): Decision tree dataframe containing the rows to process.
        leaf_threshold: entropy cutoff below which a leaf's prediction counts as "keep".
            Recomputed here from leaf_top1_entropy/leaf_decision rather than trusting the
            df's precomputed `keep` column, so it can be tuned without re-running fit_dt.
        save_to: if given, a directory to write every diagnostics bucket to, as
            "<item_type>.parquet", plus a single "meta.json" holding the scalar stats,
            the unk-drop counts, and up to max_examples sample rows per bucket
            (bucket_examples_html, under "examples") -- one file, since
            sva_trees.diagnostics reads all of it back together anyway.
        max_examples: max example rows kept per bucket under meta.json's "examples"
            (default 5).
        num_lemma, num_form: lexicon-size stats for this language.
        full_df: the language's complete per-target dataframe, used only for
            _coverage_stats' "forms of interest" population. `df` (dt_df) is
            frequently just fit_dt's ~90% train split (its test_size defaults to
            0.1 and callers rarely override it), so falling back to `df` here
            would silently undercount forms_of_interest for any language that
            got a real fitted tree. Defaults to `df` when not given (e.g. the
            __main__ example below, or trivial languages where dt_df == full_df
            already).
        unk_counts: {"head_unk": n, "nsubj_unk": n, "both_unk": n} from
            word_order.decision_tree.fit_dt, folded into meta.json as-is. None (default)
            omits those keys -- e.g. trivial languages that never called fit_dt.
        label_distribution: {label: count} across the language's full predictor_var
            column (e.g. {"Yes": 15414, "No": 62, "unk": 6488}), BEFORE fit_dt drops
            unk rows -- so it stays meaningful even though dt_df itself never has unk
            rows when drop_unk=True. Folded into meta.json under "label_distribution".
            None (default) omits the key.
    Returns:
        dict[str, pd.DataFrame]: one DataFrame per diagnostics bucket (e.g. "correct_swaps").
    """
    if full_df is None:
        full_df = df
    if "leaf_top1_entropy" in df.columns and "leaf_decision" in df.columns:
        keep = (df["leaf_top1_entropy"] < leaf_threshold) & df["leaf_decision"]
    else:
        # Trivial languages/deprels (single-class predictor, no tree fit) have no
        # leaf_top1_entropy/leaf_decision columns and are kept in full — same
        # convention as viz_deprel.py's _agreement_row_stats.
        keep = pd.Series(True, index=df.index)
    is_yes = df[swap_feat] == "Yes"
    n_raw = int(is_yes.sum())
    swap_df = df[keep & is_yes]
    n_keep = len(swap_df)

    child_deprel = target.child_deprels[0]
    ufeat, _ = inflector.inflection_map

    items_seen = 0
    feature_distribution = Counter()
    diagnostics = {
                "correct_swaps": [],
                "same_forms": [],
                "same_features": [],
                "undefined_features": [],
                "no_inflections": [],
                "no_candidates": [],
                "multi_now_valid": [],
                "ambiguous_subjects": [],
                "extra_pairs": [],
            }

    if len(swap_df) == 0:
        # Still fall through to the save_to/verbose blocks below with empty
        # buckets, so a zero-candidate language gets a diagnostics entry
        with open("error_log.txt", "a") as f:
            f.write(f"No rows to process for {swap_feat} (keep={len(df[keep])}, swap={len(df[df[swap_feat]=='Yes'])})\n")
    else:
        columns = list(swap_df.columns)
        # Precompute, once, which columns hold each swap kind's morphological
        # features (was re-matched via regex on every row before).
        kind_feat_cols = {
            kind: [
                (col, re.match(rf"^{kind}_([A-Z][a-z]+)", col).group(1))
                for col in columns
                if re.match(rf"^{kind}_[A-Z]", col)
            ]
            for kind in swap_target
        }

        for row_tuple in tqdm(swap_df.itertuples(index=False, name=None), total=swap_df.shape[0]):
            # get nsubj and head; extract their features; swap the features; re-inflect the words;
            # create a new sentence with the re-inflected words; add the new sentence to the dataframe
            base_item = dict(zip(columns, row_tuple))
            base_item["child"] = base_item.get(f"{child_deprel}_form")

            for kind in swap_target:
                og_feats = {feat: base_item[col] for col, feat in kind_feat_cols[kind]}
                # Column missing (not NaN) means target.head_feats pinned VerbForm
                # via require=... and the now-constant column got dropped; recover it.
                if "VerbForm" not in og_feats:
                    verbform_filter = (target.head_feats or {}).get("VerbForm")
                    required = getattr(verbform_filter, "keywords", {}).get("require")
                    og_feats["VerbForm"] = required if required is not None else "Fin"
                elif pd.isna(og_feats["VerbForm"]):
                    og_feats["VerbForm"] = "Fin"
                form = base_item[f"{kind}_form"]

                swap_forms, feature_vals, swap_feats = inflector.inflect(
                    form, og_feats, return_swap_feats=True
                )

                if swap_forms!=None:
                    if len(swap_forms) > 0:
                        for swap_form in swap_forms:
                            items_seen += 1
                            item_type, item = process_item(
                                dict(base_item),
                                form,
                                swap_form,
                                og_feats,
                                ufeat,
                                feature_vals,
                                feature_distribution,
                                inflector,
                                context_inflector,
                                take_features_from=kind,
                                max_num_of_pairs=max_num_of_pairs,
                                swap_bundle=swap_feats.get(swap_form, {}),
                            )
                            diagnostics[item_type].append(item)

                            if (item_type == "correct_swaps") and (len(swap_forms) > 1):
                                multi_item = dict(item)
                                multi_item["alternatives"] = swap_forms
                                diagnostics["multi_now_valid"].append(multi_item)
                    else:
                        items_seen += 1
                        diagnostics["no_inflections"].append(dict(base_item))
                else:
                    items_seen += 1
                    diagnostics["no_candidates"].append(dict(base_item))

    coverage = f"{len(diagnostics['correct_swaps'])/max(items_seen,1)*100:.1f}"
    print(f"items_seen={items_seen} correct_swaps={len(diagnostics['correct_swaps'])} coverage={coverage}%")

    diagnostic_dfs = {item_type: pd.DataFrame(items) for item_type, items in diagnostics.items()}

    feat_match = re.match(r".*_([A-Z][a-z]+)_.*", swap_feat)
    swap_feature = feat_match.group(1) if feat_match else None

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        for item_type, item_df in diagnostic_dfs.items():
            item_df.to_parquet(os.path.join(save_to, f"{item_type}.parquet"))
        num_forms_of_interest, num_covered_um, num_covered_um_ud = _coverage_stats(
            full_df, child_deprel, inflector
        )
        meta = {
            "num_ud_candidates_raw": n_raw,
            "num_ud_candidates_keep": n_keep,
            "items_seen": items_seen,
            "num_lemma": num_lemma,
            "num_form": num_form,
            "num_forms_of_interest": num_forms_of_interest,
            "num_covered_um": num_covered_um,
            "num_covered_um_ud": num_covered_um_ud,
        }
        if unk_counts is not None:
            meta.update(unk_counts)
        if label_distribution is not None:
            meta["label_distribution"] = label_distribution

        meta["examples"] = bucket_examples_html(
            diagnostic_dfs, list(diagnostic_dfs), max_examples=max_examples,
            child_deprel=child_deprel, default_kind=swap_target[0],
            swap_feature=swap_feature,
        )
        with open(os.path.join(save_to, "meta.json"), "w") as f:
            json.dump(meta, f)

    return diagnostic_dfs


if __name__ == "__main__":
    # Example usage
    resource_dir = "../../resources"
    # Load your decision tree dataframe 
    dt_df = pd.read_parquet("../../decision_trees/svNa/svNa_nsubj/German.parquet")

    swap_feat = "head_nsubj_Number_agreement"

    inflector, skip_lang, num_lemma, num_form= load_inflector(
                    lang="German", 
                    langcode="deu",
                    unimorph_args = {
                                "filter_entries": {
                                    "upos": ["V"],
                                    },
                                    "combine_um_ud": True,
                                    "remove_multiword_forms": True,
                                    },
                    inflection_map=swap_number_subj_any,
                    resource_dir=resource_dir,)

    create_pairs(dt_df, swap_feat=swap_feat, inflector=inflector, leaf_threshold=0.12,
                 save_to="../../decision_trees/svNa/svNa_nsubj/pairs/German")
