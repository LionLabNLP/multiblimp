import os
import re
import json
import html as html_lib

import pandas as pd
from collections import Counter
from typing import *
from tqdm import tqdm

from multiblimp.ud_table import clean_lemma
from agreement_trees.extract.prediction_target import PredictionTarget, nsubj_target
from agreement_trees.extract.process_treebank import resolve_layered_head_key, slot_suffixes
from agreement_trees.utils import build_grew_link
from agreement_trees.fit.decision_tree import MISSING, UNK_LABELS
from agreement_trees.fit.entropy import DEFAULT_LEAF_MIN_ACCURACY
from agreement_trees.pipeline_utils import match_casing
from agreement_trees.pairs.attractors import add_attractors_if_stored
from agreement_trees.pairs.sentence_text import add_sentence_strings
from multiblimp.script_log import append_log


def coverage_stats(df, form_cols, inflector):
    """(# forms of interest, # covered by UM alone, # covered by UM+UD combined).

    Forms of interest: every distinct value of `form_cols` across ALL of df (not just the "yes"
    swap candidates), i.e. every word that plays one of the roles. "Covered" means the form string
    appears in the inflector's own table (UM alone) or in it and its fallback (UM+UD), whether or
    not it carries the right features for a swap.
    """
    forms_of_interest = set()
    for col in form_cols:
        if col in df.columns:
            forms_of_interest.update(df[col].dropna().unique())

    um_forms = inflector.own_forms
    um_ud_forms = inflector.unique_forms

    num_covered_um = sum(1 for f in forms_of_interest if f in um_forms)
    num_covered_um_ud = sum(1 for f in forms_of_interest if f in um_ud_forms)

    return len(forms_of_interest), num_covered_um, num_covered_um_ud


def leaf_keep(df, leaf_threshold):
    """(row_threshold, keep): leaf_threshold as one cutoff per row (a number, or a {leaf_id: cutoff}
    map from a second-chance retry) and the rows in a kept leaf (leaf_top1_acc above it and
    leaf_decision). Frames without a fitted tree (single-class predictor) are kept in full."""
    row_threshold = df["leaf_id"].map(leaf_threshold) if isinstance(leaf_threshold, dict) else leaf_threshold
    if "leaf_top1_acc" in df.columns and "leaf_decision" in df.columns:
        keep = (df["leaf_top1_acc"] > row_threshold) & df["leaf_decision"]
    else:
        keep = pd.Series(True, index=df.index)
    return row_threshold, keep


# ── Example-row rendering ─────────────────────────────────────────────────────
# bucket_examples_html() renders sample rows of every diagnostics bucket as an HTML table
# (before/after sentences with the agreement tokens highlighted, key columns, grew.fr link);
# create_pairs() stores it under "examples" in each language's meta.json.

# Spacing follows the tree's SpaceAfter=No annotations when the row or a treebank has them,
# otherwise a punctuation heuristic.

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

    Only matches scalar string values: older cached fits carry list-valued "swap_" columns
    (swap_order_candidates)."""
    for col in row.index:
        if col.startswith("swap_") and isinstance(row.get(col), str) and row[col]:
            return col, col[len("swap_"):]
    return None, None


# ── Metadata: feats + treebank link ──────────────────────────────────────────

def _feats_summary(row, prefix, highlight_feat=None) -> str:
    """The node's set features (columns "{prefix}_Feat" and bracketed "{prefix}_Feat[slot]") as a collapsed
    <details> list, one Feat=Val per line, preceded by its lemma/deprel/upos/xpos (same order as the
    tree page's feature list). highlight_feat (e.g. "Number") is bolded, bracketed spellings included."""
    pat = re.compile(rf"^{re.escape(prefix)}_([A-Z][a-zA-Z]*(?:\[[^\]]+\])?)$")
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
        if highlight_feat and feat.partition("[")[0] == highlight_feat:
            line = f'<strong class="swap-feat">{line}</strong>'
            highlight_val = val_str
        pairs.append((feat.partition("[")[0], line))

    pairs.sort(key=lambda p: p[0])

    for extra_feat, col_suffix in (("lemma", "lemma"), ("deprel", "deprel"), ("upos", "pos"), ("xpos", "xpos")):
        val = row.get(f"{prefix}_{col_suffix}")
        if pd.notna(val) and val not in (None, "None", "", "_missing", "_"):
            pairs.insert(0, (extra_feat, f"{extra_feat}={html_lib.escape(str(val))}"))

    if not pairs:
        return "&mdash;"

    items = "<br>".join(line for _, line in pairs)
    badge = (
        f' &middot; <span class="swap-feat-badge">{html_lib.escape(highlight_feat)}={highlight_val}</span>'
        if highlight_val is not None else ""
    )
    return f'<details class="feat-details"><summary>{len(pairs)} feats{badge}</summary>{items}</details>'


def _treebank_link(row) -> str:
    """grew.fr query-link for a single example row — shared with word_order/
    viz_tree.py's build_treebank_links via agreement_trees.utils.build_grew_link.
    Prefers the untouched-casing "*_form_orig" columns over "*_form"."""
    orig_cols = [c for c in row.index if c.endswith("_form_orig") and pd.notna(row.get(c))]
    form_cols = orig_cols or [
        c for c in row.index if c.endswith("_form") and pd.notna(row.get(c))
    ]
    form_values = [row[c] for c in form_cols]
    link = build_grew_link(row.get("treebank"), row.get("sent_id"), form_values)
    return link if link is not None else "&mdash;"


def _highlight_indices(row, default_kind, child_deprel=None) -> set:
    """0-based positions in `sen` of both agreement tokens (the `default_kind` role and the child), always
    both: NPA pairs reinflect either role depending on the row."""
    idx = set()
    idx_col = f"{default_kind}_idx"
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
    highlight_idx = _highlight_indices(row, default_kind, child_deprel)

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


def _swap_form_after(row, kind) -> str:
    """The reinflected word form for `kind` (process_item's "swap_{kind}" column,
    e.g. "swap_head" -> the head's form post-swap), or "&mdash;" for buckets that
    never call process_item for this kind (e.g. no_candidates/no_inflections),
    where the column is entirely absent -- same missing-data convention as
    _feats_summary's "&mdash;"."""
    val = row.get(f"swap_{kind}")
    if pd.isna(val):
        return "&mdash;"
    return _fmt_cell(val)


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


def _display_kind(col: str, default_kind: str, head_label: str | None) -> str:
    """Header text only: a leading "{default_kind}_" (or bare default_kind) shown as head_label."""
    if not head_label or head_label == default_kind:
        return col
    if col == default_kind:
        return head_label
    if col.startswith(f"{default_kind}_"):
        return head_label + col[len(default_kind):]
    return col


def _examples_table_html(item_df: pd.DataFrame, max_examples: int, treebank=None,
                          child_deprel=None, default_kind="head", swap_feature=None,
                          head_label=None) -> str:
    """HTML table of up to max_examples rows of one bucket, covering as many feature_vals categories as the budget allows.

    The child's and default_kind's "_feats"/"_feats_after" columns always show that role's own
    before/after features (NPA buckets merge both directions). default_kind's "_feats_after" and
    swapped-form columns are unconditional; the child's "_feats_after" appears only if some row
    reinflected the child. head_label relabels default_kind's headers only; lookups are unchanged.
    """
    if item_df is None or len(item_df) == 0:
        return '<p class="empty">no examples</p>'

    def _kind_of(col):
        # group columns by role
        if child_deprel and col.startswith(f"{child_deprel}_"):
            return "child"
        if default_kind and col.startswith(f"{default_kind}_"):
            return "default"
        return "other"

    preferred = [c for c in item_df.columns if c.endswith("_form") or c in ("feature_vals", "alternatives", "conflicting_features")]
    forms = [c for c in preferred if c.endswith("_form")]
    other = [c for c in preferred if c not in forms]
    kind_rank = {"child": 0, "default": 1, "other": 2}
    cols = (sorted(forms, key=lambda c: kind_rank[_kind_of(c)]) + other)[:6] or list(item_df.columns[:6])

    child_form_cols = [c for c in cols if _kind_of(c) == "child"]
    default_form_cols = [c for c in cols if _kind_of(c) == "default"]
    other_cols = [c for c in cols if _kind_of(c) == "other"]

    has_sentence = "sen" in item_df.columns
    has_treebank = "treebank" in item_df.columns and "sent_id" in item_df.columns
    sample = _diverse_sample(
        item_df, max_examples,
        by="conflicting_features" if "conflicting_features" in item_df.columns else "feature_vals")

    has_child_after = bool(child_deprel) and any(
        col.startswith(f"after_{child_deprel}_") for col in item_df.columns
    )

    child_group = child_form_cols + (
        [f"{child_deprel}_feats"] + ([f"{child_deprel}_feats_after"] if has_child_after else [])
        if child_deprel else []
    )
    default_group = default_form_cols + (
        [f"{default_kind}_feats", f"{default_kind}_form_after", f"{default_kind}_feats_after"]
        if default_kind else []
    )
    # headers: default_kind's columns are relabelled (head_label); the synthetic "_form_after" column is shown as "swapped_<label>_form"
    default_group_display = [
        f"swapped_{head_label if head_label and head_label != default_kind else default_kind}_form"
        if col == f"{default_kind}_form_after"
        else "swapped_feats_after" if col == f"{default_kind}_feats_after"
        else _display_kind(col, default_kind, head_label)
        for col in default_group
    ]

    header_cols = (
        (["before", "after"] if has_sentence else []) + child_group + default_group_display + other_cols
    )
    if has_treebank:
        header_cols.append("treebank")
    # A conflict list can be long; wrap it rather than widen the table.
    wrap = ' style="min-width:9rem;max-width:11rem;white-space:normal;overflow-wrap:break-word"'
    header = "".join(
        f"<th{wrap if c == 'conflicting_features' else ''}>{html_lib.escape(c)}</th>" for c in header_cols)
    body_rows = ""
    for _, row in sample.iterrows():
        cells = ""
        if has_sentence:
            before, after = _sentence_pair(row, treebank=treebank, child_deprel=child_deprel,
                                            default_kind=default_kind)
            cells += f'<td class="sentence">{_fmt_sentence(before)}</td>'
            cells += f'<td class="sentence">{_fmt_sentence(after)}</td>'
        cells += "".join(f"<td>{_fmt_cell(row[c])}</td>" for c in child_form_cols)
        if child_deprel:
            cells += f'<td class="feats">{_feats_summary(row, child_deprel, highlight_feat=swap_feature)}</td>'
            if has_child_after:
                cells += f'<td class="feats">{_feats_summary(row, f"after_{child_deprel}", highlight_feat=swap_feature)}</td>'
        cells += "".join(f"<td>{_fmt_cell(row[c])}</td>" for c in default_form_cols)
        if default_kind:
            cells += f'<td class="feats">{_feats_summary(row, default_kind, highlight_feat=swap_feature)}</td>'
            cells += f'<td>{_swap_form_after(row, default_kind)}</td>'
            cells += f'<td class="feats">{_feats_summary(row, f"after_{default_kind}", highlight_feat=swap_feature)}</td>'
        cells += "".join(
            f"<td{wrap}>{_fmt_cell(row[c], max_len=200)}</td>" if c == "conflicting_features"
            else f"<td>{_fmt_cell(row[c])}</td>" for c in other_cols)
        if has_treebank:
            cells += f'<td class="treebank">{_treebank_link(row)}</td>'
        body_rows += f"<tr>{cells}</tr>"

    table = f'<table class="examples"><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>'
    return f'<div class="examples-scroll">{table}</div>'


def bucket_examples_html(diagnostic_dfs: dict, item_types, max_examples: int = 5, treebank=None,
                          child_deprel=None, default_kind="head", swap_feature=None,
                          head_label=None) -> dict:
    """_examples_table_html for every requested bucket, keyed by item_type; empty buckets get a "no examples" fragment."""
    return {
        item_type: _examples_table_html(
            diagnostic_dfs.get(item_type, pd.DataFrame()), max_examples,
            treebank=treebank, child_deprel=child_deprel,
            default_kind=default_kind, swap_feature=swap_feature,
            head_label=head_label,
        )
        for item_type in item_types
    }


_SWAP_DIMS = ("Person", "Number", "Gender")
_NO_VALUE = (None, "None", "", "_missing", "nan", "UNDEFINED")
_PERSON_VALUES = {"0", "1", "2", "3", "4"}


def _conflicting_features(item, kind, ufeat, slot_aliases):
    """Features on `kind` whose original value is not among the reinflected candidate's values, except the
    swapped slot itself (`ufeat`, its slot-alias spellings, and for nsubj the plain feature). A non-person
    value in a Person field (Abkhaz Person[obj]=Rel) counts as missing."""
    base = (ufeat or "").partition("[")[0]
    exempt = {ufeat}
    if "[" in (ufeat or ""):
        exempt.update(f"{base}[{s}]" for s in slot_aliases)
    if {"subj", "nsubj"} & set(slot_aliases):
        exempt.add(base)
    conflicts = []
    prefix = f"after_{kind}_"
    for col, after in item.items():
        if not col.startswith(prefix) or pd.isna(after) or after in _NO_VALUE:
            continue
        feat = col[len(prefix):]
        if feat in exempt:
            continue
        orig = item.get(f"{kind}_{feat}")
        if pd.isna(orig) or orig in _NO_VALUE:
            continue
        if feat.partition("[")[0] == "Person" and str(orig) not in _PERSON_VALUES:
            continue
        # a multi-valued original ("Masc,Neut") is compatible with a candidate carrying any of its values
        if not set(str(orig).split(",")) & set(str(after).split("/")):
            conflicts.append(f"{feat}:{orig}->{after}")
    return conflicts


def _added_slots(item, kind, slot_aliases):
    """["<slots>"] if the candidate marks more distinct bracketed argument slots (counted by bracket suffix
    across Person/Number/Gender) than the original, e.g. an extra [io] slot (`da` -> `natzaion`). Counting
    slots rather than names makes spellings ([io] vs [dat]) irrelevant. The swapped slot's aliases and the
    subject slot are not counted. Sound where the treebank annotates every slot a form carries (checked for Basque)."""
    def slots(prefix):
        out = set()
        for col, val in item.items():
            if not col.startswith(prefix) or "[" not in col or pd.isna(val) or val in _NO_VALUE:
                continue
            if col[len(prefix):].partition("[")[0] not in _SWAP_DIMS:
                continue
            suffix = col[col.index("[") + 1:-1]
            if suffix not in slot_aliases and suffix not in ("subj", "nsubj"):
                out.add(suffix)
        return out
    before, after = slots(f"{kind}_"), slots(f"after_{kind}_")
    if len(after) <= len(before):
        return []
    return ['/'.join(sorted(after - before)) or '?']


def _added_dims(item, kind):
    """Dimensions beyond Person/Number (Polite, Gender, ...) that the candidate marks on an argument bracket
    while the original lacks that dimension under the same slot (plain or bracketed), e.g. Basque allocutive
    forms (`daki` -> `dakik`). Slot-aware: the subject's plain Gender does not exempt a new Gender[obj]."""
    def dim_slots(prefix, bracketed_only):
        out = set()
        for col, val in item.items():
            if not col.startswith(prefix) or pd.isna(val) or val in _NO_VALUE:
                continue
            feat = col[len(prefix):]
            if not re.fullmatch(r"[A-Z][A-Za-z]*(\[[^\]]+\])?", feat):
                continue
            base, _, suffix = feat.partition("[")
            suffix = suffix[:-1] if suffix else ""
            if bracketed_only and (not suffix or suffix in ("subj", "nsubj")):
                continue  # the subject's reading is folded into the plain feature
            out.add((base, suffix))
        return out
    before = dim_slots(f"{kind}_", False)
    after = dim_slots(f"after_{kind}_", True)
    return sorted(
        f"{d}[{s}]" if s else d
        for d, s in after - before
        if d not in ("Person", "Number")
    )


def _misaligned_slot(item, kind, ufeat, resolved_key):
    """True if the swap was generated under a different bracket than the slot this row resolved to
    (resolve_layered_head_key) and the candidate leaves that slot unchanged, e.g. an Abs subject on a verb
    whose ergative agreement was swapped. A resolved slot the candidate does not report at all (Georgian
    [acc] vs [obj]) is not evidence of misalignment."""
    if not resolved_key or not ufeat or ufeat == resolved_key:
        return False
    orig = item.get(f"{kind}_{resolved_key}")
    after = item.get(f"after_{kind}_{resolved_key}")
    if pd.isna(orig) or orig in _NO_VALUE or pd.isna(after) or after in _NO_VALUE:
        return False
    return str(orig) in str(after).split("/")


def bucket_frame(swap_df: pd.DataFrame, entries: list) -> pd.DataFrame:
    """One bucket's DataFrame: swap_df's full-width rows for each (position, added columns) entry, plus the
    added columns in order of first appearance (NaN where an entry sets none); they overwrite existing columns."""
    if not entries:
        return pd.DataFrame()
    frame = swap_df.iloc[[pos for pos, _ in entries]].reset_index(drop=True)
    added = {}
    for i, (_, extra) in enumerate(entries):
        for col, val in extra.items():
            added.setdefault(col, {})[i] = val
    if not added:
        return frame
    new_cols = {}
    for col, by_row in added.items():
        if col in frame.columns:
            values = frame[col].astype(object)
            for i, val in by_row.items():
                values[i] = val
            frame[col] = values
        else:
            new_cols[col] = pd.Series(by_row, index=range(len(frame)))
    if new_cols:
        frame = pd.concat([frame, pd.DataFrame(new_cols)], axis=1)
    return frame


def ambiguous_readings(correct: set, readings) -> bool:
    """Whether the accepted swap forms `correct` come from more than one reading of the
    token. Readings that share a swap form are one reading (variants of one cell
    reached through different source rows)."""
    groups = []
    for reading in readings:
        merged = set(reading)
        for group in [g for g in groups if g & merged]:
            groups.remove(group)
            merged |= group
        groups.append(merged)
    return sum(1 for g in groups if g & correct) > 1


def process_item(
        item,
        form,
        swap_form,
        form_features,
        ufeat,
        feature_vals,
        feature_distribution,
        inflector,
        context_features,
        take_features_from,
        max_num_of_pairs=None,
        swap_bundle=None,
        slot_aliases=(),
        resolved_key=None,
    ) -> Tuple[str, Dict[str, str]]:
        """Classify one candidate swap form of a row. Returns (bucket, item): same_forms, same_features,
        undefined_features (no features for the swapped form), ambiguous_subjects (the fixed role's own
        form could carry the swapped value: context_features), conflicting_features, correct_swaps or extra_pairs.
        `item` carries the row's own role columns; the swap_*, after_* and feature_vals columns are added."""
        # swap_form stays the lexicon-cased string for lookups; only the displayed form is recased
        # (relative to "_form_orig", since "_form" is lowercased upstream)
        orig_form = item.get(f"{take_features_from}_form_orig")
        if not isinstance(orig_form, str):
            orig_form = form
        item[f"swap_{take_features_from}"] = match_casing(orig_form, swap_form)
        item["swap_ufeat"] = ufeat

        if swap_form == form:
            return "same_forms", item
        else:
            swap_feature_vals = inflector.get_form_features(
                swap_form, form_features, ufeat, slot_aliases=slot_aliases
            )
            feature_key = (
                "|".join(sorted(feature_vals))
                + " -> "
                + "|".join(sorted(swap_feature_vals))
            )

            # the reinflected form's other features, from the reinflection source
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

            if len(feature_vals & swap_feature_vals) == 0:
                if (len(swap_feature_vals) > 0) and add_item:
                    # the fixed role's own form could also carry the swapped value
                    if len(context_features & swap_feature_vals) > 0:
                        return "ambiguous_subjects", item
                    else:
                        # not a minimal pair if the candidate opposes the original on another feature
                        # (a missing value on either side is no conflict)
                        conflicts = _conflicting_features(item, take_features_from, ufeat, slot_aliases)
                        conflicts += [f"added_slot:{d}" for d in _added_slots(item, take_features_from, slot_aliases)]
                        conflicts += [f"added_dim:{d}" for d in _added_dims(item, take_features_from)]
                        if _misaligned_slot(item, take_features_from, ufeat, resolved_key):
                            conflicts.append(f"misaligned_slot:{resolved_key}->{ufeat}")
                        if conflicts:
                            item["conflicting_features"] = ";".join(conflicts)
                            return "conflicting_features", item
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
                wrong_item[f"swap_{take_features_from}"] = match_casing(orig_form, swap_form)
                wrong_item["feature_vals"] = feature_key

                return "same_features", wrong_item
            else:
                wrong_item = dict(item)
                wrong_item[f"swap_{take_features_from}"] = match_casing(orig_form, swap_form)
                wrong_item["feature_vals"] = feature_key

                return "undefined_features", wrong_item


class MissingColumnError(KeyError):
    """The frame lacks a column the pair loop reads."""


def create_pairs(df, swap_feat, inflector, target: PredictionTarget = nsubj_target,
                  swap_target=["head",], context_inflector=None, max_num_of_pairs=None,
                  leaf_threshold=DEFAULT_LEAF_MIN_ACCURACY, save_to=None,
                  max_examples=5, num_lemma=None, num_form=None, full_df=None,
                  unk_counts=None, label_distribution=None, head_label=None,
                  extra_meta=None, second_chance_threshold=None):
    """Re-inflect each kept "yes" row of the decision-tree frame into a minimal pair and sort the outcomes into
    diagnostics buckets.

    Args:
        df: decision-tree frame (dt_df).
        leaf_threshold: smoothed-accuracy floor for keeping a leaf's rows, computed here from
            leaf_top1_acc/leaf_decision. One cutoff, or a {leaf_id: cutoff} map (second chance).
        save_to: directory for "<bucket>.parquet" files and "meta.json" (scalar stats, unk counts, label
            distribution, up to max_examples sample rows per bucket under "examples").
        num_lemma, num_form: lexicon size of this language.
        full_df: the language's complete per-target frame, used for the forms-of-interest coverage (`df`
            is often only fit_dt's train split). Defaults to `df`.
        unk_counts: {"head_unk", "nsubj_unk", "both_unk"} from fit_dt, stored in meta.json.
        label_distribution: {label: count} over the full predictor column, before unk rows are dropped.
        head_label: display label for the head role's columns in the example tables (header text only).
        second_chance_threshold: the strict leaf_threshold; pass it on the lax retry call only. Output
            frames then get "second_chance" (row only kept because of the retry) and
            "second_chance_lax_threshold" columns.
    Returns:
        dict[str, pd.DataFrame]: one frame per diagnostics bucket.
    """
    if full_df is None:
        full_df = df
    row_threshold, keep = leaf_keep(df, leaf_threshold)
    is_yes = df[swap_feat] == "yes"
    n_raw = int(is_yes.sum())
    swap_df = df[keep & is_yes]
    n_keep = len(swap_df)
    n_keep_second_chance = 0
    if second_chance_threshold is not None:
        # rows above the strict bar are in swap_df only because the retry relaxed it
        swap_df = swap_df.copy()
        swap_df["second_chance"] = swap_df["leaf_top1_acc"] <= second_chance_threshold
        swap_df["second_chance_lax_threshold"] = (
            row_threshold.loc[swap_df.index] if isinstance(row_threshold, pd.Series) else row_threshold
        )
        # row-level count (the bucket item counts can exceed it: one row can yield several items)
        n_keep_second_chance = int(swap_df["second_chance"].sum())

    child_deprel = target.child_deprels[0]
    ufeat, _ = inflector.inflection_map

    # rows whose agreement label is unk never enter the reinflection loop; split by which side is missing,
    # as fit_dt's "dropped before fitting" counts are
    is_unk = full_df[swap_feat].isin(UNK_LABELS)
    feat_match_unk = re.match(r".*_([A-Z][a-z]+)_.*", swap_feat)
    head_col = f"head_{feat_match_unk.group(1)}" if feat_match_unk else None
    child_col = f"{child_deprel}_{feat_match_unk.group(1)}" if feat_match_unk else None
    if head_col in full_df.columns and child_col in full_df.columns:
        head_missing = full_df[head_col].isna()
        child_missing = full_df[child_col].isna()
    else:
        head_missing = child_missing = pd.Series(False, index=full_df.index)
    both_unk_df = full_df[is_unk & head_missing & child_missing]
    head_unk_df = full_df[is_unk & head_missing & ~child_missing]
    nsubj_unk_df = full_df[is_unk & ~head_missing & child_missing]

    items_seen = 0
    feature_distribution = Counter()
    diagnostics = {
                "head_unk": [],
                "nsubj_unk": [],
                "both_unk": [],
                "correct_swaps": [],
                "same_forms": [],
                "same_features": [],
                "undefined_features": [],
                "conflicting_features": [],
                "no_inflections": [],
                "no_candidates": [],
                "ambiguous_readings": [],
                "multi_now_valid": [],
                "ambiguous_subjects": [],
                "extra_pairs": [],
            }

    if len(swap_df) == 0:
        # an empty language still gets its (empty) buckets and diagnostics entry below
        append_log("create_pairs", f"no rows to process\t{save_to}\t{swap_feat}\tkeep={len(df[keep])}\tswap={len(df[df[swap_feat]=='yes'])}")
    else:
        columns = list(swap_df.columns)
        # columns holding each swap kind's features, plain (head_Number) and bracketed (head_Number[obj])
        kind_feat_cols = {
            kind: [
                (col, m.group(1))
                for col in columns
                if (m := re.match(rf"^{kind}_([A-Z][a-zA-Z]*(?:\[[^\]]+\])?)$", col))
            ]
            for kind in swap_target
        }

        target_feature = feat_match_unk.group(1) if feat_match_unk else None

        # the loop reads only the roles' form/feature columns and the child's form/Case; the full-width
        # bucket frames are built from swap_df afterwards (bucket_frame)
        feature_col = re.compile(r"[A-Z][A-Za-z]*(\[[^\]]+\])?")
        loop_cols = [
            col for col in columns
            if col in (f"{child_deprel}_form", f"{child_deprel}_Case", f"{child_deprel}_pos")
            or any(
                col in (f"{kind}_form", f"{kind}_form_orig", f"{kind}_lemma")
                or (col.startswith(f"{kind}_") and feature_col.fullmatch(col[len(kind) + 1:]))
                for kind in swap_target
            )
        ]
        loop_col_set = set(loop_cols)

        def added_cols(item):
            return {k: v for k, v in item.items() if k not in loop_col_set}

        missing = [f"{kind}_form" for kind in swap_target if f"{kind}_form" not in columns]
        if missing:
            raise MissingColumnError(missing)

        loop_rows = enumerate(swap_df[loop_cols].itertuples(index=False, name=None))
        for pos, row_tuple in tqdm(loop_rows, total=swap_df.shape[0]):
            # extract the features, swap them, re-inflect the word
            base_item = dict(zip(loop_cols, row_tuple))
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
                og_feats = {feat: val for feat, val in og_feats.items() if val != MISSING}   # unstated in dt_df
                form = base_item[f"{kind}_form"]
                lemma = clean_lemma(base_item.get(f"{kind}_lemma"), form)

                slot_aliases = (
                    slot_suffixes(child_deprel, base_item.get(f"{child_deprel}_Case"))
                    if kind == "head" else ()
                )
                swap_forms, feature_vals, swap_feats, readings = inflector.inflect(
                    form, og_feats, return_swap_feats=True, return_readings=True,
                    slot_aliases=slot_aliases, lemma=lemma,
                )
                row_ufeat = ufeat

                # ufeat (inflector.inflection_map[0]) is resolved once per language, which is wrong for
                # ergative-split languages (Basque: Number[erg] vs Number[abs] depends on this clause's
                # transitivity) and for lexicons and treebanks that bracket the same argument differently
                # (Georgian [acc]/[nom] vs [obj]/[subj]). So the row's own key (resolve_layered_head_key,
                # as used for its agreement label) is retried whenever it differs from the default, and
                # replaces the default result only if the retry finds forms; forcing it unconditionally
                # lost pairs where the lexicon lacks that column.
                layered_key = None
                if kind == "head" and target_feature:
                    layered_key = resolve_layered_head_key(
                        base_item, base_item, target_feature, child_deprel
                    )
                    if layered_key is not None and (not swap_forms or layered_key != ufeat):
                        retry_forms, retry_vals, retry_swap_feats, retry_readings = inflector.inflect(
                            form, og_feats, return_swap_feats=True, return_readings=True,
                            swap_ufeat_override=layered_key, slot_aliases=slot_aliases, lemma=lemma,
                        )
                        if retry_forms:
                            swap_forms, feature_vals, swap_feats, readings = (
                                retry_forms, retry_vals, retry_swap_feats, retry_readings,
                            )
                            row_ufeat = layered_key

                if swap_forms!=None:
                    if len(swap_forms) > 0:
                        results = []
                        context_features = set()
                        child_form = base_item["child"]
                        if context_inflector is not None and isinstance(child_form, str):
                            child_case = base_item.get(f"{child_deprel}_Case")
                            child_pos = base_item.get(f"{child_deprel}_pos")
                            child_feats = {"Case": child_case} if isinstance(child_case, str) and child_case != MISSING else {}
                            context_kwargs = dict(ufeat=row_ufeat.partition("[")[0], only_try_ud_if_no_um=True)
                            # the child's own part of speech first: lowercased forms merge pronouns with
                            # homographs (es / Es), and the treebank's tag may be one the lexicon lacks
                            context_features = (
                                context_inflector.get_form_features(child_form, child_feats, upos=child_pos, **context_kwargs)
                                if isinstance(child_pos, str) else set())
                            if not context_features:
                                context_features = context_inflector.get_form_features(child_form, child_feats, **context_kwargs)
                        for swap_form in swap_forms:
                            items_seen += 1
                            item_type, item = process_item(
                                dict(base_item),
                                form,
                                swap_form,
                                og_feats,
                                row_ufeat,
                                feature_vals,
                                feature_distribution,
                                inflector,
                                context_features,
                                take_features_from=kind,
                                max_num_of_pairs=max_num_of_pairs,
                                swap_bundle=swap_feats.get(swap_form, {}),
                                slot_aliases=slot_aliases,
                                resolved_key=layered_key,
                            )
                            results.append((item_type, swap_form, item))

                        # accepted swap forms from different readings: none of them is kept
                        correct = {sf for t, sf, _ in results if t == "correct_swaps"}
                        ambiguous = ambiguous_readings(correct, readings)
                        for item_type, swap_form, item in results:
                            if ambiguous and item_type == "correct_swaps":
                                item["alternatives"] = sorted(correct)
                                item_type = "ambiguous_readings"
                            diagnostics[item_type].append((pos, added_cols(item)))

                            if (item_type == "correct_swaps") and (len(swap_forms) > 1):
                                multi_item = added_cols(item)
                                multi_item["alternatives"] = swap_forms
                                diagnostics["multi_now_valid"].append((pos, multi_item))
                    else:
                        items_seen += 1
                        diagnostics["no_inflections"].append((pos, added_cols(base_item)))
                else:
                    items_seen += 1
                    diagnostics["no_candidates"].append((pos, added_cols(base_item)))

    coverage = f"{len(diagnostics['correct_swaps'])/max(items_seen,1)*100:.1f}"
    print(f"items_seen={items_seen} correct_swaps={len(diagnostics['correct_swaps'])} coverage={coverage}%")

    diagnostic_dfs = {item_type: bucket_frame(swap_df, items) for item_type, items in diagnostics.items()}
    if len(diagnostic_dfs["correct_swaps"]):
        diagnostic_dfs["correct_swaps"] = add_sentence_strings(diagnostic_dfs["correct_swaps"])
        diagnostic_dfs["correct_swaps"] = add_attractors_if_stored(
            diagnostic_dfs["correct_swaps"], f"{child_deprel}_attractor_slots", child_deprel)
    # unk rows are never modified: keep their dtypes, skip the records round trip
    for item_type, unk_df in (("head_unk", head_unk_df), ("nsubj_unk", nsubj_unk_df),
                              ("both_unk", both_unk_df)):
        if len(unk_df):
            diagnostic_dfs[item_type] = unk_df.reset_index(drop=True)

    feat_match = re.match(r".*_([A-Z][a-z]+)_.*", swap_feat)
    swap_feature = feat_match.group(1) if feat_match else None

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        for item_type, item_df in diagnostic_dfs.items():
            item_df.to_parquet(os.path.join(save_to, f"{item_type}.parquet"))
        num_forms_of_interest, num_covered_um, num_covered_um_ud = coverage_stats(
            full_df, ["head_form", f"{child_deprel}_form"], inflector
        )
        meta = {
            "leaf_min_acc": leaf_threshold,
            "num_ud_candidates_raw": n_raw,
            "num_ud_candidates_keep": n_keep,
            "num_ud_candidates_keep_second_chance": n_keep_second_chance,
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
        if extra_meta:
            meta.update(extra_meta)

        meta["examples"] = bucket_examples_html(
            diagnostic_dfs, list(diagnostic_dfs), max_examples=max_examples,
            child_deprel=child_deprel, default_kind=swap_target[0],
            swap_feature=swap_feature, head_label=head_label,
        )
        with open(os.path.join(save_to, "meta.json"), "w") as f:
            json.dump(meta, f)

    return diagnostic_dfs

