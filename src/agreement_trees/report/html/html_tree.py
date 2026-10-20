import html as html_lib
import json
import os
import re
import numpy as np
import pandas as pd

from agreement_trees.pipeline_utils import match_casing
from agreement_trees.utils import build_grew_link
from agreement_trees.extract.process_treebank import resolve_layered_head_key, slot_suffixes


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json(obj):
    """JSON for embedding in a <script>: "</" and "<!--" are escaped so a form or sentence containing
    "</script>" cannot end the block early."""
    return json.dumps(obj, cls=_NumpyEncoder).replace("</", "<\\/").replace("<!--", "<\\u0021--")


# ── v2 tree page ──────────────────────────────────────────────────────────
# v2 integration (see the integration plan): tree + node-click samples
# browsing, ported from the "Minimal-Pair Attention Trace" mockup's already-
# verified HTML/CSS/JS, now driven by real per-language data instead of two
# hand-picked demo trees. Step 2 adds the generated-pairs section itself --
# correct_swaps_df=None (its default) emits an empty PAIRS, which the ported
# JS still renders gracefully (a real "N leaves kept, 0 shown" header, no
# pair cards) for callers/languages that don't have one.

_V2_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "tree_v2_template.html")

# The mockup's CSS only themes this fixed agreement-classifier vocabulary
# (--yes/--no/--unk custom properties); anything else falls back to its own
# real hex color from the v1 palette rather than a var(...) reference that
# doesn't exist. Covers every SVA/NPA condition this generator targets.
_V2_SEMANTIC_COLOR_VARS = {"yes": "var(--yes)", "no": "var(--no)", "unk": "var(--unk)"}

_TREEBANK_LINK_RE = re.compile(r"href='([^']*)'.*?>([^<]*)</a>")
_FORM_KEY_RE = re.compile(r"^(.+)_form$")


def _v2_class_color(cls, hex_color):
    return _V2_SEMANTIC_COLOR_VARS.get(str(cls).lower(), hex_color)


def _v2_parse_treebank_link(html_str):
    """(href, link text) from the ready-made <a href='...'>NAME</a> string build_treebank_links puts in the
    sample rows; the template wants them separately."""
    m = _TREEBANK_LINK_RE.search(html_str or "")
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _v2_strong_to_b(s):
    # the template's featEntry() looks for <b>, the samples mark the decisive feature with <strong>
    return s.replace("<strong>", "<b>").replace("</strong>", "</b>")


def _v2_detect_role_prefixes(node_samples, head_label):
    """The two role prefixes (nsubj/head, DET/HEAD) from the first sample row, which carries exactly two
    "{prefix}_form" columns; nsubj/head if there are no samples."""
    for node_map in node_samples.values():
        for bucket in node_map.values():
            for row in bucket.get("rows", []):
                prefixes = [m.group(1) for m in (_FORM_KEY_RE.match(k) for k in row) if m]
                if len(prefixes) >= 2:
                    head_matches = [p for p in prefixes if p.lower() == str(head_label).lower()]
                    role_b = head_matches[0] if head_matches else prefixes[0]
                    role_a = next((p for p in prefixes if p != role_b), prefixes[-1])
                    return role_a, role_b
    return "nsubj", "head"


def _v2_node_samples(node_id, node_samples, role_a, role_b):
    """One node's sample rows as a flat list, each carrying its label, from get_samples's
    {label: {node_id: {rows, count, total_count}}}."""
    rows_out = []
    for label, node_map in node_samples.items():
        bucket = node_map.get(node_id, node_map.get(str(node_id)))
        if not bucket:
            continue
        for r in bucket.get("rows", []):
            href, name = _v2_parse_treebank_link(r.get("treebank_link", ""))
            rows_out.append({
                "sentence": r.get("sen_str", ""),
                "verb": r.get(f"{role_b}_form", ""),
                "verbFeats": [_v2_strong_to_b(v) for v in r.get(f"{role_b}_features", [])],
                "nsubj": r.get(f"{role_a}_form", ""),
                "nsubjFeats": [_v2_strong_to_b(v) for v in r.get(f"{role_a}_features", [])],
                "label": label,
                "route": r.get("route", {}),
                "treebankLink": href,
                "treebankName": name,
            })
    return rows_out


def _v2_layout(node_data):
    """Map compute_tree_layout's relative coordinates (leaves at integer x, y = -depth) onto the template's
    pixel canvas (y downward), sized from the actual tree."""
    NODE_SPACING_X, NODE_SPACING_Y = 190, 170
    # left margin only: the floating info panel (up to 266px wide) must not cover the leftmost nodes, whose boxes
    # extend 84px either side of x
    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_Y = 380, 100, 60
    max_leaf_x = max((nd["x"] for nd in node_data.values()), default=0)
    max_depth = max((-nd["y"] for nd in node_data.values()), default=0)
    canvas_w = int(max_leaf_x * NODE_SPACING_X + MARGIN_LEFT + MARGIN_RIGHT) or 900
    canvas_h = int(max_depth * NODE_SPACING_Y + 2 * MARGIN_Y) or 500
    screen = {
        i: (MARGIN_LEFT + nd["x"] * NODE_SPACING_X, MARGIN_Y + (-nd["y"]) * NODE_SPACING_Y)
        for i, nd in node_data.items()
    }
    return screen, canvas_w, canvas_h


def _v2_leaf_qualifying_classes(dist):
    # the majority class plus any within 50% of it (same rule as the leaf label in _finalize_tree_html)
    if not dist:
        return []
    sorted_d = sorted(dist, key=lambda d: d["cnt"], reverse=True)
    top_cnt = sorted_d[0]["cnt"]
    quals = [sorted_d[0]["cls"]]
    for d in sorted_d[1:]:
        if d["cnt"] > 0 and d["cnt"] >= 0.5 * top_cnt:
            quals.append(d["cls"])
    return quals


def _v2_raw_pair_role_prefixes(correct_swaps_df):
    """_v2_detect_role_prefixes for correct_swaps' raw columns: the head prefix is literally "head" or "HEAD"."""
    prefixes = [
        m.group(1) for m in (_FORM_KEY_RE.match(c) for c in correct_swaps_df.columns) if m
    ]
    head_matches = [p for p in prefixes if p.lower() == "head"]
    raw_role_b = head_matches[0] if head_matches else (prefixes[0] if prefixes else "head")
    raw_role_a = next((p for p in prefixes if p != raw_role_b), prefixes[-1] if len(prefixes) > 1 else "nsubj")
    return raw_role_a, raw_role_b


def _v2_pair_swap_role(row, raw_role_a, raw_role_b):
    """'A' or 'B': which role this row's reinflection swapped, from whichever swap_{role} column is set (a
    merged-direction bucket can swap either role row by row)."""
    for role, letter in ((raw_role_b, "B"), (raw_role_a, "A")):
        val = row.get(f"swap_{role}")
        if isinstance(val, str) and val:
            return letter
    return None


# Dimensions written as layered features (Number[obj], ...); same as viz_tree._LAYERED_DISPLAY_FEATS (duplicated:
# viz_tree imports this module). Case is never bracketed on the verb.
_LAYERED_DISPLAY_FEATS = ("Number", "Person", "Gender")


def _v2_pair_feat_list(row, prefix, decisive_feat, deprel=None):
    """"Feat=val" strings for one role of a correct_swaps row: lemma/deprel/upos first, then the morphological
    feats sorted (as create_pairs._feats_summary and viz_tree._build_feat_columns); the decisive feature in <b>."""
    pat = re.compile(rf"^{re.escape(prefix)}_([A-Z][a-zA-Z]*(?:\[[^\]]+\])?)$")
    # for obj/iobj targets the plain head_{feat} is the subject's folded value (see extract_instances): suppress it
    # and show the resolved bracketed keys
    resolve_bracket_feats = bool(deprel) and deprel != "nsubj" and prefix.lower() == "head"
    # the bracketed key the decisive feature resolves to (e.g. Number[abs]); only it is bolded, not a sibling
    # bracket such as the subject's Number[erg]
    decisive_layered_key = None
    if resolve_bracket_feats and decisive_feat in _LAYERED_DISPLAY_FEATS:
        decisive_layered_key = resolve_layered_head_key(row, row, decisive_feat, deprel)

    morph = []
    for col, val in row.items():
        m = pat.match(col)
        if not m or pd.isna(val) or val in (None, "None", "", "_missing", "nan"):
            continue
        feat = m.group(1)
        # feat may be bracketed; only the plain fold-in is suppressed, every real bracketed column is shown (a
        # transitive Basque verb has Number[abs] and Number[erg])
        base_feat = feat.partition("[")[0]
        if resolve_bracket_feats and base_feat in _LAYERED_DISPLAY_FEATS and "[" not in feat:
            continue
        entry = f"{feat}={val}"
        is_decisive = (decisive_feat and feat == decisive_feat) or (
            decisive_layered_key and feat == decisive_layered_key
        )
        if is_decisive:
            entry = f"<b>{entry}</b>"
        morph.append((base_feat, entry))

    morph.sort(key=lambda p: p[0])

    extra = []
    for extra_feat, suffix in (("lemma", "lemma"), ("deprel", "deprel"), ("upos", "pos"), ("xpos", "xpos")):
        val = row.get(f"{prefix}_{suffix}")
        if pd.notna(val) and val not in (None, "None", "", "_missing", "_"):
            extra.append(f"{extra_feat}={val}")
    # a dependent's deprel is its prefix and constant, so the data build dropped it
    if prefix.islower() and prefix != "head" and not any(e.startswith("deprel=") for e in extra):
        extra.append(f"deprel={prefix}")
    return extra + [e for _, e in morph]


def _v2_sentence_html(sen, highlight_idx):
    return " ".join(
        f"<strong>{tok}</strong>" if i in highlight_idx else str(tok)
        for i, tok in enumerate(sen)
    )


def _v2_pair_treebank_link(row):
    """Prefers the untouched-casing "*_form_orig" columns over "*_form"."""
    orig_values = [v for c, v in row.items() if c.endswith("_form_orig") and pd.notna(v)]
    form_values = orig_values or [
        v for c, v in row.items() if c.endswith("_form") and pd.notna(v)
    ]
    link = build_grew_link(row.get("treebank"), row.get("sent_id"), form_values)
    return _v2_parse_treebank_link(link or "")


def _has_feat_value(val):
    return pd.notna(val) and val not in (None, "None", "", "_missing", "nan")


def _v2_pair_from_row(row, raw_role_a, raw_role_b, decisive_feat):
    """One T.PAIRS entry from a correct_swaps row, or None without a sentence, a detectable swap role or a token index."""
    sen = row.get("sen")
    if sen is None:
        return None
    sen = list(sen)

    swap_role = _v2_pair_swap_role(row, raw_role_a, raw_role_b)
    if swap_role is None:
        return None
    swap_prefix = raw_role_b if swap_role == "B" else raw_role_a
    other_prefix = raw_role_a if swap_role == "B" else raw_role_b

    idx = row.get(f"{swap_prefix}_idx")
    if pd.isna(idx):
        return None
    other_idx = row.get(f"{other_prefix}_idx")

    # CoNLL-U ids are 1-indexed
    highlight_idx = {int(idx) - 1}
    if pd.notna(other_idx):
        highlight_idx.add(int(other_idx) - 1)

    swap_form = row.get(f"swap_{swap_prefix}")
    swapped_sen = list(sen)
    swapped_sen[int(idx) - 1] = match_casing(sen[int(idx) - 1], swap_form)

    href, name = _v2_pair_treebank_link(row)
    # the swapped value is often only under a bracketed after_ column (after_head_Number[abs]), not the plain name,
    # and the head_{feat}[...] columns can be absent from this trimmed frame, so scan the after_ columns with
    # resolve_layered_head_key's alias order. after_feat_key records the key it was found under, so the
    # template does not relabel a bracketed reading as plain.
    after_feat_key = decisive_feat
    after_value = row.get(f"after_{swap_prefix}_{decisive_feat}") if decisive_feat else None
    sibling_keys = set()
    if decisive_feat:
        # every spelling of this slot on the verb (the lexicon may use another one than the original token shows)
        suffixes = slot_suffixes(raw_role_a, row.get(f"{raw_role_a}_Case") if pd.notna(row.get(f"{raw_role_a}_Case")) else None)
        slot_keys = [f"{decisive_feat}[{s}]" for s in suffixes]
        # the lexicon column the swap used can be a third spelling of the slot (Georgian [acc] for [obj]/dative)
        swap_key = row.get("swap_ufeat")
        if isinstance(swap_key, str) and swap_key.startswith(f"{decisive_feat}[") and swap_key not in slot_keys:
            slot_keys.insert(0, swap_key)
        if pd.isna(after_value) and swap_prefix.lower() == "head":
            for key in slot_keys:
                candidate = row.get(f"after_{swap_prefix}_{key}")
                if pd.notna(candidate):
                    after_value, after_feat_key = candidate, key
                    break
            if pd.notna(after_value):
                # the swapped entry must replace the spelling the original row displays (nsubj may show the plain
                # name: merge_subj_layered_feats)
                shown_keys = slot_keys + ([decisive_feat] if raw_role_a == "nsubj" else [])
                after_feat_key = next(
                    (k for k in shown_keys if _has_feat_value(row.get(f"{swap_prefix}_{k}"))),
                    after_feat_key,
                )
        sibling_keys = set(slot_keys)
    sibling_keys.discard(after_feat_key)

    # bonus after-only features: a bracketed dimension with a real after_ value but no original value on this
    # token (a Basque ditransitive auxiliary fuses in an indirect-object slot that a plain transitive form lacks),
    # which applyAfterFeat would otherwise drop: it only patches existing entries. Head role, Number/Person/Gender.
    bonus_after_feats = []
    if swap_prefix.lower() == "head":
        seen_bonus_keys = {after_feat_key, *sibling_keys}
        child_case = row.get(f"{raw_role_a}_Case")
        bonus_suffixes = slot_suffixes(raw_role_a, child_case if pd.notna(child_case) else None)
        for col in row.keys():
            m = re.match(rf"^after_{re.escape(swap_prefix)}_([A-Z][a-zA-Z]*\[[^\]]+\])$", col)
            if not m:
                continue
            key = m.group(1)
            if key in seen_bonus_keys or key.split("[")[0] not in _LAYERED_DISPLAY_FEATS:
                continue
            key_base, _, key_suffix = key.partition("[")
            in_slot = key_suffix.rstrip("]") in bonus_suffixes
            same_slot = [key] + (
                [f"{key_base}[{s}]" for s in bonus_suffixes] + ([key_base] if raw_role_a == "nsubj" else [])
                if in_slot else []
            )
            seen_bonus_keys.update(same_slot)
            after_val = row.get(col)
            if pd.isna(after_val) or after_val in ("None", "", "_missing", "nan"):
                continue  # no real swap value
            if any(_has_feat_value(row.get(f"{swap_prefix}_{k}")) for k in same_slot):
                continue  # this token already had a real value here pre-swap (under any slot spelling)
            bonus_after_feats.append(f"{key}={after_val}")

    return {
        "origSentence": _v2_sentence_html(sen, highlight_idx),
        "swapSentence": _v2_sentence_html(swapped_sen, highlight_idx),
        "nsubj": row.get(f"{raw_role_a}_form", ""),
        "verb": row.get(f"{raw_role_b}_form", ""),
        "nsubjFeats": _v2_pair_feat_list(row, raw_role_a, decisive_feat),
        "verbFeats": _v2_pair_feat_list(row, raw_role_b, decisive_feat, deprel=raw_role_a),
        "ungrammatical": swap_form,
        "swapRole": swap_role,
        "afterValue": None if pd.isna(after_value) else str(after_value),
        "afterFeatKey": None if pd.isna(after_value) else after_feat_key,
        "siblingKeys": sorted(sibling_keys),
        "bonusAfterFeats": bonus_after_feats,
        "treebankLink": href,
        "treebankName": name,
    }


def _v2_leaf_role_series(valid, raw_role_b):
    """'A'/'B' per row of `valid` -- which reinflection direction that
    correct_swaps row belongs to (see swap_roles_for_target_col), detected
    the same way _v2_pair_swap_role does per-pair, just vectorized here
    since this runs over every row rather than one already-selected pair.
    All-'A' when raw_role_b's own swap_{role} column is absent or never
    set (a single-direction bucket -- SVA/subj_aux, or NPA's HEAD pairs)."""
    swap_col_b = f"swap_{raw_role_b}"
    is_b = (valid[swap_col_b].notna() if swap_col_b in valid.columns
            else pd.Series(False, index=valid.index))
    return is_b.map({True: "B", False: "A"})


def _v2_sample_per_leaf(valid, role_series, max_per_bucket):
    """Like valid.groupby("leaf_id").head(max_per_bucket), but when a
    leaf's correct_swaps rows mix both reinflection directions (e.g. NPA's
    merged non-head pairs, DET-ADJ style -- see
    agreement_trees.npa.agreement.swap_roles_for_target_col), each direction gets its own
    max_per_bucket rows rather than sharing one cap -- so a two-direction
    leaf can surface up to 2*max_per_bucket rows, and the "Determiner
    reinflected"/"Adjective reinflected" groups on the tree page are never
    starved by how many rows the other direction happens to have. A
    single-direction bucket (SVA/subj_aux, or NPA's HEAD pairs) has nothing
    to split, so it takes the plain head() path unchanged."""
    if role_series.nunique() <= 1:
        return valid.groupby("leaf_id", group_keys=False).head(max_per_bucket)

    parts = []
    for _, leaf_df in valid.groupby("leaf_id", sort=False):
        leaf_role = role_series.loc[leaf_df.index]
        if leaf_role.nunique() <= 1:
            parts.append(leaf_df.head(max_per_bucket))
            continue
        parts.append(leaf_df[leaf_role == "A"].head(max_per_bucket))
        parts.append(leaf_df[leaf_role == "B"].head(max_per_bucket))
    return pd.concat(parts) if parts else valid.head(0)


def _v2_pair_role_counts_by_leaf(valid, role_series):
    """{leaf_id_str: {"A": n, "B": n}} -- the REAL count of correct_swaps
    rows per direction per leaf, independent of any page-size cap (mirrors
    total_by_leaf's own "real total, not just what's materialized into
    PAIRS" role for entry["correctSwaps"]). Used for the Pairs tab's own
    "Determiner"/"Adjective" jump chips, the same way the Samples tab's
    Agree/Disagree chips read their counts from node.dist -- the real
    per-node distribution -- rather than from the capped sample rows."""
    counts = {}
    for (leaf_id, role), cnt in valid.groupby(["leaf_id", role_series]).size().items():
        counts.setdefault(str(int(leaf_id)), {})[role] = int(cnt)
    return counts


def _v2_build_pairs(correct_swaps_df, decisive_feat, max_per_bucket=10):
    """Builds T.PAIRS from create_pairs' own "correct_swaps" bucket, capped
    to max_per_bucket real rows per leaf *per reinflection direction*
    (page-size budget, same idea as node samples' own max_rows) -- plus the
    true per-leaf total (entry["correctSwaps"]) and true per-leaf-per-
    direction totals (entry["pairRoleCounts"]), both independent of how
    many rows are actually materialized into PAIRS.

    Returns (pairs, total_by_leaf, swapped_role, pair_role_counts) --
    swapped_role ('A'/'B', or None if no pairs) is the majority swap
    direction observed among the emitted pairs, used as T.swappedRole's
    tree-level fallback; each pair also carries its own swapRole, so a
    genuinely mixed-direction bucket (e.g. npa's merged non-head pairs)
    still renders correctly per pair.
    """
    if correct_swaps_df is None or len(correct_swaps_df) == 0 or "leaf_id" not in correct_swaps_df.columns:
        return [], {}, None, {}

    raw_role_a, raw_role_b = _v2_raw_pair_role_prefixes(correct_swaps_df)

    valid = correct_swaps_df.dropna(subset=["leaf_id"])
    total_by_leaf = {
        str(int(k)): int(v) for k, v in valid["leaf_id"].value_counts().items()
    }
    role_series = _v2_leaf_role_series(valid, raw_role_b)
    pair_role_counts = _v2_pair_role_counts_by_leaf(valid, role_series)
    sampled = _v2_sample_per_leaf(valid, role_series, max_per_bucket)

    pairs = []
    role_votes = {"A": 0, "B": 0}
    for row in sampled.to_dict("records"):
        pair = _v2_pair_from_row(row, raw_role_a, raw_role_b, decisive_feat)
        if pair is None:
            continue
        pair["leaf"] = str(int(row["leaf_id"]))
        pairs.append(pair)
        role_votes[pair["swapRole"]] += 1

    swapped_role = "A" if role_votes["A"] > role_votes["B"] else "B" if pairs else None
    return pairs, total_by_leaf, swapped_role, pair_role_counts


def _v2_leaf_threshold_for(leaf_threshold, node_id):
    """leaf_threshold is either one cutoff for every leaf, or (a
    agreement_trees.pairs.second_chance depth-aware retry) a {leaf_id: cutoff} map --
    either way, resolve to this specific node's own value."""
    if isinstance(leaf_threshold, dict):
        return leaf_threshold.get(int(node_id))
    return leaf_threshold


def _v2_build_nodes(node_data, screen, leaf_threshold, strict_leaf_threshold=None,
                     pair_counts=None, pair_role_counts=None):
    nodes = {}
    for i, nd in node_data.items():
        x, y = screen[i]
        is_leaf = nd["is_leaf"]
        branch = "root"
        if nd["parent"] is not None:
            _, is_true = nd["rule"]
            branch = "true" if is_true else "false"
        entry = {
            "x": round(x, 1), "y": round(y, 1),
            "branch": branch,
            "n": nd["n"], "H": round(nd["H"], 4), "acc": round(nd["acc"], 4),
            "leaf": is_leaf,
            "corr": nd.get("corr") or [],
            "dist": [
                {"cls": d["cls"], "cnt": d["cnt"], "color": _v2_class_color(d["cls"], d["color"])}
                for d in nd["dist"]
            ],
        }
        if is_leaf:
            quals = _v2_leaf_qualifying_classes(nd["dist"])
            entry["rule"] = "predict: " + (" / ".join(quals) if quals else "?")
            entry["predicted"] = " / ".join(quals)
            entry["majorityLabel"] = quals[0] if quals else None
            node_threshold = _v2_leaf_threshold_for(leaf_threshold, i)
            entry["keep"] = bool(node_threshold is not None and nd["acc"] > node_threshold)
            # This leaf's own cutoff, always -- whether kept or not, and
            # whether leaf_threshold is one global cutoff or a per-leaf map
            # (depth-aware second_chance gives leaves different values).
            entry["keepThresholdAcc"] = node_threshold
            # Only reached via a second_chance retry: strict_leaf_threshold is
            # the pipeline's own (pre-retry) bar, always passed regardless of
            # whether a retry happened -- so this is False both when there was
            # no retry and when one was reverted, since node_threshold then
            # equals strict_leaf_threshold and no kept leaf's acc can be <= it.
            entry["secondChance"] = bool(
                entry["keep"] and strict_leaf_threshold is not None
                and nd["acc"] <= strict_leaf_threshold
            )
            # "kept" is the majority class's own count, not the leaf's full n.
            majority_cnt = next(
                (d["cnt"] for d in nd["dist"] if d["cls"] == entry["majorityLabel"]), 0
            )
            entry["kept"] = majority_cnt if entry["keep"] else 0
            entry["correctSwaps"] = (pair_counts or {}).get(str(i), 0)
            entry["pairRoleCounts"] = (pair_role_counts or {}).get(str(i)) or {}
        else:
            entry["rule"] = nd.get("own_rule") or ""
            entry["ruleFeat"] = None
            entry["ruleRole"] = None
            entry["ruleValue"] = None
        nodes[str(i)] = entry
    return nodes


def _v2_build_edges(node_data):
    edges = []
    for i, nd in node_data.items():
        if nd["parent"] is not None:
            _, is_true = nd["rule"]
            edges.append([nd["parent"], i, "true" if is_true else "false"])
    return edges


def _v2_build_paths(node_data):
    paths = {}
    for i in node_data:
        path, cur = [i], i
        while node_data[cur]["parent"] is not None:
            cur = node_data[cur]["parent"]
            path.append(cur)
        paths[str(i)] = list(reversed(path))
    return paths


def _v2_decisive_feat(predictor_display):
    # meta["Predictor"] (_display_predictor_var's output) is a string like
    # "head_nsubj_Number" or "Verb_nsubj_Number" -- the decisive agreement
    # feature is always its last underscore segment.
    if not predictor_display:
        return None
    return str(predictor_display).rsplit("_", 1)[-1]


def _v2_build_meta(meta, leaf_threshold, strict_leaf_threshold=None):
    m = {
        "language": " · ".join(filter(None, [
            meta.get("Language") or meta.get("language"), meta.get("Treebank"), meta.get("titleSuffix")])),
        "languageName": meta.get("Language") or meta.get("language") or "",
        "accuracy": meta.get("accuracy", ""),
        "testAccuracy": meta.get("testAccuracy"),
        "testN": meta.get("testN"),
        "measure": meta.get("measure", "entropy"),
        "nodes": meta.get("Nodes", ""),
        "depth": meta.get("Depth", ""),
        "trainingSamples": meta.get("Training samples", ""),
        "predictor": meta.get("Predictor", ""),
    }
    if m["measure"] == "accuracy":
        m["rootAccuracy"] = meta.get("root accuracy", "")
        m["trainAccuracy"] = meta.get("train accuracy (smoothed)", "")
    else:
        m["baseEntropy"] = meta.get("base entropy", "")
        m["reducedEntropy"] = meta.get("reduced entropy", "")
    if isinstance(leaf_threshold, dict):
        # A depth-aware second_chance retry: no single number to show here --
        # each leaf's own 🔑/🔑2️⃣ tag already carries its own cutoff, so the
        # legend just shows the plain strict bar every leaf was first held to.
        m["keepThreshold"] = f"leaf accuracy > {strict_leaf_threshold:g}" if strict_leaf_threshold is not None else ""
        m["keepThresholdAcc"] = strict_leaf_threshold
    elif leaf_threshold is not None:
        m["keepThreshold"] = f"leaf accuracy > {leaf_threshold:g}"
        m["keepThresholdAcc"] = leaf_threshold
    if meta.get("Treebank"):
        m["treebank"] = meta["Treebank"]
    if meta.get("Excluded"):
        m["excludedReason"] = meta["Excluded"]
    if meta.get("treebankNav"):
        m["nav"] = meta["treebankNav"]
    if meta.get("excludedClasses"):
        m["excludedClasses"] = meta["excludedClasses"]
    if meta.get("rootNotice"):
        m["rootNotice"] = meta["rootNotice"]
    return m


def _v2_build_classes(classes, hex_colors, root_dist_counts):
    return [
        {"cls": cls, "cnt": int(cnt), "color": _v2_class_color(cls, hexc)}
        for cls, hexc, cnt in zip(classes, hex_colors, root_dist_counts)
    ]


def create_html_v2(meta, node_samples, node_data, hex_colors, classes, root_dist_counts,
                    leaf_threshold=None, strict_leaf_threshold=None, head_label="head",
                    correct_swaps_df=None):
    screen, canvas_w, canvas_h = _v2_layout(node_data)
    decisive_feat = _v2_decisive_feat(meta.get("Predictor"))
    pairs, pair_counts, swapped_role, pair_role_counts = _v2_build_pairs(correct_swaps_df, decisive_feat)
    nodes = _v2_build_nodes(node_data, screen, leaf_threshold, strict_leaf_threshold=strict_leaf_threshold,
                             pair_counts=pair_counts, pair_role_counts=pair_role_counts)
    edges = _v2_build_edges(node_data)
    paths = _v2_build_paths(node_data)
    role_a, role_b = _v2_detect_role_prefixes(node_samples, head_label)
    node_samples_v2 = {
        str(i): _v2_node_samples(i, node_samples, role_a, role_b) for i in node_data
    }
    tree_meta = _v2_build_meta(meta, leaf_threshold, strict_leaf_threshold=strict_leaf_threshold)
    # {"role": "a"|"b", "feat": ...} for each column kept out of the fit; the
    # head prefix is role b (verb/aux), anything else the dependent (role a).
    tree_meta["fitExcluded"] = [
        {"role": "b" if col.split("_", 1)[0].lower() == "head" else "a", "feat": col.split("_", 1)[1]}
        for col in meta.get("fitExcluded", []) if "_" in col
    ]
    tree_obj = {
        "title": " · ".join(filter(None, [
            meta.get("Language") or meta.get("language"), meta.get("Treebank"),
            meta.get("Predictor", ""), meta.get("titleSuffix")])),
        "roleA": role_a.lower(),
        "roleB": role_b.lower(),
        "decisiveFeat": decisive_feat,
        # Tree-level fallback only -- each pair also carries its own
        # swapRole, since a merged-direction bucket can swap either role row
        # by row (see _v2_build_pairs). 'B' when there are no pairs at all
        # (no visible effect, matches SVA/subj_aux's always-head convention).
        "swappedRole": swapped_role or "B",
        "defaultNode": 0,
        "CANVAS_W": canvas_w,
        "CANVAS_H": canvas_h,
        "META": tree_meta,
        "CLASSES": _v2_build_classes(classes, hex_colors, root_dist_counts),
        "NODES": nodes,
        "EDGES": edges,
        "PATHS": paths,
        "NODE_SAMPLES": node_samples_v2,
        "PAIRS": pairs,
    }
    tree_js = (
        "const TREE_DATA = " + _json(tree_obj) + ";\n"
        "  const TREES = { main: TREE_DATA };\n"
        "  let currentTreeKey = 'main';\n"
    )
    with open(_V2_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    body = template.replace("__V2_TREE_DATA_JS__", tree_js)
    return "<!doctype html>\n<html lang=\"en\">\n" + body + "\n</html>\n"


def write_html_v2(node_samples, node_data, out_file, classes, hex_colors, root_dist_counts,
                   meta, leaf_threshold=None, strict_leaf_threshold=None, head_label="head",
                   correct_swaps_df=None):
    html = create_html_v2(
        meta, node_samples, node_data, hex_colors, classes, root_dist_counts,
        leaf_threshold=leaf_threshold, strict_leaf_threshold=strict_leaf_threshold,
        head_label=head_label, correct_swaps_df=correct_swaps_df,
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)


def _placeholder_nav_html(nav):
    """Static treebank links for a placeholder page (same data as the v2 pages'
    pill row), so a treebank without a tree is a link, not a dead end."""
    def esc(x):
        return html_lib.escape(str(x))

    def pill(text, href, current=False, muted=False):
        cls = "tb-pill" + (" current" if current else "") + (" muted" if muted else "")
        return (f'<span class="{cls}">{text}</span>' if current or not href
                else f'<a class="{cls}" href="{href}">{text}</a>')

    def pct(v):
        return f"{v * 100:.1f}%"

    def rows_txt(it):
        txt = f"{it['rows']:,} rows"
        if it.get("acc") is not None:
            txt += f" &middot; {pct(it['acc'])}"
        elif it.get("placeholder"):
            txt += f" &middot; {esc(it['status'])}"
        return txt

    pooled_rows = f"{nav['pooledRows']:,} rows" + (
        f" &middot; {pct(nav['pooledAcc'])}" if nav.get("pooledAcc") is not None else "")
    parts = ['<span class="tb-nav-label">Trees</span>',
             pill(f"Pooled <small>{pooled_rows}</small>", nav.get("pooledHref"), nav.get("pooledCurrent"))]
    excluded = []
    for it in nav["items"]:
        p = pill(f"{esc(it['name'])} <small>{rows_txt(it)}</small>", it.get("href"),
                 it.get("current"), muted=it.get("placeholder", False))
        (excluded if it.get("excluded") else parts).append(p)
    if excluded:
        parts.append('<span class="tb-nav-label">Excluded from pairs</span>')
        parts.extend(excluded)
    if nav.get("counterpartHref"):
        parts.append(f'<a class="tb-nav-extra" href="{nav["counterpartHref"]}">Tree fit incl. rows missing annotation (&quot;unk&quot;) &rarr;</a>')
    return f'<div class="tb-nav">{"".join(parts)}</div>'


_CLASS_DISPLAY = {"yes": "Agree", "no": "Disagree", "unk": "unknown"}


def _display_class(cls) -> str:
    """Reader-facing name of an agreement class (the data keeps yes/no/unk)."""
    return _CLASS_DISPLAY.get(str(cls).lower(), str(cls))


def write_placeholder_html(out_file, predictor_var, label, meta=None, sample_rows=None, nav=None):
    # `label`: {value: count} across ALL training samples (build_placeholder_args
    # always passes the full value_counts, not a subsample) -- a single key means
    # every row genuinely shares one value; multiple keys means the tree wasn't
    # fit for some other reason (e.g. too few samples) despite a mixed label set.
    if len(label) == 1:
        heading = "Single label"
        description = (
            f"All training samples for <b>{predictor_var}</b> share one "
            "agreement label — no decision tree was needed."
        )
        (sole_label,) = label.keys()
        label_html = f'<div class="label">{_display_class(sole_label)}</div>'
    else:
        heading = "Too few samples to fit a tree"
        description = (
            f"Training samples for <b>{predictor_var}</b> don&rsquo;t share one "
            "agreement label, but there weren&rsquo;t enough of them to fit a "
            "decision tree — showing the raw label distribution instead."
        )
        label_html = "".join(
            f'<div class="label">{_display_class(cls)}: {n}</div>'
            for cls, n in sorted(label.items(), key=lambda kv: -kv[1])
        )

    # Machine-readable distribution for downstream consumers (e.g.
    # viz_deprel.py's overview table) to read without regex-scraping label_html.
    distribution_json = json.dumps(
        {str(k): int(n) for k, n in label.items()}, ensure_ascii=False
    )

    meta_rows = ""
    if meta:
        for key, val in meta.items():
            if val is None or key in ("titleSuffix", "treebankNav"):
                continue
            meta_rows += f'<div class="meta-row"><span class="meta-key">{key}</span><span class="meta-val">{val}</span></div>'

    table_html = ""
    if sample_rows:
        cols = list(sample_rows[0].keys())
        table_html += "<table class='ex-table'><thead><tr>"
        for c in cols:
            table_html += f"<th>{c}</th>"
        table_html += "</tr></thead><tbody>"
        for row in sample_rows:
            table_html += "<tr>"
            for c in cols:
                val = row[c]
                if val is None:
                    content = ""
                elif c.endswith("_features"):
                    # unpack list the same way as tree pages
                    if isinstance(val, list):
                        items = "<br>".join(
                            v for v in val if not str(v).endswith("=nan")
                        )
                    else:
                        # stored as string repr of list
                        import ast

                        try:
                            items = "<br>".join(
                                v
                                for v in ast.literal_eval(str(val))
                                if not str(v).endswith("=nan")
                            )
                        except Exception:
                            items = str(val)
                    content = (
                        f"<details class='feat-details'><summary>features...</summary>{items}</details>"
                    )
                else:
                    content = str(val)
                table_html += f"<td>{content}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>No decision tree to display</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f5f4; --surface: #ffffff; --surface-raised: #fafaf9;
      --border: #e7e5e4; --border-subtle: #f0efee;
      --text-primary: #1c1917; --text-secondary: #78716c; --text-tertiary: #a8a29e;
      --accent: #2563eb; --accent-light: #eff6ff;
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
      --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
      --border: #38332a; --border-subtle: #2a261e;
      --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
      --accent: #6ea8ff; --accent-light: #1c2a42;
    }}
    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        color-scheme: dark;
        --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
        --border: #38332a; --border-subtle: #2a261e;
        --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
        --accent: #6ea8ff; --accent-light: #1c2a42;
      }}
    }}
    body {{ font-family: "DM Sans", sans-serif; margin: 0; background: var(--bg); color: var(--text-primary); }}
    /* Stacked (top: title/caption/label/distribution card, bottom: example
       items), mirroring the real tree pages' layout -- there, a compact
       floating info-panel card sits above the (much larger) tree diagram
       area, with the examples section below that. This placeholder has no
       diagram to show, so the card just sits in normal flow at the top
       instead of floating, but keeps the same "compact bordered/shadowed
       card, not a full-width band" treatment -- that's what makes the
       meta-rows' space-between layout read as compact key-value pairs
       rather than stretching key and value apart across the full page. */
    .layout {{ display: flex; flex-direction: column; min-height: 100vh; }}
    .sidebar {{
        max-width: 420px; margin: 1.5rem 1.5rem 0; background: var(--surface);
        border: 1px solid var(--border); border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem;
    }}
    .overview-link {{
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 9px; border: 1px solid var(--border); border-radius: 6px;
        background: var(--surface); color: var(--text-secondary);
        font-family: "DM Sans", sans-serif; font-size: 0.75rem; font-weight: 600;
        text-decoration: none; margin-bottom: 0.7rem;
    }}
    .overview-link:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-light); }}
    h2 {{ font-size: 1.1rem; margin: 0; }}
    p  {{ color: var(--text-secondary); font-size: 0.875rem; margin: 0; line-height: 1.5; }}
    .label {{ display: inline-block; padding: 0.3rem 0.9rem;
              background: var(--bg); border-radius: 999px; font-weight: 600;
              font-size: 1rem; border: 1px solid var(--border);
              margin: 0 0.4rem 0.4rem 0; }}
    .labels {{ display: flex; flex-wrap: wrap; }}
    .meta {{ font-size: 0.8rem; }}
    .meta-row {{ display: flex; justify-content: space-between; padding: 0.2rem 0;
                 border-bottom: 1px solid var(--border); }}
    .meta-key {{ color: var(--text-tertiary); }}
    .meta-val {{ font-family: "JetBrains Mono", monospace; color: var(--text-secondary); }}
    .tb-nav {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 1.5rem 1.5rem 0; }}
    .tb-nav-label {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-tertiary); }}
    .tb-pill {{ display: inline-flex; gap: 6px; align-items: baseline; padding: 5px 12px; border-radius: 999px;
                border: 1.5px solid var(--border); background: var(--surface-raised); color: var(--text-secondary);
                font-size: 0.78rem; font-weight: 600; text-decoration: none; }}
    .tb-pill small {{ font-family: "JetBrains Mono", monospace; font-size: 0.68rem; font-weight: 500; opacity: 0.75; }}
    a.tb-pill:hover {{ border-color: var(--accent); color: var(--accent); }}
    .tb-pill.current {{ background: var(--accent-light); border-color: var(--accent); color: var(--accent); }}
    .tb-pill.muted {{ border-style: dashed; opacity: 0.8; }}
    .tb-nav-extra {{ margin-left: auto; font-size: 0.78rem; font-weight: 600; color: var(--accent); text-decoration: none; }}
    .main {{ padding: 1.5rem; }}
    .main h3 {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.05em; color: var(--text-secondary); margin: 0 0 0.75rem; }}
    .ex-table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem;
                 border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    .ex-table th {{ background: var(--surface-raised); color: var(--text-secondary); font-size: 0.7rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.04em; padding: 6px 10px;
                    text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    .ex-table td {{ padding: 5px 10px; border-bottom: 1px solid var(--border-subtle); vertical-align: top;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }}
    .ex-table td:first-child {{ white-space: normal; word-break: break-word; max-width: none; }}
    .ex-table tbody tr:hover td {{ background: var(--accent-light); }}
    .ex-table a {{ color: var(--accent); text-decoration: none; font-weight: 500; }}
    .ex-table a:hover {{ text-decoration: underline; }}
    .ex-table tbody tr:last-child td {{ border-bottom: none; }}
    .ex-table td strong {{ color: var(--accent); font-weight: 700; }}
  </style>
</head>
<body>
  <script type="application/json" id="label-distribution">{distribution_json}</script>
  <div class="layout">
    {_placeholder_nav_html(nav) if nav else ""}
    <div class="sidebar">
      <div>
        <a href="{nav["overviewHref"] if nav else "index.html"}" class="overview-link" title="Back to the overview for this dependency relation">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 12L6 8L10 4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Overview
        </a>
        <h2>{heading}</h2>
        <p>{description}</p>
      </div>
      <div class="labels">{label_html}</div>
      <div class="meta">{meta_rows}</div>
    </div>
    <div class="main">
      <h3>Example items (sample of {len(sample_rows) if sample_rows else 0})</h3>
      {table_html}
    </div>
  </div>
  <script>
    // Feature lists for different roles in the same example row (e.g. subject
    // vs. verb) are separate <details> elements. Link them per-row so that
    // opening/closing one toggles the others in that row to match.
    document.querySelectorAll(".ex-table tr").forEach(tr => {{
      const detailsEls = tr.querySelectorAll("details.feat-details");
      if (detailsEls.length < 2) return;
      detailsEls.forEach(d => {{
        d.addEventListener("toggle", () => {{
          detailsEls.forEach(other => {{
            if (other !== d) other.open = d.open;
          }});
        }});
      }});
    }});
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
