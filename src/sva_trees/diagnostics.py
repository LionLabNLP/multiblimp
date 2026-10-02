"""
Per-language diagnostics table for a create_pairs() run: how many minimal pairs
were produced, where UniMorph/UD coverage broke down, and how reliably the
inflector actually flipped the target feature (e.g. Number SG<->PL).

Reads the "<item_type>.parquet" buckets and "meta.json" that sva_trees.pipeline
writes into "<pairs_dir>/<language>/" via sva_trees.create_pairs.create_pairs,
one row per language -- including languages with zero swappable rows, which
still get an all-empty-buckets entry (see create_pairs' zero-candidate path).
"""

import json
import os
from collections import Counter
from glob import glob

import pandas as pd
import pyarrow.parquet as pq

BUCKET_LABELS = {
    "no_candidates": "no match",
    "no_inflections": "no inflection",
    "same_forms": "same inflection",
    "same_features": "same feature",
    "undefined_features": "undefined feature",
    "ambiguous_subjects": "ambiguous subject",
    "conflicting_features": "conflicting features",
}

_BUCKET_JSON_KEYS = {
    "no_candidates": "no_match",
    "no_inflections": "no_inflection",
    "same_forms": "same_inflection",
    "same_features": "same_feature",
    "undefined_features": "undefined_feature",
    "ambiguous_subjects": "ambiguous_subject",
    "conflicting_features": "conflicting_features",
}

# We excludes multi_now_valid (a logging duplicate of some correct_swaps rows) so nothing
# is double-counted.
DISTRIBUTION_BUCKETS = ["correct_swaps", "same_features", "ambiguous_subjects", "undefined_features"]

ROW_SCOPED_BUCKETS = {"no_candidates", "no_inflections"}


def _read_bucket(lang_dir: str, item_type: str, columns=None) -> pd.DataFrame:
    """One bucket's parquet (only `columns` of it, those that exist, when
    given -- these files can be thousands of columns wide)."""
    fn = os.path.join(lang_dir, f"{item_type}.parquet")
    if not os.path.exists(fn):
        return pd.DataFrame()
    if columns is not None:
        columns = [c for c in columns if c in pq.read_schema(fn).names]
    return pd.read_parquet(fn, columns=columns)


def _bucket_len(lang_dir: str, item_type: str) -> int:
    """Row count from the parquet footer, without reading any data."""
    fn = os.path.join(lang_dir, f"{item_type}.parquet")
    return pq.ParquetFile(fn).metadata.num_rows if os.path.exists(fn) else 0


def _bucket_second_chance_count(lang_dir: str, item_type: str) -> int:
    """How many of this bucket's rows only exist because sva_trees.
    second_chance retried this language at a laxer bar, vs. how many already
    cleared the strict first pass -- 0 if the language was never retried (no
    "second_chance" column at all, see create_pairs) or the bucket is empty.
    """
    df = _read_bucket(lang_dir, item_type, columns=["second_chance"])
    if "second_chance" not in df.columns:
        return 0
    return int(df["second_chance"].sum())


def _read_meta(lang_dir: str) -> dict:
    fn = os.path.join(lang_dir, "meta.json")
    if not os.path.exists(fn):
        return {}
    with open(fn) as f:
        return json.load(f)


def _split_feature_key(key: str):
    """'SG -> PL' -> ({'SG'}, {'PL'}); handles multi-valued sides like
    'PL -> PL|SG' -> ({'PL'}, {'PL', 'SG'})."""
    frm, to = key.split(" -> ")
    return set(frm.split("|")), set(to.split("|"))


def _distribution_and_probs(buckets: dict):
    """Combine feature_vals across DISTRIBUTION_BUCKETS into:
      - dist: Counter of 'FROM -> TO' -> row count
      - probs: dict[value] -> P(value|value), i.e. among attempts that started
        from `value`, the fraction that landed in same_features (the reinflected
        form's value still overlapped the original) rather than genuinely
        changing away from it.
    """
    dist = Counter()
    denom = Counter()
    stayed = Counter()

    for item_type in DISTRIBUTION_BUCKETS:
        df = buckets.get(item_type)
        if df is None or not len(df) or "feature_vals" not in df.columns:
            continue
        for key, n in df["feature_vals"].value_counts().items():
            n = int(n)
            dist[key] += n
            from_set, to_set = _split_feature_key(key)
            for v in from_set:
                denom[v] += n
                if item_type == "same_features" and v in to_set:
                    stayed[v] += n

    probs = {v: (stayed[v] / denom[v] if denom[v] else None) for v in denom}
    return dist, probs


def _format_distribution(dist: Counter) -> str:
    if not dist:
        return ""
    return "; ".join(f"{key}: {n}" for key, n in sorted(dist.items(), key=lambda kv: -kv[1]))


def language_diagnostics_row(lang: str, lang_dir: str) -> dict:
    bucket_names = list(BUCKET_LABELS) + ["correct_swaps", "multi_now_valid"]
    counts = {name: _bucket_len(lang_dir, name) for name in bucket_names}
    buckets = {
        name: _read_bucket(lang_dir, name, columns=["feature_vals"])
        for name in DISTRIBUTION_BUCKETS if counts[name]
    }

    meta = _read_meta(lang_dir)
    n_raw = meta.get("num_ud_candidates_raw") or 0
    n_keep = meta.get("num_ud_candidates_keep") or 0
    n_keep_second_chance = meta.get("num_ud_candidates_keep_second_chance") or 0
    items_seen = meta.get("items_seen") or 0
    swap_items = items_seen - counts["no_candidates"] - counts["no_inflections"]
    n_pairs = counts["correct_swaps"]

    def pct(n, denom):
        return round(n / denom * 100, 1) if denom else 0

    def bucket_pct(item_type, n):
        return pct(n, n_keep if item_type in ROW_SCOPED_BUCKETS else swap_items)

    # UM/UM+UD coverage over every distinct head/child form that's a candidate
    # for this prediction target at all (not just the "Yes"-labeled ones)
    num_forms_of_interest = meta.get("num_forms_of_interest") or 0
    num_covered_um = meta.get("num_covered_um") or 0
    num_covered_um_ud = meta.get("num_covered_um_ud") or 0

    dist, probs = _distribution_and_probs(buckets)

    # A dict here means a second_chance depth-aware retry gave leaves their
    # own individual cutoffs -- no single number represents the language for
    # this per-language table (each leaf's own tag on the tree page has it).
    leaf_threshold = meta.get("leaf_threshold")
    row = {
        "Language": lang,
        "leaf_threshold": None if isinstance(leaf_threshold, dict) else leaf_threshold,
        "# minimal pairs": n_pairs,
        "# forms of interest": num_forms_of_interest,
        "% covered by UM": pct(num_covered_um, num_forms_of_interest),
        "% covered by UM+UD": pct(num_covered_um_ud, num_forms_of_interest),
        "# UD candidates (raw)": n_raw,
        "# UD candidates (kept)": n_keep,
        "# UD candidates (kept, 2nd chance)": n_keep_second_chance,
        "# UM lemmas": meta.get("num_lemma"),
        "# UM Forms": meta.get("num_form"),
        # unk ≈ missing annotation, can be dropped from fit_dt
        "# head unk": meta.get("head_unk") or 0,
        "# nsubj unk": meta.get("nsubj_unk") or 0,
        "# both unk": meta.get("both_unk") or 0,
    }
    for item_type, label in BUCKET_LABELS.items():
        row[f"# {label}"] = counts[item_type]
        row[f"% {label}"] = bucket_pct(item_type, counts[item_type])
    row["# valid from multi"] = counts["multi_now_valid"]
    row["#valid_from_multi / #valid"] = (
        round(counts["multi_now_valid"] / n_pairs, 3) if n_pairs else 0
    )
    row["distribution"] = _format_distribution(dist)
    for value, prob in sorted(probs.items()):
        row[f"P({value}|{value})"] = round(prob, 3) if prob is not None else None

    return row


def generate_diagnostics_table(pairs_dir: str) -> pd.DataFrame:
    """One row per language subdirectory found directly under `pairs_dir`
    (e.g. '../../minimal_pairs/svNa/svNa_nsubj'), sorted by language name.
    """
    base_cols = [
        "Language", "leaf_threshold", "# minimal pairs", "# forms of interest",
        "% covered by UM", "% covered by UM+UD",
        "# UD candidates (raw)", "# UD candidates (kept)",
        "# UD candidates (kept, 2nd chance)",
        "# UM lemmas", "# UM Forms",
        "# head unk", "# nsubj unk", "# both unk",
    ]
    for label in BUCKET_LABELS.values():
        base_cols += [f"# {label}", f"% {label}"]
    base_cols += ["# valid from multi", "#valid_from_multi / #valid", "distribution"]

    rows = []
    prob_cols = []
    for lang_dir in sorted(glob(os.path.join(pairs_dir, "*"))):
        if not os.path.isdir(lang_dir):
            continue
        lang = os.path.basename(lang_dir)
        row = language_diagnostics_row(lang, lang_dir)
        rows.append(row)
        for col in row:
            if col.startswith("P(") and col not in prob_cols:
                prob_cols.append(col)

    columns = base_cols + sorted(prob_cols)
    return pd.DataFrame(rows, columns=columns)


def write_diagnostics_csv(df: pd.DataFrame, out_path: str) -> None:
    """Write a Google Sheets-importable CSV (File > Import > Upload)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def _num(row, key, default=0):
    """row.get(key, default), but also treats NaN as `default`. Plain `or
    default` doesn't work for this: NaN is truthy in Python, so `nan or 0`
    evaluates to nan, not 0 -- and generate_diagnostics_table's DataFrame
    construction fills any column a given language's row didn't set (e.g. a
    P(value|value) column only some languages have) with NaN, not None.
    """
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _read_examples(lang_dir: str) -> dict:
    """{item_type: html_fragment}, sva_trees.create_pairs.create_pairs's
    "examples" key inside meta.json (via create_pairs.bucket_examples_html). {}
    for a language whose create_pairs run never reached the save_to block, or
    predates "examples" living in meta.json rather than its own file.
    """
    return _read_meta(lang_dir).get("examples") or {}


def _read_label_distribution(lang_dir: str) -> dict:
    """{label: count} across the language's full predictor_var column, from
    meta.json's "label_distribution" key (sva_trees.pipeline, computed before
    fit_dt drops unk rows). {} for a language predating this, or one whose
    create_pairs run never reached the save_to block.
    """
    return _read_meta(lang_dir).get("label_distribution") or {}


def diagnostics_row_to_json(row, lang_dir: str | None = None) -> dict:
    """Reshape one row of generate_diagnostics_table's DataFrame (pretty,
    spreadsheet-facing column names, e.g. "# same feature") into the compact
    structure word_order.html.html_deprel's report embeds per language --
    bucket counts/percents keyed by _BUCKET_JSON_KEYS, and probs keyed by
    bare feature value ("P(SG|SG)" -> "SG": 0.041).

    `row` is anything supporting .get() with default -- a pandas Series (from
    df.iterrows()) or a plain dict both work.

    lang_dir: that language's "<pairs_dir>/<language>" directory. When given,
    also reads meta.json's "examples" key and includes it (remapped to the
    same _BUCKET_JSON_KEYS as "buckets") as "examples" -- kept out of the
    DataFrame/CSV path entirely (generate_diagnostics_table never sees this
    function), so the spreadsheet export never ends up with raw HTML in a
    cell.
    """
    buckets = {}
    for item_type, label in BUCKET_LABELS.items():
        n = _num(row, f"# {label}")
        pct = _num(row, f"% {label}")
        n_second = _bucket_second_chance_count(lang_dir, item_type) if lang_dir else 0
        buckets[_BUCKET_JSON_KEYS[item_type]] = [int(n), float(pct), n_second]

    probs = {}
    for col, val in dict(row).items():
        if col.startswith("P(") and pd.notna(val):
            value = col[2:-1].split("|")[0]  # "P(SG|SG)" -> "SG"
            probs[value] = round(float(val), 3)

    leaf_threshold = row.get("leaf_threshold")
    # A dict here means a second_chance depth-aware retry gave leaves their
    # own individual cutoffs -- no single number represents the language, so
    # this per-language summary table leaves it out rather than picking one
    # leaf's value arbitrarily (see each leaf's own tag on the tree page).
    is_per_leaf = isinstance(leaf_threshold, dict)
    # correct_swaps rows are 1:1 with kept rows (no fan-out), so its own
    # bucket count is a safe proxy for the pairs-level split. nKeep's split
    # is NOT derivable the same way from the buckets: several of them (e.g.
    # same_features, undefined_features) can hold more than one diagnostic
    # item per source row (one per inflection candidate tried), so their
    # counts don't sum back to n_keep's own row count -- read the row-level
    # figure create_pairs itself already computed instead (see meta.json's
    # num_ud_candidates_keep_second_chance).
    n_pairs_second = _bucket_second_chance_count(lang_dir, "correct_swaps") if lang_dir else 0
    result = {
        "leafThreshold": (
            None if is_per_leaf or leaf_threshold is None or pd.isna(leaf_threshold)
            else float(leaf_threshold)
        ),
        "nPairs": int(_num(row, "# minimal pairs")),
        "nPairsSecondChance": n_pairs_second,
        "nForms": int(_num(row, "# forms of interest")),
        "pctUM": float(_num(row, "% covered by UM")),
        "pctUMUD": float(_num(row, "% covered by UM+UD")),
        "nRaw": int(_num(row, "# UD candidates (raw)")),
        "nKeep": int(_num(row, "# UD candidates (kept)")),
        "nKeepSecondChance": int(_num(row, "# UD candidates (kept, 2nd chance)")),
        "nLemma": int(_num(row, "# UM lemmas")),
        "nForm": int(_num(row, "# UM Forms")),
        "nValidFromMulti": int(_num(row, "# valid from multi")),
        "ratioValidFromMulti": float(_num(row, "#valid_from_multi / #valid")),
        "buckets": buckets,
        "distribution": row.get("distribution") or "",
        "probs": probs,
        "headUnk": int(_num(row, "# head unk")),
        "nsubjUnk": int(_num(row, "# nsubj unk")),
        "bothUnk": int(_num(row, "# both unk")),
    }

    if lang_dir is not None:
        raw_examples = _read_examples(lang_dir)
        # The 6 problem buckets remap to _BUCKET_JSON_KEYS
        result["examples"] = {
            _BUCKET_JSON_KEYS.get(item_type, item_type): html
            for item_type, html in raw_examples.items()
        }
        # Kept out of the DataFrame/CSV path
        result["labelDistribution"] = _read_label_distribution(lang_dir)

    return result
