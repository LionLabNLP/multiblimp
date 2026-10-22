"""Per-language diagnostics table for a create_pairs() run: how many minimal pairs were produced,
where lexicon coverage broke down, and how reliably the inflector flipped the target feature.

Reads the "<item_type>.parquet" buckets and "meta.json" that create_pairs writes into
"<pairs_dir>/<language>/"; one row per language, including languages with zero swappable rows.
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
    "ambiguous_readings": "ambiguous reading",
    "same_forms": "same inflection",
    "same_features": "same feature",
    "undefined_features": "undefined feature",
    "ambiguous_subjects": "ambiguous subject",
    "conflicting_features": "conflicting features",
    "invalid_surface": "invalid surface",
}

BUCKET_JSON_KEYS = {
    "no_candidates": "no_match",
    "no_inflections": "no_inflection",
    "ambiguous_readings": "ambiguous_reading",
    "same_forms": "same_inflection",
    "same_features": "same_feature",
    "undefined_features": "undefined_feature",
    "ambiguous_subjects": "ambiguous_subject",
    "conflicting_features": "conflicting_features",
    "invalid_surface": "invalid_surface",
}

# multi_now_valid is left out: it duplicates some correct_swaps rows.
DISTRIBUTION_BUCKETS = ["correct_swaps", "same_features", "ambiguous_subjects", "undefined_features"]

ROW_SCOPED_BUCKETS = {"no_candidates", "no_inflections"}


def _read_bucket(lang_dir: str, item_type: str, columns=None) -> pd.DataFrame:
    """One bucket's parquet (only the `columns` that exist, when given: these files can be thousands of columns wide)."""
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
    """How many of this bucket's rows exist only because agreement_trees.pairs.second_chance retried the
    language at a laxer bar; 0 if it was never retried (no "second_chance" column) or the bucket is empty."""
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
    """'Sing -> Plur' -> ({'Sing'}, {'Plur'}); handles multi-valued sides like
    'Plur -> Plur|Sing' -> ({'Plur'}, {'Plur', 'Sing'})."""
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

    # coverage over every distinct head/child form that is a candidate for the target, not just the "Yes" ones
    num_forms_of_interest = meta.get("num_forms_of_interest") or 0
    num_covered_um = meta.get("num_covered_um") or 0
    num_covered_um_ud = meta.get("num_covered_um_ud") or 0

    dist, probs = _distribution_and_probs(buckets)

    # a dict means leaves have their own cutoffs (second chance): no single number for the language
    leaf_threshold = meta.get("leaf_min_acc")
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
    """One row per language subdirectory directly under `pairs_dir`, sorted by language name."""
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
    """row.get(key, default), with NaN also counting as missing (the DataFrame fills columns a
    language's row didn't set with NaN, which is truthy)."""
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _read_examples(lang_dir: str) -> dict:
    """{item_type: html_fragment}, meta.json's "examples" key (create_pairs.bucket_examples_html); {} if absent."""
    return _read_meta(lang_dir).get("examples") or {}


def _read_label_distribution(lang_dir: str) -> dict:
    """{label: count} over the language's full predictor_var column (meta.json "label_distribution",
    computed before fit_dt drops unk rows); {} if absent."""
    return _read_meta(lang_dir).get("label_distribution") or {}


def diagnostics_row_to_json(row, lang_dir: str | None = None) -> dict:
    """One row of generate_diagnostics_table (spreadsheet column names like "# same feature") as the
    compact per-language structure agreement_trees.report.html.html_deprel embeds: bucket counts/percents
    keyed by BUCKET_JSON_KEYS, probs keyed by bare value ("P(Sing|Sing)" -> "Sing").

    row: a pandas Series or a dict. lang_dir: the language's "<pairs_dir>/<language>" directory;
    when given, meta.json's examples (raw HTML, kept out of the CSV) and label distribution are included.
    """
    buckets = {}
    for item_type, label in BUCKET_LABELS.items():
        n = _num(row, f"# {label}")
        pct = _num(row, f"% {label}")
        n_second = _bucket_second_chance_count(lang_dir, item_type) if lang_dir else 0
        buckets[BUCKET_JSON_KEYS[item_type]] = [int(n), float(pct), n_second]

    probs = {}
    for col, val in dict(row).items():
        if col.startswith("P(") and pd.notna(val):
            value = col[2:-1].split("|")[0]  # "P(Sing|Sing)" -> "Sing"
            probs[value] = round(float(val), 3)

    leaf_threshold = row.get("leaf_threshold")
    is_per_leaf = isinstance(leaf_threshold, dict)
    # correct_swaps is one row per kept row, so its second-chance count is the pairs-level split;
    # the other buckets can hold several items per source row, hence nKeep's comes from meta.json
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
        result["examples"] = {
            BUCKET_JSON_KEYS.get(item_type, item_type): html
            for item_type, html in raw_examples.items()
        }
        result["labelDistribution"] = _read_label_distribution(lang_dir)

    return result
