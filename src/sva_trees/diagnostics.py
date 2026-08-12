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

# item_type -> requested column label
BUCKET_LABELS = {
    "no_candidates": "no match",
    "no_inflections": "no inflection",
    "same_forms": "same inflection",
    "same_features": "same feature",
    "undefined_features": "undefined feature",
    "ambiguous_subjects": "ambiguous subject",
}

# Buckets whose rows carry a "feature_vals" ("FROM -> TO") column, i.e. every
# row that reached process_item's feature comparison step. Excludes
# multi_now_valid (a logging duplicate of some correct_swaps rows) so nothing
# is double-counted.
DISTRIBUTION_BUCKETS = ["correct_swaps", "same_features", "ambiguous_subjects", "undefined_features"]


def _read_bucket(lang_dir: str, item_type: str) -> pd.DataFrame:
    fn = os.path.join(lang_dir, f"{item_type}.parquet")
    return pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame()


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
    buckets = {name: _read_bucket(lang_dir, name) for name in bucket_names}
    counts = {name: len(df) for name, df in buckets.items()}
    meta = _read_meta(lang_dir)

    num_ud_candidates = meta.get("num_ud_candidates")
    n_pairs = counts["correct_swaps"]

    def pct(n):
        return round(n / num_ud_candidates * 100, 1) if num_ud_candidates else None

    dist, probs = _distribution_and_probs(buckets)

    row = {
        "Language": lang,
        "# minimal pairs": n_pairs,
        "% covered by UniMorph": (
            round((num_ud_candidates - counts["no_candidates"]) / num_ud_candidates * 100, 1)
            if num_ud_candidates else None
        ),
        "# UD candidates": num_ud_candidates,
        "# UM lemmas": meta.get("num_lemma"),
        "# UM Forms": meta.get("num_form"),
    }
    for item_type, label in BUCKET_LABELS.items():
        row[f"# {label}"] = counts[item_type]
        row[f"% {label}"] = pct(counts[item_type])
    row["# valid from multi"] = counts["multi_now_valid"]
    row["#valid_from_multi / #valid"] = (
        round(counts["multi_now_valid"] / n_pairs, 3) if n_pairs else None
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
        "Language", "# minimal pairs", "% covered by UniMorph", "# UD candidates",
        "# UM lemmas", "# UM Forms",
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
