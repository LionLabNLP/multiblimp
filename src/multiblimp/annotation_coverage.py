"""Annotation coverage statistics, one entry per language in annotation_coverage.json
(next to the scan results; annotation_coverage_overview.tsv is the readable summary):

  ud        token counts from the UD treebanks, written by scripts/ud_tables.py: tokens and
            tokens with any feats; per UPOS the tokens and, per feature, how many carry
            it; tokens skipped when building the UD-derived table, by reason; feature
            values with no UniMorph tag; per treebank the tokens with feats
  um_table  the UniMorph-derived table (resources/um_tables), written by scripts/um2ud.py:
            rows per UD upos and, per feature, the rows that state a real value, i.e.
            what annotation filling can supply
  ud_table  the same for the UD-derived table (resources/ud_tables), written by scripts/ud_tables.py
"""
import json
import os
from collections import Counter, defaultdict

from .config import SCAN_RESULTS_DIR

COVERAGE_PATH = os.path.join(SCAN_RESULTS_DIR, "annotation_coverage.json")
OVERVIEW_PATH = os.path.join(SCAN_RESULTS_DIR, "annotation_coverage_overview.tsv")

# (UPOS, feature) pairs shown as % of that UPOS's tokens in the overview
OVERVIEW_FEATURES = [
    ("VERB", f) for f in ("VerbForm", "Mood", "Tense", "Person", "Number", "Voice")
] + [("AUX", f) for f in ("VerbForm", "Mood", "Person", "Number")] + [
    ("NOUN", f) for f in ("Number", "Case", "Gender")
] + [("ADJ", f) for f in ("Number", "Case", "Gender")] + [("PRON", f) for f in ("Person", "Number")]


def new_ud_stats() -> dict:
    return {
        "tokens": 0, "tokens_with_feats": 0,
        "upos": defaultdict(lambda: {"tokens": 0, "feats": Counter()}),
        "skipped": Counter(), "unmapped": Counter(),
        "treebanks": defaultdict(lambda: Counter()),
        "_unmapped_cache": {},
    }


def count_token(stats: dict, treebank: str, upos: str, feats: dict) -> None:
    """Count one token before any skip rule applies."""
    stats["tokens"] += 1
    entry = stats["upos"][upos]
    entry["tokens"] += 1
    entry["feats"].update(feats.keys())
    stats["treebanks"][treebank]["tokens"] += 1
    if feats:
        stats["tokens_with_feats"] += 1
        stats["treebanks"][treebank]["with_feats"] += 1


def count_unmapped(stats: dict, feats: dict, value_is_mapped) -> None:
    """Count (feature, value) pairs with no UniMorph tag; value_is_mapped(feat, raw) says
    whether one has (cached per pair)."""
    cache = stats["_unmapped_cache"]
    for feat, raw in feats.items():
        key = (feat, raw)
        if key not in cache:
            cache[key] = not value_is_mapped(feat, raw)
        if cache[key]:
            stats["unmapped"][f"{feat}={raw}"] += 1


def finish_ud_stats(stats: dict) -> dict:
    return {
        "tokens": stats["tokens"], "tokens_with_feats": stats["tokens_with_feats"],
        "rows_raw": stats.get("rows_raw", 0), "rows": stats.get("rows", 0),
        "skipped": dict(stats["skipped"]),
        "unmapped": dict(stats["unmapped"].most_common(100)),
        "upos": {u: {"tokens": e["tokens"], "feats": dict(e["feats"])} for u, e in sorted(stats["upos"].items())},
        "treebanks": {t: dict(c) for t, c in sorted(stats["treebanks"].items())},
    }


def table_coverage(path: str) -> dict:
    """Per-UPOS rows and, per feature, the rows that state a real UD value in a parquet table
    (um_tables / ud_tables), i.e. what annotation filling can supply. A UniMorph code with
    no UD value ("UM:...") and the shadow/UNK columns do not count."""
    import pandas as pd
    from .ud_table import META_COLUMNS, SHADOW_PREFIX, UNMAPPED_PREFIX
    if not os.path.isfile(path):
        return {"rows": 0, "upos": {}}
    df = pd.read_parquet(path)
    features = [c for c in df.columns if c not in META_COLUMNS and c != "UNK" and not c.startswith(SHADOW_PREFIX)]
    upos_stats = {}
    for upos, part in df.groupby("upos", observed=True):
        feats = {}
        for col in features:
            cells = part[col].astype(object)
            real = int(cells.map(lambda v: isinstance(v, str) and not v.startswith(UNMAPPED_PREFIX)).sum())
            if real:
                feats[col] = real
        upos_stats[str(upos)] = {"rows": len(part), "feats": feats}
    return {"rows": len(df), "upos": upos_stats}


def load_coverage() -> dict:
    try:
        with open(COVERAGE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def update_coverage(lang: str, key: str, data: dict) -> None:
    """Set one language's `key` section, leaving everything else in the file."""
    coverage = load_coverage()
    coverage.setdefault(lang, {})[key] = data
    os.makedirs(os.path.dirname(COVERAGE_PATH), exist_ok=True)
    tmp = COVERAGE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(coverage, f, indent=1, sort_keys=True)
    os.replace(tmp, COVERAGE_PATH)


def write_overview() -> None:
    """One row per language: UD token coverage and table sizes."""
    coverage = load_coverage()
    header = ["language", "ud_tokens", "ud_pct_tokens_with_feats"] + [
        f"{u}:{f}%" for u, f in OVERVIEW_FEATURES] + ["um_rows", "ud_table_rows", "no_feats_no_um"]
    lines = ["\t".join(header)]
    for lang in sorted(coverage):
        entry = coverage[lang]
        ud = entry.get("ud")
        cells = [lang]
        if ud:
            cells += [str(ud["tokens"]), f"{ud['tokens_with_feats'] / max(ud['tokens'], 1) * 100:.1f}"]
            for upos, feat in OVERVIEW_FEATURES:
                u = ud["upos"].get(upos)
                cells.append(f"{u['feats'].get(feat, 0) / u['tokens'] * 100:.1f}" if u and u["tokens"] else "")
        else:
            cells += [""] * (2 + len(OVERVIEW_FEATURES))
        um_rows = entry.get("um_table", {}).get("rows")
        ud_rows = entry.get("ud_table", {}).get("rows")
        cells += ["" if um_rows is None else str(um_rows), "" if ud_rows is None else str(ud_rows)]
        none_ud = ud is not None and ud["tokens_with_feats"] == 0
        cells.append("yes" if none_ud and um_rows == 0 else "")
        lines.append("\t".join(cells))
    with open(OVERVIEW_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
