"""Noun-phrase feature cache: one np_instances parquet per language (plus an empty
sentinel when a language has no NP heads, and optional np_type count CSVs),
shared by cache_treebank_features.py and the NP fit."""
import csv
import os
import traceback
from collections import Counter

import pyarrow as pa

from agreement_trees.pipeline_utils import (
    drop_empty_columns, parquet_current, sentinel_current, write_sentinel, write_table,
)
from multiblimp.config import NP_INSTANCES_DIR, RESOURCE_DIR
from multiblimp.languages import gblang2udlang
from multiblimp.script_log import log_cache_error
from agreement_trees.npa.np_types import build_np_data, build_np_data_streaming


def save_counts(counts: Counter, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = sum(counts.values())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["np_type", "count", "proportion"])
        for np_type, count in counts.most_common():
            writer.writerow([np_type, count, f"{count / total:.6f}"])


def read_counts(path: str) -> Counter:
    counts = Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[row["np_type"]] = int(row["count"])
    return counts


def np_paths(lang: str, report_dir: str | None) -> tuple[str, str, str | None]:
    """(instances parquet, empty sentinel, report CSV path or None) for `lang`."""
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    return (
        os.path.join(NP_INSTANCES_DIR, f"{cached_lang}.parquet"),
        os.path.join(NP_INSTANCES_DIR, f"{cached_lang}.empty.json"),
        os.path.join(report_dir, f"{cached_lang}.csv") if report_dir else None,
    )


def np_cache_status(lang: str, never_skip: bool) -> str | None:
    """"skipped" if `lang`'s instances parquet exists, "empty" if it was extracted
    and had no NP heads (sentinel), else None; a cache from another extraction
    version counts as missing. The optional report CSV plays no part."""
    if never_skip:
        return None
    instances_path, empty_marker, _ = np_paths(lang, None)
    if os.path.exists(instances_path):
        return "skipped" if parquet_current(instances_path) else None
    if os.path.exists(empty_marker) and sentinel_current(empty_marker):
        return "empty"
    return None


def cache_np_language(lang: str, report_dir: str | None, max_treebank_len: int | None,
                      streaming: bool, chunk_size: int,
                      fetch_all: bool = False) -> tuple[str, Counter | None]:
    """Extract one language's NP instances. Status is "processed", "empty" (no
    NOUN/PROPN/PRON heads, or none with a qualifying dependent) or "failed".
    Streaming and in-memory modes write the same on-disk shape. fetch_all fills
    missing annotation from UniMorph (ambiguous readings become "A,B")."""
    instances_path, empty_marker, csv_path = np_paths(lang, report_dir)
    try:
        if streaming:
            os.makedirs(NP_INSTANCES_DIR, exist_ok=True)
            counts = build_np_data_streaming(
                lang, RESOURCE_DIR, instances_path,
                max_treebank_len=max_treebank_len, chunk_size=chunk_size, fetch_all=fetch_all,
            )
        else:
            counts, df = build_np_data(lang, RESOURCE_DIR, max_treebank_len=max_treebank_len, fetch_all=fetch_all)
    except Exception as e:
        print(f"  FAILED (np): {lang}", flush=True)
        traceback.print_exc()
        log_cache_error("np", lang, f"{type(e).__name__}: {e}", traceback.format_exc())
        return "failed", None

    os.makedirs(NP_INSTANCES_DIR, exist_ok=True)
    if counts:
        if csv_path:
            save_counts(counts, csv_path)
        if not streaming:
            write_table(drop_empty_columns(pa.Table.from_pandas(df, preserve_index=False)), instances_path)
    else:
        print(f"  no NOUN/PROPN/PRON heads found for {lang}", flush=True)
        if os.path.exists(instances_path):
            os.remove(instances_path)

    if os.path.exists(instances_path):
        if os.path.exists(empty_marker):
            os.remove(empty_marker)
    else:
        write_sentinel(empty_marker, {"has_np_heads": bool(counts)})
    return ("processed", counts) if counts else ("empty", None)
