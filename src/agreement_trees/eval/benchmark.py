"""Pair ids, loading of correct_swaps, and the benchmark directory built from them (sampling: multiblimp.downsample)."""
import hashlib
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multiblimp.downsample import (
    CATEGORICAL, LEMMA_CAP, MIN_PAIRS, N_ITEMS, Condition, balance_columns, balance_features, condition_roles, language_seed,
    select_pairs,
)

PAIR_COLUMNS = ["treebank", "sent_id", "sen_str", "wrong_sen_str", "swap_kind", "swap_idx", "swap_ufeat", "pair_key"]
SURFACE_FILE = "surface_strings.parquet"
SURFACE_COLUMNS = ["pair_key", "sen_str", "wrong_sen_str", "surface_status"]


def _hash(text: str, length: int) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def pair_id(lang: str, cond: str, deprel: str, pair_key: str) -> str:
    """Stable id of a pair: lang, condition, deprel and pair_key (treebank, sentence id, swapped token,
    swapped feature, swapped form, see add_sentence_strings)."""
    return _hash(f"{lang}|{cond}|{deprel}|{pair_key}", 16)


def pair_files(pairs_dir, langs=None, conds=None):
    """(path, lang, cond, deprel) of every <cond>/<cond>_<deprel>/<lang>/correct_swaps.parquet under pairs_dir."""
    for path in sorted(Path(pairs_dir).glob("*/*/*/correct_swaps.parquet")):
        lang, target, family = path.parent.name, path.parent.parent.name, path.parent.parent.parent.name
        # sva: <cond>/<cond>_<deprel>; npa: npa/<role pair>_<feature>, where the target is the condition
        cond, deprel = (family, target[len(family) + 1:]) if target.startswith(family + "_") else (target, "")
        if (langs and lang not in langs) or (conds and cond not in conds):
            continue
        yield path, lang, cond, deprel


def apply_surface(df: pd.DataFrame, surface_path) -> pd.DataFrame:
    """Replace sen_str/wrong_sen_str of the rows listed in a surface_strings.parquet and drop the dropped ones."""
    surface = pd.read_parquet(surface_path).drop_duplicates("pair_key").set_index("pair_key")
    status = df["pair_key"].map(surface["surface_status"])
    df = df[~status.fillna("").str.startswith("dropped")].copy()
    hit = df["pair_key"].isin(surface.index)
    for col in ("sen_str", "wrong_sen_str"):
        df.loc[hit, col] = df.loc[hit, "pair_key"].map(surface[col])
    return df


def load_pairs(pairs_dir, langs=None, conds=None) -> pd.DataFrame:
    """The correct_swaps of every <cond>/<cond>_<deprel>/<lang>/ under pairs_dir, reduced to the scoring
    columns plus pair_id, lang, cond, deprel and sen_hash (the original sentence's, to spot UD edits)."""
    frames = []
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        if pq.ParquetFile(path).metadata.num_rows == 0:  # empty buckets are written without columns
            continue
        df = pd.read_parquet(path, columns=PAIR_COLUMNS)
        if (path.parent / SURFACE_FILE).exists():
            df = apply_surface(df, path.parent / SURFACE_FILE)
        for col in ("treebank", "sent_id"):
            df[col] = df[col].astype(str)
        df.insert(0, "lang", lang)
        df.insert(1, "cond", cond)
        df.insert(2, "deprel", deprel)
        df.insert(0, "pair_id", [pair_id(lang, cond, deprel, key) for key in df["pair_key"]])
        df["sen_hash"] = [_hash(sen, 12) for sen in df["sen_str"]]
        frames.append(df.drop(columns=["pair_key"]))
    if not frames:
        raise FileNotFoundError(f"no correct_swaps.parquet with sentence strings under {pairs_dir}")
    pairs = pd.concat(frames, ignore_index=True)
    return pairs.drop_duplicates("pair_id").reset_index(drop=True)


def cap_per_cond(pairs: pd.DataFrame, max_per_cond: int, seed: int = 0) -> pd.DataFrame:
    """At most max_per_cond pairs per (lang, cond), in sha1(seed|pair_id) order: a quick, deterministic cap for
    test runs (eval_model.py --max_per_cond). The benchmark itself is sampled by build_benchmark."""
    order = pairs["pair_id"].map(lambda pid: _hash(f"{seed}|{pid}", 8))
    ranked = pairs.assign(_order=order).sort_values("_order")
    kept = ranked.groupby(["lang", "cond"]).head(max_per_cond)
    return kept.drop(columns="_order").sort_values("pair_id").reset_index(drop=True)


MARKERS = (*CATEGORICAL, "second_chance", "surface_status")
DISTANCE_LABELS = ("1", "2", "3-4", "5-8", "9+")


def _clean(path: Path, columns):
    """The given columns of a correct_swaps file with the surface strings applied and repeated sentence pairs
    dropped once."""
    df = pq.read_table(path, columns=columns).to_pandas()
    if (path.parent / SURFACE_FILE).exists():
        df = apply_surface(df, path.parent / SURFACE_FILE)
    # the same sentence pair under several sentence ids (duplicated sentences, overlapping treebanks) counts once
    return df.drop_duplicates(["sen_str", "wrong_sen_str"]).drop_duplicates("pair_key").reset_index(drop=True)


def _ambiguous(features: pd.DataFrame) -> int:
    """Pairs whose swap value is ambiguous (feature_vals like "3 -> 1|2": the swapped form also spells another value)."""
    return int(features["feature_vals"].str.contains("|", regex=False).sum())


def _distribution(features: pd.DataFrame) -> dict:
    out = {}
    for marker in MARKERS:
        if marker not in features:
            continue
        values = features[marker].astype(str)
        if marker == "distance":
            values = values.map(lambda v: DISTANCE_LABELS[int(v)])
        out[marker] = {k: int(v) for k, v in values.value_counts().items()}
    return out


def build_benchmark(pairs_dir, out_dir, langs=None, conds=None, n_items=N_ITEMS, lemma_cap=LEMMA_CAP, seed=0,
                    min_pairs=MIN_PAIRS):
    """Sample up to n_items pairs per language across all its conditions (multiblimp.downsample) from the
    correct_swaps under pairs_dir into out_dir, keeping the <cond>/<cond>_<deprel>/<lang>/ layout.

    Each file holds the chosen rows with all their columns plus pair_id and sen_hash, with the surface strings
    applied; meta.json is copied without its HTML examples. manifest.json records the settings and the
    treebank counts before and after (and the languages left out for having fewer than min_pairs pairs); stats.json the per language x condition counts and marker distributions
    (the benchmark page, scripts/benchmark/render_page.py, is built from it)."""
    by_lang = defaultdict(list)
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        if pq.ParquetFile(path).metadata.num_rows:   # empty buckets are written without columns
            by_lang[lang].append((path, cond, deprel))

    cells, languages, before, after, n_pairs = [], {}, {}, {}, 0
    too_small = {}
    for lang, files in sorted(by_lang.items()):
        conditions, kept_files = [], []
        for path, cond, deprel in files:
            name = f"{cond}/{deprel}" if deprel else cond
            roles = condition_roles(cond, deprel)
            names = set(pq.ParquetFile(path).schema_arrow.names)
            df = _clean(path, [c for c in balance_columns(roles) + ["wrong_sen_str"] if c in names])
            if not len(df):
                continue
            conditions.append(Condition(name, roles, balance_features(df, roles, name)))
            kept_files.append((path, cond, deprel))
        if not conditions:
            continue
        available = sum(len(c) for c in conditions)
        if available <= n_items:
            chosen = {c.name: np.arange(len(c)) for c in conditions}
            limits = {"lemma_cap_count": None, "sentence_limit": None}
        else:
            chosen, limits = select_pairs(conditions, n_items, lemma_cap, np.random.default_rng(language_seed(lang, seed)))
        total = sum(len(v) for v in chosen.values())
        if total < min_pairs:
            too_small[lang] = total
            cells = [c for c in cells if c["lang"] != lang]
            print(f"{lang}: {total} pairs, left out", flush=True)
            continue
        n_pairs += total

        selected_rows = []
        for (path, cond, deprel), c in zip(kept_files, conditions):
            rows = chosen[c.name]
            picked = c.features.iloc[rows]
            keep_ids = {pair_id(lang, cond, deprel, key) for key in picked["pair_key"]}
            cells.append({"lang": lang, "cond": cond, "deprel": deprel, "available": len(c), "selected": len(rows),
                          "ambiguous": _ambiguous(c.features), "ambiguous_selected": _ambiguous(picked),
                          "available_markers": _distribution(c.features), "selected_markers": _distribution(picked)})
            selected_rows.append(picked)
            before[f"{lang}/{cond}"] = {tb: int(n) for tb, n in c.features["treebank"].value_counts().items()}
            after[f"{lang}/{cond}"] = {tb: int(n) for tb, n in picked["treebank"].value_counts().items()}
            if not keep_ids:
                continue
            df = pd.read_parquet(path)
            if (path.parent / SURFACE_FILE).exists():
                df = apply_surface(df, path.parent / SURFACE_FILE)
            df.insert(0, "pair_id", [pair_id(lang, cond, deprel, key) for key in df["pair_key"]])
            df["sen_hash"] = [_hash(sen, 12) for sen in df["sen_str"]]
            df = df[df["pair_id"].isin(keep_ids)].drop_duplicates("pair_id")

            target = os.path.join(out_dir, *path.relative_to(pairs_dir).parts[:-1])
            os.makedirs(target, exist_ok=True)
            df.to_parquet(os.path.join(target, "correct_swaps.parquet"), index=False)
            meta_path = path.parent / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta.pop("examples", None)
                with open(os.path.join(target, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=1)

        chosen_all = pd.concat(selected_rows)
        languages[lang] = {
            "available": available, "selected": int(len(chosen_all)), "kept_all": available <= n_items, **limits,
            "conditions": sum(1 for r in selected_rows if len(r)),
            "treebanks": int(chosen_all["treebank"].nunique()),
            "head_lemmas": int(chosen_all["lemma_a"].nunique()), "child_lemmas": int(chosen_all["lemma_b"].nunique()),
        }
        print(f"{lang}: {len(chosen_all)} of {available} pairs", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "built": date.today().isoformat(),
        "source": os.path.abspath(pairs_dir),
        "n_items": n_items,
        "lemma_cap": lemma_cap,
        "seed": seed,
        "method": "per language: conditions served round-robin; within a condition a greedy draw with weight "
                  "prod(1 / (1 + times the value is already sampled)) over " + ", ".join((*CATEGORICAL, "lemmas")) +
                  "; a lemma takes at most lemma_cap of the sample, a sentence one pair (both relaxed if the quota is not met); pairs with an ambiguous swap value (feature_vals with '|') are drawn only once a condition has no other pair left and count fractionally for each of their values",
        "n_pairs": n_pairs,
        "ambiguous_available": sum(c["ambiguous"] for c in cells),
        "ambiguous_selected": sum(c["ambiguous_selected"] for c in cells),
        "min_pairs": min_pairs,
        "languages_left_out": too_small,
        "treebank_counts_before": before,
        "treebank_counts_after": after,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({"built": manifest["built"], "n_items": n_items, "lemma_cap": lemma_cap, "seed": seed,
                   "languages": languages, "cells": cells}, f)
    return manifest
