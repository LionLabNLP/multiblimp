"""Pair ids, loading of correct_swaps, and the benchmark directory built from them (sampling: multiblimp.downsample)."""
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multiblimp.downsample import (
    CATEGORICAL, LEMMA_CAP, MIN_PAIRS, N_ITEMS, Condition, balance_columns, balance_features, condition_roles, language_seed,
    select_pairs,
)

PAIR_COLUMNS = ["treebank", "sent_id", "sen_str", "wrong_sen_str", "swap_kind", "swap_idx", "swap_ufeat", "pair_key"]
ATTRACTOR_COLUMNS = ["attractors_congruent", "attractors_weakly_congruent", "attractors_incongruent", "attractors_span_len",
                     "has_attractors", "only_has_fully_congruent", "only_has_congruent", "only_has_weakly_congruent",
                     "only_has_incongruent"]
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


SLIM_IDENTITY = ["pair_id", "lang", "phenomenon", "deprel", "treebank", "sent_id", "sen_str", "wrong_sen_str", "sen",
                 "pair_key", "sen_hash", "swap_kind", "swap_idx", "swap_ufeat", "feature_vals", "leaf_id", "leaf_top1_acc",
                 "surface_status", "alt_phenomena", *ATTRACTOR_COLUMNS]
SLIM_DERIVED = ["idx_a", "idx_b", "word_order", "distance", "length", "pos_a", "pos_b", "attractors", "lemma_a", "lemma_b",
                "second_chance"]
SCORE_COLUMNS = ["pair_id", "lang", "phenomenon", "deprel", "treebank", "sen", "wrong_sen"]
# the pipeline calls the token list sen and the written sentences sen_str/wrong_sen_str; the benchmark calls the sentences sen/wrong_sen
BENCHMARK_NAMES = {"sen": "sen_tokens", "sen_str": "sen", "wrong_sen_str": "wrong_sen"}
FEATURES_DIR = "features"
META_DIR = "meta"


def slim_frame(df: pd.DataFrame, lang: str, cond: str, deprel: str) -> pd.DataFrame:
    """The benchmark table of one condition's pairs (a correct_swaps frame with pair_id): the sentences, the pair's
    identity and the markers the sampling balances; the pair's two role positions are idx_a/idx_b."""
    df = df.reset_index(drop=True)
    roles = condition_roles(cond, deprel)
    feats = balance_features(df, roles, cond)
    out = df.assign(lang=lang, phenomenon=cond, deprel=deprel)
    out = out[[c for c in SLIM_IDENTITY if c in out]].copy()
    out["treebank"] = out["treebank"].astype(str)
    out["sent_id"] = out["sent_id"].astype(str)
    for col in ("idx_a", "idx_b"):
        role = roles[0 if col == "idx_a" else 1]
        out[col] = pd.to_numeric(df.get(f"{role}_idx"), errors="coerce").astype("Int64")
    for col in SLIM_DERIVED[2:]:
        out[col] = feats[col]
    return out


def write_parquet_safe(df: pd.DataFrame, path) -> None:
    """to_parquet, with the mixed-type object columns of a union of conditions (bool in one, string in another) as strings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.copy()
    for col in df.columns[df.dtypes == object]:
        try:
            pa.array(df[col])
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            df[col] = df[col].map(lambda v: v if v is None or (isinstance(v, float) and v != v) else str(v))
    df.to_parquet(path, index=False)


def write_language(out_dir, lang: str, parts: list) -> None:
    """<out_dir>/<lang>.parquet (the slim table of every selected pair of the language, one row per pair),
    features/<lang>.parquet (the same pairs with all columns, joined by pair_id; sen_tokens is the token list) and meta/<lang>.json.
    parts: (frame with pair_id, cond, deprel, meta) for each condition."""
    slim = pd.concat([slim_frame(df, lang, cond, deprel) for df, cond, deprel, _ in parts], ignore_index=True)
    if not slim["pair_id"].is_unique:
        raise ValueError(f"{lang}: pair_id is not unique across the conditions")
    wide = pd.concat([df.assign(lang=lang, phenomenon=cond, deprel=deprel) for df, cond, deprel, _ in parts], ignore_index=True)
    first = ["pair_id", "lang", "phenomenon", "deprel"]
    wide = wide[first + [c for c in wide.columns if c not in first]]
    slim, wide = slim.rename(columns=BENCHMARK_NAMES), wide.rename(columns=BENCHMARK_NAMES)
    out_dir = str(out_dir)
    write_parquet_safe(slim, os.path.join(out_dir, f"{lang}.parquet"))
    write_parquet_safe(wide, os.path.join(out_dir, FEATURES_DIR, f"{lang}.parquet"))
    meta = {f"{cond}/{deprel}" if deprel else cond: meta for _, cond, deprel, meta in parts if meta}
    if meta:
        os.makedirs(os.path.join(out_dir, META_DIR), exist_ok=True)
        with open(os.path.join(out_dir, META_DIR, f"{lang}.json"), "w") as f:
            json.dump(meta, f, indent=1)


def load_benchmark(bench_dir, langs=None, phenomena=None, columns=SCORE_COLUMNS) -> pd.DataFrame:
    """The benchmark pairs: <bench_dir>/<Language>.parquet of every language (all columns with columns=None)."""
    available = {path.stem for path in Path(bench_dir).glob("*.parquet")}
    if langs and set(langs) - available:
        raise ValueError(f"not in the benchmark: {sorted(set(langs) - available)}")
    frames = []
    for path in sorted(Path(bench_dir).glob("*.parquet")):
        if langs and path.stem not in langs:
            continue
        df = pd.read_parquet(path, columns=columns)
        if phenomena:
            df = df[df["phenomenon"].isin(phenomena)]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no <Language>.parquet under {bench_dir}")
    out = pd.concat(frames, ignore_index=True)
    if phenomena and set(phenomena) - set(out["phenomenon"]):
        raise ValueError(f"no pairs for the phenomena {sorted(set(phenomena) - set(out['phenomenon']))}")
    return out


def cap_per_cond(pairs: pd.DataFrame, max_per_cond: int, seed: int = 0) -> pd.DataFrame:
    """At most max_per_cond pairs per (lang, phenomenon), in sha1(seed|pair_id) order: a quick, deterministic cap for
    test runs (eval_model.py --max_per_cond). The benchmark itself is sampled by build_benchmark."""
    order = pairs["pair_id"].map(lambda pid: _hash(f"{seed}|{pid}", 8))
    ranked = pairs.assign(_order=order).sort_values("_order")
    kept = ranked.groupby(["lang", "phenomenon"]).head(max_per_cond)
    return kept.drop(columns="_order").sort_values("pair_id").reset_index(drop=True)


MARKERS = (*CATEGORICAL, "second_chance", "surface_status")
DISTANCE_LABELS = ("1", "2", "3-4", "5-8", "9+")
LENGTH_LABELS = ("up to 40", "41-80", "81-140", "141-240", "241+")


def _clean(path: Path, columns, counts: Counter = None):
    """The given columns of a correct_swaps file with the surface strings applied and repeated sentence pairs
    dropped once; `counts` (if given) gains the pairs written, without a written form, identical and repeated."""
    df = pq.read_table(path, columns=columns).to_pandas()
    written = len(df)
    if (path.parent / SURFACE_FILE).exists():
        df = apply_surface(df, path.parent / SURFACE_FILE)
    unwritten = written - len(df)
    # a pair whose two written sentences coincide (a contraction that absorbs the swap) cannot be scored
    same = df["sen_str"] == df["wrong_sen_str"]
    df = df[~same]
    # the same sentence pair under several sentence ids (duplicated sentences, overlapping treebanks) counts once
    unique = df.drop_duplicates(["sen_str", "wrong_sen_str"]).drop_duplicates("pair_key").reset_index(drop=True)
    if counts is not None:
        counts.update({"written": written, "no_written_form": unwritten, "identical": int(same.sum()),
                       "repeated_in_condition": len(df) - len(unique)})
    return unique


def dedupe_across_conditions(frames: dict, phenomena: dict):
    """(frames without repeats, {name: {pair_key: other phenomena}}, number removed): a sentence pair (sen_str,
    wrong_sen_str) found in several conditions of a language stays only in the one with the fewest pairs (the rarer
    condition; ties by name). The pair that stays lists the other phenomena it was found in, and only those."""
    owners = defaultdict(list)
    for name, df in frames.items():
        for key in zip(df["sen_str"], df["wrong_sen_str"]):
            owners[key].append(name)
    size = {name: len(df) for name, df in frames.items()}
    home = {key: min(names, key=lambda n: (size[n], n)) for key, names in owners.items() if len(names) > 1}
    out, alternatives, removed = {}, {}, 0
    for name, df in frames.items():
        keys = list(zip(df["sen_str"], df["wrong_sen_str"]))
        drop = np.array([key in home and home[key] != name for key in keys], dtype=bool)
        removed += int(drop.sum())
        alternatives[name] = {
            pair_key: sorted({phenomena[n] for n in owners[key]} - {phenomena[name]})
            for pair_key, key in zip(df["pair_key"], keys) if key in home and home[key] == name}
        out[name] = df[~drop].reset_index(drop=True)
    return out, alternatives, removed


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
        elif marker == "length":
            values = values.map(lambda v: LENGTH_LABELS[int(v)])
        out[marker] = {k: int(v) for k, v in values.value_counts().items()}
    return out


def convert_benchmark(old_dir, out_dir, langs=None) -> dict:
    """Rewrite a benchmark in the old <cond>/<cond>_<deprel>/<Language>/ layout as the per-language files of
    build_benchmark; manifest.json and stats.json are copied. Returns {language: number of pairs}."""
    by_lang = defaultdict(list)
    for path, lang, cond, deprel in pair_files(old_dir, langs):
        by_lang[lang].append((path, cond, deprel))
    counts = {}
    for lang, files in sorted(by_lang.items()):
        parts = []
        for path, cond, deprel in files:
            df = pd.read_parquet(path)
            same = df["sen_str"] == df["wrong_sen_str"]
            if same.any():
                print(f"{lang} {cond}: dropped {int(same.sum())} pair(s) whose two sentences are identical", flush=True)
                df = df[~same]
            meta_path = path.parent / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            parts.append((df, cond, deprel, meta))
        write_language(out_dir, lang, parts)
        counts[lang] = sum(len(df) for df, *_ in parts)
        print(f"{lang}: {counts[lang]} pairs", flush=True)
    for name in ("manifest.json", "stats.json"):
        if os.path.exists(os.path.join(old_dir, name)) and str(old_dir) != str(out_dir):
            shutil.copy(os.path.join(old_dir, name), os.path.join(out_dir, name))
    return counts


def build_benchmark(pairs_dir, out_dir, langs=None, conds=None, n_items=N_ITEMS, lemma_cap=LEMMA_CAP, seed=0,
                    min_pairs=MIN_PAIRS):
    """Sample up to n_items pairs per language across all its conditions (multiblimp.downsample) from the
    correct_swaps under pairs_dir into out_dir as two files per language (write_language): <Language>.parquet with the
    sentences, the phenomenon (the condition, e.g. svNa) and the balance markers, and features/<Language>.parquet with all
    columns, both keyed by pair_id and with the surface strings applied; meta/<Language>.json holds the pipeline's
    meta.json of each condition without its HTML examples. manifest.json records the settings and the
    treebank counts before and after (and the languages left out for having fewer than min_pairs pairs); stats.json the per language x condition counts and marker distributions
    (the benchmark page, scripts/benchmark/render_page.py, is built from it)."""
    by_lang = defaultdict(list)
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        if pq.ParquetFile(path).metadata.num_rows:   # empty buckets are written without columns
            by_lang[lang].append((path, cond, deprel))

    cells, languages, before, after, n_pairs = [], {}, {}, {}, 0
    too_small, n_repeats, cleaning = {}, 0, Counter()
    for lang, files in sorted(by_lang.items()):
        conditions, kept_files, cleaned, phenomena = [], [], {}, {}
        for path, cond, deprel in files:
            name = f"{cond}/{deprel}" if deprel else cond
            names = set(pq.ParquetFile(path).schema_arrow.names)
            df = _clean(path, [c for c in balance_columns(condition_roles(cond, deprel)) + ["wrong_sen_str"] if c in names], cleaning)
            if len(df):
                cleaned[name], phenomena[name] = df, cond
        cleaned, alternatives, removed = dedupe_across_conditions(cleaned, phenomena)
        n_repeats += removed
        for path, cond, deprel in files:
            name = f"{cond}/{deprel}" if deprel else cond
            if name in cleaned and len(cleaned[name]):
                conditions.append(Condition(name, condition_roles(cond, deprel), balance_features(cleaned[name], condition_roles(cond, deprel), name)))
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
            cleaning["left_out_languages"] += available
            cells = [c for c in cells if c["lang"] != lang]
            print(f"{lang}: {total} pairs, left out", flush=True)
            continue
        n_pairs += total

        selected_rows, parts = [], []
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
            alt = alternatives[c.name]
            df["alt_phenomena"] = [alt.get(key, []) for key in df["pair_key"]]
            meta_path = path.parent / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta.pop("examples", None)
            parts.append((df, cond, deprel, meta))

        if parts:
            write_language(out_dir, lang, parts)
        chosen_all = pd.concat(selected_rows)
        languages[lang] = {
            "available": available, "selected": int(len(chosen_all)), "kept_all": available <= n_items, **limits,
            "conditions": sum(1 for r in selected_rows if len(r)),
            "repeats_removed": removed,
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
                  "; a sentence pair found in several conditions stays in the rarest one (alt_phenomena lists the others); a lemma takes at most lemma_cap of the sample, a sentence one pair (both relaxed if the quota is not met); pairs with an ambiguous swap value (feature_vals with '|') are drawn only once a condition has no other pair left and count fractionally for each of their values",
        "n_pairs": n_pairs,
        "ambiguous_available": sum(c["ambiguous"] for c in cells),
        "ambiguous_selected": sum(c["ambiguous_selected"] for c in cells),
        "min_pairs": min_pairs,
        "cross_condition_repeats_removed": n_repeats,
        "cleaning": dict(cleaning),
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
