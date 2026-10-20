"""Pair ids, loading and downsampling of correct_swaps, and the benchmark directory built from them."""
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

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


def downsample(pairs: pd.DataFrame, max_per_cond: int = 1000, seed: int = 0) -> pd.DataFrame:
    """At most max_per_cond pairs per (lang, cond), taken round-robin over treebanks in a pair_id-hash order.

    Deterministic, and independent of the other rows: a pair's position depends only on its own id and
    its rank among its treebank's ids, so data changes only move the pairs next to the changed ones.
    """
    order = pairs["pair_id"].map(lambda pid: _hash(f"{seed}|{pid}", 8))
    ranked = pairs.assign(_order=order)
    # the same sentence pair under several sentence ids (duplicated sentences, overlapping treebanks) counts once
    ranked = ranked.sort_values("_order").drop_duplicates(["lang", "cond", "sen_str", "wrong_sen_str"])
    ranked["_rank"] = ranked.groupby(["lang", "cond", "treebank"])["_order"].rank(method="first")
    ranked = ranked.sort_values(["lang", "cond", "_rank", "_order"])
    kept = ranked.groupby(["lang", "cond"]).head(max_per_cond)
    return kept.drop(columns=["_order", "_rank"]).sort_values("pair_id").reset_index(drop=True)


def build_benchmark(pairs_dir, out_dir, langs=None, conds=None, max_per_cond=1000, seed=0):
    """Downsample the correct_swaps under pairs_dir into out_dir, keeping the <cond>/<cond>_<deprel>/<lang>/ layout.

    Each file holds the kept rows with all their columns plus pair_id and sen_hash, with the surface strings
    applied; meta.json is copied without its HTML examples. manifest.json records the settings and the
    treebank counts before and after."""
    pairs = load_pairs(pairs_dir, langs, conds)
    kept = downsample(pairs, max_per_cond, seed)
    print(f"{len(kept)} of {len(pairs)} pairs kept")

    # one source file at a time: the full frames are ~1,350 columns wide
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        keep_ids = set(kept.loc[(kept["lang"] == lang) & (kept["cond"] == cond) & (kept["deprel"] == deprel), "pair_id"])
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

    def counts(df):
        return {
            f"{lang}/{cond}": {tb: int(n) for tb, n in grp["treebank"].value_counts().items()}
            for (lang, cond), grp in df.groupby(["lang", "cond"])
        }

    manifest = {
        "built": date.today().isoformat(),
        "source": os.path.abspath(pairs_dir),
        "max_per_cond": max_per_cond,
        "seed": seed,
        "method": "round-robin over treebanks in sha1(seed|pair_id) order",
        "n_pairs": len(kept),
        "treebank_counts_before": counts(pairs),
        "treebank_counts_after": counts(kept),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
