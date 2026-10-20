"""Attractor features of the minimal pairs, computed from the slot columns extraction stored.

create_pairs and create_npa_pairs_for_target_col call add_attractors_if_stored on each correct_swaps frame once
the pairs are built; this script recomputes the columns of existing correct_swaps.parquet files in place (after
changing the bucketing). Adds, per row, for the pair's swap feature (a layered one like Number[erg] counts as its
plain feature):
  attractors_congruent / attractors_weakly_congruent / attractors_incongruent   slot counts (attractor_bucket)
  attractors_span_len                                                           tokens between controller and target
  has_attractors, only_has_fully_congruent, only_has_congruent, only_has_weakly_congruent, only_has_incongruent
                                                                                flags over those counts (FLAG_COLS)
against the controller's own value, which is the child's in a verb phrase and the unswapped role's in a noun
phrase. NA where the controller has no value; the output columns are replaced on a re-run.

Usage (any directory):
  python3 attractors.py                                  every correct_swaps.parquet under output/minimal_pairs
  python3 attractors.py --conds svNa npa --langs Basque
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "."))

from agreement_trees.extract.process_treebank import ATTRACTOR_BUCKETS, attractor_bucket, slot_folds_into
from agreement_trees.fit.decision_tree import MISSING
from multiblimp.config import OUTPUT_MINIMAL_PAIRS_DIR

COUNT_COLS = [f"attractors_{bucket}" for bucket in ATTRACTOR_BUCKETS] + ["attractors_span_len"]
FLAG_COLS = ["has_attractors", "only_has_fully_congruent", "only_has_congruent", "only_has_weakly_congruent",
             "only_has_incongruent"]
OUTPUT_COLS = COUNT_COLS + FLAG_COLS


def attractor_flags(full: pd.Series, weak: pd.Series, incongruent: pd.Series) -> dict:
    """The flags over the bucket counts (NA where a count is NA). "Congruent" means full or partial match;
    each only_* flag also needs at least one attractor of its kind."""
    return {
        "has_attractors": (full + weak + incongruent) > 0,
        "only_has_fully_congruent": (full > 0) & (weak == 0) & (incongruent == 0),
        "only_has_congruent": ((full + weak) > 0) & (incongruent == 0),
        "only_has_weakly_congruent": (weak > 0) & (full == 0) & (incongruent == 0),
        "only_has_incongruent": (incongruent > 0) & (full == 0) & (weak == 0),
    }


def parse_slot(entry: str):
    """(token id, UPOS, key, value, lexicon-filled) of a stored entry; key and value are None for a token
    without a slot."""
    token_id, upos, *rest = entry.split(":", 2)
    if not rest:
        return int(token_id), upos, None, None, False
    key, value = rest[0].split("=", 1)
    filled = value.endswith("*")
    return int(token_id), upos, key, value.rstrip("*"), filled


def classify_entries(entries, feat: str, ref) -> tuple[dict, int]:
    """({bucket: count}, span length) of one row's stored entries for `feat` against the controller's value."""
    counts = dict.fromkeys(ATTRACTOR_BUCKETS, 0)
    ids = set()
    for entry in entries if entries is not None else ():
        token_id, _, key, value, filled = parse_slot(entry)
        ids.add(token_id)
        if key is not None and slot_folds_into(feat, key, filled):
            counts[attractor_bucket(ref, value)] += 1
    return counts, len(ids)


def pair_context(path: str, pairs_dir: str) -> tuple[str, str | tuple]:
    """(slots column, controller) of a correct_swaps file from its place in the pairs layout:
    <cond>/<cond>_<deprel>/<lang>/ (verb phrase: the child relation is the controller) or
    npa/<ROLE1>-<ROLE2>_<F>/<lang>/ (noun phrase: the controller is whichever role was not swapped, per row,
    so the controller is the role pair)."""
    parts = os.path.relpath(path, pairs_dir).split(os.sep)
    if parts[0] == "npa":
        pair = parts[1].rsplit("_", 1)[0]
        return f"{pair}_attractor_slots", tuple(pair.split("-"))
    deprel = parts[1].split("_", 1)[1]
    return f"{deprel}_attractor_slots", deprel


def controller_roles(df: pd.DataFrame, controller) -> np.ndarray:
    """The controller role of each row: the given relation, or for a role pair the role the row did not swap
    (None where neither or both were swapped)."""
    if isinstance(controller, str):
        return np.full(len(df), controller, dtype=object)
    roles = np.full(len(df), None, dtype=object)
    first, second = controller
    swapped_first = df[f"swap_{first}"].notna().to_numpy() if f"swap_{first}" in df else np.zeros(len(df), bool)
    swapped_second = df[f"swap_{second}"].notna().to_numpy() if f"swap_{second}" in df else np.zeros(len(df), bool)
    roles[swapped_first & ~swapped_second] = second
    roles[swapped_second & ~swapped_first] = first
    return roles


def add_attractor_features(df: pd.DataFrame, slots_col: str, controller) -> pd.DataFrame:
    """df with the OUTPUT_COLS columns (replacing existing ones), per row for its swap_ufeat."""
    df = df.drop(columns=[c for c in OUTPUT_COLS if c in df.columns])
    values = {col: [pd.NA] * len(df) for col in COUNT_COLS}
    features = df["swap_ufeat"].astype(str).str.split("[").str[0].to_numpy()
    roles = controller_roles(df, controller)
    slots = df[slots_col].to_numpy()
    for role in {r for r in roles if r is not None}:
        for feat in np.unique(features[roles == role]):
            column = df.get(f"{role}_{feat}")
            if column is None:
                continue
            refs = column.to_numpy()
            for i in ((roles == role) & (features == feat)).nonzero()[0]:
                ref = refs[i]
                if pd.isna(ref) or ref == MISSING:
                    continue
                counts, span_len = classify_entries(slots[i], feat, ref)
                for bucket in ATTRACTOR_BUCKETS:
                    values[f"attractors_{bucket}"][i] = counts[bucket]
                values["attractors_span_len"][i] = span_len
    new = pd.DataFrame({col: pd.array(vals, dtype="Int16") for col, vals in values.items()}, index=df.index)
    flags = attractor_flags(*(new[f"attractors_{bucket}"] for bucket in ATTRACTOR_BUCKETS))
    return pd.concat([df, new, pd.DataFrame(flags, index=df.index).astype("boolean")], axis=1)


def add_attractors_if_stored(df: pd.DataFrame, slots_col: str, controller) -> pd.DataFrame:
    """add_attractor_features, or df unchanged (with a note) if the frame has no slots column, i.e. its
    extraction cache predates the attractor slots."""
    if slots_col not in df.columns:
        print(f"no {slots_col} in the pairs frame: attractor features skipped (re-extract the cache)")
        return df
    return add_attractor_features(df, slots_col, controller)


def process_file(path: str, pairs_dir: str) -> str:
    df = pd.read_parquet(path)
    if not len(df):
        return "empty"
    slots_col, controller = pair_context(path, pairs_dir)
    if slots_col not in df.columns:
        return f"skipped: no {slots_col} (pairs built before attractor slots)"
    df = add_attractor_features(df, slots_col, controller)
    df.to_parquet(path)
    return f"{len(df)} rows (controller: {controller if isinstance(controller, str) else 'unswapped role of ' + '-'.join(controller)})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    parser.add_argument("--conds", nargs="+", default=None, help="condition ids (svNa, ...) and/or npa")
    parser.add_argument("--langs", nargs="+", default=None)
    args = parser.parse_args()

    for path in sorted(glob.glob(os.path.join(args.pairs_dir, "**", "correct_swaps.parquet"), recursive=True)):
        parts = os.path.relpath(path, args.pairs_dir).split(os.sep)
        if (args.conds and parts[0] not in args.conds) or (args.langs and parts[-2] not in args.langs):
            continue
        print(os.path.relpath(path, args.pairs_dir), "->", process_file(path, args.pairs_dir))


if __name__ == "__main__":
    main()
