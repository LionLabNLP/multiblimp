import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

sys.path.append("../../src/")

from sva_trees.conditions import CONDITIONS, FEATURES as KNOWN_FEATURES
from word_order.process_treebank import values_overlap, slot_suffixes
from multiblimp.languages import is_treebank_row_excluded
from multiblimp.agreement_pipeline_utils import wilson_lower_bound, passes_agreement_bar

# prefix -> (word_order cache dir, child deprel). One "Na" condition per
# family is enough to read off word_order_dir()/deprel -- svNa/svGa/svPa all
# point at the same cache (see sva_trees.conditions._build_conditions: only
# feature_suffix differs), because extract_node_features writes every
# attested feature into head_{Feat}/{deprel}_{Feat} columns on every run,
# not just the ones the condition being run actually asked for. So these
# caches already hold everything this scan needs -- no pipeline rerun.
FAMILIES = {
    prefix: (CONDITIONS[f"{prefix}Na"].word_order_dir(), CONDITIONS[f"{prefix}Na"].deprel)
    for prefix in ("sv", "sp", "sa", "ov", "op", "oa", "iov", "iop", "ioa")
}
KNOWN_FEATURE_NAMES = set(KNOWN_FEATURES.values())  # {"Number", "Gender", "Person"}

# obj/iobj agreement is marked in a per-argument BRACKETED feature (e.g.
# head_Number[obj], head_Number[erg] -- see process_treebank.
# resolve_layered_head_key), never in the plain head_{feat}: plain
# head_{feat} for these deprels is the SUBJECT's own merged value (via
# merge_subj_layered_feats), so comparing it against obj_{feat}/iobj_{feat}
# produces false "agreement" purely from both sides defaulting to the same
# common value (typically 3rd person / singular), not real signal -- this
# project's own Georgian ovNa run hit exactly this (see
# process_treebank._resolve_layered_head_val's docstring: head_obj_Person_
# agreement was "Yes" 2442/2442 times over the plain feature, including rows
# where the real per-argument marker was missing entirely). These deprels
# NEVER fall back to plain at all -- see _resolve_head_series, used
# unconditionally for them in scan_language.
#
# nsubj is NOT in this set, but still needs bracket awareness: merge_subj_
# layered_feats only merges [subj]/[nsubj]-suffixed keys into the plain
# feature, never case-named ones like [erg]/[abs], so an ergative-alignment
# language can leave plain head_{feat} almost entirely null (e.g. Basque's
# Number) while the real signal sits in head_{feat}[erg]/[abs]. scan_language
# handles nsubj as its own case: plain FIRST, bracket fallback only where
# plain is null -- the same order process_treebank.extract_instances already
# uses for its real agreement-label column.
LAYERED_DEPRELS = {"obj", "iobj"}

OUT_DIR = "../../sva_exploration"
OUT_CSV = os.path.join(OUT_DIR, "agreement_candidates.csv")
OUT_LANG_CSV = os.path.join(OUT_DIR, "agreement_candidates_by_lang.csv")
OUT_CONFIG = "../../resources/sva_agreement_config.json"

_FEAT_RE = re.compile(r"^[A-Z][a-zA-Z]*$")


_BRACKET_RE = re.compile(r"^head_([A-Z][a-zA-Z]*)\[[^\]]+\]$")


def _feat_cols(schema_names: set, deprel: str) -> list[str]:
    """Every morphological-feature suffix jointly present on the head and
    child side of this parquet's schema. Anchoring to ^[A-Z][a-zA-Z]*$ (UD
    feature names are always capitalized, e.g. "Number", "VerbForm") is what
    excludes every structural column extract_node_features also writes for
    "head"/deprel prefixes -- lowercase ones (form, lemma, pos, upos, xpos,
    dir, idx, head_deprel, head_pos, grandhead_deprel) and multi-underscore
    ones (child-deprel_*, child-feat_*, under_*, and the condition's own
    head_{deprel}_{Feat}_agreement column) -- without an explicit denylist,
    same trick np_types/agreement_candidates.py uses for NPA's role-pair
    columns.

    The head side also counts a feature present ONLY as a bracketed
    head_{X}[...] column (e.g. head_Number[obj] or, for an ergative-
    alignment nsubj like Basque's, head_Number[erg]/head_Number[abs], with
    no plain head_Number at all) -- otherwise a feature exclusively used
    for polypersonal/ergative argument-marking, with no independent plain-
    head channel, would never surface as a candidate at all. Checked for
    every deprel, not just LAYERED_DEPRELS: nsubj can have bracket-only
    features too (see scan_language's own nsubj handling -- plain first,
    bracket fallback -- for why nsubj isn't in LAYERED_DEPRELS despite
    needing bracket awareness here).
    """
    plain_head_feats = {c[len("head_"):] for c in schema_names
                        if c.startswith("head_") and _FEAT_RE.match(c[len("head_"):])}
    head_feats = set(plain_head_feats)
    head_feats |= {m.group(1) for c in schema_names if (m := _BRACKET_RE.match(c))}
    child_prefix = f"{deprel}_"
    child_feats = {c[len(child_prefix):] for c in schema_names
                   if c.startswith(child_prefix) and _FEAT_RE.match(c[len(child_prefix):])}
    return sorted(head_feats & child_feats)


def _resolve_head_series(df: pd.DataFrame, feat: str, deprel: str, schema_set: set) -> pd.Series:
    """Vectorized equivalent of process_treebank._resolve_layered_head_val
    for deprel in LAYERED_DEPRELS: the per-argument bracketed value of
    `feat` on the head, resolved in the same priority order slot_suffixes
    documents (relation-named bracket first, e.g. head_Number[obj]/
    head_Number[io], then the child's own Case-named bracket, e.g.
    head_Number[erg]) -- never the plain head_{feat} (see LAYERED_DEPRELS).

    Grouped by the child's Case value (a handful of distinct values even for
    a case-rich language) rather than resolved row by row: every row sharing
    one Case value resolves through the exact same ordered suffix list, so
    the per-suffix column lookup/coalesce can run once per group instead of
    once per row.
    """
    bracket_cols = {c for c in schema_set if c.startswith(f"head_{feat}[") and c.endswith("]")}
    if not bracket_cols:
        return pd.Series([None] * len(df), index=df.index, dtype=object)

    case_col = f"{deprel}_Case"
    cases = df[case_col] if case_col in df.columns else pd.Series([None] * len(df), index=df.index)

    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    unresolved = pd.Series(True, index=df.index)
    for case_val in [None] + sorted(cases.dropna().unique().tolist()):
        mask = unresolved & (cases.isna() if case_val is None else cases == case_val)
        if not mask.any():
            continue
        for suffix in slot_suffixes(deprel, case_val):
            col = f"head_{feat}[{suffix}]"
            if col not in bracket_cols:
                continue
            hit = mask & df[col].notna()
            if hit.any():
                result.loc[hit] = df.loc[hit, col]
                unresolved.loc[hit] = False
                mask = mask & unresolved
            if not mask.any():
                break
    return result


def _pair_counts(h: pd.Series, c: pd.Series) -> tuple[int, int, int, int, int, int]:
    """(yes, no, head_only, child_only, neither, total) for one feature's
    head/child value series. Unlike NPA's np_instances (where a row may pair
    up roles that don't actually co-occur), every row here already IS one
    head-child instance by construction (extract_instances only emits a row
    once all of the target's required deprels were found under that head) --
    so total is simply len(h), and "neither" is just the leftover after
    yes/no/head_only/child_only, no separate co-occurrence check needed.
    """
    total = len(h)
    both = h.notna() & c.notna()
    if not both.any():
        head_only = int((h.notna() & c.isna()).sum())
        child_only = int((h.isna() & c.notna()).sum())
        return 0, 0, head_only, child_only, total - head_only - child_only, total

    # astype(str): head_col/child_col land as pandas Categorical from
    # parquet with independent (differently-ordered, differently-sized)
    # category sets per column, which pandas refuses to compare directly
    # ("Categoricals can only be compared if 'categories' are the same").
    h_b, c_b = h[both].astype(str), c[both].astype(str)
    eq = h_b == c_b
    # values_overlap only matters for comma-valued features (e.g. coordinated
    # "Masc,Fem") -- rare, so only re-check those rows with the slower
    # per-row set-overlap comparison instead of applying it to every row.
    has_comma = h_b.str.contains(",", regex=False) | c_b.str.contains(",", regex=False)
    if has_comma.any():
        idx = has_comma[has_comma].index
        eq = eq.copy()
        eq.loc[idx] = [values_overlap(h_b[i], c_b[i]) for i in idx]

    yes = int(eq.sum())
    no = int(both.sum() - yes)
    head_only = int((h.notna() & c.isna()).sum())
    child_only = int((h.isna() & c.notna()).sum())
    neither = total - yes - no - head_only - child_only
    return yes, no, head_only, child_only, neither, total


def scan_language(parquet_path: str, deprel: str, stats: dict) -> str | None:
    """Update `stats` (feat -> {"lang_counts": {lang: Counter}}) in place from
    one language's word_order cache parquet. Mirrors npa/agreement_candidates
    .scan_language: schema-only check first, then a single column-projected
    read covering every candidate feature at once (not one read per feature).

    The cache parquet holds EVERY treebank, selected and excluded alike
    (sva_trees.pipeline.Pipeline._build_df -- one unified per-language
    extraction, split by "treebank" at read time rather than two separately-
    cached parquets). Excluded rows are dropped here before counting: this
    scan is meant to answer "is this feature worth a real condition on the
    treebanks the pipeline actually trains on", and excluded treebanks
    (learner data, code-switching, spoken transcripts, ...) are exactly the
    ones whose agreement signal shouldn't count toward that.
    """
    lang = os.path.basename(parquet_path)[: -len(".parquet")]
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    schema_set = set(schema_names)
    feats = _feat_cols(schema_set, deprel)
    if not feats:
        return None

    layered = deprel in LAYERED_DEPRELS
    nsubj_fallback = deprel == "nsubj"
    needs_brackets = layered or nsubj_fallback
    wanted = {f"{deprel}_{feat}" for feat in feats}
    for feat in feats:
        if needs_brackets:
            wanted.update(c for c in schema_set if c.startswith(f"head_{feat}["))
        if not layered:
            wanted.add(f"head_{feat}")
    if needs_brackets:
        wanted.add(f"{deprel}_Case")
    if "treebank" in schema_set:
        wanted.add("treebank")
    df = pd.read_parquet(parquet_path, columns=[c for c in wanted if c in schema_set])
    if "treebank" in df.columns:
        # astype(bool): "treebank" is categorical dtype, and Series.map on a
        # categorical Series returns a categorical result -- "~" on that
        # raises TypeError rather than negating it (see sva_trees.pipeline.
        # Pipeline._split_excluded's identical fix). is_treebank_row_excluded
        # checks the treebank's OWN embedded language, not this scan's `lang`
        # -- see multiblimp.languages._split_treebank_row's docstring for why
        # (a few bilingual/code-switched treebanks are bundled into a
        # different language's own cache than the one their
        # excluded_treebanks entry is registered under).
        mask = df["treebank"].map(is_treebank_row_excluded).astype(bool)
        df = df[~mask]

    for feat in feats:
        if layered:
            head_series = _resolve_head_series(df, feat, deprel, schema_set)
        elif nsubj_fallback:
            bracket = _resolve_head_series(df, feat, deprel, schema_set)
            plain = df[f"head_{feat}"].astype(object) if f"head_{feat}" in df.columns else bracket
            head_series = plain.where(plain.notna(), bracket)
        else:
            head_series = df[f"head_{feat}"]
        yes, no, head_only, child_only, neither, total = _pair_counts(
            head_series, df[f"{deprel}_{feat}"]
        )
        lc = stats[feat]["lang_counts"][lang]
        lc["yes"] += yes
        lc["no"] += no
        lc["head_only"] += head_only
        lc["child_only"] += child_only
        lc["neither"] += neither
        lc["total"] += total
    return lang


def build_summary(stats: dict, family: str) -> pd.DataFrame:
    rows = []
    for feat, entry in stats.items():
        yes = sum(c["yes"] for c in entry["lang_counts"].values())
        no = sum(c["no"] for c in entry["lang_counts"].values())
        unk = sum(c["head_only"] + c["child_only"] for c in entry["lang_counts"].values())
        total = sum(c["total"] for c in entry["lang_counts"].values())
        joint = yes + no
        n_langs_joint = sum(1 for c in entry["lang_counts"].values() if c["yes"] + c["no"] > 0)
        rows.append({
            "family": family,
            "feature": feat,
            "known": feat in KNOWN_FEATURE_NAMES,
            "yes": yes,
            "no": no,
            "unk": unk,
            "joint_attested": joint,
            "pct_joint": round(joint / total * 100, 2) if total else 0.0,
            "pct_agree": round(yes / joint * 100, 2) if joint else 0.0,
            "n_langs_joint": n_langs_joint,
            "n_langs_seen": len(entry["lang_counts"]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["known", "joint_attested"], ascending=[True, False]).reset_index(drop=True)


def build_lang_config(stats: dict, wilson_floor: float, min_both: int,
                       min_yes: int, yes_rate_floor: float) -> dict[str, dict]:
    """feat -> {lang: {...}} -- same shape/semantics as npa's
    build_lang_config (see that docstring for the Wilson-bar rationale),
    just keyed by plain feature name instead of a role-pair target_col,
    since here there's only ever one head/child pair per family."""
    config = {}
    for feat, entry in stats.items():
        lang_entries = {}
        for lang, c in sorted(entry["lang_counts"].items()):
            both = c["yes"] + c["no"]
            total = c["total"]
            if not passes_agreement_bar(c["yes"], c["no"], total, wilson_floor,
                                        min_both, min_yes, yes_rate_floor):
                continue

            def pct(x, total=total):
                return round(x / total * 100, 2) if total else 0.0

            lang_entries[lang] = {
                "neither": {"abs": c["neither"], "rel": pct(c["neither"])},
                "head_only": {"abs": c["head_only"], "rel": pct(c["head_only"])},
                "child_only": {"abs": c["child_only"], "rel": pct(c["child_only"])},
                "both": {"abs": both, "rel": pct(both)},
                "total": total,
            }
        if lang_entries:
            config[feat] = lang_entries
    return config


def build_lang_csv(family: str, config: dict) -> pd.DataFrame:
    rows = []
    for feat, lang_entries in config.items():
        for lang, s in sorted(lang_entries.items()):
            rows.append({
                "family": family, "feature": feat, "lang": lang,
                "head_only": s["head_only"]["abs"], "%": s["head_only"]["rel"],
                "child_only": s["child_only"]["abs"], "% ": s["child_only"]["rel"],
                "both": s["both"]["abs"], "%  ": s["both"]["rel"],
                "neither": s["neither"]["abs"], "%   ": s["neither"]["rel"],
                "total": s["total"],
            })
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, family: str, min_langs: int) -> None:
    print(f"\n=== {family} ===")
    known = df[df["known"]]
    candidates = df[~df["known"]]
    keep = candidates[candidates["n_langs_joint"] >= min_langs]
    drop = candidates[candidates["n_langs_joint"] < min_langs]
    if len(known):
        print("  known (Number/Gender/Person, sanity check):")
        for _, r in known.iterrows():
            print(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                  f"agree={r['pct_agree']}% in {r['n_langs_joint']}/{r['n_langs_seen']} langs")
    if len(keep):
        print("  NEW candidates worth an agreement check:")
        for _, r in keep.iterrows():
            print(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                  f"agree={r['pct_agree']}% in {r['n_langs_joint']}/{r['n_langs_seen']} langs")
    if len(drop):
        print("  skip (never/rarely jointly tagged):")
        for _, r in drop.iterrows():
            print(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                  f"in {r['n_langs_joint']}/{r['n_langs_seen']} langs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan every language's existing word_order cache "
                     "(nsubj_fin/nsubj_part/subj_aux, obj_*, iobj_* -- "
                     "already on disk from prior Na/Ga/Pa condition runs, "
                     "no rerun needed) and report, per s/o/io x v/a/p "
                     "family, which morphological features besides Number/"
                     "Gender/Person are ever jointly attested and agreeing "
                     "between head and child -- the sv/sp/sa/.../ov/op/oa/"
                     "iov/iop/ioa analogue of scripts/npa/"
                     "agreement_candidates.py."
    )
    parser.add_argument("--families", "-f", nargs="*", default=list(FAMILIES),
                         help=f"Subset of families to scan (default: all {list(FAMILIES)}).")
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    parser.add_argument("--min_langs", type=int, default=1)
    parser.add_argument("--wilson_floor", type=float, default=0.01)
    parser.add_argument("--min_both", type=int, default=10)
    parser.add_argument("--min_yes", type=int, default=10)
    parser.add_argument("--yes_rate_floor", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_summaries, all_configs, all_lang_csvs = [], {}, []

    for family in args.families:
        cache_dir, deprel = FAMILIES[family]
        parquet_paths = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
        if args.langs:
            wanted = set(args.langs)
            parquet_paths = [p for p in parquet_paths
                              if os.path.basename(p)[: -len(".parquet")] in wanted]
        if not parquet_paths:
            print(f"=== {family}: no cache found at {cache_dir}, skipping ===")
            continue

        stats = defaultdict(
            lambda: {"lang_counts": defaultdict(lambda: Counter(
                yes=0, no=0, head_only=0, child_only=0, neither=0, total=0))}
        )
        n_done = 0
        for parquet_path in parquet_paths:
            if scan_language(parquet_path, deprel, stats) is not None:
                n_done += 1
        print(f"{family}: scanned {n_done}/{len(parquet_paths)} languages, "
              f"{len(stats)} distinct features found")

        summary = build_summary(stats, family)
        all_summaries.append(summary)

        config = build_lang_config(stats, args.wilson_floor, args.min_both,
                                    args.min_yes, args.yes_rate_floor)
        all_configs[family] = config
        all_lang_csvs.append(build_lang_csv(family, config))

        print_report(summary, family, args.min_langs)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(OUT_CSV, index=False)
        print(f"\nwrote {OUT_CSV}")
    with open(OUT_CONFIG, "w") as f:
        json.dump(all_configs, f, indent=2, sort_keys=True)
    print(f"wrote {OUT_CONFIG}")
    if all_lang_csvs:
        pd.concat(all_lang_csvs, ignore_index=True).to_csv(OUT_LANG_CSV, index=False)
        print(f"wrote {OUT_LANG_CSV}")
