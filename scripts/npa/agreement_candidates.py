import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

sys.path.append("../../src/")

from npa.np_types import ROLE_PRIORITY
from multiblimp.config import TREEBANK_FEATURES_DIR

INSTANCES_DIR = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")
OUT_CSV = "../../npa_exploration/agreement_candidates.csv"
OUT_LANG_CSV = "../../npa_exploration/agreement_candidates_by_lang.csv"
OUT_CONFIG = "../../resources/npa_config.json"

# Matches exactly np_instance()'s pairwise agreement columns ("HEAD-DET_Case",
# "DET-ADJ_Gender", ...) -- role names come from the same small fixed
# vocabulary np_roles() ever emits (ROLE_PRIORITY), so this can't accidentally
# pick up the far more numerous "{role}_sibling-deprel_*"/"{role}_child-*"
# node-feature columns extract_node_features also writes (those never start
# with a *pair* of role names joined by "-").
_ROLES = list(ROLE_PRIORITY)
_PAIR_COL_RE = re.compile(
    r"^(" + "|".join(_ROLES) + r")-(" + "|".join(_ROLES) + r")_([A-Z][a-zA-Z]*)$"
)


def _split_col(col: str) -> tuple[str, str, str]:
    """"HEAD-DET_Gender" -> ("HEAD", "DET", "Gender"); also handles the
    synthetic per-HEAD-UPOS column names --split_head_upos produces
    ("HEAD:NOUN-DET_Gender" -> ("HEAD:NOUN", "DET", "Gender")), unlike
    _PAIR_COL_RE (which only matches real, on-disk column names and is used
    solely to find them in a parquet's schema -- see scan_language).
    """
    role1, rest = col.split("-", 1)
    role2, feat = rest.split("_", 1)
    return role1, role2, feat


def _role_priority(role: str) -> int:
    """Sort key for a (possibly synthetic, "HEAD:NOUN"-style) role name --
    strips any ":UPOS" suffix before looking up ROLE_PRIORITY, so
    --split_head_upos's per-UPOS HEAD variants still sort where plain "HEAD"
    would.
    """
    return ROLE_PRIORITY.get(role.split(":", 1)[0], 99)


def wilson_lower_bound(both: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score confidence interval (default z=1.96,
    i.e. ~95%) for the true rate of `both`/`total` -- used by
    build_lang_config in place of a flat min-count or min-percentage cutoff
    to decide whether a (target_col, lang)'s joint attestation is real
    signal or plausible annotation noise (see conversation: a raw `both`
    count alone can't tell "8/50, a real minority pattern" apart from
    "8/50000, probably a few stray tags", and a raw percentage alone is
    unstable at small `total` -- 2/20 and 2000/20000 are both "10%" but
    very different strength of evidence). The Wilson lower bound adapts
    to both at once: a small `total` needs a much higher observed rate to
    clear a given floor than a large `total` does, roughly matching how
    much a human would trust each case.

    0.0 for total==0 (no co-occurrences at all -- shouldn't reach here in
    practice, since build_lang_config only calls this for target_cols that
    already have a nonzero `total`, but kept safe for direct callers).
    """
    if total == 0:
        return 0.0
    p = both / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, center - margin)


def _pair_counts(df: pd.DataFrame, col: str, role1: str, role2: str,
                  feat: str) -> tuple[int, int, int, int, int, int]:
    """(yes, no, role1_only, role2_only, neither, cooccur_total) for one
    pairwise agreement column over `df` (or any row-subset of it, e.g. one
    HEAD-UPOS group -- see scan_language). Split out of scan_language so it
    can be applied either to a language's full df or to a per-HEAD-UPOS
    slice of it without duplicating the counting logic.
    """
    counts = df[col].value_counts(dropna=True)
    yes = int(counts.get("yes", 0))
    no = int(counts.get("no", 0))
    unk = int(counts.get("unk", 0))

    r1_feat_col = f"{role1}_{feat}"
    if unk and r1_feat_col in df.columns:
        unk_mask = df[col] == "unk"
        role1_only = int((unk_mask & df[r1_feat_col].notna()).sum())
    else:
        role1_only = 0
    role2_only = unk - role1_only

    r1_form_col, r2_form_col = f"{role1}_form", f"{role2}_form"
    if r1_form_col in df.columns and r2_form_col in df.columns:
        cooccur_total = int((df[r1_form_col].notna() & df[r2_form_col].notna()).sum())
    else:
        cooccur_total = yes + no + unk  # can't detect "neither"; assume 0
    neither = max(cooccur_total - yes - no - unk, 0)

    return yes, no, role1_only, role2_only, neither, cooccur_total


def scan_qualifying_langs(target_col: str, instances_dir: str = INSTANCES_DIR,
                           wilson_floor: float = 0.01, min_both: int = 10) -> list[str]:
    """Live, single-target_col equivalent of one build_lang_config entry --
    which languages of npa_config.json's languages worth attempting for
    `target_col`, computed on the fly straight from the np_instances
    parquets instead of a prebuilt npa_config.json.

    Exists because npa_config.json only ever has entries for exactly the
    target_cols a full agreement_candidates.py --split_head_upos scan
    produced (e.g. "HEAD:NOUN-DET_Case"), never the plain pooled form
    ("HEAD-DET_Case") -- np_instance() itself always builds the pooled
    column regardless of any scan-time option, so a pooled target_col is
    perfectly fittable, it's just never been through the pooled scan.
    Before this, getting its qualifying-language list meant hand-writing a
    one-off script reimplementing this exact loop each time (see
    conversation) -- fit_trees.py now calls this automatically instead
    (see its own langs-resolution logic) whenever a requested target_col
    isn't already a key in the loaded config, so this only runs for
    target_cols actually missing from it, not as a blanket replacement for
    the (much cheaper, since it's one pass over every column at once) full
    scan.

    Uses the exact same wilson_floor/min_both criteria as
    build_lang_config, defaulting to the same values, so a language that
    would clear the bar in a full agreement_candidates.py rerun clears it
    here too -- just computed for one target_col rather than every one at
    once, and without --split_head_upos's per-head-UPOS splitting (since
    the whole point is resolving the plain pooled form).

    Only reads the target_col itself and role1's raw feature/form columns
    plus role2's form column per language (same column-projection
    discipline as scan_language) -- skips a language's parquet entirely
    (via a schema-only check) when it doesn't even have target_col.
    """
    role1, role2, feat = _split_col(target_col)
    qualifying = []
    for path in sorted(glob.glob(os.path.join(instances_dir, "*.parquet"))):
        lang = os.path.basename(path)[: -len(".parquet")]
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        if target_col not in schema_names:
            continue
        wanted = [
            c for c in (target_col, f"{role1}_{feat}", f"{role1}_form", f"{role2}_form")
            if c in schema_names
        ]
        df = pd.read_parquet(path, columns=wanted)
        yes, no, *_rest = _pair_counts(df, target_col, role1, role2, feat)
        _role1_only, _role2_only, _neither, total = _rest
        both = yes + no
        if both >= min_both and wilson_lower_bound(both, total) >= wilson_floor:
            qualifying.append(lang)
    return qualifying


def scan_language(parquet_path: str, stats: dict, split_head_upos: bool = False) -> str | None:
    """Update `stats` (target_col -> {"lang_counts": {lang: Counter}}) in
    place from one language's np_instances parquet. Returns the language
    name, or None if the parquet has no pairwise agreement columns at all
    (e.g. a degenerate, no-qualifying-dependents parquet -- see
    np_morph_pct.py's own handling of this case).

    Reads column names from the parquet's schema (metadata only, no data)
    first, then loads only the matching columns (the pairwise column itself,
    role1's own raw feature column, and both roles' "_form" columns) --
    avoids pulling in the other ~13k node-feature columns np_instance() also
    writes per language, which is most of these files' size (see
    np_types.np_instance).

    Tracks, per (target_col, lang), the full 4-way partition of every row
    where role1 and role2's tokens actually co-occur in the NP (role1_form
    and role2_form both set -- the same presence signal npa.agreement.
    _coverage_stats_npa uses, not the pairwise column itself, since a row
    where NEITHER side has this particular feature tagged never gets a
    pairwise column value at all -- see np_instance's `for feat in
    feats1.keys() | feats2.keys()` union: that key is simply absent from
    such a row, not "unk"):

    - "yes"/"no": both sides tagged, matching / mismatching (from the
      pairwise column's own value -- "both" in build_lang_config's output is
      yes+no).
    - "role1_only"/"role2_only": exactly one side tagged (pairwise == "unk").
      Determined from role1's own raw feature column alone: pairwise_
      agreement() returns "unk" iff exactly one side is None (never both --
      if both were None the key wouldn't exist in this row at all), so
      role1_only = (unk & role1_feat notna), role2_only = unk - role1_only.
    - "neither": co-occurs but neither side tagged for this feature at all
      (invisible to the pairwise column itself; only visible against the
      form-based co-occurrence count).

    split_head_upos: when True, every column with role1=="HEAD" (HEAD, when
    present in a pair, always sorts first -- see np_types.ROLE_PRIORITY) is
    additionally split by the head token's own UPOS (its "HEAD_pos" column,
    e.g. NOUN/PROPN/PRON -- np_roles() pools all three under one "HEAD" role
    label, see its own docstring) into separate synthetic target_cols
    "HEAD:NOUN-DET_Gender", "HEAD:PROPN-DET_Gender", etc, tallied
    independently instead of pooled into plain "HEAD-DET_Gender". Column
    LOOKUPS still use the real, on-disk role name ("HEAD_Gender",
    "HEAD_form") throughout -- only the stats dict KEY becomes synthetic;
    _split_col (not _PAIR_COL_RE, which only matches real column names)
    parses that synthetic key back apart downstream. A language whose
    schema lacks "HEAD_pos" (shouldn't happen -- extract_node_features
    always writes it for a present role -- but guarded anyway) falls back
    to the unsplit column for that language, same as split_head_upos=False.
    """
    lang = os.path.basename(parquet_path)[: -len(".parquet")]
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    schema_set = set(schema_names)
    matches = [
        (m.group(1), m.group(2), m.group(3), c)
        for c in schema_names if (m := _PAIR_COL_RE.match(c))
    ]
    if not matches:
        return None

    wanted = {"HEAD_pos"} if split_head_upos else set()
    for role1, role2, feat, col in matches:
        wanted.update((col, f"{role1}_{feat}", f"{role1}_form", f"{role2}_form"))
    df = pd.read_parquet(parquet_path, columns=[c for c in wanted if c in schema_set])

    for role1, role2, feat, col in matches:
        if split_head_upos and role1 == "HEAD" and "HEAD_pos" in df.columns:
            groups = [(f"HEAD:{upos}-{role2}_{feat}", sub_df)
                      for upos, sub_df in df.groupby("HEAD_pos", observed=True)]
        else:
            groups = [(col, df)]

        for stats_key, group_df in groups:
            yes, no, role1_only, role2_only, neither, total = _pair_counts(
                group_df, col, role1, role2, feat
            )
            lc = stats[stats_key]["lang_counts"][lang]
            lc["yes"] += yes
            lc["no"] += no
            lc["role1_only"] += role1_only
            lc["role2_only"] += role2_only
            lc["neither"] += neither
            lc["total"] += total
    return lang


def build_summary(stats: dict) -> pd.DataFrame:
    rows = []
    for col, entry in stats.items():
        role1, role2, feat = _split_col(col)
        yes = sum(c["yes"] for c in entry["lang_counts"].values())
        no = sum(c["no"] for c in entry["lang_counts"].values())
        unk = sum(c["role1_only"] + c["role2_only"] for c in entry["lang_counts"].values())
        joint = yes + no
        total = joint + unk
        n_langs_joint = sum(
            1 for c in entry["lang_counts"].values() if c["yes"] + c["no"] > 0
        )
        rows.append({
            "role1": role1,
            "role2": role2,
            "feature": feat,
            "target_col": col,
            "yes": yes,
            "no": no,
            "unk": unk,
            "joint_attested": joint,
            "pct_joint": round(joint / total * 100, 2) if total else 0.0,
            "n_langs_joint": n_langs_joint,
            "n_langs_seen": len(entry["lang_counts"]),
        })
    df = pd.DataFrame(rows)
    df["role1_pri"] = df["role1"].map(_role_priority)
    df["role2_pri"] = df["role2"].map(_role_priority)
    df = df.sort_values(
        ["role1_pri", "role2_pri", "joint_attested"], ascending=[True, True, False]
    ).drop(columns=["role1_pri", "role2_pri"])
    return df.reset_index(drop=True)


def build_lang_config(stats: dict, wilson_floor: float = 0.01,
                       min_both: int = 10) -> dict[str, dict[str, dict]]:
    """target_col -> {lang: {"neither": {abs,rel}, "{role1}_only": {abs,rel},
    "{role2}_only": {abs,rel}, "both": {abs,rel}, "total": int}} for every
    language where the feature's joint attestation ("both", i.e. "yes" or
    "no", never just one-sided or untagged) looks like real signal rather
    than annotation noise -- the npa_config.json src.npa.npa_config.
    load_npa_config reads, and scripts/npa/fit_trees.py uses (in place of
    get_ud_langs' full ~190 language list) so run_agreement_pipeline only
    ever opens a language's np_instances parquet when there's a real
    chance of it contributing.

    "total" is every row where role1 and role2's tokens actually co-occur in
    the NP, regardless of tagging -- the four abs counts (neither/role1_only/
    role2_only/both) partition it exactly (see scan_language), so each "rel"
    is abs/total*100. E.g. {"German": {"neither": {"abs": 6, "rel": 1.08},
    "DET_only": {"abs": 12, "rel": 2.17}, "ADJ_only": {"abs": 29,
    "rel": 5.24}, "both": {"abs": 506, "rel": 91.5}, "total": 553}} for
    DET-ADJ_Gender reads as "506/553 (91.5%) of DET-ADJ co-occurrences in
    German had Gender annotated on both sides; 12 had it on DET only, 29 on
    ADJ only, 6 on neither" -- kept per language (not just pooled, see
    build_summary) since a pooled percentage would hide a language where the
    pair co-occurs often but the feature is rarely double-tagged, or vice
    versa.

    A language clears the bar iff BOTH (a) wilson_lower_bound(both, total)
    >= wilson_floor (default 1%) and (b) both >= min_both (default 10).

    (a) alone can't distinguish "real signal" from "too little data to say
    anything" -- a raw `both` count can't distinguish "8/50, a real
    minority pattern" from "8/50000, plausibly a handful of stray
    annotation tags" (same count, very different evidence), and a raw
    percentage is unstable at small `total` (2/20 and 2000/20000 are both
    "10%" but nowhere near equally trustworthy); the Wilson lower bound
    adapts to both at once (see wilson_lower_bound's own docstring;
    empirically, a 1% floor cleanly separates cases like Ukrainian's
    HEAD:NOUN-ADJ_Abbr, 5/11341 -> 0.02%, from Latvian's DET-ADJ_Degree,
    3/39 -> 2.7%, keeping the latter as a genuine small-but-real pattern).
    But (a) has its own blind spot at very small `both`: a 100%-agreement
    "both=2, total=2" clears wilson_lower_bound(2,2)=34% -- comfortably
    above even a strict floor -- despite 2 data points being nowhere near
    enough to call anything "real" (Wilson correctly estimates confidence
    in the true rate GIVEN the sample, but has no opinion on whether the
    sample itself is big enough to trust at all; verified empirically --
    313/3002 rows in one scan had total<=2). (b) closes that gap: it also
    matches a real downstream constraint exactly rather than being
    arbitrary -- npa.agreement.fit_npa_tree hard-requires
    len(sub_df)>=10 after dropping "unk" (i.e., at the default
    drop_unk=True, exactly this `both` count), so a target_col that can
    never reach both>=10 could never produce a fitted tree regardless of
    what (a) says; keeping it in npa_config.json would just be wasted work
    for fit_trees.py to discover and skip later.

    A language failing either condition -- most simply, zero "both" rows
    at all -- can never pass fit_npa_tree's own checks anyway, so excluding
    it here just moves that skip earlier, before the (expensive,
    full-width) parquet read run_agreement_pipeline currently does
    unconditionally per language. (The filter is on "both" vs "total"
    only -- "neither"/role-only volume doesn't otherwise affect whether a
    language is worth attempting, since those rows can never contribute a
    "yes"/"no" tree label.)

    A target_col with NO language clearing the bar at all (e.g.
    "DET-ADJ_Degree") gets no entry -- fit_trees.py's lookup falls back to
    the full language list in that case, same as for a target_col this scan
    never encountered at all (e.g. a typo).
    """
    config = {}
    for col, entry in stats.items():
        role1, role2, _feat = _split_col(col)
        lang_entries = {}
        for lang, c in sorted(entry["lang_counts"].items()):
            both = c["yes"] + c["no"]
            total = c["total"]
            if both < min_both or wilson_lower_bound(both, total) < wilson_floor:
                continue

            def pct(x, total=total):
                return round(x / total * 100, 2) if total else 0.0

            lang_entries[lang] = {
                "neither": {"abs": c["neither"], "rel": pct(c["neither"])},
                f"{role1}_only": {"abs": c["role1_only"], "rel": pct(c["role1_only"])},
                f"{role2}_only": {"abs": c["role2_only"], "rel": pct(c["role2_only"])},
                "both": {"abs": both, "rel": pct(both)},
                "total": total,
            }
        if lang_entries:
            config[col] = lang_entries
    return config


def build_lang_csv(config: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Flat, one-row-per-(target_col, language) CSV view of npa_config.json
    (see build_lang_config) -- same neither/role1_only/role2_only/both/total
    numbers, just tabular instead of nested, for spreadsheet/pandas
    filtering (e.g. "every language where DET_only exceeds 20%").

    Uses generic "role1_only"/"role2_only" column names rather than the
    config's own per-pair "{role1}_only"/"{role2}_only" keys (e.g.
    "DET_only" for one row, "HEAD_only" for another) -- a fixed column set
    keeps this one flat table rather than a sparse one with a different
    column per role name; the separate "role1"/"role2" columns already say
    which UPOS each one is for that row.

    Each abs/rel pair is emitted as two adjacent columns, the second one
    literally named "%" (so the header reads "role1_only,%,role2_only,%,
    both,%,neither,%,total") rather than "*_abs"/"*_rel" -- duplicate
    headers are unusual but pandas reads/writes them fine positionally; a
    caller after column names rather than position should use
    build_lang_config's nested dict form instead.
    """
    rows = []
    for col, lang_entries in config.items():
        role1, role2, feat = _split_col(col)
        for lang, stats in sorted(lang_entries.items()):
            rows.append({
                "target_col": col,
                "role1": role1,
                "role2": role2,
                "feature": feat,
                "lang": lang,
                "role1_only": stats[f"{role1}_only"]["abs"],
                "role1_only_rel": stats[f"{role1}_only"]["rel"],
                "role2_only": stats[f"{role2}_only"]["abs"],
                "role2_only_rel": stats[f"{role2}_only"]["rel"],
                "both": stats["both"]["abs"],
                "both_rel": stats["both"]["rel"],
                "neither": stats["neither"]["abs"],
                "neither_rel": stats["neither"]["rel"],
                "total": stats["total"],
            })
    df = pd.DataFrame(rows)
    df["role1_pri"] = df["role1"].map(_role_priority)
    df["role2_pri"] = df["role2"].map(_role_priority)
    df = df.sort_values(
        ["role1_pri", "role2_pri", "feature", "lang"]
    ).drop(columns=["role1_pri", "role2_pri"])
    df = df.reset_index(drop=True)
    # Rename *after* sorting/computation (which needs unique names) --
    # "role1_only_rel" etc become a bare "%" right after their abs column,
    # matching the requested header exactly.
    df.columns = [
        "%" if c.endswith("_rel") else c for c in df.columns
    ]
    return df


def print_report(df: pd.DataFrame, min_langs: int) -> None:
    """Grouped-by-role-pair report: which features are worth an agreement
    check (jointly attested in >= min_langs languages) vs which are never (or
    only incidentally, in fewer languages than that) jointly tagged -- e.g.
    DET-ADJ_Degree should land in the second group, HEAD-ADJ_Gender in the
    first.
    """
    for (role1, role2), group in df.groupby(["role1", "role2"], sort=False):
        print(f"\n=== {role1}-{role2} ===")
        keep = group[group["n_langs_joint"] >= min_langs]
        drop = group[group["n_langs_joint"] < min_langs]
        if len(keep):
            print("  run agreement check:")
            for _, r in keep.iterrows():
                print(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                      f"({r['pct_joint']}% of co-occurrences) "
                      f"in {r['n_langs_joint']}/{r['n_langs_seen']} langs")
        if len(drop):
            print("  skip (never/rarely jointly tagged):")
            for _, r in drop.iterrows():
                print(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                      f"in {r['n_langs_joint']}/{r['n_langs_seen']} langs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan every language's np_instances parquet "
                     "(treebank_features/npa/np_instances) and report, per "
                     "role1-role2 UPOS pair, which morphological features are "
                     "ever jointly attested on both sides (i.e. worth fitting "
                     "an agreement target_col for) vs. only ever tagged on "
                     "one side (never worth running, e.g. DET-ADJ_Degree)."
    )
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    parser.add_argument("--min_langs", type=int, default=1,
                         help="Minimum number of languages a feature must be "
                              "jointly attested in to count as a real "
                              "agreement candidate in the printed report "
                              "(default 1 -- any language at all). Does not "
                              "affect the saved CSV, which keeps every "
                              "pairwise column found.")
    parser.add_argument("--wilson_floor", type=float, default=0.01,
                         help="Minimum Wilson-score lower bound (95%%) on "
                              "both/total for a language to be listed under "
                              "a given target_col in npa_config.json "
                              "(default 0.01 = 1%%). Adapts to sample size "
                              "rather than using a flat count or percentage "
                              "cutoff -- see build_lang_config/"
                              "wilson_lower_bound.")
    parser.add_argument("--min_both", type=int, default=10,
                         help="Minimum raw 'both' (jointly-tagged) count a "
                              "language must have for a given target_col, "
                              "on top of --wilson_floor -- catches small-n/"
                              "100%%-rate cases (e.g. both=2,total=2) that "
                              "clear the Wilson floor despite too little "
                              "data to trust. Default 10 matches "
                              "npa.agreement.fit_npa_tree's own hard "
                              "len(sub_df)>=10 requirement, so anything "
                              "below it could never be fit anyway -- see "
                              "build_lang_config.")
    parser.add_argument("--split_head_upos", action="store_true",
                         help="Split every HEAD-involving pair (e.g. "
                              "HEAD-DET_Gender) by the head token's own UPOS "
                              "(NOUN/PROPN/PRON, from its HEAD_pos column) "
                              "into separate target_cols (HEAD:NOUN-DET_"
                              "Gender, HEAD:PROPN-DET_Gender, ...) instead of "
                              "pooling all three under plain HEAD-*. Off by "
                              "default, matching np_types.np_roles' own "
                              "pooled HEAD role.")
    args = parser.parse_args()

    parquet_paths = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.parquet")))
    if args.langs:
        wanted = set(args.langs)
        parquet_paths = [
            p for p in parquet_paths
            if os.path.basename(p)[: -len(".parquet")] in wanted
        ]

    stats = defaultdict(
        lambda: {"lang_counts": defaultdict(lambda: Counter(yes=0, no=0, unk=0))}
    )
    n_done = 0
    for parquet_path in parquet_paths:
        lang = scan_language(parquet_path, stats, split_head_upos=args.split_head_upos)
        if lang is None:
            continue
        n_done += 1
        if n_done % 25 == 0:
            print(f"...{n_done}/{len(parquet_paths)} languages scanned")

    print(f"scanned {n_done}/{len(parquet_paths)} languages, "
          f"{len(stats)} distinct role-pair_feature columns found")

    summary = build_summary(stats)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    config = build_lang_config(stats, wilson_floor=args.wilson_floor, min_both=args.min_both)
    os.makedirs(os.path.dirname(OUT_CONFIG), exist_ok=True)
    with open(OUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"wrote {OUT_CONFIG} ({len(config)} target_cols with >=1 usable language)")

    lang_csv = build_lang_csv(config)
    lang_csv.to_csv(OUT_LANG_CSV, index=False)
    print(f"wrote {OUT_LANG_CSV} ({len(lang_csv)} target_col/language rows)")

    print_report(summary, args.min_langs)
