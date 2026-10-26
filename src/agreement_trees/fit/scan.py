"""Which (target, language) pairs have enough real agreement evidence to be
worth fitting. Both pipelines are scanned the same way and judged by one bar
(agreement_trees.pipeline_utils.passes_agreement_bar):

  vp  one scan target per (family, feature), keyed "sv:Number": the stored
      head_{deprel}_{feat}_agreement column of the family's cache
  np  one scan target per role pair and feature, keyed "HEAD-DET_Number": the
      stored pairwise column of the np_instances parquet

Per (target, language) the rows are split into yes / no (both sides tagged,
agreeing or not), one-side-only (two buckets, named after the roles), and
neither, all out of the rows where the two roles actually co-occur. Rows of
excluded treebanks are not counted. Each language entry also carries two diagnostics (nothing is filtered on
them): head_values, how many distinct values the head's feature takes over the yes/no rows, and chance_adjusted,
the agreement beyond what independent head and child values would give.
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

from agreement_trees.config import (
    LANGS_PATH, VP_FAMILIES, load_scan_results, parse_vp_target, vp_scan_key,
)
from agreement_trees.pipeline_utils import parquet_current, sentinel_current
from agreement_trees.pipeline_utils import passes_agreement_bar
from multiblimp.config import NP_INSTANCES_DIR
from multiblimp.languages import gblang2udlang
from agreement_trees.extract.per_treebank import split_excluded

KNOWN_FEATURES = {"Number", "Gender", "Person"}
COUNT_KEYS = ("yes", "no", "a_only", "b_only", "neither", "total")


def new_stats():
    return defaultdict(lambda: {
        "roles": None,
        "lang_counts": defaultdict(lambda: Counter({k: 0 for k in COUNT_KEYS})),
        "lang_values": defaultdict(lambda: {"head": Counter(), "child": Counter()}),
    })


def _included(path: str, schema_names, columns) -> pd.DataFrame:
    """The projected `columns` of one cache parquet minus excluded-treebank rows."""
    wanted = list(columns)
    if "treebank" in schema_names and "treebank" not in wanted:
        wanted.append("treebank")
    df, _ = split_excluded(pd.read_parquet(path, columns=wanted))
    return df.drop(columns="treebank", errors="ignore")


def _add(stats: dict, key: str, roles: tuple[str, str], lang: str, counts: tuple, values: tuple | None = None) -> None:
    entry = stats[key]
    entry["roles"] = roles
    lc = entry["lang_counts"][lang]
    for name, value in zip(COUNT_KEYS, counts):
        lc[name] += value
    if values:
        entry["lang_values"][lang]["head"].update(values[0])
        entry["lang_values"][lang]["child"].update(values[1])


def value_pairs(agreement: pd.Series, head: pd.Series | None, child: pd.Series | None) -> tuple | None:
    """Counters of the head's and the child's feature values over the rows labelled yes or no."""
    if head is None or child is None:
        return None
    rows = agreement.isin(["yes", "no"])
    return Counter(head[rows].dropna().astype(str)), Counter(child[rows].dropna().astype(str))


def agreement_diagnostics(head: Counter, child: Counter, yes: int, no: int) -> tuple:
    """(distinct head values, agreement beyond chance): (observed - expected) / (1 - expected), expected being the
    agreement of a head and a child value drawn independently; None when it cannot be told. Diagnostics only: a head
    that never varies, or agreement no better than chance, shows up here, but nothing is filtered on it."""
    n_head, n_child = sum(head.values()), sum(child.values())
    if not n_head or not n_child or not yes + no:
        return len(head), None
    expected = sum(head[v] / n_head * child.get(v, 0) / n_child for v in head)
    return len(head), None if expected >= 1 else round((yes / (yes + no) - expected) / (1 - expected), 3)


# ---------------------------------------------------------------- vp

def vp_family_dirs() -> dict[str, tuple[str, str]]:
    """family -> (word_order cache dir, child deprel). One Na condition per
    family is enough: Na/Ga/Pa share the cache."""
    from agreement_trees.vp.conditions import CONDITIONS
    return {
        family: (CONDITIONS[f"{family}Na"].word_order_dir(), CONDITIONS[f"{family}Na"].deprel)
        for family in VP_FAMILIES
    }


def vp_features(schema_names, deprel: str, exclude_features=()) -> list[str]:
    prefix, suffix = f"head_{deprel}_", "_agreement"
    return sorted(feat for c in schema_names
                  if c.startswith(prefix) and c.endswith(suffix) and len(c) > len(prefix) + len(suffix)
                  and (feat := c[len(prefix):-len(suffix)]) not in exclude_features)


def vp_counts(agreement: pd.Series, child: pd.Series | None) -> tuple:
    """(yes, no, head_only, child_only, neither, total) from a stored
    agreement column: "yes"/"no" both sides tagged, "+-" exactly one side (split
    by whether the child side carries the feature), anything else ("--" or no
    value) neither."""
    total = len(agreement)
    yes, no = int((agreement == "yes").sum()), int((agreement == "no").sum())
    one_side = agreement == "+-"
    child_only = int((one_side & child.notna()).sum()) if child is not None else 0
    n_one_side = int(one_side.sum())
    return yes, no, n_one_side - child_only, child_only, total - yes - no - n_one_side, total


def scan_vp_language(parquet_path: str, family: str, deprel: str, stats: dict,
                     only_feature: str | None = None, exclude_features=()) -> str | None:
    lang = os.path.basename(parquet_path)[: -len(".parquet")]
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    feats = vp_features(schema_names, deprel, exclude_features)
    if only_feature is not None:
        feats = [f for f in feats if f == only_feature]
    if not feats:
        return None
    wanted = [f"head_{deprel}_{f}_agreement" for f in feats]
    wanted += [f"{deprel}_{f}" for f in feats if f"{deprel}_{f}" in schema_names]
    wanted += [f"head_{f}" for f in feats if f"head_{f}" in schema_names]
    df = _included(parquet_path, set(schema_names), wanted)
    for feat in feats:
        child_col = f"{deprel}_{feat}"
        counts = vp_counts(df[f"head_{deprel}_{feat}_agreement"],
                           df[child_col] if child_col in df.columns else None)
        values = value_pairs(df[f"head_{deprel}_{feat}_agreement"], df.get(f"head_{feat}"), df.get(child_col))
        _add(stats, vp_scan_key(family, feat), ("head", deprel), lang, counts, values)
    return lang


# ---------------------------------------------------------------- np

def _np_roles():
    from agreement_trees.npa.np_types import ROLE_PRIORITY
    return ROLE_PRIORITY


def np_pair_re():
    roles = list(_np_roles())
    return re.compile(r"^(" + "|".join(roles) + r")-(" + "|".join(roles) + r")_([A-Z][a-zA-Z]*)$")


def split_np_col(col: str) -> tuple[str, str, str]:
    """"HEAD-DET_Gender" -> ("HEAD", "DET", "Gender"); also the per-UPOS
    split names ("HEAD:NOUN-DET_Gender" -> ("HEAD:NOUN", "DET", "Gender"))."""
    role1, rest = col.split("-", 1)
    role2, feat = rest.split("_", 1)
    return role1, role2, feat


def role_priority(role: str) -> int:
    return _np_roles().get(role.split(":", 1)[0], 99)


def np_counts(df: pd.DataFrame, col: str, role1: str, role2: str, feat: str) -> tuple:
    """(yes, no, role1_only, role2_only, neither, total) for one pairwise column
    over `df` (or a row subset such as one head UPOS). "total" is the rows
    where both role tokens are present."""
    counts = df[col].value_counts(dropna=True)
    yes, no, unk = int(counts.get("yes", 0)), int(counts.get("no", 0)), int(counts.get("unk", 0))
    r1_feat_col = f"{role1}_{feat}"
    if unk and r1_feat_col in df.columns:
        role1_only = int(((df[col] == "unk") & df[r1_feat_col].notna()).sum())
    else:
        role1_only = 0
    r1_form, r2_form = f"{role1}_form", f"{role2}_form"
    if r1_form in df.columns and r2_form in df.columns:
        total = int((df[r1_form].notna() & df[r2_form].notna()).sum())
    else:
        total = yes + no + unk
    return yes, no, role1_only, unk - role1_only, max(total - yes - no - unk, 0), total


def scan_np_language(parquet_path: str, stats: dict, split_head_upos: bool = False,
                     only_col: str | None = None, exclude_features=()) -> str | None:
    lang = os.path.basename(parquet_path)[: -len(".parquet")]
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    schema_set = set(schema_names)
    pair_re = np_pair_re()
    matches = [(m.group(1), m.group(2), m.group(3), c)
               for c in schema_names if (m := pair_re.match(c)) and m.group(3) not in exclude_features]
    if only_col is not None:
        matches = [m for m in matches if m[3] == only_col]
    if not matches:
        return None

    wanted = {"HEAD_pos"} if split_head_upos else set()
    for role1, role2, feat, col in matches:
        wanted.update((col, f"{role1}_{feat}", f"{role2}_{feat}", f"{role1}_form", f"{role2}_form"))
    df = _included(parquet_path, schema_set, [c for c in wanted if c in schema_set])

    for role1, role2, feat, col in matches:
        if split_head_upos and role1 == "HEAD" and "HEAD_pos" in df.columns:
            groups = [(f"HEAD:{upos}-{role2}_{feat}", sub_df)
                      for upos, sub_df in df.groupby("HEAD_pos", observed=True)]
        else:
            groups = [(col, df)]
        for key, group_df in groups:
            _add(stats, key, (key.split("-", 1)[0], role2), lang, np_counts(group_df, col, role1, role2, feat),
                 value_pairs(group_df[col], group_df.get(f"{role1}_{feat}"), group_df.get(f"{role2}_{feat}")))
    return lang


# ---------------------------------------------------------------- scans

def cache_file(mode: str, family: str | None, lang: str) -> str:
    """Path of the cache parquet a scan of (family, lang) reads."""
    name = gblang2udlang.get(lang, lang).replace(" ", "_") + ".parquet"
    return os.path.join(vp_family_dirs()[family][0] if mode == "vp" else NP_INSTANCES_DIR, name)


def unit_key(mode: str, family: str | None, lang: str) -> str:
    return f"{family}/{lang}" if mode == "vp" else lang


# settings an earlier version recorded in the saved scans; they no longer count when a saved scan is compared
RETIRED_THRESHOLD_KEYS = ("min_agree_share", "min_agree_share_families")


def scan_thresholds(scan_cfg: dict) -> dict:
    keys = ("wilson_floor", "min_both", "min_yes", "yes_rate_floor", "split_head_upos")
    return {**{k: scan_cfg[k] for k in keys}, "exclude_features": sorted(scan_cfg["exclude_features"])}


def same_thresholds(saved: dict | None, current: dict) -> bool:
    """Whether a saved scan's settings equal the current ones, ignoring the retired keys."""
    return {k: v for k, v in (saved or {}).items() if k not in RETIRED_THRESHOLD_KEYS} == current


def scan_vp(langs: list[str] | None = None, families: list[str] | None = None,
            only_target: str | None = None, exclude_features=()) -> tuple[dict, dict, dict]:
    """(stats, {family: (scanned, found)}, {unit: cache mtime}) over the VP caches."""
    stats, counts, units = new_stats(), {}, {}
    only_family, only_feature = (parse_vp_target(only_target) or (None, None)) if only_target else (None, None)
    for family, (cache_dir, deprel) in vp_family_dirs().items():
        if (families is not None and family not in families) or (only_family and family != only_family):
            continue
        paths = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
        if langs:
            paths = [p for p in paths if os.path.basename(p)[: -len(".parquet")] in set(langs)]
        done = 0
        for path in paths:
            lang = os.path.basename(path)[: -len(".parquet")]
            units[unit_key("vp", family, lang)] = os.path.getmtime(path)
            done += scan_vp_language(path, family, deprel, stats, only_feature, exclude_features) is not None
        counts[family] = (done, len(paths))
    return stats, counts, units


def scan_np(langs: list[str] | None = None, split_head_upos: bool = False,
            only_target: str | None = None, exclude_features=()) -> tuple[dict, tuple[int, int], dict]:
    stats, units, done = new_stats(), {}, 0
    paths = sorted(glob.glob(os.path.join(NP_INSTANCES_DIR, "*.parquet")))
    if langs:
        paths = [p for p in paths if os.path.basename(p)[: -len(".parquet")] in set(langs)]
    for path in paths:
        units[unit_key("np", None, os.path.basename(path)[: -len(".parquet")])] = os.path.getmtime(path)
        done += scan_np_language(path, stats, split_head_upos, only_target, exclude_features) is not None
    return stats, (done, len(paths)), units


# ---------------------------------------------------------------- results

def build_summary(stats: dict, mode: str) -> pd.DataFrame:
    rows = []
    for key, entry in stats.items():
        counts = entry["lang_counts"].values()
        yes, no = sum(c["yes"] for c in counts), sum(c["no"] for c in counts)
        unk = sum(c["a_only"] + c["b_only"] for c in counts)
        total = sum(c["total"] for c in counts)
        joint = yes + no
        if mode == "vp":
            group, feature = key.split(":", 1)
        else:
            role1, role2, feature = split_np_col(key)
            group = f"{role1}-{role2}"
        rows.append({
            "group": group, "feature": feature, "target": key,
            "known": feature in KNOWN_FEATURES,
            "yes": yes, "no": no, "unk": unk, "joint_attested": joint,
            "pct_joint": round(joint / total * 100, 2) if total else 0.0,
            "pct_agree": round(yes / joint * 100, 2) if joint else 0.0,
            "n_langs_joint": sum(1 for c in entry["lang_counts"].values() if c["yes"] + c["no"] > 0),
            "n_langs_seen": len(entry["lang_counts"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if mode == "np":
        df["_pri"] = df["group"].map(lambda g: tuple(role_priority(r) for r in g.split("-")))
        df = df.sort_values(["_pri", "joint_attested"], ascending=[True, False]).drop(columns="_pri")
    else:
        df = df.sort_values(["group", "known", "joint_attested"], ascending=[True, True, False])
    return df.reset_index(drop=True)


def langs_passing(entry: dict, scan_cfg: dict) -> dict[str, Counter]:
    return {
        lang: c for lang, c in sorted(entry["lang_counts"].items())
        if passes_agreement_bar(c["yes"], c["no"], c["total"], scan_cfg["wilson_floor"],
                                scan_cfg["min_both"], scan_cfg["min_yes"], scan_cfg["yes_rate_floor"])
    }


def build_lang_config(stats: dict, scan_cfg: dict) -> dict[str, dict[str, dict]]:
    """target -> {lang: {"neither", "{role}_only" x2, "both", "total"}}, each
    count as {"abs", "rel" (% of total)}, for every language clearing the bar."""
    config = {}
    for key, entry in stats.items():
        role_a, role_b = entry["roles"]
        lang_entries = {}
        for lang, c in langs_passing(entry, scan_cfg).items():
            total = c["total"]

            def cell(n, total=total):
                return {"abs": n, "rel": round(n / total * 100, 2) if total else 0.0}

            lang_entries[lang] = {
                "neither": cell(c["neither"]),
                f"{role_a}_only": cell(c["a_only"]),
                f"{role_b}_only": cell(c["b_only"]),
                "both": cell(c["yes"] + c["no"]),
                "total": total,
            }
            values = entry["lang_values"].get(lang)
            if values:
                lang_entries[lang]["head_values"], lang_entries[lang]["chance_adjusted"] = agreement_diagnostics(
                    values["head"], values["child"], c["yes"], c["no"])
        if lang_entries:
            config[key] = lang_entries
    return config


def build_lang_csv(config: dict, stats: dict) -> pd.DataFrame:
    rows = []
    for key, lang_entries in config.items():
        role_a, role_b = stats[key]["roles"]
        for lang, s in sorted(lang_entries.items()):
            rows.append({
                "target": key, "lang": lang,
                "a_role": role_a, "a_only": s[f"{role_a}_only"]["abs"], "a_only_%": s[f"{role_a}_only"]["rel"],
                "b_role": role_b, "b_only": s[f"{role_b}_only"]["abs"], "b_only_%": s[f"{role_b}_only"]["rel"],
                "both": s["both"]["abs"], "both_%": s["both"]["rel"],
                "neither": s["neither"]["abs"], "neither_%": s["neither"]["rel"],
                "total": s["total"],
                "head_values": s.get("head_values"), "chance_adjusted": s.get("chance_adjusted"),
            })
    return pd.DataFrame(rows)


def report_lines(summary: pd.DataFrame, min_langs: int) -> list[str]:
    lines = []
    for group, rows in summary.groupby("group", sort=False):
        lines.append(f"\n=== {group} ===")
        keep, drop = rows[rows["n_langs_joint"] >= min_langs], rows[rows["n_langs_joint"] < min_langs]
        for title, part in (("worth an agreement check:", keep), ("skip (never/rarely jointly tagged):", drop)):
            if len(part):
                lines.append(f"  {title}")
                for _, r in part.iterrows():
                    tag = " (known)" if r["known"] else ""
                    lines.append(f"    {r['feature']:<12} joint={r['joint_attested']:<8} "
                                 f"agree={r['pct_agree']}% in {r['n_langs_joint']}/{r['n_langs_seen']} langs{tag}")
    return lines


def cache_state(mode: str, family: str | None, lang: str) -> str:
    """"cached", "empty" (extracted, nothing to cache: sentinel), "outdated" (a cache
    from another extraction version) or "missing"."""
    path = cache_file(mode, family, lang)
    if os.path.exists(path):
        return "cached" if parquet_current(path) else "outdated"
    return "empty" if sentinel_current(path[: -len(".parquet")] + ".empty.json") else "missing"


def save_results(mode: str, results: dict, units: dict, thresholds: dict, scanned_langs: set,
                 scanned_families: set, log=None) -> None:
    """Write the scan file. A partial scan (some languages, some vp families) replaces
    only what it scanned, unless the thresholds changed: then nothing older is valid
    and the file starts over."""
    old = load_scan_results(mode)
    if old and not same_thresholds(old.get("thresholds"), thresholds):
        (log.info if log else print)(
            f"{mode}: thresholds changed since the saved scan; starting the file over "
            "(languages not scanned now count as unscanned)")
        old = {}

    def in_scope(target):
        return mode != "vp" or target.split(":", 1)[0] in scanned_families

    targets = {t: dict(langs) for t, langs in old.get("targets", {}).items()}
    for target in list(targets):
        if in_scope(target):
            if scanned_langs:
                for lang in scanned_langs:
                    targets[target].pop(lang, None)
            else:
                targets.pop(target)
    for target, langs in results.items():
        targets.setdefault(target, {}).update(langs)
    out = {"thresholds": thresholds, "scanned": {**old.get("scanned", {}), **units},
           "targets": {t: langs for t, langs in targets.items() if langs}}

    os.makedirs(os.path.dirname(LANGS_PATH[mode]), exist_ok=True)
    tmp = LANGS_PATH[mode] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, LANGS_PATH[mode])


def resolve_target_langs(mode: str, target: str, universe: list[str], scan_cfg: dict) -> tuple[list[str], dict]:
    """The languages worth fitting for `target`, from the saved scan where it is
    still valid, scanning the rest now (and saving that scan, so it is not repeated).

    A language is judged from the saved file if its cache file is no newer than
    when it was scanned and the file's thresholds equal the config's; otherwise
    (never scanned, cache rebuilt since, thresholds changed) it is scanned now.
    A language without a current cache file (empty, missing, outdated) is not
    considered. Returns (languages, info with n_cached, n_saved, n_scanned_now)."""
    saved = load_scan_results(mode)
    current = same_thresholds(saved.get("thresholds"), scan_thresholds(scan_cfg))
    scanned = saved.get("scanned", {}) if current else {}
    key = vp_scan_key(*parse_vp_target(target)) if mode == "vp" else target
    qualifying = saved.get("targets", {}).get(key, {}) if current else {}
    family = parse_vp_target(target)[0] if mode == "vp" else None

    valid, to_scan, n_cached = [], [], 0
    for lang in universe:
        if cache_state(mode, family, lang) != "cached":
            continue
        n_cached += 1
        if scanned.get(unit_key(mode, family, lang), -1) >= os.path.getmtime(cache_file(mode, family, lang)):
            if lang in qualifying:
                valid.append(lang)
        else:
            to_scan.append(lang)
    if to_scan:
        if mode == "vp":
            stats, _, units = scan_vp(to_scan, families=[family], exclude_features=scan_cfg["exclude_features"])
        else:
            stats, _, units = scan_np(to_scan, split_head_upos=scan_cfg["split_head_upos"],
                                    exclude_features=scan_cfg["exclude_features"])
        results = build_lang_config(stats, scan_cfg)
        save_results(mode, results, units, scan_thresholds(scan_cfg), set(to_scan), {family})
        valid += list(results.get(key, {}))
    return sorted(valid), {"n_cached": n_cached, "n_saved": n_cached - len(to_scan), "n_scanned_now": len(to_scan)}
