import json


def load_npa_config(path: str) -> dict[str, dict[str, dict]]:
    """target_col -> {lang: {"neither": {"abs": int, "rel": float},
    "{role1}_only": {...}, "{role2}_only": {...}, "both": {...},
    "total": int}} for every language worth attempting for it (see
    scripts/npa/agreement_candidates.py's build_lang_config, which produces
    this file from a pooled scan of every language's np_instances parquet).

    "total" is every co-occurrence of the role pair (feature-tagged or not);
    "neither"/"{role1}_only"/"{role2}_only"/"both" partition it exactly, each
    with "abs" (raw count) and "rel" (that count as % of total). E.g.
    {"German": {"both": {"abs": 506, "rel": 91.5}, "total": 553, ...}} for
    DET-ADJ_Gender means 506/553 (91.5%) of German's DET-ADJ co-occurrences
    had Gender annotated on both sides.

    {} if `path` doesn't exist -- callers should treat that the same as an
    empty config (fall back to their own full language list), not an error,
    since the config is an optional speed-up, not a required input.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def config_langs_for(config: dict[str, dict[str, dict]], target_col: str,
                      fallback: list[str]) -> list[str]:
    """Languages to attempt for `target_col`: config's own language keys when
    it has an entry (even an empty result never happens -- build_lang_config
    only ever writes non-empty entries), else `fallback` (e.g. the caller's
    full get_ud_langs() list) -- covers both a target_col the scan never saw
    (typo, or a genuinely new feature pair) and a stale config predating a
    newly-added language's np_instances parquet.
    """
    entry = config.get(target_col)
    return list(entry.keys()) if entry else fallback


def describe(config: dict[str, dict[str, dict]], target_col: str, lang: str) -> str:
    """"506/553 (91.5%)" for one (target_col, lang)'s both/total -- the
    exact reading build_lang_config's docstring spells out for its "both"
    bucket -- or "n/a" if this (target_col, lang) has no entry (below the
    Wilson-score floor at scan time, or never co-occurred at all).
    """
    stats = config.get(target_col, {}).get(lang)
    if stats is None:
        return "n/a"
    both = stats["both"]
    return f"{both['abs']}/{stats['total']} ({both['rel']}%)"
