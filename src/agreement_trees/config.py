"""Settings for the agreement pipelines (caching, candidate scan, tree fit), read
from resources/agreement_config.json (keys: modes, langs, workers, vp, np, scan,
fit), and the per-target language lists the scans write.

Target ids: "sv:Number" (verb-phrase family + feature; the Number/Gender/Person
ones are also written svNa/svGa/svPa, and the output directories use that
form) or "HEAD-DET_Number" (noun-phrase role pair + feature).
"""
import functools
import json
import os

from agreement_trees.taxonomy import FEATURE_SUFFIXES, GROUP_PREFIXES
from multiblimp.config import RESOURCE_DIR, SCAN_RESULTS_DIR

AGREEMENT_CONFIG_PATH = os.path.join(RESOURCE_DIR, "agreement_config.json")
LANGS_PATH = {
    "vp": os.path.join(SCAN_RESULTS_DIR, "agreement_langs_vp.json"),
    "np": os.path.join(SCAN_RESULTS_DIR, "agreement_langs_np.json"),
}

VP_FAMILIES = list(GROUP_PREFIXES)
VP_GROUPS = {
    "verb": ["sv", "ov", "iov"],
    "participle": ["sp", "op", "iop"],
    "aux": ["sa", "oa", "ioa"],
    "other": ["sx"],
}
_SUFFIX_OF_FEATURE = {feature: suffix for suffix, feature in FEATURE_SUFFIXES.items()}

def expand_families(names: list[str]) -> list[str]:
    """Family names and/or group names (verb, participle, aux) -> family ids."""
    families = []
    for name in names:
        families += VP_GROUPS.get(name, [name])
    unknown = set(families) - set(VP_FAMILIES)
    if unknown:
        raise SystemExit(f"unknown vp families {sorted(unknown)} (valid: {VP_FAMILIES}, groups {sorted(VP_GROUPS)})")
    return list(dict.fromkeys(families))


def load_agreement_config(path: str | None = None) -> dict:
    """The JSON config with vp.families expanded to family ids. Every setting
    lives in the file; there are no code defaults."""
    path = path or AGREEMENT_CONFIG_PATH
    with open(path) as f:
        config = json.load(f)
    bad_modes = set(config["modes"]) - {"vp", "np"}
    if bad_modes:
        raise SystemExit(f"{path}: modes must be 'vp' and/or 'np', got {sorted(bad_modes)}")
    config["vp"]["families"] = expand_families(config["vp"]["families"])
    return config


@functools.lru_cache(maxsize=1)
def lexical_features() -> frozenset:
    """Features that never count as agreement (scan.exclude_features); no agreement
    column is extracted for them."""
    return frozenset(load_agreement_config()["scan"]["exclude_features"])


def apply_mode_flags(config: dict, vp: bool, np: bool) -> dict:
    """--vp / --np on the command line override the config's modes."""
    if vp or np:
        config["modes"] = [m for m, on in (("vp", vp), ("np", np)) if on]
    return config


def parse_vp_target(target: str) -> tuple[str, str] | None:
    """(family, feature) for "sv:Number" or a fixed id like "svNa"; None if it is
    not a verb-phrase target."""
    family, sep, feature = target.partition(":")
    if sep and family in VP_FAMILIES:
        return family, feature
    for prefix in sorted(VP_FAMILIES, key=len, reverse=True):
        suffix = target[len(prefix):]
        if target.startswith(prefix) and suffix in FEATURE_SUFFIXES:
            return prefix, FEATURE_SUFFIXES[suffix]
    return None


def vp_target_id(family: str, feature: str) -> str:
    """Output directory name: "svNa" for the Number/Gender/Person conditions,
    else "svCase"."""
    return f"{family}{_SUFFIX_OF_FEATURE.get(feature, feature)}"


def vp_scan_key(family: str, feature: str) -> str:
    return f"{family}:{feature}"


def load_scan_results(mode: str) -> dict:
    """The saved scan file written by find_agreement_candidates.py, {} if the scan
    has not run:
      "thresholds"  the scan settings it was computed with
      "scanned"     cache unit -> modification time of the cache file scanned
                    (vp: "family/lang", np: "lang")
      "targets"     target -> {lang: counts} for the languages that qualify
    A cache unit that is scanned but absent from a target's languages was
    found lacking."""
    try:
        with open(LANGS_PATH[mode]) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
