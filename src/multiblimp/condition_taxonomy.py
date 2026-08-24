"""Naming convention shared by every agreement condition this project runs:
sv/sp/sa + Na/Ga/Pa for subject-verb/participle/auxiliary agreement (e.g.
"svNa" = Subject-Verb, Number), and npa/{ROLE1}-{ROLE2}_{FeatAbbrev} for
noun-phrase agreement role pairs (e.g. "npa/HEAD-DET_N" = Head-Determiner
Number). Both scripts/overview/build_stats.py (the published-site stats
scraper) and word_order.viz_overview.py (the local decision-tree index page)
independently classified conditions by this same convention before this
module existed; it's the single source of truth for it now, so the two
pages can't drift apart on what a condition id means.
"""

GROUP_PREFIXES = {
    "sv": "Subject–Verb",
    "sp": "Subject–Participle",
    "sa": "Subject–Auxiliary",
}
FEATURE_SUFFIXES = {
    "Na": "Number",
    "Ga": "Gender",
    "Pa": "Person",
}
FEATURE_ORDER = {"Na": 0, "Ga": 1, "Pa": 2}

NPA_ROLE_PROSE = {
    "HEAD": "Head", "DET": "Determiner", "NUM": "Numeral",
    "ADJ": "Adjective", "ADP": "Adposition", "PRON": "Pronoun",
}
NPA_ROLE_ORDER = {"HEAD": 0, "DET": 1, "NUM": 2, "ADJ": 3, "ADP": 4, "PRON": 5}
NPA_FEATURE_NAMES = {
    "N": "Number", "G": "Gender", "P": "Person", "C": "Case",
    "Def": "Definite", "Deg": "Degree", "PT": "PronType",
    "NT": "NumType", "Pos": "Poss",
}
NPA_FEATURE_ORDER = {"N": 0, "G": 1, "P": 2, "C": 3}

# (group, label) for every flat sv/sp/sa condition, in GROUP_PREFIXES x
# FEATURE_SUFFIXES reading order (svNa, svGa, svPa, spNa, ..., saPa) --
# callers that also track NPA's own aggregate condition (e.g.
# build_stats.py's "npa": ("Noun Phrase", "Agreement (all role pairs)"))
# add that entry themselves, since NPA has no single flat id here.
FLAT_CONDITION_META = {
    f"{prefix}{suffix}": (group, FEATURE_SUFFIXES[suffix])
    for prefix, group in GROUP_PREFIXES.items()
    for suffix in FEATURE_SUFFIXES
}


def npa_subgroup_label(subgroup: str) -> str:
    # e.g. "HEAD-DET_N" -> "Head–Determiner Number"
    try:
        roles, feat = subgroup.split("_", 1)
        role1, role2 = roles.split("-")
        role_label = f"{NPA_ROLE_PROSE.get(role1, role1)}–{NPA_ROLE_PROSE.get(role2, role2)}"
        return f"{role_label} {NPA_FEATURE_NAMES.get(feat, feat)}"
    except ValueError:
        return subgroup
