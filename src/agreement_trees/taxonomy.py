"""Naming convention shared by every agreement condition this project runs:
sv/sp/sa + Na/Ga/Pa for subject-verb/participle/auxiliary agreement,
ov/op/oa for object-verb/participle/auxiliary, iov/iop/ioa for indirect-
object-verb/participle/auxiliary agreement (e.g. "svNa" = Subject-Verb, Number), sx<form> for subject agreement
with a head of another VerbForm (OTHER_VERB_FORMS, e.g. "sxinfNa" = Subject-Infinitive, Number), and
npa/{ROLE1}-{ROLE2}_{FeatAbbrev} for noun-phrase agreement role pairs
(e.g. "npa/HEAD-DET_N" = Head-Determiner Number). Both
scripts/overview/build_stats.py (the stats-overview aggregator) and
agreement_trees.report.viz_overview.py (the local decision-tree index page)
independently classified conditions by this same convention before this
module existed; it's the single source of truth for it now, so the two
pages can't drift apart on what a condition id means.
"""

import re

# Subject agreement with a head whose VerbForm is neither Fin nor Part: one family per VerbForm value, all shown
# under one group. family prefix -> (VerbForm value, label)
OTHER_VERB_FORMS = {
    "sxinf": ("Inf", "Infinitive"),
    "sxvnoun": ("Vnoun", "Verbal noun"),
    "sxgdv": ("Gdv", "Gerundive"),
    "sxger": ("Ger", "Gerund"),
    "sxconv": ("Conv", "Converb"),
    "sxsup": ("Sup", "Supine"),
    "sxnonfin": ("NonFin", "Non-finite"),
    "sxstat": ("Stat", "Stative"),
    "sxnomag": ("NomAg", "Agent noun"),
}
OTHER_VERB_FORMS_GROUP = "Subject–Other verb forms"

GROUP_PREFIXES = {
    "sv": "Subject–Verb",
    "sp": "Subject–Participle",
    **{prefix: OTHER_VERB_FORMS_GROUP for prefix in OTHER_VERB_FORMS},
    "sa": "Subject–Auxiliary",
    "ov": "Object–Verb",
    "op": "Object–Participle",
    "oa": "Object–Auxiliary",
    "iov": "Indirect Object–Verb",
    "iop": "Indirect Object–Participle",
    "ioa": "Indirect Object–Auxiliary",
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
def _feature_label(prefix: str, feature: str) -> str:
    return f"{OTHER_VERB_FORMS[prefix][1]} ({feature})" if prefix in OTHER_VERB_FORMS else feature


FLAT_CONDITION_META = {
    f"{prefix}{suffix}": (group, _feature_label(prefix, FEATURE_SUFFIXES[suffix]))
    for prefix, group in GROUP_PREFIXES.items()
    for suffix in FEATURE_SUFFIXES
}


_FLAT_ID_RE = re.compile(
    rf"^({'|'.join(sorted(GROUP_PREFIXES, key=len, reverse=True))})([A-Z][A-Za-z]*)(?:_(.+))?$"
)
_CANDIDATE_ORDER = len(FEATURE_ORDER)


def classify_flat_condition(cid: str):
    """(group, feature label, sort order, variant) for a fixed (svNa, spGa_keepunk)
    or ad-hoc candidate (svCase, saNounClass) condition id, else None. Candidates
    go in the "Other" group, labelled with their family ("Subject–Verb Case")."""
    match = _FLAT_ID_RE.match(cid)
    if not match:
        return None
    prefix, suffix, variant = match.groups()
    if suffix in FEATURE_SUFFIXES:
        return GROUP_PREFIXES[prefix], _feature_label(prefix, FEATURE_SUFFIXES[suffix]), FEATURE_ORDER[suffix], variant
    family = f"Subject–{OTHER_VERB_FORMS[prefix][1]}" if prefix in OTHER_VERB_FORMS else GROUP_PREFIXES[prefix]
    return "Other", f"{family} {suffix}", _CANDIDATE_ORDER, variant


def other_verb_form(cid: str):
    """(VerbForm label, feature label) of a condition id of the other-verb-forms families ("sxinfNa" ->
    ("Infinitive", "Number"), "sxinfCase" -> ("Infinitive", "Case")), else None."""
    match = _FLAT_ID_RE.match(cid)
    if not match or match.group(1) not in OTHER_VERB_FORMS:
        return None
    suffix = match.group(2)
    return OTHER_VERB_FORMS[match.group(1)][1], FEATURE_SUFFIXES.get(suffix, suffix)


def npa_subgroup_label(subgroup: str) -> str:
    # e.g. "HEAD-DET_N" -> "Head–Determiner (Number)"
    try:
        roles, feat = subgroup.split("_", 1)
        role1, role2 = roles.split("-")
        role_label = f"{NPA_ROLE_PROSE.get(role1, role1)}–{NPA_ROLE_PROSE.get(role2, role2)}"
        return f"{role_label} ({NPA_FEATURE_NAMES.get(feat, feat)})"
    except ValueError:
        return subgroup
