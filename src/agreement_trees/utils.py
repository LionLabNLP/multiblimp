import html as html_lib
import itertools
import re
from urllib.parse import quote

import pandas as pd

from agreement_trees.extract.prediction_target import PredictionTarget


ALL_CORE_ARGS = ["vos", "vso", "ovs", "svo", "osv", "sov"]   # LEGACY: only get_all_orders reads it

MAX_TREEBANK_LEN = 30_000   # LEGACY: unused

# A predictor_var names an agreement relation either as VP's "head_{deprel}_{Feature}_agreement" or as the
# pairwise "{Role1}-{Role2}_{Feature}" of NPA ("yes" = the agreement holds). VP names never contain "-",
# so checking the pairwise pattern first is unambiguous. Shared by viz_tree and viz_deprel.
_PAIRWISE_AGREEMENT_RE = re.compile(r"^([A-Za-z]+)-([A-Za-z]+)_([A-Z][a-zA-Z]+)$")


def split_pairwise_predictor(predictor_var: str) -> tuple[str, str, str] | None:
    """(role1, role2, feature) for a "{Role1}-{Role2}_{Feature}" predictor_var,
    or None if it doesn't match that convention."""
    m = _PAIRWISE_AGREEMENT_RE.match(predictor_var)
    return m.groups() if m else None


def is_agreement_predictor(predictor_var: str) -> bool:
    """True for either agreement predictor_var convention: VP's "..._agreement" suffix or the pairwise one."""
    return bool(predictor_var) and (
        "agreement" in predictor_var or split_pairwise_predictor(predictor_var) is not None
    )


def capitalize_first(word: str) -> str:
    """LEGACY (only the dead word_order create_pairs/create_tree_pairs use it)."""
    if len(word) == 0:
        return ""
    return word[0].upper() + word[1:]


def shorten_cls(cls, target):
    if isinstance(target.head_pos, list) and len(target.head_pos) == 1:
        head_pos = target.head_pos[0][0]
    else:
        head_pos = "H"

    cls_swap = {
        "det": "d",
        "nummod": "n",
        "amod": "a",
        "head": head_pos,
        "nsubj": "s",
        "obj": "o",
        "_": "",
        "acl:relcl": "v",
        "obl": "c",
        "iobj": "i",
        "advmod": "a",
        "nmod": "n",
    }

    result = cls
    for old, new in cls_swap.items():
        result = result.replace(old, new)

    return result


def build_grew_link(treebank, sent_id, form_values, link_text=None) -> str | None:
    """HTML <a> link to a universal.grew.fr query matching this sentence and constraining the given ordered
    word forms (one grew "form" slot per value; use the untouched "*_form_orig" spelling so one exact match
    suffices). The query values are percent-encoded (a sent_id can contain "+", which would decode to a
    space). Shared by viz_tree and agreement_trees.pairs.create_pairs. None if treebank or sent_id is missing.
    """
    href = build_grew_url(treebank, sent_id, form_values)
    if href is None:
        return None
    text = html_lib.escape(str(link_text if link_text is not None else treebank))
    return f"<a href='{href}' target='_blank' rel='noopener'>{text}</a>"


def build_grew_url(treebank, sent_id, form_values) -> str | None:
    """The universal.grew.fr URL of build_grew_link's query, or None if treebank or
    sent_id is missing."""
    if treebank is None or sent_id is None or pd.isna(treebank) or pd.isna(sent_id):
        return None

    letters = [chr(ord("A") + i) for i in range(len(form_values))]
    slot = ";".join(
        f' {letter} [form="{form}"] '
        for letter, form in zip(letters, form_values)
    )
    request_value = f'pattern {{ meta.sent_id = "{sent_id}" ;{slot} }}'
    corpus_value = f"{treebank}@2.18"
    return (
        f"https://universal.grew.fr/?corpus={quote(corpus_value, safe='')}"
        f"&request={quote(request_value, safe='')}"
    )


def get_all_orders(predictor_var: str, target: PredictionTarget):
    """LEGACY (word-order block of decision_tree.set_dt_features_in_df, dead create_pairs): every order of
    the target's deprels and head, shortened."""
    if predictor_var == "core_args":
        all_orders = ALL_CORE_ARGS
    else:
        deprels = target.child_deprels + ["head"]
        all_orders = [
            shorten_cls("_".join(permutation), target)
            for permutation in sorted(
                itertools.permutations(deprels, len(deprels)),
                key=lambda p: (p.index("head"), "_".join(p)),
            )
        ]

    return all_orders
