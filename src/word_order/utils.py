import html as html_lib
import itertools
from urllib.parse import quote

import pandas as pd

from .prediction_target import PredictionTarget


ALL_CORE_ARGS = ["vos", "vso", "ovs", "svo", "osv", "sov"]

MAX_TREEBANK_LEN = 30_000


def capitalize_first(word: str) -> str:
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
    """Build an HTML <a> link to a universal.grew.fr query matching this sentence,
    constraining the given ordered word forms (one grew "form" slot per value).

    Shared by word_order/viz_tree.py's build_treebank_links and sva_trees/
    flowchart.py's example tables, which both need this exact query construction.

    Percent-encodes the query values (not just interpolating them raw) since
    sent_id can itself contain characters like "+" (e.g. Abkhaz) that a naive
    f-string would leave unescaped in the URL; standard query-string decoding
    then turns that "+" into a space server-side, silently corrupting the
    sent_id being matched.

    Returns None if treebank or sent_id is missing.
    """
    if treebank is None or sent_id is None or pd.isna(treebank) or pd.isna(sent_id):
        return None

    letters = [chr(ord("A") + i) for i in range(len(form_values))]
    slot = ";".join(
        f' {letter} [form="{form}"|"{str(form).capitalize()}"] '
        for letter, form in zip(letters, form_values)
    )
    request_value = f'pattern {{ meta.sent_id = "{sent_id}" ;{slot} }}'
    corpus_value = f"{treebank}@2.18"
    href = (
        f"https://universal.grew.fr/?corpus={quote(corpus_value, safe='')}"
        f"&request={quote(request_value, safe='')}"
    )
    text = html_lib.escape(str(link_text if link_text is not None else treebank))
    return f"<a href='{href}' target='_blank' rel='noopener'>{text}</a>"


def get_all_orders(predictor_var: str, target: PredictionTarget):
    if predictor_var == "core_args":
        all_orders = ALL_CORE_ARGS
    else:
        #print("target", target, type(target))

        deprels = target.child_deprels + ["head"]
        all_orders = [
            shorten_cls("_".join(permutation), target)
            for permutation in sorted(
                itertools.permutations(deprels, len(deprels)),
                key=lambda p: (p.index("head"), "_".join(p)),
            )
        ]

    return all_orders
