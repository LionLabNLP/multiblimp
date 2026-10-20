import sys
import os
import random
import re
import gc

from collections import defaultdict, Counter
from dataclasses import dataclass

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from typing import *

from agreement_trees.config import lexical_features
from agreement_trees.pipeline_utils import write_parquet
from multiblimp.config import REPO_ROOT, SEED
from multiblimp.treebank import sentence_line, token_line
from multiblimp.ud_table import clean_lemma
from multiblimp.ud_table import UDTable
from multiblimp.treebank import Treebank
from multiblimp.languages import remove_diacritics_langs, gblang2udlang

sys.path.append(REPO_ROOT)   # for resources.*
from resources.um2ud_annotation.UD2UM_mapper import expand_ud_feats

from agreement_trees.extract.prediction_target import PredictionTarget, CHILD_DEPREL_ALIASES, child_slot
from agreement_trees.utils import shorten_cls


# Emit "{prefix}_xpos" as a node feature (SVA_DT_XPOS=0 turns it off). Some tagsets (Czech positional
# tags) spell out gender/number/case, so it can leak the agreement label; the fits omit it.
INCLUDE_XPOS = os.environ.get("SVA_DT_XPOS", "1") != "0"

META_FEATURES = ["sen", "no_space_after", "treebank", "sent_id", "tree_idx", "conllu_file", "conllu_sent_line", "treebank_link", "sen_str", "filled_feats"]


def is_attractor_col(col: str) -> bool:
    """A "{role}_attractors_{Feat}_{bucket}" count column; metadata, never a fit predictor."""
    return "_attractors_" in col


def _fill_span_token(tok, feats, um_data, ud_data) -> None:
    """Lexicon fill (as for the role tokens) for an in-between token, only if the lexicon can supply one of
    `feats` for its UPOS and the token lacks it; tried once per token."""
    if tok.get("attr_filled"):
        return
    tok["attr_filled"] = True
    if type(tok.get("feats")) != dict:
        tok["feats"] = dict()
    for data in (um_data, ud_data):
        table = data.get(tok["upos"]) if isinstance(data, UDTable) else None
        if table is not None and any(f in table.ud_features and not tok["feats"].get(f) for f in feats):
            fill_node_feats(tok, um_data, ud_data)
            return


def is_attractor_slots_col(col: str) -> bool:
    """A "{role}_attractor_slots" list column (see attractor_cols); metadata, never a fit predictor."""
    return col.endswith("_attractor_slots")


ATTRACTOR_BUCKETS = ("congruent", "weakly_congruent", "incongruent")


def attractor_bucket(ref, val) -> str:
    """Bucket of one attractor slot value against the controller's value. A comma value is a set of readings:
    disjoint sets are incongruent, equal single readings congruent, any other overlap weakly congruent."""
    ref_readings, readings = set(str(ref).split(",")), set(str(val).split(","))
    if not ref_readings & readings:
        return "incongruent"
    return "congruent" if len(ref_readings) == len(readings) == 1 else "weakly_congruent"


def slot_folds_into(feat: str, key: str, filled: bool) -> bool:
    """Whether a token's slot `key` counts as a slot of `feat`: the key itself, and for a plain feature its
    raw layered variants (Number[iobj] for Number). [psor] is the possessor's, and a lexicon-filled layered key
    only mirrors the plain value."""
    if key == feat:
        return True
    return "[" not in feat and key.startswith(feat + "[") and not filled and not key.endswith("[psor]")


def attractor_cols(tree, id_a: int, id_b: int, ref_vals: dict, prefix: str, features,
                   um_data=None, ud_data=None, fetch_all: bool = False) -> dict:
    """Attractor columns for the tokens strictly between ids `id_a` and `id_b` (either order).

    "{prefix}_attractor_slots": every token of the span as "id:UPOS:Key=Value" per slot of `features` (a
    trailing "*" marks a lexicon-filled value), or a bare "id:UPOS" for a token without any; None for an empty
    span. Independent of the controller, so any bucketing can be recomputed from it.
    "{prefix}_attractors_{feat}_{bucket}": counts of the slots of `feat` per attractor_bucket against
    `ref_vals[feat]`, the controller's value; none for a feature without one.
    With fetch_all, in-between tokens are lexicon-filled like the role tokens."""
    lo, hi = sorted((id_a, id_b))
    span = tree[lo:hi - 1]   # ids lo+1..hi-1
    bases = {f.split("[", 1)[0] for f in features}
    if fetch_all and bases:
        for tok in span:
            _fill_span_token(tok, bases, um_data, ud_data)
    entries, slots = [], []
    for tok in span:
        filled = tok.get("enriched_feats", ())
        found = False
        for key, val in (tok.get("feats") or {}).items():
            if key.split("[", 1)[0] in bases:
                found = True
                slots.append((key, val, key in filled))
                entries.append(f"{tok['id']}:{tok['upos']}:{key}={val}{'*' if key in filled else ''}")
        if not found:
            entries.append(f"{tok['id']}:{tok['upos']}")
    cols = {f"{prefix}_attractor_slots": entries or None}
    for feat, ref in ref_vals.items():
        if ref is None:
            continue
        counts = Counter(attractor_bucket(ref, val) for key, val, filled in slots
                         if slot_folds_into(feat, key, filled))
        for bucket in ATTRACTOR_BUCKETS:
            cols[f"{prefix}_attractors_{feat}_{bucket}"] = counts[bucket]
    return cols


ATTRACTOR_COUNT_DTYPE = "Int16"


def cast_attractor_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Attractor counts as nullable Int16 (NaN -> NA), set in place on `df` (an astype over the frame would copy
    every column). A count cannot exceed a sentence's token count; the assert makes an impossible overflow
    loud instead of wrapping."""
    for col in [c for c in df.columns if is_attractor_col(c)]:
        top = df[col].max()
        assert pd.isna(top) or top <= np.iinfo(np.int16).max, f"attractor count {top} overflows Int16"
        df[col] = df[col].astype(ATTRACTOR_COUNT_DTYPE)
    return df


@dataclass
class TreeMaps:
    child2root: dict
    head2child_ids: dict
    head2child_deprels: dict
    head2child_lemmas: dict
    head2left_deprels: dict
    head2right_deprels: dict
    head2child_pos: dict
    head2child_feats: dict


def load_treebank(
    lang: str, resource_dir: str | None = None, max_treebank_len: int | None = None,
    only_excluded: bool = False, use_selected_treebanks: bool = True,
) -> Treebank:
    """The language's treebank (from the pickle). use_selected_treebanks=False loads every treebank,
    selected and excluded alike, so callers can split them after extraction (agreement_trees.vp.pipeline); the
    pickle is re-filtered against the live exclusion config either way. Ignored with only_excluded.
    max_treebank_len samples that many trees (fixed seed)."""
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    treebank = Treebank(
        lang,
        remove_diacritics=(lang in remove_diacritics_langs),
        resource_dir=resource_dir,
        remove_typo=True,
        pickle_path="ud/ud_pickles",
        load_from_pickle=True,
        only_excluded=only_excluded,
        use_selected_treebanks=use_selected_treebanks,
    )
    if max_treebank_len is not None and len(treebank) > max_treebank_len:
        rng = random.Random(SEED)
        treebank = rng.sample(treebank, max_treebank_len)

    return treebank


def read_df(lang, word_order_dir=None, deprels=None) -> pd.DataFrame:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    deprel_suffix = "" if deprels is None else "_" + "_".join(deprels)

    df = pd.read_parquet(
        os.path.join(word_order_dir or "", f"{lang}{deprel_suffix}.parquet"),
    )

    return df


def tree2sen(tree):
    return [x["form"] for x in tree]


def conllu_sentence_meta(tree) -> dict:
    """Source file (UD_X/file.conllu) and first line of the sentence's block."""
    path = tree.metadata.get("treebank", "")
    return {"conllu_file": path, "conllu_sent_line": sentence_line(path, tree.metadata.get("sent_id"))}


def conllu_token_line(tree, token_id):
    return token_line(tree.metadata.get("treebank", ""), tree.metadata.get("sent_id"), token_id)


def tree_no_space_after(tree):
    """Per-token SpaceAfter=No flags, parallel to tree2sen(tree), saved so example tables can respace the sentence."""
    return [(tok["misc"] or {}).get("SpaceAfter") == "No" for tok in tree]


_SUBJ_FEAT_SUFFIXES = ("[subj]", "[nsubj]")


def merge_subj_layered_feats(treebank) -> None:
    """Fold a layered "[subj]" / "[nsubj]" feature (Abkhaz Number[subj]: plain Number is set on ~5% of rows,
    Number[subj] on all) into the plain key where that is absent, and drop the layered key. Other layered
    features (Number[psor]) are left alone. Mutates token["feats"]; must run before get_all_feats/extract_instances.
    """
    for tree in treebank:
        for token in tree:
            feats = token.get("feats")
            if not feats:
                continue
            for key in list(feats.keys()):
                for suffix in _SUBJ_FEAT_SUFFIXES:
                    if key.endswith(suffix):
                        base = key[: -len(suffix)]
                        feats.setdefault(base, feats[key])
                        del feats[key]
                        break


def get_all_feats(treebank, min_freq=0.005):
    all_feats = Counter()
    all_deprel = set()
    all_lemma_freqs = Counter()
    all_pos = set()
    num_tokens = 0

    for tree in treebank:
        for token in tree:
            all_lemma_freqs[token["lemma"]] += 1
            for feat in token["feats"] or {}:
                all_feats[feat] += 1

            all_deprel.add(token["deprel"])
            all_pos.add(token["upos"])
            num_tokens += 1

    all_feats = Counter(
        {x: c / num_tokens for x, c in all_feats.items() if (c / num_tokens) > min_freq}
    )

    return set(all_feats), all_lemma_freqs, all_deprel, all_pos


def expand_anno(node, morph_feats, um_split):
    add_feats, contested = candidate_fills(node, morph_feats, um_split)
    if contested:
        node["verbform_contested"] = True
    if add_feats:
        for feat, val in add_feats.items():
            morph_feats[feat] = val
        # provenance: link_head_slot_feats prefers treebank-annotated
        # values over lexicon-derived ones when merging slot spellings
        node.setdefault("enriched_feats", set()).update(add_feats)
    return morph_feats


def candidate_fills(node, morph_feats, um_split):
    """(features to add or False, contested): what the tightest-matching lexicon rows
    say about features the token lacks. Rows that don't state a feature say nothing
    about it; competing values become one ambiguous value ("Plur,Sing"). Nothing is
    added if a value conflicts with the token's own.

    A verb without VerbForm is matched as it is, so the lexicon can supply VerbForm.
    If its rows disagree on VerbForm it is contested: with a Mood it is matched again
    as Fin, without one it gets only the features all rows agree on."""
    inflect_feats = dict(expand_ud_feats(morph_feats) if um_split.expand_values else morph_feats)
    inflect_feats["upos"] = node["upos"]
    # no lemma in UD: match on form alone
    inflect_feats["lemma"] = node.get("lemma", None) if node.get("lemma", None) != "_" else None

    unified = _unify_rows(node, inflect_feats, um_split)
    if unified is None:
        return False, False
    query_feats = inflect_feats
    contested = len(unified.get("VerbForm", ())) > 1
    if contested and inflect_feats.get("Mood"):
        query_feats = {**inflect_feats, "VerbForm": "Fin"}
        unified = _unify_rows(node, query_feats, um_split)
        if unified is None:
            return False, True
    elif contested:
        unified = {k: v for k, v in unified.items() if k != "VerbForm" and len(v) == 1}
    unified = {k: ",".join(v) for k, v in unified.items()}

    if any(inflect_feats.get(k) and not _readings_overlap(inflect_feats[k], v) for k, v in unified.items()):
        return False, contested
    add_feats = {k: v for k, v in unified.items() if k not in query_feats}
    return (add_feats or False), contested


def _unify_rows(node, query_feats, um_split):
    """{feature: sorted values} over the tightest rows matching the query, None if
    there are none."""
    return um_split.unify(node["form"].lower(), clean_lemma(query_feats["lemma"], node["form"]),
                          {k: v for k, v in query_feats.items() if k not in ("upos", "lemma")})


def filled_feats(role_nodes: dict, extra=()) -> str | None:
    """"role:Feature" for every feature of an instance's tokens that is not in the
    original annotation (filled from the lexicon, or assumed), comma-joined, for
    reference at later stages; a META_FEATURES column, so never a fit predictor."""
    keys = {f"{role}:{feat}" for role, node in role_nodes.items()
            for feat in set(node.get("enriched_feats", ())) | set(node.get("assumed_feats", ()))}
    return ",".join(sorted(keys | set(extra))) or None


def _readings_overlap(a, b) -> bool:
    """Equal, or sharing a reading when either is a comma-joined ambiguous value."""
    return a == b or bool(set(str(a).split(",")) & set(str(b).split(",")))


def fill_node_feats(node, um_data, ud_data) -> None:
    """Missing annotation is filled from the UniMorph-derived table first, then from the UD-derived one
    (which only gets the features the first pass left open). Mutates node["feats"]."""
    if isinstance(um_data, UDTable):
        um_slice = um_data.get(node["upos"])
        if um_slice is not None and not all(node["feats"].get(k, False) for k in um_slice.ud_features):
            node["feats"] = expand_anno(node, node["feats"], um_slice)

    if isinstance(ud_data, UDTable):
        ud_split = ud_data.get(node["upos"])
        if ud_split is not None and not all(node["feats"].get(k, False) for k in ud_split.ud_features):
            node["feats"] = expand_anno(node, node["feats"], ud_split)


def extract_node_features(
    node,
    tree,
    tree_maps: TreeMaps,
    prefix: str,
    all_feats: set,
    all_deprel: set,
    all_pos: set,
    target: PredictionTarget,
    excluded_deprels: set | None = None,
    lexicalize: bool = False,
    um_data: UDTable | None = None,
    ud_data: UDTable | None = None,
    fetch_all=False,
    include_xpos: bool | None = None,
) -> dict[str, str | bool | int]:
    """
    Extract all features for a specific node (whether it's a child, head, or co-child).
    This ensures consistent feature extraction across all node types.
    include_xpos: None uses INCLUDE_XPOS.
    """
    features = {}
    if type(node.get("feats", None))!=dict: node["feats"] = dict()
    excluded_deprels = excluded_deprels or set()

    # Basic node features
    features[f"{prefix}_deprel"] = node["deprel"]
    features[f"{prefix}_pos"] = node["upos"]
    features[f"{prefix}_idx"] = node["id"]
    features[f"{prefix}_idx_line"] = conllu_token_line(tree, node["id"])
    if INCLUDE_XPOS if include_xpos is None else include_xpos:
        features[f"{prefix}_xpos"] = node.get("xpos")

    if lexicalize:
        features[f"{prefix}_form"] = (
            node["form"].lower() if node["upos"] != "PROPN" else node["form"]
        )
        # untouched spelling, for exact grew.fr links; omitted from the fits
        features[f"{prefix}_form_orig"] = node["form"]
        features[f"{prefix}_lemma"] = node["lemma"]

    if fetch_all:
        fill_node_feats(node, um_data, ud_data)

    all_feats.update(set(node.get("feats", {}).keys()))

    for feat in all_feats:
        features[f"{prefix}_{feat}"] = (node["feats"] or {}).get(feat)

    # A verb still without VerbForm after annotation filling counts as Fin, unless the lexicon
    # rows disagreed on it and the token has no Mood to settle it
    if (prefix == "head" and node["upos"] in ("VERB", "AUX")
            and "VerbForm" in (target.head_feats or {})
            and features.get("head_VerbForm") is None
            and not (node.get("verbform_contested") and features.get("head_Mood") is None)):
        features["head_VerbForm"] = "Fin"
        node.setdefault("assumed_feats", set()).add("VerbForm")

    # PredictionTarget.head_feats predicates on the head
    if prefix=="head" and target.head_feats is not None:
        if not all(
            [val_filter(features.get(f"{prefix}_{feat}", None)) 
             for feat, val_filter in target.head_feats.items()]
             ):
            return None # item does not fulfil PredictionTarget feature filters, drop instance

    # Features about this node's relationship to its head
    if node["head"] != 0:
        head = tree[node["head"] - 1]
        features[f"{prefix}_head_pos"] = head["upos"]
        features[f"{prefix}_head_deprel"] = head["deprel"]
        features[f"{prefix}_dir"] = int(node["id"] > node["head"])

        # Grandhead features
        if head["head"] != 0:
            features[f"{prefix}_grandhead_deprel"] = tree[head["head"] - 1]["deprel"]

    # Path to root features
    for deprel in all_deprel:
        features[f"{prefix}_under_{deprel}"] = (
            deprel in tree_maps.child2root[node["id"]]
        )

    # Sibling features (if node has siblings)
    if node["head"] != 0:
        # Get siblings excluding this node
        def remove_first_occurrence(lst, x):
            result = []
            removed = False
            for item in lst:
                if not removed and item == x:
                    removed = True
                    continue
                result.append(item)
            return result

        sibling_deprels = remove_first_occurrence(
            tree_maps.head2child_deprels[node["head"]], node["deprel"]
        )
        sibling_pos = tree_maps.head2child_pos[node["head"]].copy()

        sibling_deprel_candidates = set(all_deprel)

        # # Sibling deprel presence
        for deprel in sibling_deprel_candidates:
            features[f"{prefix}_sibling-deprel_{deprel}"] = deprel in sibling_deprels

        # # Sibling POS presence
        for pos in all_pos:
            features[f"{prefix}_sibling-pos_{pos}"] = pos in sibling_pos

        # Directional sibling deprels
        for sibling_dir in ["L", "R"]:
            if sibling_dir == "L":
                dir_sibling_deprels = list(
                    tree_maps.head2left_deprels[node["head"]]
                )
                if node["id"] < node["head"]:
                    dir_sibling_deprels = remove_first_occurrence(
                        dir_sibling_deprels, node["deprel"]
                    )
            else:
                dir_sibling_deprels = list(
                    tree_maps.head2right_deprels[node["head"]]
                )
                if node["id"] > node["head"]:
                    dir_sibling_deprels = remove_first_occurrence(
                        dir_sibling_deprels, node["deprel"]
                    )

            for deprel in sibling_deprel_candidates:
                features[f"{prefix}_sibling-deprel-{sibling_dir}_{deprel}"] = (
                    deprel in dir_sibling_deprels
                )

        # Sibling lemma for specific deprels
        if lexicalize:
            sibling_lemmas = tree_maps.head2child_lemmas[node["head"]]
            for deprel in sibling_deprel_candidates:
                lemma_val = "None"
                for dep, lem in zip(
                    tree_maps.head2child_deprels[node["head"]], sibling_lemmas
                ):
                    if dep == deprel and dep != node["deprel"]:
                        lemma_val = lem
                        break
                features[f"{prefix}_sibling-lemma_{deprel}"] = lemma_val

        # Sibling morphological features
        for feat in all_feats:
            for deprel in sibling_deprel_candidates:
                if deprel == node["deprel"]:
                    features[f"{prefix}_sibling-feat_{deprel}_{feat}"] = None
                else:
                    features[f"{prefix}_sibling-feat_{deprel}_{feat}"] = (
                        tree_maps.head2child_feats[node["head"]]
                        .get(deprel, {})
                        .get(feat)
                    )

    # Child features (node's own dependents)
    child_deprel_candidates = set(all_deprel) - excluded_deprels
    child_deprels = tree_maps.head2child_deprels[node["id"]]
    child_pos = tree_maps.head2child_pos[node["id"]]

    for deprel in child_deprel_candidates:
        features[f"{prefix}_child-deprel_{deprel}"] = deprel in child_deprels

    for pos in all_pos:
        features[f"{prefix}_child-pos_{pos}"] = pos in child_pos

    if lexicalize:
        child_lemmas = tree_maps.head2child_lemmas[node["id"]]
        for deprel in child_deprel_candidates:
            lemma_val = "None"
            for dep, lem in zip(child_deprels, child_lemmas):
                if dep == deprel:
                    lemma_val = lem
                    break
            features[f"{prefix}_child-lemma_{deprel}"] = lemma_val

    for feat in all_feats:
        for deprel in child_deprel_candidates:
            features[f"{prefix}_child-feat_{deprel}_{feat}"] = (
                tree_maps.head2child_feats[node["id"]].get(deprel, {}).get(feat)
            )

    return features


def extract_sen_features(tree):
    """
    Extract sentence-level features that apply to the whole sentence.
    Sets core argument structure of main clause and `in_question` features.

    Returns:
        dict of feature_name -> feature_value
    """
    feature2val = {}

    subj_idx = None
    verb_idx = None
    obj_idx = None

    for token in tree:
        head = token["head"]
        if head == 0:
            continue
        head_is_root = tree[head - 1]["deprel"] == "root"

        if head_is_root:
            if token["deprel"] == "nsubj":
                subj_idx = token["id"]
                verb_idx = head
            elif token["deprel"] == "obj":
                obj_idx = token["id"]

    if subj_idx is None or verb_idx is None:
        core_arg_order = None
    elif obj_idx is None:
        core_arg_order = "sv" if subj_idx < verb_idx else "vs"
    else:
        arg_order = np.argsort([subj_idx, verb_idx, obj_idx])
        core_arg_order = "".join(np.array(["s", "v", "o"])[arg_order])

    feature2val["core_args"] = core_arg_order
    feature2val["subject_idx"] = subj_idx
    feature2val["object_idx"] = obj_idx
    feature2val["verb_idx"] = verb_idx

    # Question detection
    feature2val["in_question"] = tree[-1]["form"] == "?"


    return feature2val


def _build_tree_maps(tree) -> TreeMaps:
    """Build per-tree index structures needed by extract_node_features."""
    child2head = {}
    child2root = defaultdict(set)
    head2child_ids = defaultdict(list)
    head2child_deprels = defaultdict(list)
    head2child_pos = defaultdict(set)
    head2child_lemmas = defaultdict(list)
    head2child_feats = defaultdict(dict)
    head2left_deprels = defaultdict(list)
    head2right_deprels = defaultdict(list)

    for child in tree:
        child2head[child["id"]] = child["head"]
        head2child_ids[child["head"]].append(child["id"])
        head2child_deprels[child["head"]].append(child["deprel"])
        head2child_pos[child["head"]].add(child["upos"])
        head2child_lemmas[child["head"]].append(child["lemma"])

        head2child_feats[child["head"]][child["deprel"]] = {}
        for feat, value in (child["feats"] or {}).items():
            head2child_feats[child["head"]][child["deprel"]][feat] = value

        if child["id"] < child["head"]:
            head2left_deprels[child["head"]].append(child["deprel"])
        else:
            head2right_deprels[child["head"]].append(child["deprel"])

    for child, head in child2head.items():
        child2root[child].add(tree[child - 1]["deprel"])
        while head != 0:
            child2root[child].add(tree[head - 1]["deprel"])
            head = child2head[head]

    return TreeMaps(
        child2root=child2root,
        head2child_ids=head2child_ids,
        head2child_deprels=head2child_deprels,
        head2child_lemmas=head2child_lemmas,
        head2left_deprels=head2left_deprels,
        head2right_deprels=head2right_deprels,
        head2child_pos=head2child_pos,
        head2child_feats=head2child_feats,
    )


# Bracket suffixes some treebanks use for a deprel's slot (Georgian: [io] for iobj; nsubj is already
# folded by merge_subj_layered_feats).
_DEPREL_BRACKET_ALIASES = {
    "nsubj": ["subj", "nsubj"],
    "iobj": ["iobj", "io"],
}


def relation_slot(child, head, deprel_slot):
    """(slot of `child`, subtype) for the dependent of `head`. A dependent labelled `iobj:X` counts as an
    indirect object, with the verb slot [X], when the verb carries a [X] key (Abkhaz iobj:lo <-> Number[lo]);
    other subtypes (iobj:agent, iobj:appl) keep their own relation name and stay out of the iobj slot."""
    rel = child["deprel"]
    if rel.startswith("iobj:") and head is not None:
        sub = rel.split(":", 1)[1]
        if any(str(k).endswith(f"[{sub}]") for k in (head.get("feats") or {})):
            return "iobj", sub
    return deprel_slot(rel), None


def slot_suffixes(deprel, child_case):
    """Bracket suffixes that can name `deprel`'s slot on the verb, in priority order: relation-named (the
    deprel and its aliases), then the child's own Case lowercased ("dat"). Which case-named bracket is
    right is only knowable per instance, from the child token."""
    sufs = list(_DEPREL_BRACKET_ALIASES.get(deprel, [deprel]))
    if child_case is not None:
        sufs.append(str(child_case).lower())
    return list(dict.fromkeys(sufs))


def resolve_layered_head_key(head_features, child_features, target_feature, deprel):
    """The bracketed feature name (unprefixed, e.g. "Number[erg]", as in create_pairs' og_feats) holding the
    verb's agreement value for this relation, or None. create_pairs reuses it to pick the lexicon column
    to reinflect. Tried in order:

    1. relation-named: head_{feat}[{deprel}] or an alias (Georgian Person[obj] / Person[io]);
    2. case-named: head_{feat}[{child's Case lowercased}] (Basque Number[erg] / Number[abs]); the child's
       case already says which slot it fills, so no transitivity check is needed.

    None for languages without that convention.
    """
    # a dependent whose relation subtype names the verb slot (iobj:lo) is compared with that slot only
    named = child_features.get(f"{deprel}_idx_slot")
    suffixes = [named] if isinstance(named, str) and named else slot_suffixes(
        deprel, child_features.get(f"{deprel}_Case"))
    for suffix in suffixes:
        feat_name = f"{target_feature}[{suffix}]"
        # create_pairs passes DataFrame rows: an unset column is NaN, not None
        if pd.notna(head_features.get(f"head_{feat_name}")):
            return feat_name
    return None


def link_head_slot_feats(head_features, instance, deprel, child_features, enriched=()):
    """Rename the verb's relation-named brackets for `deprel`'s slot (Number[io]) to the child's case-named
    one (Number[dat]) when the child has a Case, so one slot is one feature. If the case-named bracket is
    already set the two are the same slot recorded twice (treebank vs lexicon spelling): keep one value,
    preferring the treebank's (`enriched` = keys the lexicon added). Without a child Case nothing is
    renamed. Mutates head_features and instance."""
    child_case = child_features.get(f"{deprel}_Case")
    rel_sufs = set(slot_suffixes(deprel, None))
    bracket_cols = [c for c in head_features if c.startswith("head_") and c.endswith("]")]

    def merge(rel_col, case_col):
        feat, _, suffix = rel_col[len("head_"):-1].partition("[")
        case_suf = case_col[case_col.index("[") + 1:-1]
        rel_key, case_key = f"{feat}[{suffix}]", f"{feat}[{case_suf}]"
        if head_features.get(case_col) is None or (
            case_key in enriched and rel_key not in enriched
        ):
            head_features[case_col] = head_features[rel_col]
        head_features[rel_col] = None
        instance[case_col] = head_features[case_col]
        instance[rel_col] = None

    if child_case is not None:
        case_suf = str(child_case).lower()
        for col in bracket_cols:
            feat, _, suffix = col[len("head_"):-1].partition("[")
            if suffix in rel_sufs and suffix != case_suf and head_features[col] is not None:
                merge(col, f"head_{feat}[{case_suf}]")
        return


def _resolve_layered_head_val(head_features, child_features, target_feature, deprel):
    """The verb's layered agreement value for extract_instances' label comparison (the value at the key
    resolve_layered_head_key finds). For nsubj this is only a fallback when the plain head_{feat} is
    missing; for obj/iobj it is used instead of the plain feature, which holds the subject's merged value
    and would be compared with the wrong argument (Georgian ovNa was "yes" 2442/2442 that way, both sides
    defaulting to Person 3).
    """
    feat_name = resolve_layered_head_key(head_features, child_features, target_feature, deprel)
    return head_features.get(f"head_{feat_name}") if feat_name is not None else None


_AGREEMENT_FEAT_RE = re.compile(r"^[A-Z][a-zA-Z]*$")


def values_overlap(val1, val2) -> bool:
    """True if two UD feature values share a member, treating a comma value
    as the set of readings it lists ("Masc,Neut" agrees with "Masc" and with
    "Neut"). False if either is missing."""
    if val1 is None or val2 is None:
        return False
    return bool(set(str(val1).split(",")) & set(str(val2).split(",")))


def extract_instances(
    tree,
    tree_idx,
    target: PredictionTarget,
    tree_metadata,
    predictor_var: str | None = None,
    agreement_feats: list[str] | str | None = None,
    lexicalize: bool = False,
    um_data: UDTable | None = None,
    ud_data: UDTable | None = None,
    fetch_all: bool = False,
):
    """One instance per head of `tree` that has all of the target's child deprels and passes the head and
    child filters: node features of the head and each child, the sentence-level features, and, if requested,
    the agreement labels (yes / no / +- one side unset / -- both unset). Returns a list of dicts.
    """
    instances = []

    tree_maps = _build_tree_maps(tree)

    all_feats, _, all_deprel, all_pos = tree_metadata

    # Extract sentence-level features (core_args, in_question, etc.)
    sen_features = extract_sen_features(tree)

    # Group children by head, keeping only those matching target deprels
    head2children = defaultdict(list)
    slot_of = {}
    for child in tree:
        head = tree[child["head"] - 1] if child["head"] > 0 else None
        slot, sub = relation_slot(child, head, child_slot)
        if slot in target.child_deprels:
            slot_of[id(child)] = (slot, sub)
            head2children[child["head"]].append(child)
    # several dependents of one relation where some are slot-named subtypes (iobj:lo next to iobj:ro):
    # which slot belongs to which dependent is not knowable, so only the plain ones are kept
    for head_id, kids in head2children.items():
        for rel in {slot_of[id(c)][0] for c in kids}:
            same = [c for c in kids if slot_of[id(c)][0] == rel]
            if len(same) > 1 and any(slot_of[id(c)][1] for c in same):
                head2children[head_id] = [c for c in kids if not (slot_of[id(c)][0] == rel and slot_of[id(c)][1])]
                kids = head2children[head_id]

    # For each head that has the required child deprels present
    for head_id, children in head2children.items():
        if not children:
            continue
        child_deprels_present = {slot_of[id(c)][0] for c in children}
        if not all(deprel in child_deprels_present for deprel in target.child_deprels):
            continue

        head = tree[head_id - 1]

        # Create a mapping from deprel to child for this head
        deprel2child = {slot_of[id(c)][0]: c for c in children}

        # Optional filters
        if target.head_deprel is not None and head["deprel"] != target.head_deprel:
            continue
        if target.head_pos is not None and head["upos"] not in target.head_pos:
            continue
        if target.child_pos is not None:
            skip = False
            for deprel, child in deprel2child.items():
                if isinstance(target.child_pos, list):
                    if child["upos"] not in target.child_pos:
                        skip = True
                elif isinstance(target.child_pos, dict):
                    if child["upos"] not in target.child_pos[deprel]:
                        skip = True
            if skip:
                continue

        sen = tree2sen(tree)
        instance = {
            "sen": sen,
            "no_space_after": tree_no_space_after(tree),
            "treebank": tree.metadata["treebank"].split("/")[0],
            "sent_id": tree.metadata["sent_id"],
            "tree_idx": tree_idx,
            **conllu_sentence_meta(tree),
        }

        excluded_deprels = set(target.child_deprels) | {
            alias for alias, slot in CHILD_DEPREL_ALIASES.items() if slot in target.child_deprels}

        # Extract features for the head
        head_features = extract_node_features(
            head,
            tree,
            tree_maps,
            "head",
            all_feats,
            all_deprel,
            all_pos,
            target,
            lexicalize=lexicalize,
            excluded_deprels=excluded_deprels,
            um_data=um_data,
            ud_data=ud_data,
            fetch_all=fetch_all
        )
        if not head_features:   # failed the PredictionTarget's head_feats filters
            continue
        instance.update(head_features)

        role_nodes = {"head": head}

        # Extract features for each child type; use deprel as prefix (e.g. "nsubj_pos", "amod_pos")
        for deprel in target.child_deprels:
            if deprel not in deprel2child:
                continue

            child = deprel2child[deprel]
            child_features = extract_node_features(
                child,
                tree,
                tree_maps,
                deprel,
                all_feats,
                all_deprel,
                all_pos,
                target,
                lexicalize=lexicalize,
                um_data=um_data,
                ud_data=ud_data,
                fetch_all=fetch_all,
            )
            child_features[f"{deprel}_idx_slot"] = slot_of[id(child)][1]   # verb slot named by the subtype, if any
            instance.update(child_features)
            role_nodes[deprel] = child
            link_head_slot_feats(
                head_features, instance, deprel, child_features,
                enriched=head.get("enriched_feats", ()),
            )

            # add SV agreement variable(s) (e.g. "head_nsubj_Number_agreement")
            if agreement_feats == "all":
                # every plain feature seen so far on either side; one not yet
                # seen anywhere has no value here, same as a "--" label
                seen = {k[len("head_"):] for k in head_features} | {
                    k[len(deprel) + 1:] for k in child_features if k.startswith(f"{deprel}_")}
                target_features = sorted(f for f in seen - lexical_features() if _AGREEMENT_FEAT_RE.match(f))
            elif agreement_feats is not None:
                target_features = agreement_feats
            elif predictor_var and "agreement" in predictor_var:
                target_features = [re.match(r".*_([A-Z][a-z]+)_.*", predictor_var).group(1)]
            else:
                target_features = []

            attractor_refs = {}
            for target_feature in target_features:
                if deprel == "nsubj":
                    # plain head_{feat} first (merge_subj_layered_feats folds [subj] into it); layered only as a fallback
                    head_val = head_features.get(f"head_{target_feature}", None)
                    if head_val is None:
                        head_val = _resolve_layered_head_val(
                            head_features, child_features, target_feature, deprel
                        )
                else:
                    # other relations: never the plain feature, see _resolve_layered_head_val
                    head_val = _resolve_layered_head_val(
                        head_features, child_features, target_feature, deprel
                    )
                child_val = child_features.get(f"{deprel}_{target_feature}", None)

                # a NOUN/PROPN without Person is assumed 3rd person when its counterpart states one
                if target_feature == "Person":
                    if head_val is None and child_val is not None and head["upos"] in ("NOUN", "PROPN"):
                        head_val = "3"
                        head_features[f"head_{target_feature}"] = head_val
                        instance[f"head_{target_feature}"] = head_val
                    elif child_val is None and head_val is not None and child["upos"] in ("NOUN", "PROPN"):
                        child_val = "3"
                        child_features[f"{deprel}_{target_feature}"] = child_val
                        instance[f"{deprel}_{target_feature}"] = child_val

                if head_val == child_val or values_overlap(head_val, child_val):
                    if head_val != None:
                        agreement_label = "yes"  # both set and agreeing
                    else:
                        agreement_label = "--"  # both undefined
                elif head_val != None:
                    if child_val != None:
                        agreement_label = "no"  # both set but disagreement
                    else:
                        agreement_label = "+-"  # child feat not set
                elif head_val == None:
                    agreement_label = "+-"  # head feat notset
                else:
                    raise ValueError(
                        "Value combination inadmissable"
                    )  # should not occur

                instance[f"head_{deprel}_{target_feature}_agreement"] = agreement_label
                attractor_refs[target_feature] = child_val if child_val is not None else head_val

            instance.update(attractor_cols(tree, child["id"], head["id"], attractor_refs, deprel,
                                             all_feats - lexical_features(), um_data, ud_data, fetch_all))

        instance["filled_feats"] = filled_feats(role_nodes)

        # Add sentence-level features
        instance.update(sen_features)

        deprel_ids = {
            deprel: instance[f"{deprel}_idx"]
            for deprel in target.child_deprels + ["head"]
            if deprel in deprel2child or deprel == "head"
        }
        deprel_order = "_".join(sorted(deprel_ids, key=deprel_ids.get))
        instance["deprel_order"] = shorten_cls(deprel_order, target)

        instances.append(instance)

    return instances


def _categorize(df: pd.DataFrame, fill_true_only: bool = True) -> pd.DataFrame:
    """Convert object-dtype columns to categorical, skipping columns with unhashable values. A column whose
    only value is True has its gaps filled with False; with `fill_true_only=False` (a partial frame) that is
    left to the caller, since a part cannot tell whether the column holds False elsewhere."""
    obj_cols = df.select_dtypes(include="object").columns
    # numpy arrays and one concat instead of per-column Series work and
    # df[col] = ... assignments, which dominate on frames this wide
    converted = {}
    if len(obj_cols):
        not_all_na = ~df[obj_cols].isna().all().to_numpy()
        for col in obj_cols[not_all_na]:
            values = df[col].to_numpy()
            is_na = pd.isna(values)
            try:
                non_null = pd.unique(values[~is_na])
                if fill_true_only and len(non_null) == 1 and non_null[0] is True:
                    values = np.where(is_na, False, values).astype(object)

                converted[col] = pd.Categorical(values)
            except TypeError:
                # some columns are lists of strings, we leave those as is
                pass

    if converted:
        df = pd.concat(
            [df.drop(columns=list(converted)), pd.DataFrame(converted, index=df.index)],
            axis=1,
        )

    # Feature records are built by iterating sets, so their column order
    # varies with every process (string hashing); a fixed order keeps
    # cached tables, and the trees fit on them, reproducible.
    return cast_attractor_cols(df[sorted(df.columns)])


def records_to_df(records: list[dict], fill_true_only: bool = True) -> pd.DataFrame:
    """Create a DataFrame from feature records with object columns as categorical."""
    return _categorize(pd.DataFrame(records), fill_true_only)


def extract_features(
    treebank,
    target: PredictionTarget | None = None,
    predictor_var: str | None = None,
    agreement_feats: list[str] | None = None,
    lexicalize: bool = False,
    batch_size: int = 500,
    um_data: UDTable | None = None,
    ud_data: UDTable | None = None,
    fetch_all: bool = False,
):
    """
    Extract features from treebank based on prediction target.

    Flushes instance dicts to a plain DataFrame every batch_size trees to
    avoid holding the full list of dicts in memory at once. Returns a list
    of partial DataFrames; the caller is responsible for concatenating and
    categorizing them (see create_word_order_df).

    """
    merge_subj_layered_feats(treebank)
    all_feats, all_lemma_freqs, all_deprel, all_pos = get_all_feats(treebank)
    tree_metadata = (all_feats, all_lemma_freqs, all_deprel, all_pos)

    dfs = []
    batch = []

    for tree_idx, tree in tqdm(enumerate(treebank), total=len(treebank)):
        batch.extend(
            extract_instances(
                tree,
                tree_idx,
                target,
                tree_metadata,
                lexicalize=lexicalize,
                predictor_var=predictor_var,
                agreement_feats=agreement_feats,
                um_data=um_data,
                ud_data=ud_data,
                fetch_all=fetch_all,
            )
        )

        if (tree_idx + 1) % batch_size == 0 and batch:
            batch_df = _categorize(pd.DataFrame(batch))
            dfs.append(batch_df.dropna(axis=1, how="all"))
            batch.clear()

    if batch:
        batch_df = _categorize(pd.DataFrame(batch))
        dfs.append(batch_df.dropna(axis=1, how="all"))

    return dfs


def is_feature_col(col: str) -> bool:
    """A morphological feature column (head_Number, iobj_Case,
    head_Person[io], ...): its last underscore-separated segment is a
    capitalised feature name, optionally bracketed."""
    return re.fullmatch(r"[A-Z][A-Za-z]*(\[[^\]]+\])?", col.rsplit("_", 1)[-1]) is not None


def drop_singleton_cols(
    df: pd.DataFrame, target: "PredictionTarget | None" = None, extra_always_keep=()
) -> pd.DataFrame:
    """Drop columns with a single distinct value (NaN counts as a value: a column alternating between one
    value and "not annotated" is informative). Shared by create_word_order_df and agreement_trees.aux.pipeline.

    Always kept: deprel_order/sen/no_space_after/treebank/sent_id/tree_idx/conllu_*, attractor columns, columns ending in
    "agreement", "_idx", "_form", "_form_orig", "_pos" (constant upos is still shown in the tree page) or
    "_deprel", every non-empty feature column (a constant like Person[io]=3 is still read by
    resolve_layered_head_key), target.head_feats columns, and extra_always_keep (subj_aux's aux count).
    """
    if len(df) == 0:
        return df
    always_keep = {"deprel_order", "sen", "no_space_after", "treebank", "sent_id", "tree_idx", "conllu_file", "conllu_sent_line"}
    always_keep.update(col for col in df.columns if col.endswith("agreement"))
    always_keep.update(col for col in df.columns if col.endswith("_idx"))
    always_keep.update(col for col in df.columns if col.endswith("_form"))
    always_keep.update(col for col in df.columns if col.endswith("_form_orig"))
    always_keep.update(col for col in df.columns if col.endswith("_pos"))
    always_keep.update(col for col in df.columns if col.endswith("_deprel"))
    always_keep.update(col for col in df.columns if is_attractor_col(col) or is_attractor_slots_col(col))
    if target is not None and target.head_feats:
        always_keep.update(f"head_{feat}" for feat in target.head_feats)
    always_keep.update(
        col for col in df.columns if is_feature_col(col) and df[col].notna().any()
    )
    always_keep.update(extra_always_keep)
    cols_to_check = df.columns.difference(list(always_keep))
    keep = df[cols_to_check].nunique(dropna=False) > 1
    kept_always = [c for c in df.columns if c in always_keep]  # df order, not set order
    return df[[*kept_always, *keep.index[keep]]].copy()


def create_word_order_df(
    target: PredictionTarget | None = None,
    treebank: Treebank | None = None,
    lang: str | None = None,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
    drop_singleton_columns: bool = False,
    predictor_var = None,
    agreement_feats: list[str] | None = None,
    lexicalize: bool = False,
    batch_size: int = 500,
    um_data: UDTable | None = None,
    ud_data: UDTable | None = None,
    fetch_all: bool = False,
    extra_always_keep=(),
) -> pd.DataFrame:
    """DataFrame of word-order/agreement features for a language and prediction target (one row per instance).

    treebank, or lang (+ resource_dir, max_treebank_len) to load one. save_to: directory for "<lang>.parquet".
    um_data / ud_data: the UniMorph- and UD-derived tables for annotation filling (fetch_all).
    predictor_var names the agreement column to build; agreement_feats overrides it with a list of features
    ("all": every plain feature on either side).
    extra_always_keep: columns drop_singleton_columns must keep (see drop_singleton_cols).
    """
    if save_to is not None:
        assert lang is not None, "lang must be provided for saving"

    if treebank is None:
        assert lang is not None, "lang must be provided for loading treebank"
        treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

    dfs = extract_features(
        treebank,
        target,
        predictor_var=predictor_var,
        agreement_feats=agreement_feats,
        lexicalize=lexicalize,
        batch_size=batch_size,
        um_data=um_data,
        ud_data=ud_data,
        fetch_all=fetch_all
    )
    del treebank
    gc.collect()

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    del dfs
    gc.collect()

    df = _categorize(df)

    if drop_singleton_columns and len(df) > 0:
        df = drop_singleton_cols(df, target=target, extra_always_keep=extra_always_keep)

    if save_to is not None and len(df) > 0:
        output_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        output_path = os.path.join(save_to, f"{output_lang}.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_parquet(df, output_path)

    return df


