from pyexpat import features
import sys
import os
import random
import re
import gc

from collections import defaultdict, Counter
from dataclasses import dataclass

sys.path.append("src")

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from typing import *

from multiblimp.treebank import Treebank
from multiblimp.languages import remove_diacritics_langs, gblang2udlang
from multiblimp.unimorph import UM2UD, UD2UM, UnimorphInflector
from resources.um2ud_morpho.UM2UD_mapper import map_um_value_to_ud

from .prediction_target import PredictionTarget
from .utils import shorten_cls


META_FEATURES = ["sen", "no_space_after", "treebank", "sent_id", "tree_idx", "treebank_link", "sen_str"]


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
    lang: str, resource_dir: str | None = None, max_treebank_len: int | None = None
) -> Treebank:
    lang = gblang2udlang.get(lang, lang).replace(" ", "_")

    treebank = Treebank(
        lang,
        remove_diacritics=(lang in remove_diacritics_langs),
        resource_dir=resource_dir,
        remove_typo=True,
        load_from_pickle=True,
    )
    if max_treebank_len is not None and len(treebank) > max_treebank_len:
        rng = random.Random(42)
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


def tree_no_space_after(tree):
    """Per-token SpaceAfter=No flags, parallel to tree2sen(tree). Saved alongside
    "sen" so downstream consumers (e.g. sva_trees' flowchart) can reconstruct the
    correctly-spaced surface string without reloading the source treebank."""
    return [(tok["misc"] or {}).get("SpaceAfter") == "No" for tok in tree]


_SUBJ_FEAT_SUFFIXES = ("[subj]", "[nsubj]")


def merge_subj_layered_feats(treebank) -> None:
    """UD sometimes marks a token's subject-agreement value with a layered
    "[subj]" suffix (or "[nsubj]", depending on the treebank) instead of the
    plain feature name — e.g. Abkhaz annotates verbs with Number[subj] rather
    than plain Number: in the shipped data, head_Number is set on only ~5% of
    Abkhaz rows, vs. 100% for head_Number[subj]. For SVA purposes this is the
    same underlying property as the plain feature, so fold it into the plain
    key wherever the plain key is itself absent, and drop the layered key —
    "Number[subj]" never survives as its own separate feature/column.

    Deliberately narrow: only [subj]/[nsubj] are converted. Other layered
    features (e.g. Number[psor]) are left untouched — they stay as their own
    distinct feature rather than being folded into anything.

    Mutates token["feats"] in place; must run before get_all_feats/
    extract_instances read them, since both only see whatever's already there.
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


UNDEFINED = "UNDEFINED"

# Cached groupby("form") per UniMorph POS-slice, keyed by object identity. The
# same POS-slice DataFrame (from sva_trees.pipeline.get_um_lookup_table) is
# reused across every node of that POS for the whole treebank, so grouping it
# once and reusing .get_group() turns each partial_df_match call from a full
# linear scan of a (often 100k+ row) DataFrame into an indexed lookup.
_form_groups_cache: dict = {}


def _get_form_groups(um_data):
    key = id(um_data)
    cached = _form_groups_cache.get(key)
    if cached is not None and cached[0] is um_data:
        return cached[1]
    groups = um_data.groupby("form", observed=False)
    _form_groups_cache[key] = (um_data, groups)
    return groups


def clear_form_groups_cache() -> None:
    """Drop all cached groupby("form") results and their source DataFrames.

    The cache holds a strong reference to every um_data slice it's ever seen
    (see _get_form_groups), so it grows for as long as new slices keep coming
    in. Callers that process a sequence of languages where each language's
    um_data is never revisited (e.g. sva_trees.pipeline.Pipeline, one
    get_um_lookup_table() call per language) should call this once they're
    done with a language's data, so memory doesn't accumulate across the run.
    """
    _form_groups_cache.clear()


def partial_df_match(
        um_data,
        match_feats : dict,
        ufeat,
        features: Dict[str, str],
        prefer_tight_match: Optional[bool] = None,
    ):
        """Find all rows in the morphology dataframe that match the
        group_index (lemma or form) and the features in the provided
        `features` dictionary.
        """
        remaining_match_feats = dict(match_feats)
        form_val = remaining_match_feats.pop("form", None)
        if (form_val not in (None, "_")) and ("form" in um_data.columns):
            groups = _get_form_groups(um_data)
            if form_val not in groups.groups:
                return um_data.iloc[0:0]
            um_data = groups.get_group(form_val)

        for filter_type, filter_value in remaining_match_feats.items():
            if filter_value not in [None, "_"]:
                um_data = um_data[um_data[filter_type] == filter_value]
            if len(um_data) == 0:
                return um_data.iloc[0:0]
        sub_df = um_data

        mask = np.ones(len(sub_df), dtype=bool)

        for col, val in features.items():
            if isinstance(val, list):
                sub_mask = np.zeros_like(mask)
                for subval in val:
                    sub_mask |= sub_df[col] == subval
                mask &= sub_mask
            elif (val is not None) and (pd.notna(val)):
                if val.startswith("-"):
                    mask &= (sub_df[col] != val[1:]) & (sub_df[col] != UNDEFINED)
                elif col == ufeat:
                    mask &= sub_df[col] == val
                else:
                    try:
                        mask &= (
                            (sub_df[col] == val)
                            | (sub_df[col] == UNDEFINED)
                            | pd.isna(sub_df[col])
                        )
                    except:
                        pass

        candidate_rows = sub_df[mask]

        if (not prefer_tight_match) or (len(candidate_rows) < 2):
            return candidate_rows

        # Vectorized equivalent of the per-row/per-column loop below (avoids
        # candidate_rows.iterrows()):
        #   for each cell (row, col):
        #     if cell is set (not-nan or UNDEFINED): +1 if col not in `features`
        #     else (cell is nan): +1 if `features` has a real (non-nan) value for col
        is_set_df = candidate_rows.notna() | (candidate_rows == UNDEFINED)
        not_in_features = pd.Series(
            {col: (col not in features) for col in candidate_rows.columns}
        )
        wants_feature = pd.Series(
            {
                col: (col in features) and pd.notna(features.get(col))
                for col in candidate_rows.columns
            }
        )
        penalty = is_set_df.mul(not_in_features, axis=1).astype(int) + (
            (~is_set_df).mul(wants_feature, axis=1).astype(int)
        )
        features_added = penalty.sum(axis=1).to_numpy()

        min_features_added = min(features_added)
        min_features_added_mask = features_added == min_features_added

        return candidate_rows[min_features_added_mask]


def expand_anno(node, morph_feats, target, um_split):
    # make copy to keep upos/lemma separate from conllu Token's feats
    inflect_feats = morph_feats
    inflect_feats["upos"] = node["upos"]
    # allows for soft matching if no lemma was specified in UD
    inflect_feats["lemma"] = node.get("lemma", None) if node.get("lemma", None) != "_" else None

    um_feats = {f: (UD2UM.get((f,v), None) if f!="lemma" else v) for f, v in inflect_feats.items()}

    form_rows = partial_df_match(
        um_split,
        {"form": node["form"], "lemma":inflect_feats["lemma"]},
        target,
        um_feats,
        prefer_tight_match=True
    )
    add_feats=False
    if len(form_rows):
        unified = dict()
        for i, row in form_rows.iterrows():
            transformed =  map_um_value_to_ud(row["ufeat"])
            for k, v in transformed["morpho"].items():
                unified[k] = unified.get(k, list()) + [v]
            unified["upos"] = unified.get("upos", list()) + [transformed["upos"]]

        unified = {k: list(set(v))[0] for k, v in unified.items() if len(list(set(v)))==1}

        if( not any(
                    [1 if (inflect_feats.get(k, False) and 
                            inflect_feats.get(k, False)!=v and
                            inflect_feats.get(k, False)!="UNDEFINED"
                            ) else 0 for k, v in unified.items()]
                )):
                    #unanimous feat matches
                    #check for conflict with og data stil
                    for k,v in unified.items():
                        if not k in um_feats:#s and len(set(v))==1:
                            if add_feats==False: add_feats = dict()
                            add_feats[k] = v

        if add_feats:
            for feat, val in add_feats.items():
                morph_feats[feat] = val

    return morph_feats


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
    encode_positional_features: bool = False,
    um_data: pd.DataFrame=None,
    fetch_all=False
) -> dict[str, str | bool | int]:
    """
    Extract all features for a specific node (whether it's a child, head, or co-child).
    This ensures consistent feature extraction across all node types.
    """
    features = {}
    if type(node.get("feats", None))!=dict: node["feats"] = dict()
    excluded_deprels = excluded_deprels or set()

    # Basic node features
    features[f"{prefix}_deprel"] = node["deprel"]
    features[f"{prefix}_pos"] = node["upos"]
    features[f"{prefix}_idx"] = node["id"]

    if lexicalize:
        features[f"{prefix}_form"] = (
            node["form"].lower() if node["upos"] != "PROPN" else node["form"]
        )
        features[f"{prefix}_lemma"] = node["lemma"]

    # Morphological features
    # First: Try to get missing anno from Unimorph
    if type(um_data)==dict and fetch_all:
        # first check if we could even get any more anno from UM for this upos
        um_split = um_data.get(UD2UM["upos", node["upos"]], None)
        if type(um_split)==pd.DataFrame and (
            not all([node["feats"].get(k, False) for k in [x for x in um_split.columns if x[0].isupper()]])):
            node["feats"] = expand_anno(node, node["feats"], target, um_split)

    all_feats.update(set(node.get("feats", {}).keys()))

    for feat in all_feats:
        features[f"{prefix}_{feat}"] = (node["feats"] or {}).get(feat)

    # Further optional filters for head from PredictionTarget; filter is (lamnda x: condition)
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
        if encode_positional_features:
            features["child_head_distance"] = abs(node["head"] - node["id"])

            sibling_ids = tree_maps.head2child_ids[node["head"]]
            for idx in sibling_ids:
                sibling_deprel = tree[idx - 1]["deprel"]
                sibling_pos = tree[idx - 1]["upos"]

                if idx != node["id"]:
                    if idx < node["id"] < node["head"]:
                        features[f"{prefix}_sibling-deprel-sCH_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-sCH_{sibling_pos}"] = True
                    elif node["id"] < idx < node["head"]:
                        features[f"{prefix}_sibling-deprel-CsH_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-CsH_{sibling_pos}"] = True
                    elif node["id"] < node["head"] < idx:
                        features[f"{prefix}_sibling-deprel-CHs_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-CHs_{sibling_pos}"] = True
                    elif idx < node["head"] < node["id"]:
                        features[f"{prefix}_sibling-deprel-sHC_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-sHC_{sibling_pos}"] = True
                    elif node["head"] < idx < node["id"]:
                        features[f"{prefix}_sibling-deprel-HsC_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-HsC_{sibling_pos}"] = True
                    elif node["head"] < node["id"] < idx:
                        features[f"{prefix}_sibling-deprel-HCs_{sibling_deprel}"] = True
                        features[f"{prefix}_sibling-pos-HCs_{sibling_pos}"] = True
        else:
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


def extract_instances(
    tree,
    tree_idx,
    target: PredictionTarget,
    tree_metadata,
    predictor_var: str | None = None,
    lexicalize: bool = False,
    um_data: pd.DataFrame | None = None,
    fetch_all: bool = False,
):
    """
    Extract training instances from a tree.

    When target is None, returns one instance per (head, child) pair across all
    non-root tokens, using prefix 'head' and 'child'. When target is provided,
    filters to matching head-child groups and uses the deprel as the child prefix.

    Returns:
        list of dicts, where each dict contains features for one instance
    """
    instances = []

    tree_maps = _build_tree_maps(tree)

    all_feats, _, all_deprel, all_pos = tree_metadata

    # Extract sentence-level features (core_args, in_question, etc.)
    sen_features = extract_sen_features(tree)

    # Group children by head, keeping only those matching target deprels
    head2children = defaultdict(list)
    for child in tree:
        if child["deprel"] in target.child_deprels:
            head2children[child["head"]].append(child)

    # For each head that has all required child deprels present
    for head_id, children in head2children.items():
        # Check if we have all required deprels
        child_deprels_present = {c["deprel"] for c in children}
        if not all(deprel in child_deprels_present for deprel in target.child_deprels):
            continue

        head = tree[head_id - 1]

        # Create a mapping from deprel to child for this head
        deprel2child = {c["deprel"]: c for c in children}

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
        }

        excluded_deprels = set(target.child_deprels)

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
            fetch_all=fetch_all
        )
        if head_features == None: # item failed PredictionTarget filters
            continue
        instance.update(head_features)

        if len(head_features) == 0:  # item failed PredictionTarget filters
            continue

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
            )
            instance.update(child_features)

            # add SV agreement variable for predictor_var (e.g. "head_nsubj_Number_agreement")
            # 1. is feature annotated on head & target child?, 2. is value the same? -> TRUE else False
            if predictor_var and "agreement" in predictor_var:
                target_feature = re.match(r".*_([A-Z][a-z]+)_.*", predictor_var).group(
                    1
                )
                head_val = head_features.get(f"head_{target_feature}", None)
                child_val = child_features.get(f"{deprel}_{target_feature}", None)
                if head_val == child_val:
                    if head_val != None:
                        agreement_label = "Yes"  # both set and agreeing
                    else:
                        agreement_label = "--"  # both undefined
                elif head_val != None:
                    if child_val != None:
                        agreement_label = "No"  # both set but disagreement
                    else:
                        agreement_label = "+-"  # child feat not set
                elif head_val == None:
                    agreement_label = "+-"  # head feat notset
                else:
                    raise ValueError(
                        "Value combination inadmissable"
                    )  # should not occur

                instance[f"head_{deprel}_{target_feature}_agreement"] = agreement_label

        # Add sentence-level features
        instance.update(sen_features)

        deprel_ids = {
            deprel: instance[f"{deprel}_idx"]
            for deprel in target.child_deprels + ["head"]
        }
        deprel_order = "_".join(sorted(deprel_ids, key=deprel_ids.get))
        instance["deprel_order"] = shorten_cls(deprel_order, target)

        instances.append(instance)

    return instances


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-dtype columns to categorical, skipping columns with unhashable values."""
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().all():
            continue
        try:
            non_null = df[col].dropna().unique()
            if len(non_null) == 1 and non_null[0] is True:
                df.loc[df[col].isna(), col] = False

            df[col] = df[col].astype("category")
        except TypeError:
            # some columns are lists of strings, we leave those as is
            pass

    return df


def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Create a DataFrame from feature records with object columns as categorical."""
    return _categorize(pd.DataFrame(records))


def extract_features(
    treebank,
    target: PredictionTarget | None = None,
    predictor_var: str | None = None,
    lexicalize: bool = False,
    batch_size: int = 500,
    um_data: pd.DataFrame | None = None,
    fetch_all: bool = False
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
                um_data=um_data,
                fetch_all=fetch_all
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


def create_word_order_df(
    target: PredictionTarget | None = None,
    treebank: Treebank | None = None,
    lang: str | None = None,
    resource_dir: str | None = None,
    save_to: str | None = None,
    max_treebank_len: int | None = None,
    drop_singleton_columns: bool = False,
    predictor_var = None,
    lexicalize: bool = False,
    batch_size: int = 500,
    um_data: pd.DataFrame | None = None,
    fetch_all: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame with word order features for a given language and prediction target.

    Args:
        target: PredictionTarget specifying what word order to predict
        treebank: Treebank for language, can be provided optionally
        lang: Language code, must be provided if Treebank is not passed
        resource_dir: Directory containing treebank resources
        save_to: Directory to save CSV to (optional)
        max_treebank_len: Maximum number of sentences to process
        drop_singleton_columns: drop columns with only a single value
        um_data: all unimorph data for current language
        predictor_var: variable name to fit the decision tree on
        fetch_all: if True, try to enhance feature annotation via Unimorph

    Returns:
        DataFrame with extracted features
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
        lexicalize=lexicalize,
        batch_size=batch_size,
        um_data=um_data,
        fetch_all=fetch_all
    )
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    del dfs
    gc.collect()

    df = _categorize(df)

    if drop_singleton_columns and len(df) > 0:
        always_keep = {"deprel_order", "sen", "no_space_after", "treebank", "sent_id", "tree_idx"}
        always_keep.update(set([col for col in df.columns if col.endswith("agreement")]))
        always_keep.update(set([col for col in df.columns if col.endswith("_idx")]))
        cols_to_check = df.columns.difference(list(always_keep))
        keep = df[cols_to_check].nunique() > 1
        kept_always = [c for c in always_keep if c in df.columns]
        df = df[[*kept_always, *keep.index[keep]]].copy()

    if save_to is not None and len(df) > 0:
        output_path = os.path.join(save_to, f"{lang.replace(' ', '_')}.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)

    return df


def split_node_dfs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a combined (head, child) DataFrame — as produced by create_word_order_df(target=None)
    — into the head_df and child_df expected by fit_reattachment_classifiers.

    head_df: head_* columns, deduplicated by (tree_idx, head_idx), with a child_deprels
             column listing every child deprel observed for that head in df, and a
             deprel_directions column mapping each unambiguously-directed deprel to "L" or "R".
    child_df: child_* columns + deprel label (= child_deprel) + metadata.
              child_deprel is kept as a feature column; the separate 'deprel' column
              is used as the classification label in build_training_data.
    """
    meta = ["tree_idx", "sent_id"]

    head_cols = meta + [c for c in df.columns if c.startswith("head_")]
    child_cols = meta + [c for c in df.columns if c.startswith("child_")]

    child_deprels_per_head = (
        df.groupby(["tree_idx", "head_idx"])["child_deprel"]
        .apply(list)
        .reset_index()
        .rename(columns={"child_deprel": "child_deprels"})
    )

    # Build deprel_directions: {deprel: "L"/"R"} per head.
    # child_dir (0=left, 1=right) is set by extract_node_features.
    # Deprels where children appear on both sides are excluded (ambiguous).
    dir_agg = (
        df.groupby(["tree_idx", "head_idx", "child_deprel"])["child_dir"]
        .agg(["first", "nunique"])
        .reset_index()
    )
    unambiguous = dir_agg[dir_agg["nunique"] == 1].copy()
    unambiguous["_dir"] = unambiguous["first"].map({0: "L", 1: "R"})
    deprel_directions_per_head = (
        unambiguous.groupby(["tree_idx", "head_idx"])
        .apply(lambda g: dict(zip(g["child_deprel"], g["_dir"])))
        .reset_index(name="deprel_directions")
    )

    head_df = (
        df[head_cols]
        .drop_duplicates(subset=["tree_idx", "head_idx"])
        .merge(child_deprels_per_head, on=["tree_idx", "head_idx"])
        .merge(deprel_directions_per_head, on=["tree_idx", "head_idx"], how="left")
        .reset_index(drop=True)
    )
    # Heads with no unambiguous-direction children get an empty dict
    head_df["deprel_directions"] = head_df["deprel_directions"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    child_df = df[child_cols].copy()
    child_df["deprel"] = df["child_deprel"].values

    return head_df, child_df
