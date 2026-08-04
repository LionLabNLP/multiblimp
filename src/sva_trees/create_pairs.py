import sys
import os
import re

import pandas as pd
from collections import Counter
from typing import *
from tqdm import tqdm

sys.path.append("../")
from multiblimp.swap_features import *
from word_order.prediction_target import PredictionTarget, nsubj_target
from sva_trees.flowchart import render_flowchart_html
from multiblimp.unimorph import load_inflector


UNDEFINED="UNDEFINED"
def process_item(
        item,
        form,
        swap_form,
        form_features,
        ufeat,
        feature_vals,
        feature_distribution,
        inflector,
        context_inflector,
        take_features_from,
        max_num_of_pairs=None,
    ) -> Tuple[str, Dict[str, str]]:
        item[f"swap_{take_features_from}"] = swap_form

        if swap_form == form:
            return "same_forms", item
        else:
            swap_feature_vals = inflector.get_form_features(
                swap_form, form_features, ufeat
            )
            feature_key = (
                "|".join(sorted(feature_vals))
                + " -> "
                + "|".join(sorted(swap_feature_vals))
            )

            add_item = False
            for feat1 in feature_vals:
                for feat2 in swap_feature_vals:
                    if (
                        max_num_of_pairs is None
                        or feature_distribution[f"{feat1} -> {feat2}"]
                        < max_num_of_pairs
                    ):
                        add_item = True
                    else:
                        add_item = None

            if (len(feature_vals & swap_feature_vals) == 0) and (
                UNDEFINED not in swap_feature_vals
            ):
                if (len(swap_feature_vals) > 0) and add_item:
                    if context_inflector is None:
                        child_features = set()
                    else:
                        child_features = context_inflector.get_form_features(
                            item["child"],
                            {"Case": "Nom"},  # ergative?
                            ufeat,
                            only_try_ud_if_no_um=True,
                        )
                    if len(child_features & swap_feature_vals) > 0:
                        return "ambiguous_subjects", item
                    else:
                        item["feature_vals"] = feature_key
                        num_combinations = len(feature_vals) * len(swap_feature_vals)
                        for feat1 in feature_vals:
                            for feat2 in swap_feature_vals:
                                feature_distribution[f"{feat1} -> {feat2}"] += (
                                    1 / num_combinations
                                )
                        if num_combinations > 1:
                            feature_distribution[feature_key] += 1
                        return "correct_swaps", item
                elif add_item is None:
                    return "extra_pairs", item
                else:
                    return "undefined_features", item
            elif len(feature_vals & swap_feature_vals) > 0:
                wrong_item = dict(item)
                wrong_item[f"swap_{take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "same_features", wrong_item
            else:
                wrong_item = dict(item)
                wrong_item[f"swap_{take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "undefined_features", wrong_item


def create_pairs(df, swap_feat, inflector, target: PredictionTarget = nsubj_target,
                  swap_target=["head",], context_inflector=None, max_num_of_pairs=None,
                  leaf_threshold=0.1, save_to=None, verbose=False, flowchart_path=None,
                  flowchart_examples=5):
    """
    Create re-inflected minimal sentence pairs for each row in the decision tree dataframe.

    Args:
        dt_df (pd.DataFrame): Decision tree dataframe containing the rows to process.
        leaf_threshold: entropy cutoff below which a leaf's prediction counts as "keep".
            Recomputed here from leaf_top1_entropy/leaf_decision rather than trusting the
            df's precomputed `keep` column, so it can be tuned without re-running fit_dt.
        save_to: if given, a directory to write every diagnostics bucket to, as
            "<item_type>.parquet".
        verbose: if True, also render an HTML flowchart of how each diagnostics bucket
            was reached, with up to flowchart_examples example rows per bucket.
        flowchart_path: where to write the flowchart HTML. Defaults to
            "<save_to>/flowchart.html" if save_to is given, else "pairs_flowchart.html".
        flowchart_examples: max example rows shown per bucket in the flowchart (default 5).

    Returns:
        dict[str, pd.DataFrame]: one DataFrame per diagnostics bucket (e.g. "correct_swaps").
    """
    if "leaf_top1_entropy" in df.columns and "leaf_decision" in df.columns:
        keep = (df["leaf_top1_entropy"] < leaf_threshold) & df["leaf_decision"]
    else:
        # Trivial languages/deprels (single-class predictor, no tree fit) have no
        # leaf_top1_entropy/leaf_decision columns and are kept in full — same
        # convention as viz_deprel.py's _agreement_row_stats.
        keep = pd.Series(True, index=df.index)
    swap_df = df[keep & (df[swap_feat]=="Yes")]
    if len(swap_df) == 0:
        with open("error_log.txt", "a") as f:
            f.write(f"No rows to process for {swap_feat} (keep={len(df[keep])}, swap={len(df[df[swap_feat]=='Yes'])})\n")
        return {bucket: pd.DataFrame() for bucket in [
            "correct_swaps", "same_forms", "same_features", "undefined_features",
            "no_inflections", "no_candidates", "multi_now_valid", "ambiguous_subjects",
            "extra_pairs"
        ]}

    child_deprel = target.child_deprels[0]
    ufeat, _ = inflector.inflection_map

    items_seen = 0
    feature_distribution = Counter()
    diagnostics = {
                "correct_swaps": [],
                "same_forms": [],
                "same_features": [],
                "undefined_features": [],
                "no_inflections": [],
                "no_candidates": [],
                "multi_now_valid": [],
                "ambiguous_subjects": [],
                "extra_pairs": [],
            }

    columns = list(swap_df.columns)
    # Precompute, once, which columns hold each swap kind's morphological
    # features (was re-matched via regex on every row before).
    kind_feat_cols = {
        kind: [
            (col, re.match(rf"^{kind}_([A-Z][a-z]+)", col).group(1))
            for col in columns
            if re.match(rf"^{kind}_[A-Z]", col)
        ]
        for kind in swap_target
    }

    # iter over each row in the decision tree dataframe. itertuples() (vs.
    # iterrows()) avoids rebuilding a per-row Series from this wide, mixed-dtype
    # dataframe on every iteration, which otherwise dominates runtime.
    for row_tuple in tqdm(swap_df.itertuples(index=False, name=None), total=swap_df.shape[0]):
        # get nsubj and head; extract their features; swap the features; re-inflect the words;
        # create a new sentence with the re-inflected words; add the new sentence to the dataframe
        base_item = dict(zip(columns, row_tuple))
        base_item["child"] = base_item.get(f"{child_deprel}_form")

        for kind in swap_target:
            og_feats = {feat: base_item[col] for col, feat in kind_feat_cols[kind]}
            form = base_item[f"{kind}_form"]

            swap_forms, feature_vals = inflector.inflect(form, og_feats)

            if swap_forms!=None:
                if len(swap_forms) > 0:
                    for swap_form in swap_forms:
                        items_seen += 1
                        item_type, item = process_item(
                            dict(base_item),
                            form,
                            swap_form,
                            og_feats,
                            ufeat,
                            feature_vals,
                            feature_distribution,
                            inflector,
                            context_inflector,
                            take_features_from=kind,
                            max_num_of_pairs=max_num_of_pairs,
                        )
                        diagnostics[item_type].append(item)

                        if (item_type == "correct_swaps") and (len(swap_forms) > 1):
                            multi_item = dict(item)
                            multi_item["alternatives"] = swap_forms
                            diagnostics["multi_now_valid"].append(multi_item)
                else:
                    items_seen += 1
                    diagnostics["no_inflections"].append(dict(base_item))
            else:
                items_seen += 1
                diagnostics["no_candidates"].append(dict(base_item))

    coverage = f"{len(diagnostics['correct_swaps'])/max(items_seen,1)*100:.1f}"
    print(f"items_seen={items_seen} correct_swaps={len(diagnostics['correct_swaps'])} coverage={coverage}%")

    diagnostic_dfs = {item_type: pd.DataFrame(items) for item_type, items in diagnostics.items()}

    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        for item_type, item_df in diagnostic_dfs.items():
            item_df.to_parquet(os.path.join(save_to, f"{item_type}.parquet"))

    if verbose:
        if flowchart_path is not None:
            out_dir =flowchart_path.rsplit("/", 1)[0]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        out_file = flowchart_path or (
            os.path.join(save_to, f"../flowcharts/pairs_flowchart.html") if save_to is not None
            else f"pairs_flowchart.html"
        )
        feat_match = re.match(r".*_([A-Z][a-z]+)_.*", swap_feat)
        render_flowchart_html(diagnostic_dfs, out_file, max_examples=flowchart_examples,
                               child_deprel=child_deprel, default_kind=swap_target[0],
                               swap_feature=feat_match.group(1) if feat_match else None)
        print(f"wrote flowchart to {out_file}")

    return diagnostic_dfs


if __name__ == "__main__":
    # Example usage
    resource_dir = "../../resources"
    # Load your decision tree dataframe 
    dt_df = pd.read_parquet("../../decision_trees/svNa/svNa_nsubj/German.parquet")

    swap_feat = "head_nsubj_Number_agreement"

    inflector, skip_lang, num_lemma, num_form= load_inflector(
                    lang="German", 
                    langcode="deu",
                    unimorph_args = {
                                "filter_entries": {
                                    "upos": ["V"],
                                    },
                                    "combine_um_ud": True,
                                    "remove_multiword_forms": True,
                                    },
                    inflection_map=swap_number_subj_any,
                    resource_dir=resource_dir,)

    create_pairs(dt_df, swap_feat=swap_feat, inflector=inflector, leaf_threshold=0.12,
                 save_to="../../decision_trees/svNa/svNa_nsubj/pairs/German",
                 verbose=True)
