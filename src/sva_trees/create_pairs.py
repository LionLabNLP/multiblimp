import sys
import io

import pandas as pd

import re

import ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("../")
from multiblimp.languages import remove_diacritics_langs, remove_multiples_langs, lang2langcode
from multiblimp.unimorph import UnimorphInflector
from multiblimp.swap_features import *
from word_order.prediction_target import PredictionTarget, nsubj_target




def load_inflector(lang: str, langcode: str, unimorph_args, inflection_map: dict, 
                   resource_dir: str): # TODO put in utils?
    num_form = 0
    num_lemma = 0
    skip_lang = False

    remove_diacritics = lang in remove_diacritics_langs
    remove_multiples = lang in remove_multiples_langs

    inflector = UnimorphInflector(
        langcode=langcode,
        inflection_map=inflection_map,
        resource_dir=resource_dir,
        load_from_pickle=True,
        remove_diacritics=remove_diacritics,
        remove_multiples=remove_multiples,
        fill_unk_values=False,
        **unimorph_args,
    )

    if len(inflector) == 0:
        skip_lang = True
        num_lemma = 0
        num_form = 0
    else:
        num_lemma = inflector.num_lemmas
        num_form = inflector.num_forms

    if not inflector.can_feature_swap:
        skip_lang = True

    return inflector, skip_lang, num_lemma, num_form


from typing import *
UNDEFINED="UNDEFINED"
def process_item(
        self,
        item,
        form,
        swap_form,
        form_features,
        ufeat,
        feature_vals,
        feature_distribution,
        inflector,
        context_inflector,
    ) -> Tuple[str, Dict[str, str]]:
        item[f"swap_{self.take_features_from}"] = swap_form

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
                        feature_distribution[f"{feat1} -> {feat2}"]
                        < self.max_num_of_pairs
                    ):
                        add_item = True
                    else:
                        add_item = None

            if (len(feature_vals & swap_feature_vals) == 0) and (
                UNDEFINED not in swap_feature_vals
            ):
                if (len(swap_feature_vals) > 0) and add_item:
                    # todo: take the child features from an argument
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
                wrong_item[f"swap_{self.take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "same_features", wrong_item
            else:
                wrong_item = dict(item)
                wrong_item[f"swap_{self.take_features_from}"] = swap_form
                wrong_item["feature_vals"] = feature_key

                return "undefined_features", wrong_item


def create_pairs(df, swap_feat, inflector, swap_target=["head",]):
    """
    Create re-inflected minimal sentence pairs for each row in the decision tree dataframe.

    Args:
        dt_df (pd.DataFrame): Decision tree dataframe containing the rows to process.
        treebank: The treebank object containing the original sentences.

    Returns:
        pd.DataFrame: A new dataframe with re-inflected minimal sentence pairs added.
    """
    print()
    swap_df = df[(df["keep"]==True) & (df[swap_feat]=="Yes")]

    # iter over each row in the decision tree dataframe
    for df_idx, row in swap_df.iterrows():
        original_sentence = ast.literal_eval(row["sen"])
        # get nsubj and head; extract their features; swap the features; re-inflect the words; create a new sentence with the re-inflected words; add the new sentence to the dataframe

        items_seen = 0
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
        for kind in swap_target:
            og_feats = {re.match(rf"^{kind}_([A-Z][a-z]+)", k).group(1): v for k, v in row.filter(regex=rf"^{kind}_[A-Z]").to_dict().items() }

            with open("swap_debug.txt", "a", encoding="utf-8") as f:
                print(row[f"{kind}_form"],file=f)
                print("og feats",og_feats, file=f)

                swap_forms, feature_vals = inflector.inflect(
                            row[f"{kind}_form"],
                            {re.match(rf"^{kind}_([A-Z][a-z]+)", k).group(1): v for k, v in row.filter(regex=rf"^{kind}_[A-Z]").to_dict().items() },
                            #strategies=strategies,
                        )
                print("swap_forms", swap_forms, file=f)
                print("feature_vals", feature_vals, file=f)
                print("-"*50, file=f)


                if swap_forms!=None:
                    if len(swap_forms) > 0:
                        items_seen += 1
                        # rebuild process_item?
                        pass
                    # if len(swap_forms) == 1:
                    #     print(" ".join([w if i!=row[f"{kind}_idx"]-1 else swap_forms[0]
                    #             for i, w in  enumerate(original_sentence)]),
                    #             file=f)
                    # elif len(swap_forms) > 1:

                    #     for swap_form in swap_forms:
                    #         print(f"Re-inflected {kind} form: {swap_form}", file=f)
                    #         raise IndexError(f"Multiple re-inflected forms for {kind} form: {swap_form}")

                    else:
                        items_seen += 1
                        diagnostics["no_inflections"].append(None)  #?
                else:
                    items_seen += 1
                    diagnostics["no_candidates"].append(None)  #?

            # with open("swap_debug.txt", "w", encoding="utf-8") as f:
            #     print("original_sentence", original_sentence, file=f)
            #     with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            #         print("row", row, file=f)

           # break
        #break


if __name__ == "__main__":
    # Example usage
    resource_dir = "../../resources"
    # Load your decision tree dataframe 
    dt_df = pd.read_csv("../../decision_trees/svNa/svNa_nsubj/German.csv")
    #treebank = load_treebank("German", resource_dir, max_treebank_len=30_000)

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

    create_pairs(dt_df, swap_feat=swap_feat, inflector=inflector)
