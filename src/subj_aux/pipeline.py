import os
import sys

sys.path.append("../")

from word_order.process_treebank import (
    load_treebank, create_word_order_df, drop_singleton_cols,
)
from multiblimp.languages import gblang2udlang
from sva_trees.pipeline import Pipeline, get_um_lookup_table, get_ud_lookup_table

from .redirect import redirect_to_aux, AUX_COUNT_FEAT


class AuxPipeline(Pipeline):
    """Subject/object/indirect-object-auxiliary agreement:
    sva_trees.pipeline.Pipeline on a treebank whose nsubj/obj/iobj tokens
    (the target's child deprel) were first re-pointed at their clause's
    auxiliary (redirect_to_aux), so that the ordinary dependent-vs-head
    extraction compares the argument with the token that carries the
    agreement. `aux_target` is the extraction target (redirect.py).

    include_multi_aux: single-aux and stacked-aux (2+) clauses together
    (default) or single-aux only. Extraction always caches every stream;
    this only selects downstream, so it is baked into the default target_id
    (a "_single" suffix) to keep the tree/pair caches apart.
    """

    head_label = "Aux"

    def __init__(self, *args, aux_target, include_multi_aux=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_target = aux_target
        self.include_multi_aux = include_multi_aux

    def _select_stream(self, df):
        aux_count_col = f"head_{AUX_COUNT_FEAT}"
        if aux_count_col not in df.columns:
            return df.iloc[0:0]
        keep = ["1", "2+"] if self.include_multi_aux else ["1"]
        return df[df[aux_count_col].isin(keep)]

    def _build_df(self, lang, save_to, inflector):
        """Extraction goes through aux_target rather than self.target
        (which only carries the swap feature); the aux count column is kept
        through singleton-column dropping. Loads EVERY treebank (selected
        and excluded alike) in one pass, same as the parent class's own
        _build_df -- see that docstring for why (one unified per-language
        cache, split by "treebank" at read time instead of two separately-
        cached extractions)."""
        treebank = load_treebank(
            lang, self.resource_dir, self.max_treebank_len, use_selected_treebanks=False
        )
        redirect_to_aux(treebank, self.aux_target.child_deprels)
        df = create_word_order_df(
            treebank=treebank,
            lang=lang,
            target=self.aux_target,
            resource_dir=self.resource_dir,
            save_to=None,
            agreement_feats=self.agreement_feats,
            lexicalize=True,
            drop_singleton_columns=False,
            um_data=get_um_lookup_table(inflector),
            ud_data=get_ud_lookup_table(inflector),
            fetch_all=self.fetch_all,
        )
        df = drop_singleton_cols(df, target=self.aux_target,
                                 extra_always_keep={f"head_{AUX_COUNT_FEAT}"})
        if save_to is not None and len(df) > 0:
            os.makedirs(save_to, exist_ok=True)
            cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
            df.to_parquet(os.path.join(save_to, f"{cached_lang}.parquet"), index=False)
        return df
