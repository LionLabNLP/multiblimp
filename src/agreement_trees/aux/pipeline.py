from agreement_trees.vp.pipeline import Pipeline

from agreement_trees.aux.redirect import redirect_to_aux, AUX_COUNT_FEAT


class AuxPipeline(Pipeline):
    """Subject/object/indirect-object-auxiliary agreement: Pipeline on a treebank whose argument tokens
    were re-pointed at their clause's auxiliary (redirect_to_aux). `aux_target` is the extraction target.

    include_multi_aux: single-aux and stacked-aux (2+) clauses together (default) or single-aux only.
    Extraction always caches every stream; the choice only selects downstream, so a "_single" suffix in
    the target_id keeps the tree/pair caches apart.
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

    def _extraction_setup(self, treebank):
        """Re-point the argument at its clause's auxiliary and extract through aux_target (self.target
        only carries the swap feature); the aux count column survives singleton-column dropping. The
        treebank is mutated, so it must be freshly loaded (cache_treebank_features loads its own)."""
        redirect_to_aux(treebank, self.aux_target.child_deprels)
        return treebank, self.aux_target, {f"head_{AUX_COUNT_FEAT}"}
