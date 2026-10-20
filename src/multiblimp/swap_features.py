"""Inflection maps: (swap feature candidates, value map or None) per agreement relation, used through
agreement_trees.vp.conditions.INFLECTION_MAPS. A None value map means any other value of the feature is a swap.

Candidates are tried in order, bracketed columns first (UDInflector); the one that applies is picked per
row from the clause's own alignment (agreement_trees.extract.process_treebank.resolve_layered_head_key).
Subject: [subj] (UD relation spelling), plain (nominative-accusative), [erg]/[abs] (ergative splits,
e.g. Basque, where the clause's transitivity decides) and [nom] (UniMorph's case spelling, e.g. Georgian).
Object / indirect object: no plain fallback, since plain Number/Person/Gender on the verb is already the
subject's (merge_subj_layered_feats folds [subj] into it); covers relation brackets ([obj]/[iobj]/[io])
and case brackets ([abs]/[acc]/[dat]).
"""

swap_number_subj_any = (["Number[subj]", "Number", "Number[erg]", "Number[abs]", "Number[nom]"], None)
swap_any_person = (["Person[subj]", "Person", "Person[erg]", "Person[abs]", "Person[nom]"], None)
swap_gender_any = (["Gender[subj]", "Gender", "Gender[erg]", "Gender[abs]", "Gender[nom]"], None)

swap_number_obj_any = (["Number[obj]", "Number[abs]", "Number[acc]"], None)
swap_person_obj_any = (["Person[obj]", "Person[abs]", "Person[acc]"], None)
swap_gender_obj_any = (["Gender[obj]", "Gender[abs]", "Gender[acc]"], None)

swap_number_iobj_any = (["Number[iobj]", "Number[io]", "Number[dat]"], None)
swap_person_iobj_any = (["Person[iobj]", "Person[io]", "Person[dat]"], None)
swap_gender_iobj_any = (["Gender[iobj]", "Gender[io]", "Gender[dat]"], None)

