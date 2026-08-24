# NUMBER
swap_number_subj = (
    ["Number[subj]", "Number"],
    {
        "SG": "PL",
        "PL": "SG",
    },
)
swap_number_subj_any = (["Number[subj]", "Number"], None)
swap_number = (
    "Number",
    {
        "SG": "PL",
        "PL": "SG",
    },
)

# NUMBER -- object-verb / indirect-object-verb agreement (polypersonal
# marking, e.g. Basque, Georgian). No plain "Number" fallback entry, unlike
# swap_number_subj_any above: in these languages, plain Number on the verb
# is already the subject's number (merge_subj_layered_feats -- see
# word_order.process_treebank -- folds [subj]-suffixed features into the
# plain feature name), so falling back to it here would silently swap
# subject agreement while claiming to swap object agreement instead of
# finding nothing. The candidate list has to cover several conventions
# different sources use for the same phenomenon: UD treebanks disagree
# between relation-based brackets (Georgian's Number[obj]/Number[io]) and
# case-based ones (Basque's absolutive/dative, marking transitive objects/
# indirect objects under ergative alignment) -- and real UniMorph paradigm
# data can use yet another case-based convention even for a
# relation-bracketed treebank (Georgian's own UniMorph data uses
# Number[acc]/Number[nom], not [obj]/[subj] -- see multiblimp.ud2um.
# LAYERED_FEAT_SEP and UnimorphInflector.ufeats2dict's ARG*/PSS* handling
# for where each convention's columns come from).
swap_number_obj_any = (["Number[obj]", "Number[abs]", "Number[acc]"], None)
swap_number_iobj_any = (["Number[iobj]", "Number[io]", "Number[dat]"], None)

# PERSON
swap_person = (
    "Person",
    {
        "1": "3",
        "3": "1",
    },
)
swap_any_person = ("Person", None)

# PERSON -- object-verb / indirect-object-verb agreement. See the NUMBER
# object/indirect-object entries above for why there's no plain "Person"
# fallback, and for the multiple bracket conventions covered here. In
# practice Person dominates Number for polypersonal object marking in the
# languages UD covers (e.g. Georgian: object agreement is almost always
# realized via Person, rarely Number).
swap_person_obj_any = (["Person[obj]", "Person[abs]", "Person[acc]"], None)
swap_person_iobj_any = (["Person[iobj]", "Person[io]", "Person[dat]"], None)

# CASE
swap_case_any = ("Case", None)
swap_case = (
    "Case",
    {
        "NOM": "ACC",
        "ACC": "NOM",
    },
)

# GENDER
swap_gender_any = ("Gender", None)
swap_gender_f2m = (
    "Gender",
    {
        "MASC": "FEM",
        "FEM": "MASC",
    },
)
swap_gender_fm2n = (
    "Gender",
    {
        "MASC": "NEUT",
        "FEM": "NEUT",
        "COM": "NEUT",
        "NEUT": "COM",
    },
)
