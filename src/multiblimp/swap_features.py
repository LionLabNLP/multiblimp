# NUMBER
swap_number_subj = (
    ["Number[subj]", "Number"],
    {
        "SG": "PL",
        "PL": "SG",
    },
)
# "Number[erg]"/"Number[abs]" cover ergative-split languages (e.g. Basque),
# where plain Number is essentially never set on finite verbs at all --
# these candidates only matter as sva_trees.create_pairs' per-row fallback
# (word_order.process_treebank.resolve_layered_head_key), since which one
# applies depends on this clause's own transitivity, not the language as a
# whole; a global pick here (update_inflection_map picking whichever's
# best-populated corpus-wide) would only ever be right for one of the two.
# "Number[nom]" covers real UniMorph paradigm data's own case-based
# convention for the same role (e.g. Georgian's real UM lexicon uses
# Person[nom]/Person[acc], never [subj]/[obj] -- that bracket spelling is
# specific to the UD-derived fallback lexicon). Without it, the primary
# (real-UM) inflector's own update_inflection_map resolution finds none of
# its candidates present at all, inherits the UD-fallback's "[subj]"
# resolution instead (see UnimorphInflector.__init__), and then every
# lookup against the primary lexicon fails outright (it has no "[subj]"
# column), silently making the richer real-UM data unusable for subject
# reinflection -- confirmed on Georgian: primary lexicon columns are
# ['Person[acc]', 'Person[nom]'], nothing named "[subj]" at all.
swap_number_subj_any = (["Number[subj]", "Number", "Number[erg]", "Number[abs]", "Number[nom]"], None)
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
# "Person[subj]" matches swap_number_subj_any's own first candidate --
# omitted here until it was caught as a real gap: Georgian's real data has
# Person[subj] populated 3939/23500 times in the reinflection lexicon vs.
# only 129/23500 for plain Person, so without this entry in the candidate
# list, update_inflection_map could never even consider the column that
# actually carries the signal for this language. "Person[nom]" is here for
# the same reason "Number[nom]" is in swap_number_subj_any -- see that
# entry's comment for both. See swap_number_subj_any above for why [erg]/
# [abs] are also here, and why a per-row (not global) resolution is what
# actually makes all of these useful.
swap_any_person = (["Person[subj]", "Person", "Person[erg]", "Person[abs]", "Person[nom]"], None)

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
# See swap_any_person above for why "Gender[subj]" is included, and
# swap_number_subj_any for why [erg]/[abs]/[nom] are too. No Georgian
# Gender[nom] data was found in practice (Georgian has no grammatical
# gender at all), but kept for consistency/completeness in case another
# language's real UniMorph data does mark it this way.
swap_gender_any = (["Gender[subj]", "Gender", "Gender[erg]", "Gender[abs]", "Gender[nom]"], None)

# GENDER -- object-verb / indirect-object-verb agreement. See the NUMBER
# object/indirect-object entries above for why there's no plain "Gender"
# fallback, and for the multiple bracket conventions covered here. Rarer
# signal than Number/Person for polypersonal object marking (e.g. Basque:
# Gender[erg]/Gender[dat] each occur a couple dozen times vs. thousands for
# Number/Person), so expect low candidate counts even where it's present.
swap_gender_obj_any = (["Gender[obj]", "Gender[abs]", "Gender[acc]"], None)
swap_gender_iobj_any = (["Gender[iobj]", "Gender[io]", "Gender[dat]"], None)
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
