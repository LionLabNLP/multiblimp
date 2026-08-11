import os
from typing import *
from unidecode import unidecode

import numpy as np
import pandas as pd

from .inflection_maps import InflectionMap
from .languages import latin_to_cyrillic, remove_diacritics_langs, remove_multiples_langs
from .unimorph_features import load_um_features

import sys
sys.path.append("../../")
from resources.um2ud_annotation.UM2UD_mapper import UM2UD_values


unmarked_features = {
    "Degree",
}
implicit_features = {
    "Number",
    "Gender",
    "Person",
}
BASE = "BASE"
UNDEFINED = "UNDEFINED"
DEFAULTS = {
    "Mood": "IND",
    "Voice": "ACT",
}
# raw UM upos codes (not UD tags) for the two verbal POS
VERB_UPOS_VALUES = {"V", "AUX"}

# UM2UD[tag] is the tag's full {UD_feature: UD_value} dict, e.g.
# UM2UD["V.PTCP"] == {"upos": "VERB", "VerbForm": "Part"}.
UM2UD = UM2UD_values
# Pass 1: (feature, value) -> UM tag via each tag's first dict item (matches
# UM2UD_values' longest-tag-first order, e.g. Mood/Ind -> "IND" over "REAL").
UD2UM = {tuple(v.items())[0]: k for k, v in UM2UD_values.items() if v.items()}
# Pass 2: backfill (feature, value) pairs compound tags set beyond their
# first (e.g. VerbForm/Part for "V.PTCP"), without overwriting pass 1's picks.
for um_tag, ud_feats in UM2UD_values.items():
    for feat, val in ud_feats.items():
        UD2UM.setdefault((feat, val), um_tag)



def allval2um(val):
    if val == UNDEFINED:
        return UNDEFINED

    # val is already a real UM tag (unimorph/ and ud_unimorph/ both store
    # real UM tags now, see multiblimp.ud2um), so just normalize casing.
    return val.upper()


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
        fill_unk_values=True,
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



class UnimorphInflector:
    def __init__(
        self,
        langcode: str,
        inflection_map: InflectionMap,
        resource_dir: Optional[str] = None,
        filter_entries: Dict[str, List[str]] = {},
        use_ud_inflections: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        fill_unk_values: bool = True,
        combine_um_ud: bool = False,
        inflect_wo_ud_features: bool = False,
        prefer_tight_match: bool = True,
        remove_diacritics: bool = False,
        remove_multiples: bool = False,
        remove_multiword_forms: bool = False,
    ):
        """
        UnimorphInflector loads in a UniMorph file and provides utility
        for inflecting features of a word.

        :param langcode: ISO-639-3 code of the language to be loaded in
        :param inflection_map: Tuple containing the feature to be
        inflected for and the mapping for what each value should be
        inflected to. If this mapping is set to None we inflect to any
        other value.
        :param resource_dir: Directory of the UD and UM files.
        :param filter_entries: Filter on the features of the UM data to be
        loaded in. Only the filtered data is considered for creating
        inflections, allowing us to constrain the inflection search to
        a subset of the data (e.g. only Verbs).
        :param use_ud_inflections: Set to true to also consider the UD
        morphological features that are computed using `ud2um.py`.
        :param verbose: Toggle to print out intermediate output.
        :param load_from_pickle: Load UM + UD files from a pre-computed
        pickle file to save overhead.
        :param fill_unk_values: Fill in undefined values in the UM data
        with default values.
        :param combine_um_ud: Toggle to find inflections in both the UM
        and UD data. UM takes precedence over UD: if a valid inflection
        is found in UM we do not look for it anymore in UD data.
        :param inflect_wo_ud_features: By default we pass along the
        morphological features from UD to steer the lemma matching
        (allowing us to know if 'saw' is a past tense of 'see' or present
        tense of 'sawing'), by setting this to True we also search for a
        match without these features
        :param prefer_tight_match: Sometimes we find multiple inflected
        forms that can be plausible, but one of them has additional
        features over the other (e.g. in Yakut negation is marked on the
        verb). By toggling this we prefer the inflected form that has the
        least amount of additional features on the word wrt the original
        form.
        :param remove_diacritics: Remove diacritics from UM data.
        :param remove_multiples: Some UM files (e.g. Greek) contain
        multiple forms per entry; we only take the first form.
        :param remove_multiword_forms: Remove word forms the contain
        multiple words. For example, in German the UM entries for verbs+
        prepositions are encoded as a single entry, but these are not
        suitable for inflecting.
        """
        self.langcode = langcode
        self.resource_dir = resource_dir or "."

        # ud_unimorph/ now stores real UM tag syntax too (multiblimp.ud2um),
        # so use_ud_inflections only changes which file gets read, not the vocabulary.
        self.feat2val, self.val2feat = load_um_features()

        self.inflection_map = inflection_map
        self.use_ud_inflections = use_ud_inflections
        self.verbose = verbose
        self.fill_unk_values = fill_unk_values
        self.combine_um_ud = combine_um_ud
        self.inflect_wo_ud_features = inflect_wo_ud_features
        self.prefer_tight_match = prefer_tight_match
        self.remove_multiword_forms = remove_multiword_forms

        self.ud_inflector = None
        self.unimorph_df = self.load_unimorph(
            remove_diacritics,
            filter_entries,
            remove_multiples,
            load_from_pickle=load_from_pickle,
        )
        self.lemma_groups, self.form_groups = self.create_df_groups()

        if inflection_map is not None:
            self.update_inflection_map()

        # (form, ud_features) -> (inflected_forms, inflected_features)
        self.prev_inflections: Dict[
            Tuple[str, Dict[str, str]], Tuple[List[str], List[str]]
        ] = {}
        # (form, features, ufeat, only_try_ud_if_no_um, prefer_tight_match, fetch_all) -> Set[str]
        self.prev_form_features: Dict[Tuple, Set[str]] = {}

        if combine_um_ud:
            assert (
                not use_ud_inflections
            ), "Set `use_ud_inflections` to False if combining UM & UD"
            self.ud_inflector = UnimorphInflector(
                langcode,
                inflection_map,
                remove_diacritics=remove_diacritics,
                filter_entries=filter_entries,
                remove_multiples=remove_multiples,
                use_ud_inflections=True,
                verbose=verbose,
                resource_dir=resource_dir,
                load_from_pickle=load_from_pickle,
                fill_unk_values=fill_unk_values,
                combine_um_ud=False,
                inflect_wo_ud_features=inflect_wo_ud_features,
                prefer_tight_match=prefer_tight_match,
                remove_multiword_forms=remove_multiword_forms,
            )
        else:
            self.ud_inflector = None

    def __len__(self) -> int:
        um_len = 0 if self.unimorph_df is None else len(self.unimorph_df)
        ud_len = 0 if self.ud_inflector is None else len(self.ud_inflector)

        return um_len + ud_len

    @property
    def ufeat(self):
        return self.inflection_map[0]

    @property
    def columns(self) -> Set[str]:
        return set() if self.unimorph_df is None else set(self.unimorph_df.columns)

    @property
    def all_columns(self) -> Set[str]:
        um_columns = (
            set() if self.unimorph_df is None else set(self.unimorph_df.columns)
        )
        ud_columns = set() if self.ud_inflector is None else self.ud_inflector.columns

        return um_columns.union(ud_columns)

    @property
    def unique_lemmas(self) -> Set[str]:
        um_lemmas = (
            set()
            if (self.unimorph_df is None or len(self.unimorph_df) == 0)
            else set(self.unimorph_df.lemma.unique())
        )
        ud_lemmas = (
            set() if self.ud_inflector is None else self.ud_inflector.unique_lemmas
        )

        return um_lemmas.union(ud_lemmas)

    @property
    def num_lemmas(self) -> int:
        return len(self.unique_lemmas)

    @property
    def unique_forms(self) -> Set[str]:
        um_forms = (
            set()
            if (self.unimorph_df is None or len(self.unimorph_df) == 0)
            else set(self.unimorph_df.form.unique())
        )
        ud_forms = (
            set() if self.ud_inflector is None else self.ud_inflector.unique_forms
        )

        return um_forms.union(ud_forms)

    @property
    def num_forms(self) -> int:
        return len(self.unique_forms)

    def unique_values(self, column: str, upos: Optional[str] = None) -> List[str]:
        unique_values = set()
        if column in self.columns:
            df = self.unimorph_df
            if upos is not None:
                df = df[df.upos == upos]

            unique_values.update(df[column].unique())

        if self.ud_inflector is not None:
            unique_values.update(self.ud_inflector.unique_values(column, upos=upos))

        unique_values = [
            allval2um(val)
            for val in unique_values
            if not (val == UNDEFINED or pd.isna(val))
        ]

        return unique_values

    @property
    def can_feature_swap(self) -> bool:
        """Returns True if the feature of the inflection map is present
        in either the UM or UD dataframe columns.
        """
        um_can_feature_swap = isinstance(self.ufeat, str)

        ud_can_feature_swap = False
        if self.ud_inflector is not None:
            ud_can_feature_swap = isinstance(self.ud_inflector.ufeat, str)

        return um_can_feature_swap or ud_can_feature_swap

    @property
    def has_unimorph_df(self) -> bool:
        return (self.unimorph_df is not None) and (len(self.unimorph_df) > 0)

    def load_unimorph(
        self,
        remove_diacritics: bool,
        filter: Dict[str, List[str]],
        remove_multiples: bool,
        load_from_pickle: bool = False,
    ) -> pd.DataFrame:
        if load_from_pickle:
            if self.use_ud_inflections:
                return self.load_unimorph_pickle("ud_unimorph/ud_pickles", filter)
            else:
                return self.load_unimorph_pickle("unimorph/um_pickles", filter)

        if self.use_ud_inflections:
            path = os.path.join(self.resource_dir, f"ud_unimorph/{self.langcode}")
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(
                path, sep="\t", names=["lemma", "form", "ufeat", "frequency"], header=1
            )
            df = df.drop("frequency", axis=1)
        else:
            path = os.path.join(
                self.resource_dir, f"unimorph/{self.langcode}/{self.langcode}"
            )
            if not os.path.isfile(path):
                return None
            df = pd.read_csv(path, sep="\t", names=["lemma", "form", "ufeat"])

            if os.path.isfile(path + ".segmentations"):
                path += ".segmentations"
                df_seg = pd.read_csv(
                    path, sep="\t", names=["lemma", "form", "ufeat", "segmentation"]
                )
                df_seg = df_seg.drop("segmentation", axis=1)
                df_seg["ufeat"] = [
                    str(ufeat).replace("|", ";") for ufeat in df_seg.ufeat
                ]

                df = pd.concat([df, df_seg], ignore_index=True).drop_duplicates()

        if remove_diacritics:
            df.lemma = [unidecode(lemma) for lemma in df.lemma]
            df.form = [unidecode(lemma) for lemma in df.form]

        if (self.langcode == "tat") and (not self.use_ud_inflections):
            df.lemma = [latin_to_cyrillic(lemma) for lemma in df.lemma]
            df.form = [latin_to_cyrillic(lemma) for lemma in df.form]

        if remove_multiples:
            df.form = [form.split(", ")[0] for form in df.form]

        ufeat_cols = {x: [] for x in self.feat2val}

        for ufeat in df.ufeat:
            row_ufeats = self.ufeats2dict(ufeat)
            for ufeat_col in self.feat2val:
                ufeat_cols[ufeat_col].append(row_ufeats.get(ufeat_col))

        # Only set the columns for ufeats that *do* occur
        for ufeat_col, ufeat_vals in ufeat_cols.items():
            num_not_none = len([x for x in ufeat_vals if x])
            if num_not_none > 0:
                df[ufeat_col] = ufeat_vals

        df = self.filter_entries(df, filter)
        df = self.expand_multiple_values(df)
        df = self.set_unk_values(df)

        # Plain object dtype, not "category": partial_df_match's equality
        # lookups compare these columns against literal strings on every
        # inflection attempt, and category dtype pays a conversion cost on
        # each such comparison (profiled: ~50s of a 94s create_pairs run).
        for column in df.columns:
            df[column] = df[column].astype(object)

        return df

    def load_unimorph_pickle(
        self, pickle_path: str, filter: Dict[str, List[str]]
    ) -> pd.DataFrame:
        path = os.path.join(self.resource_dir, pickle_path, f"{self.langcode}.pickle")
        if not os.path.isfile(path):
            if self.verbose:
                print(f"UM Pickle not found at {path}")
            return None
        df = pd.read_pickle(path)

        # Pickled dataframes were saved with "category" dtype columns; convert to
        # plain object dtype for the same reason as load_unimorph above.
        for column in df.columns:
            df[column] = df[column].astype(object)

        df = self.filter_entries(df, filter)
        df = self.set_unk_values(df)

        return df

    def pickle_unimorph_df(self, path: str) -> None:
        self.unimorph_df.to_pickle(path)

    def filter_entries(self, df, filter: Dict[str, List[str]]):
        """Only keep entries of a particular feature tag to speed up inflections later."""
        if self.remove_multiword_forms:
            df = df[~df.form.str.contains(" ", na=False)]

        if len(filter) == 0:
            return df

        for ufeat, values in filter.items():
            if ufeat in df.columns:
                values = self.val2ud_um(ufeat, values)
                pos_values = [val for val in values if not val.startswith("-")]
                neg_values = [val[1:] for val in values if val.startswith("-")]

                if len(pos_values) > 0:
                    df = df[df[ufeat].isin(pos_values)]
                if len(neg_values) > 0:
                    df = df[~df[ufeat].isin(neg_values)]
                nan_cols = [
                    col for col in df.columns if sum(pd.isna(df[col])) == len(df)
                ]
                for column in nan_cols:
                    df = df.drop(column, axis=1)

        return df

    def set_unk_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if isinstance(df[column].dtype, pd.CategoricalDtype):
                if BASE not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([BASE])
                if UNDEFINED not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([UNDEFINED])

        if self.fill_unk_values:
            # Missingness of a feature in UniMorph either indicates that
            # i) the feature is an (unmarked) base class that is distinct from marked ones (e.g. ADJ Degree in Dutch)
            # ii) the form is the same for all instantiations of the feature (ADJ Gender in Dutch)
            for column in unmarked_features & set(df.columns):
                df.loc[pd.isna(df[column]), column] = BASE
            for column in implicit_features & set(df.columns):
                df.loc[pd.isna(df[column]), column] = UNDEFINED

            # Plain finite verbs get no VerbForm token (only V.PTCP/INF/... do),
            # leaving NaN -- a partial_df_match wildcard that could match Part.
            # Resolved before DEFAULTS below, which needs finiteness to decide
            # whether Mood/Voice defaults apply (mood/voice are meaningless on
            # participles etc., so must not be defaulted onto them).
            fin_val = None
            if "VerbForm" in df.columns and "upos" in df.columns:
                fin_val = self.val2ud_um("VerbForm", "Fin")
                is_verb = df["upos"].isin(VERB_UPOS_VALUES)
                if (
                    isinstance(df["VerbForm"].dtype, pd.CategoricalDtype)
                    and fin_val not in df["VerbForm"].cat.categories
                ):
                    df["VerbForm"] = df["VerbForm"].cat.add_categories([fin_val])
                df.loc[is_verb & pd.isna(df["VerbForm"]), "VerbForm"] = fin_val

            # Mood/Voice defaults only make sense on finite forms; skip
            # non-finite rows (participles etc.) if we know which are which.
            is_finite = (
                df["VerbForm"] == fin_val if fin_val is not None else True
            )
            for column in set(DEFAULTS.keys()) & set(df.columns):
                column_val = self.val2ud_um(column, DEFAULTS[column])

                if (
                    isinstance(df[column].dtype, pd.CategoricalDtype)
                    and column_val not in df[column].cat.categories
                ):
                    df[column] = df[column].cat.add_categories([column_val])
                df.loc[is_finite & pd.isna(df[column]), column] = column_val

        return df

    def expand_multiple_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_rows = df.to_dict(orient="records")

        idx = 0
        delete_rows = []

        while idx < len(df_rows):
            row = df_rows[idx]
            for ufeat, val in row.items():
                if (
                    (ufeat not in {"lemma", "form", "ufeat"})
                    and isinstance(val, str)
                    and ("+" in val)
                ):
                    all_vals = val.split("+")
                    for extra_val in all_vals:
                        extra_row = row.copy()
                        extra_row[ufeat] = self.val2ud_um(ufeat, extra_val)
                        df_rows.append(extra_row)

                    # We expand only one ufeat per loop, if multiple ufeats are disjunctions we get to those
                    # later since the row is appended to the end of the list of extra_rows.
                    # The row with the "x+y+z" value is removed.
                    delete_rows.append(idx)
                    break

            idx += 1

        df = pd.DataFrame(df_rows)
        df = df.drop(index=delete_rows)

        return df

    def create_df_groups(self):
        lemma_groups = None
        form_groups = None

        if self.has_unimorph_df:
            lemma_groups = self.unimorph_df.groupby("lemma", observed=False)
            form_groups = self.unimorph_df.groupby("form", observed=False)

        return lemma_groups, form_groups

    def update_inflection_map(self) -> None:
        """UD morphological features are encoded different than UM
        features (e.g. Sing vs. SG). We need to ensure the inflection
        map uses the right feature names, which are set here."""

        # Some languages have subcategories for features (e.g. Number[subj])
        # To use a universal inflection map, we look for the feature
        # that is present the *most* in the unimorph_df columns for inflection.
        if isinstance(self.ufeat, list):
            inflection_column = None
            min_feats_per_column = 0
            for column in self.ufeat:
                if column in self.columns:
                    feature_counts = Counter(self.unimorph_df[column])
                    non_unk_counts = []
                    for ufeat, counts in feature_counts.items():
                        if (not pd.isna(ufeat)) and (ufeat != UNDEFINED):
                            non_unk_counts.append(counts)

                    if (len(non_unk_counts) > 1) and (
                        min(non_unk_counts) > min_feats_per_column
                    ):
                        min_feats_per_column = min(non_unk_counts)
                        inflection_column = column

            self.inflection_map = (inflection_column, self.inflection_map[1])

        swap_ufeat, swap_map = self.inflection_map
        if isinstance(swap_ufeat, list):
            # None of the inflection map features are present in this language
            swap_ufeat = None
            self.inflection_map = (swap_ufeat, swap_map)


    def ufeats2dict(self, ufeats: str) -> Dict[str, str]: # replace with um2ud_mapper 
        """Translates the unimorph X;Y;Z format to a dictionary"""
        ufeats = (
            str(ufeats)
            .replace("/", "+")
            .replace(",", "+")
            .replace("{", "")
            .replace("}", "")
            .replace("V.PTCP", "V;V.PTCP")
            .replace("V.PCTP", "V;V.PTCP")
        )
        ufeats = ufeats.strip()
        ufeat_vals = ufeats.split(";")
        ufeat_dict = {}

        for val in ufeat_vals:
            if len(val) == 0:
                continue
            elif val in self.val2feat:
                ufeat = self.val2feat[val]
            elif val.strip() in self.val2feat:
                ufeat = self.val2feat[val.strip()]  # Livvi
            elif "." in val:
                val1, val2 = val.split(".")
                ufeat1 = self.val2feat[val1]
                ufeat2 = self.val2feat[val2]
                ufeat_dict[ufeat1] = val1
                ufeat_dict[ufeat2] = val2
                continue
            elif "+" in val:
                subval = val.split("+")[0]
                ufeat = self.val2feat[subval]
            elif val.upper() in self.val2feat:
                ufeat = self.val2feat[val.upper()]
            else:
                ufeat = "UNK"

            if ufeat in ufeat_dict:
                ufeat_dict[ufeat] += f"+{val}"
            else:
                ufeat_dict[ufeat] = val

        return ufeat_dict

    def inflect(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]] = [],
        return_swap_feats: bool = False,
    ) -> Union[
        Tuple[Union[None, str, List[str]], Optional[Set[str]]],
        Tuple[Union[None, str, List[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]],
    ]:
        """return_swap_feats: also return {swap_form: {feature: {values}}},
        e.g. for flowchart display -- captured directly from the rows that
        produced each swap_form, instead of a separate get_form_feature_bundle
        lookup by result form. Default False keeps the old 2-tuple return
        shape so existing callers (e.g. multiblimp.pipeline) don't break.
        """
        prev_inflect_key = (form, frozenset(ud_features.items()), return_swap_feats)
        if prev_inflect_key in self.prev_inflections:
            return self.prev_inflections[prev_inflect_key]

        if self.unimorph_df is None or len(self.unimorph_df) == 0:
            swap_forms = None
            feature_vals = None
            swap_feats = {}
        else:
            swap_forms, feature_vals, swap_feats = self.inflect_features(
                form, ud_features, strategies, return_swap_feats
            )

            if not self.form_found(swap_forms) and self.inflect_wo_ud_features:
                swap_forms, feature_vals, swap_feats = self.inflect_features(
                    form, {}, strategies, return_swap_feats
                )

        if self.ud_inflector is not None and not self.form_found(swap_forms):
            ud_result = self.ud_inflector.inflect(
                form, ud_features, strategies=strategies,
                return_swap_feats=return_swap_feats,
            )
            if return_swap_feats:
                ud_forms, ud_feature_vals, ud_swap_feats = ud_result
            else:
                ud_forms, ud_feature_vals = ud_result
                ud_swap_feats = {}
            if swap_forms is None:
                swap_forms = ud_forms
                feature_vals = ud_feature_vals
                swap_feats = ud_swap_feats
            elif ud_forms is not None:
                swap_forms.update(ud_forms)
                feature_vals.update(ud_feature_vals)
                for swap_form, bundle in ud_swap_feats.items():
                    merged = swap_feats.setdefault(swap_form, {})
                    for col, vals in bundle.items():
                        merged.setdefault(col, set()).update(vals)

        if isinstance(swap_forms, set):
            swap_forms = list(swap_forms)

        result = (
            (swap_forms, feature_vals, swap_feats)
            if return_swap_feats
            else (swap_forms, feature_vals)
        )
        self.prev_inflections[prev_inflect_key] = result

        return result

    def inflect_features(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]],
        return_swap_feats: bool = False,
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]]:
        """Inflect form based on provided `ud_features`."""
        swap_forms, feature_vals, swap_feats = self.inflect_strategy(
            form, ud_features, return_swap_feats=return_swap_feats
        )
        if not self.form_found(swap_forms):
            for strat in strategies:
                swap_forms, feature_vals, swap_feats = self.inflect_strategy(
                    form, ud_features, strategy=strat, return_swap_feats=return_swap_feats
                )
                if self.form_found(swap_forms):
                    break

        return swap_forms, feature_vals, swap_feats

    @staticmethod
    def form_found(forms: Optional[List[str]]) -> bool:
        return (forms is not None) and (len(forms) == 1)

    def inflect_strategy(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategy: Dict[str, Optional[str]] = {},
        return_swap_feats: bool = False,
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]]:
        um_features = self.ud2um_features(ud_features, strategy)
        form_rows = self.form2rows(form, um_features)

        # We skip inflection if no matching rows were found
        if len(form_rows) == 0:
            return None, None, {}

        forms = set()
        feature_vals = set()
        swap_feats = {}

        for row_features, feature_val in self.yield_row_features(
            um_features, form_rows, strategy
        ):
            inflected_forms, bundles = self.lemma2form(row_features, return_swap_feats)

            forms.update(inflected_forms)
            feature_vals.add(feature_val)
            # Union per form: the same swap form can be reached via more than
            # one candidate row (e.g. different original-form matches).
            for swap_form, bundle in bundles.items():
                merged = swap_feats.setdefault(swap_form, {})
                for col, vals in bundle.items():
                    merged.setdefault(col, set()).update(vals)

        return forms, feature_vals, swap_feats

    def ud2um_features(
        self, ud_features: Dict[str, str], strategy={}, set_defaults=True
    ) -> Dict[str, str]:
        um_features = {}

        for ufeat, val in ud_features.items():
            if ufeat not in self.columns:
                continue
            elif ufeat == "lemma":
                um_features[ufeat] = val
            elif ufeat in strategy:
                um_features[ufeat] = strategy[ufeat]
            else:
                um_features[ufeat] = UD2UM.get((ufeat, val), None)#self.val2ud_um(ufeat, val)

        if set_defaults:
            for ufeat, val in DEFAULTS.items():
                if (ufeat in self.columns) and (ufeat not in um_features):
                    um_features[ufeat] = self.val2ud_um(ufeat, val)

        if self.verbose:
            print("UM Features:", um_features)

        return um_features

    def val2ud_um(self, feat, val) -> Union[str, List[str]]:
        """Maps a value from a UD/UM feature map to the compatible
        feature format of the inflector (which can be either UM/UD)
        """
        if ("," in val) or ("|" in val) or ("/" in val):
            val = val.replace(",", "+").replace("|", "+").replace("/", "+")
            subvals = val.split("+")
            um_vals = []
            for subval in subvals:
                um_vals.append(self.val2ud_um(feat, subval))
            return um_vals

        if f"{feat}_{val}" in self.val2feat:
            return f"{feat}_{val}"
        elif feat == "lemma":
            return val
        elif isinstance(val, list):
            return [self.val2ud_um(feat, subval) for subval in val]
        elif val in self.val2feat:
            return val
        elif (feat, val) in UD2UM:
            # Must come before val.upper() below: e.g. "PART" is also a valid
            # val2feat entry (upos Particle), which would misresolve VerbForm=Part.
            return UD2UM[(feat, val)]
        elif val in UM2UD and feat in UM2UD[val]:
            # val is already a UM tag carrying a value for this feature.
            return self.val2ud_um(feat, UM2UD[val][feat])
        elif val.upper() in self.val2feat:
            return val.upper()
        elif val.startswith("-"):
            return "-" + self.val2ud_um(feat, val[1:])
        elif f"{feat}_{val.title()}" in self.val2feat:
            return f"{feat}_{val.title()}"
        else:
            raise ValueError(f"Unknown {feat}: {val}")

    def form2rows(self, form, um_features: Dict[str, str]):
        sub_df = self.partial_df_match(self.form_groups, form, um_features)

        if len(sub_df) > 1:
            sub_df = sub_df.drop_duplicates()

        if self.verbose:
            print("Matched Rows:", sub_df)

        return sub_df

    def lemma2form(self, row_features: Dict[str, str], return_bundle: bool = False):
        lemma = row_features.pop("lemma")
        sub_df = self.partial_df_match(self.lemma_groups, lemma, row_features)
        inflected_forms = set(sub_df.form)

        if self.verbose:
            print("Matched Inflected Rows:", sub_df)
            print("Matched Forms:", inflected_forms)

        bundles = {}
        if return_bundle:
            for form_val, rows in sub_df.groupby("form"):
                bundle = self._rows_to_bundle(rows)
                if bundle:
                    bundles[form_val] = bundle

        return inflected_forms, bundles

    @staticmethod
    def _rows_to_bundle(rows_df) -> Dict[str, Set[str]]:
        """Every UM feature column's value(s) across `rows_df`, e.g. for
        displaying a reinflected form's other features (flowchart's "after"
        column). UNDEFINED means "no info", same as NaN -- excluded."""
        bundle = {}
        for col in rows_df.columns:
            if col in ("lemma", "form", "ufeat", "upos"):
                continue
            values = {
                allval2um(val) for val in set(rows_df[col])
                if isinstance(val, str) and val != UNDEFINED
            }
            if values:
                bundle[col] = values
        return bundle

    def partial_df_match(
        self,
        groups,
        group_index,
        features: Dict[str, str],
        prefer_tight_match: Optional[bool] = None,
    ):
        """Find all rows in the morphology dataframe that match the
        group_index (lemma or form) and the features in the provided
        `features` dictionary.
        """
        if len(groups.indices.get(group_index, [])) == 0:
            empty_df = self.unimorph_df.iloc[0:0]
            return empty_df

        sub_df = groups.get_group(group_index)

        if len(sub_df) == 0:
            return sub_df

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
                elif col == self.ufeat:
                    mask &= sub_df[col] == val
                else:
                    mask &= (
                        (sub_df[col] == val)
                        | (sub_df[col] == UNDEFINED)
                        | pd.isna(sub_df[col])
                    )

        candidate_rows = sub_df[mask]

        if prefer_tight_match is None:
            prefer_tight_match = self.prefer_tight_match
        if (not prefer_tight_match) or (len(candidate_rows) < 2):
            return candidate_rows

        # Vectorized equivalent of the per-row/per-column loop below (avoids
        # candidate_rows.iterrows(), expensive on wide/mixed-dtype frames):
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

        if self.verbose:
            print("Tight matching #added features:", features, features_added)
            print(
                "Tight matching yielded:",
                candidate_rows,
                "to",
                candidate_rows[min_features_added_mask],
                sep="\n",
            )

        return candidate_rows[min_features_added_mask]

    def yield_row_features(
        self,
        um_features: Dict[str, str],
        form_rows: pd.DataFrame,
        strategy: Dict[str, Optional[str]],
    ):
        """
        Based on all matching rows, set and yield the (swapped) features
        we use for finding the inflected form.
        """
        swap_ufeat, swap_map = self.inflection_map

        for _, row in form_rows.iterrows():
            row_features = dict(um_features)  # make a copy
            skip_row = False
            feature_val = None

            if (
                (swap_ufeat not in row.keys())
                or (row[swap_ufeat] == UNDEFINED)
                or pd.isna(row[swap_ufeat])
            ):
                skip_row = True

            for ufeat, val in row.items():
                if (ufeat in {"ufeat", "form"}) or (pd.isna(val)):
                    continue

                if ufeat == swap_ufeat:
                    if swap_map is None:
                        feature_val = allval2um(val)
                        row_features[ufeat] = f"-{val}"
                    elif val not in swap_map:
                        skip_row = True
                        continue
                    else:
                        feature_val = allval2um(val)
                        row_features[ufeat] = self.val2ud_um(ufeat, swap_map[val])
                elif val == UNDEFINED:
                    continue
                else:
                    row_features[ufeat] = val

            # Override values if strategy is set
            for ufeat, val in strategy.items():
                row_features[ufeat] = val

            if self.verbose:
                print("Matched Features", row_features)

            if (not skip_row) and (feature_val is not None):
                yield row_features, feature_val

    def get_form_features(
        self,
        form: str,
        features: Dict[str, str],
        ufeat: Optional[str] = None,
        only_try_ud_if_no_um: bool = False,
        prefer_tight_match: bool = False,
        fetch_all = False
    ) -> Set[str]:
        # Same form/features/ufeat combos recur constantly (common subjects, child
        # forms, agreement targets) and each uncached call does a full UniMorph
        # lookup — including a recursive self.ud_inflector.get_form_features call —
        # so this is memoized the same way inflect() already is.
        cache_key = (
            form, frozenset(features.items()), ufeat,
            only_try_ud_if_no_um, prefer_tight_match, fetch_all,
        )
        if cache_key in self.prev_form_features:
            # Return a copy — callers must not mutate the cached set in place.
            return set(self.prev_form_features[cache_key])

        result = self._get_form_features_uncached(
            form, features, ufeat, only_try_ud_if_no_um, prefer_tight_match, fetch_all
        )
        self.prev_form_features[cache_key] = result
        return set(result)

    def _get_form_features_uncached(
        self,
        form: str,
        features: Dict[str, str],
        ufeat: Optional[str] = None,
        only_try_ud_if_no_um: bool = False,
        prefer_tight_match: bool = False,
        fetch_all = False
    ) -> Set[str]:
        um_features = self.ud2um_features(features, set_defaults=False)
        if ufeat in um_features:
            del um_features[ufeat]

        form_features = set()

        if self.has_unimorph_df:
            df_ufeat = self.ufeat if ufeat is None else ufeat
            if df_ufeat is not None and fetch_all==False:
                form_rows = self.partial_df_match(
                    self.form_groups, form, um_features,
                    prefer_tight_match=prefer_tight_match
                )
                if self.verbose:
                    print(form_rows)
                if (len(form_rows) > 0) and (df_ufeat in form_rows.columns):
                    form_features.update(
                        {
                            allval2um(val)
                            for val in set(form_rows[df_ufeat])
                            # UNDEFINED means "no info", same as NaN -- exclude it.
                            if isinstance(val, str) and val != UNDEFINED
                        }
                    )
                if self.ud_inflector is not None:
                    if only_try_ud_if_no_um and len(form_features) > 0:
                        return form_features
                    ud_form_features = self.ud_inflector.get_form_features(
                        form, features, ufeat
                    )
                    form_features.update(ud_form_features)

        return form_features