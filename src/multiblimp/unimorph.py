import os
import shutil
from typing import *
from urllib.parse import quote, unquote
from unidecode import unidecode

import numpy as np
import pandas as pd
from collections import Counter

from .ud_table import UDTable
from .config import REPO_ROOT
from .languages import latin_to_cyrillic, remove_diacritics_langs, remove_multiples_langs, unimorph_code

import sys
from functools import lru_cache
sys.path.append(REPO_ROOT)
from resources.um2ud_annotation.UM2UD_mapper import (
    COMPOUND_TAGS, LAYERED_FEAT_SEP, SHORTENED_UM_VALS, UM2UD, UM2UD_values, UPOS_VALUES,
    fix_typos as FIX_TYPOS, val2feat as PKG_VAL2FEAT, feat2val as PKG_FEAT2VAL, feature_for,
    um_tag_upos, pos_is_degree,
)
from resources.um2ud_annotation.UD2UM_mapper import BASE, UD2UM, expand_ud_feats, stored_upos, to_um_value, ufeats_to_dict


# (swap feature or candidate list, {value: swapped value} or None for any other value)
InflectionMap = Tuple[Union[str, List[str]], Optional[Dict[str, str]]]

# Tables above this many rows are pickled per UPOS (see pickle_unimorph_df).
SPLIT_PICKLE_MIN_ROWS = 5_000_000


class PosLookup(dict):
    """LEGACY (pickle lookups). {upos: DataFrame} whose slices may be loaded on first .get() (a split
    pickle) instead of up front."""

    def __init__(self, loader=None, names=()):
        super().__init__()
        self._loader = loader
        self._names = frozenset(names)

    def upos_names(self):
        return set(self) | self._names

    def get(self, key, default=None):
        if key not in self and key in self._names:
            self[key] = self._loader(key)
        return super().get(key, default)


unmarked_features = {
    "Degree",
}
implicit_features = {
    "Number",
    "Gender",
    "Person",
}
UNDEFINED = "UNDEFINED"
DEFAULTS = {
    "Mood": "IND",
    "Voice": "ACT",
    # unannotated Degree is positive; resolves to BASE via the ("Degree", "Pos") entry in UD2UM
    "Degree": "Pos",
}
# raw UM upos codes (not UD tags) for the two verbal POS
VERB_UPOS_VALUES = {"V", "AUX"}


def ud_value_to_um(feat: str, val):
    """UD value -> UM code for `feat` (bracket suffix ignored). A comma value ("Masc,Neut")
    becomes the list of its readings' codes; so does a "/"-joined code (Perf -> "PFV/PRF").
    None if any reading has no UM code (the constraint is then dropped)."""
    base = feat.partition("[")[0]

    def _expand(code):
        return code.split("/") if isinstance(code, str) and "/" in code else [code]

    if isinstance(val, str) and "," in val:
        codes = [UD2UM.get((base, v)) for v in val.split(",")]
        if any(c is None for c in codes):
            return None
        flat = [c for code in codes for c in _expand(code)]
        return flat if len(flat) > 1 else flat[0]

    code = UD2UM.get((base, val), None)
    if code is None:
        return None
    expanded = _expand(code)
    return expanded if len(expanded) > 1 else expanded[0]


def _has_value(val) -> bool:
    return len(val) > 0 if isinstance(val, list) else bool(pd.notna(val))


def allval2um(val):
    if val == UNDEFINED:
        return UNDEFINED

    # val is already a real UM tag (unimorph/ and ud_unimorph/ both store
    # real UM tags now, see the ud_unimorph TSV writer, removed), so just normalize casing.
    return val.upper()


def um_val_to_ud(feat: str, val: str) -> str:
    """A UM value ("PL", "NOM") for UD feature `feat` -> its UD value ("Plur", "Nom"), looking
    in SHORTENED_UM_VALS first, then UM2UD. `feat` may be bracketed (the bracket is stripped).
    Returns `val` unchanged if neither table has it under `feat` (e.g. UNDEFINED)."""
    base_feat = feat.partition("[")[0]
    source = SHORTENED_UM_VALS if val in SHORTENED_UM_VALS else UM2UD
    return source.get(val, {}).get(base_feat, val)


def load_inflector(lang: str, langcode: str, unimorph_args, inflection_map: dict,
                   resource_dir: str):
    """LEGACY (the live pipeline uses ud_inflector.load_ud_inflector).
    (inflector, number of lemmas, number of forms); both counts are 0 for an empty table."""
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
        return inflector, 0, 0
    return inflector, inflector.num_lemmas, inflector.num_forms


class UnimorphInflector:
    """LEGACY for inference: inflect(), get_form_features() and the matching helpers
    (partial_df_match, form2rows, lemma2form, yield_row_features) are used only by the old
    multiblimp.pipeline chain; the live pipeline inflects with multiblimp.ud_inflector.UDInflector.
    Shared vocabulary and the UniMorph text parser moved to resources/um2ud_annotation and
    multiblimp.um2ud; the methods left here (stored_upos, ufeats2dict, val2ud_um) delegate to them."""

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
        skip_load: bool = False,
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
        morphological features that are computed using the old ud2um.py.
        :param verbose: Toggle to print out intermediate output.
        :param load_from_pickle: Load UM + UD files from a pre-computed
        pickle file to save overhead.
        :param fill_unk_values: Fill in undefined values in the UM data
        with default values.
        :param combine_um_ud: Toggle to find inflections in both the UM
        and UD data. Both are always queried and their candidate forms
        unioned (see inflect()) -- UM finding a match does not stop UD
        from being consulted too, since each lexicon can independently
        cover a case the other misses (e.g. a syncretic/ambiguous UM
        match vs. a usable UD-derived form for the same row).
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
        :param skip_load: Don't read any table (unimorph_df stays None); for
        callers that only use the processing methods, see
        pickle_unimorph_streaming.
        """
        self.langcode = langcode
        self.um_langcode = unimorph_code(langcode)
        self.resource_dir = resource_dir or "."

        # use_ud_inflections only changes which file is read: ud_unimorph/ uses UM tags too.
        # feat2val/val2feat are the vendored package's tables, shared and never mutated.
        self.feat2val, self.val2feat = PKG_FEAT2VAL, PKG_VAL2FEAT

        self.inflection_map = inflection_map
        self.use_ud_inflections = use_ud_inflections
        self.verbose = verbose
        self.fill_unk_values = fill_unk_values
        self.combine_um_ud = combine_um_ud
        self.inflect_wo_ud_features = inflect_wo_ud_features
        self.prefer_tight_match = prefer_tight_match
        self.remove_multiword_forms = remove_multiword_forms

        self.ud_inflector = None
        self.unimorph_df = (
            None
            if skip_load
            else self.load_unimorph(
                remove_diacritics,
                filter_entries,
                remove_multiples,
                load_from_pickle=load_from_pickle,
            )
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
        # (id(groups), group_index) -> (group sub-df, {col: object ndarray},
        # {col: is-set ndarray}), see partial_df_match
        self._group_cache: Dict[Tuple, Tuple] = {}

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
            # callers read ufeat off this outer instance, so take the UD table's
            # resolution when the UM table (empty or filtered away) left it None
            if (self.inflection_map[0] is None) and (
                self.ud_inflector.inflection_map[0] is not None
            ):
                self.inflection_map = (
                    self.ud_inflector.inflection_map[0], self.inflection_map[1]
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

    def resolve_column(self, key: str, slot_aliases: Sequence[str] = (), columns: Optional[Set[str]] = None) -> str:
        """`key` itself if it's a lexicon column here, else the same feature
        under another bracket suffix in `slot_aliases` -- the suffixes that
        all name one argument slot for the row being processed (relation-
        named and case-named, see agreement_trees.extract.process_treebank.slot_suffixes)
        -- that IS a column; `key` unchanged if none is."""
        columns = self.columns if columns is None else columns
        base, bracket, suffix = key.partition("[")
        if key in columns or not bracket or suffix.rstrip("]") not in slot_aliases:
            return key
        return next(
            (f"{base}[{s}]" for s in slot_aliases if f"{base}[{s}]" in columns), key
        )

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
                path, sep="\t", names=["lemma", "form", "ufeat", "frequency"], header=0
            )
            df = df.drop("frequency", axis=1)
        else:
            path = os.path.join(
                self.resource_dir, f"unimorph/{self.um_langcode}/{self.um_langcode}"
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

        return self._build_df(df, remove_diacritics, filter, remove_multiples)

    def _build_df(
        self,
        df: pd.DataFrame,
        remove_diacritics: bool,
        filter: Dict[str, List[str]],
        remove_multiples: bool,
    ) -> pd.DataFrame:
        if remove_diacritics:
            df.lemma = [unidecode(lemma) for lemma in df.lemma]
            df.form = [unidecode(lemma) for lemma in df.form]

        if (self.langcode == "tat") and (not self.use_ud_inflections):
            df.lemma = [latin_to_cyrillic(lemma) for lemma in df.lemma]
            df.form = [latin_to_cyrillic(lemma) for lemma in df.form]

        if remove_multiples:
            df.form = [form.split(", ")[0] for form in df.form]

        for column in ("lemma", "form"):
            df[column] = df[column].map(lambda v: v.lower() if isinstance(v, str) else v)

        row_ufeats = [self.ufeats2dict(ufeat) for ufeat in df.ufeat]

        # "{feat}[{suffix}]" columns (ARG*/PSS* tags, LAYERED_FEAT_SEP) are not in feat2val
        bracketed_cols = {k for d in row_ufeats for k in d if "[" in k}
        ufeat_cols = {x: [] for x in sorted(set(self.feat2val) | bracketed_cols)}

        for d in row_ufeats:
            for ufeat_col in ufeat_cols:
                ufeat_cols[ufeat_col].append(d.get(ufeat_col))

        # Only set the columns for ufeats that *do* occur
        for ufeat_col, ufeat_vals in ufeat_cols.items():
            num_not_none = len([x for x in ufeat_vals if x])
            if num_not_none > 0:
                df[ufeat_col] = ufeat_vals

        df = self.filter_entries(df, filter)
        df = self.expand_multiple_values(df)
        df = self.set_unk_values(df)

        # object dtype, not category: partial_df_match compares against literals on every call
        for column in df.columns:
            df[column] = df[column].astype(object)

        return df

    def _finalize_pickle_df(self, df, filter: Dict[str, List[str]]):
        df = self.filter_entries(df, filter)

        # object dtype (see _build_df)
        for column in df.columns:
            df[column] = df[column].astype(object)

        return self.set_unk_values(df)

    def _table_code(self, pickle_path: str) -> str:
        return self.langcode if pickle_path.startswith("ud_unimorph") else self.um_langcode

    def _split_pickle_dir(self, pickle_path: str) -> str:
        return os.path.join(self.resource_dir, pickle_path, f"{self._table_code(pickle_path)}.split")

    def _split_pickle_upos(self, split_dir: str) -> List[str]:
        if not os.path.isdir(split_dir):
            return []
        return [
            unquote(f[: -len(".pickle")])
            for f in os.listdir(split_dir)
            if f.endswith(".pickle")
        ]

    def _split_pickle_file(self, split_dir: str, upos: str) -> str:
        return os.path.join(split_dir, f"{quote(upos, safe='')}.pickle")

    def load_unimorph_pickle(
        self, pickle_path: str, filter: Dict[str, List[str]]
    ) -> pd.DataFrame:
        path = os.path.join(self.resource_dir, pickle_path, f"{self._table_code(pickle_path)}.pickle")
        if os.path.isfile(path):
            return self._finalize_pickle_df(pd.read_pickle(path), filter)

        split_dir = self._split_pickle_dir(pickle_path)
        available = self._split_pickle_upos(split_dir)
        if len(available) == 0:
            if self.verbose:
                print(f"UM Pickle not found at {path}")
            return None

        wanted = available
        if "upos" in filter:
            values = self.val2ud_um("upos", filter["upos"])
            pos_values = [val for val in values if not val.startswith("-")]
            neg_values = [val[1:] for val in values if val.startswith("-")]
            wanted = [
                u for u in available
                if (len(pos_values) == 0 or u in pos_values) and u not in neg_values
            ]
        if len(wanted) == 0:
            return None
        df = pd.concat(
            [pd.read_pickle(self._split_pickle_file(split_dir, u)) for u in wanted],
            ignore_index=True,
        )
        return self._finalize_pickle_df(df, filter)

    def load_pos_lookup(self, pickle_path: str) -> Optional["PosLookup"]:
        """LEGACY (pickle lookups, no live user). {upos: DataFrame slice} lookup over a pickled table. A split pickle
        (see pickle_unimorph_df) is read one UPOS at a time, on first use."""
        split_dir = self._split_pickle_dir(pickle_path)
        names = self._split_pickle_upos(split_dir)
        if len(names) > 0 and not os.path.isfile(
            os.path.join(self.resource_dir, pickle_path, f"{self._table_code(pickle_path)}.pickle")
        ):
            def loader(upos):
                df = self._finalize_pickle_df(
                    pd.read_pickle(self._split_pickle_file(split_dir, upos)), {}
                )
                return df.dropna(axis=1, how="all")

            return PosLookup(loader, names)

        df = self.load_unimorph_pickle(pickle_path, {})
        if not isinstance(df, pd.DataFrame):
            return None
        lookup = PosLookup()
        for pos in df["upos"].unique():
            lookup[pos] = df[df["upos"] == pos].dropna(axis=1, how="all")
        return lookup

    def pickle_unimorph_streaming(
        self,
        pickle_path: str,
        remove_diacritics: bool,
        remove_multiples: bool,
        chunk_rows: int = 1_000_000,
    ) -> int:
        """pickle_unimorph_df's split layout for a UniMorph text file too big to
        parse in one go (load_unimorph holds a dict per row plus a copy of the
        frame): parse `chunk_rows` rows at a time -- every step of _build_df
        is row-wise -- and assemble one pickle per UPOS at the end.
        Returns the row count."""
        path = os.path.join(
            self.resource_dir, f"unimorph/{self.um_langcode}/{self.um_langcode}"
        )
        split_dir = pickle_path[: -len(".pickle")] + ".split"
        tmp_dir = split_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)

        columns: List[str] = []
        parts: Dict[str, List[str]] = {}
        num_rows = 0
        reader = pd.read_csv(
            path, sep="\t", names=["lemma", "form", "ufeat"], dtype=str,
            chunksize=chunk_rows,
        )
        for chunk_idx, chunk in enumerate(reader):
            df = self._build_df(chunk, remove_diacritics, {}, remove_multiples)
            columns += [c for c in df.columns if c not in columns]
            # category -> object shares one string object per distinct value
            for column in df.columns:
                if column not in ("lemma", "form", "ufeat"):
                    df[column] = df[column].astype("category").astype(object)
            for upos, group in df.groupby("upos"):
                part = os.path.join(
                    tmp_dir, f"{quote(str(upos), safe='')}.{chunk_idx}.pickle"
                )
                group.to_pickle(part)
                parts.setdefault(str(upos), []).append(part)
            num_rows += len(df)
            del df, chunk

        if os.path.isfile(pickle_path):
            os.remove(pickle_path)
        shutil.rmtree(split_dir, ignore_errors=True)
        os.makedirs(split_dir)
        for upos, files in parts.items():
            piece = pd.concat(
                [pd.read_pickle(f) for f in files], ignore_index=True
            ).reindex(columns=columns)
            piece.to_pickle(self._split_pickle_file(split_dir, upos))
            del piece
        shutil.rmtree(tmp_dir)
        return num_rows

    def pickle_unimorph_df(self, path: str) -> None:
        """Pickle to `path`; tables over SPLIT_PICKLE_MIN_ROWS rows are written
        as one pickle per UPOS in a sibling `<langcode>.split/` dir instead, so
        readers can load just the UPOS they need."""
        split_dir = path[: -len(".pickle")] + ".split"
        if len(self.unimorph_df) <= SPLIT_PICKLE_MIN_ROWS:
            shutil.rmtree(split_dir, ignore_errors=True)
            self.unimorph_df.to_pickle(path)
            return

        if os.path.isfile(path):
            os.remove(path)
        shutil.rmtree(split_dir, ignore_errors=True)
        os.makedirs(split_dir)
        for upos, group in self.unimorph_df.groupby("upos", observed=True):
            group.to_pickle(self._split_pickle_file(split_dir, str(upos)))

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
            # Only rows with a Mood are taken to be finite. Resolved before
            # DEFAULTS below, which needs finiteness to decide whether Mood/Voice
            # defaults apply (mood/voice are meaningless on participles etc.).
            fin_val = None
            if "VerbForm" in df.columns and "upos" in df.columns:
                fin_val = self.val2ud_um("VerbForm", "Fin")
                is_verb = df["upos"].isin(VERB_UPOS_VALUES)
                if "Mood" in df.columns:
                    is_verb = is_verb & ~pd.isna(df["Mood"])
                else:
                    is_verb = is_verb & False
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
            for column in sorted(set(DEFAULTS.keys()) & set(df.columns)):
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
                    and val not in COMPOUND_TAGS
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


    def stored_upos(self, tag: str) -> str:
        return stored_upos(tag)

    def ufeats2dict(self, ufeats: str) -> Dict[str, str]:
        return ufeats_to_dict(ufeats)

    def inflect(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]] = [],
        return_swap_feats: bool = False,
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
        lemma: Optional[str] = None,
        return_readings: bool = False,
    ) -> Union[
        Tuple[Union[None, str, List[str]], Optional[Set[str]]],
        Tuple[Union[None, str, List[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]],
    ]:
        """return_swap_feats: also return {swap_form: {feature: {values}}} from the rows that
        produced each swap form.

        swap_ufeat_override: swap column for this call instead of inflection_map[0] (per-row
        choice for ergative-split languages); also passed to ud_inflector.

        lemma: the token's lemma; the source rows are looked up by form, lemma and
        features, and by form and features alone if the lexicon has none for that lemma.

        return_readings: append the distinct readings to the result, one swap-form set
        per differing source row (more than one means the token is ambiguous).
        """
        form = form.lower()
        lemma = lemma.lower() if isinstance(lemma, str) else lemma
        prev_inflect_key = (
            form, frozenset(ud_features.items()), return_swap_feats, swap_ufeat_override,
            tuple(slot_aliases), lemma, return_readings,
        )
        if prev_inflect_key in self.prev_inflections:
            return self.prev_inflections[prev_inflect_key]

        if self.unimorph_df is None or len(self.unimorph_df) == 0:
            swap_forms = None
            feature_vals = None
            swap_feats = {}
            readings = set()
        else:
            swap_forms, feature_vals, swap_feats, readings = self.inflect_features(
                form, ud_features, strategies, return_swap_feats,
                swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
            )

            if not self.form_found(swap_forms) and self.inflect_wo_ud_features:
                swap_forms, feature_vals, swap_feats, readings = self.inflect_features(
                    form, {}, strategies, return_swap_feats,
                    swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
                )

        # the UD-derived table is always consulted and its candidates unioned in
        if self.ud_inflector is not None:
            ud_forms, ud_feature_vals, ud_swap_feats, ud_readings = self.ud_inflector.inflect(
                form, ud_features, strategies=strategies,
                return_swap_feats=True, return_readings=True,
                swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
            )
            readings = set(readings) | set(ud_readings)
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
            # sorted: set order changes with every process (string hashing)
            swap_forms = sorted(swap_forms)

        result = (
            (swap_forms, feature_vals, swap_feats)
            if return_swap_feats
            else (swap_forms, feature_vals)
        )
        if return_readings:
            result = (swap_forms, feature_vals, swap_feats, readings)
        self.prev_inflections[prev_inflect_key] = result

        return result

    def inflect_features(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategies: List[Dict[str, Optional[str]]],
        return_swap_feats: bool = False,
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
        lemma: Optional[str] = None,
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]], Set[frozenset]]:
        """Inflect form based on provided `ud_features`: first without the
        query-side DEFAULTS, so the lexicon can supply Mood/Voice/VerbForm itself,
        then with them if that did not give exactly one form."""
        swap_forms, feature_vals, swap_feats, readings = self.inflect_strategy(
            form, ud_features, return_swap_feats=return_swap_feats,
            swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
            set_defaults=False,
        )
        if not self.form_found(swap_forms):
            swap_forms, feature_vals, swap_feats, readings = self.inflect_strategy(
                form, ud_features, return_swap_feats=return_swap_feats,
                swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
            )
        if not self.form_found(swap_forms):
            for strat in strategies:
                swap_forms, feature_vals, swap_feats, readings = self.inflect_strategy(
                    form, ud_features, strategy=strat, return_swap_feats=return_swap_feats,
                    swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases, lemma=lemma,
                )
                if self.form_found(swap_forms):
                    break

        return swap_forms, feature_vals, swap_feats, readings

    @staticmethod
    def form_found(forms: Optional[List[str]]) -> bool:
        return (forms is not None) and (len(forms) == 1)

    def inflect_strategy(
        self,
        form: str,
        ud_features: Dict[str, str],
        strategy: Dict[str, Optional[str]] = {},
        return_swap_feats: bool = False,
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
        set_defaults: bool = True,
        lemma: Optional[str] = None,
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]], Set[frozenset]]:
        um_features = self.ud2um_features(ud_features, strategy, set_defaults=set_defaults,
                                          slot_aliases=slot_aliases)
        form_rows = self.form2rows(form, um_features, lemma)

        # We skip inflection if no matching rows were found
        if len(form_rows) == 0:
            return None, None, {}, set()

        forms = set()
        feature_vals = set()
        swap_feats = {}
        readings = set()

        for row_features, feature_val in self.yield_row_features(
            um_features, form_rows, strategy, swap_ufeat_override=swap_ufeat_override,
            slot_aliases=slot_aliases,
        ):
            inflected_forms, bundles = self.lemma2form(row_features, return_swap_feats)

            if inflected_forms:
                readings.add(frozenset(inflected_forms))
            forms.update(inflected_forms)
            feature_vals.add(feature_val)
            # Union per form: the same swap form can be reached via more than
            # one candidate row (e.g. different original-form matches).
            for swap_form, bundle in bundles.items():
                merged = swap_feats.setdefault(swap_form, {})
                for col, vals in bundle.items():
                    merged.setdefault(col, set()).update(vals)

        return forms, feature_vals, swap_feats, readings

    def ud2um_features(
        self, ud_features: Dict[str, str], strategy={}, set_defaults=True,
        slot_aliases: Sequence[str] = (),
    ) -> Dict[str, str]:
        um_features = {}
        columns = self.columns
        ud_features = expand_ud_feats(ud_features)

        for src_ufeat, val in ud_features.items():
            # slot-alias spellings ([dat] vs [io]) resolve to the lexicon's own column
            ufeat = self.resolve_column(src_ufeat, slot_aliases, columns)
            if ufeat not in columns or um_features.get(ufeat) is not None:
                continue
            elif ufeat == "lemma":
                um_features[ufeat] = val
            elif ufeat in strategy:
                um_features[ufeat] = strategy[ufeat]
            else:
                # UD2UM is keyed by plain feature names; the key stays the bracketed column
                um_features[ufeat] = ud_value_to_um(ufeat, val)

        if set_defaults:
            for ufeat, val in DEFAULTS.items():
                if (ufeat in self.columns) and (ufeat not in um_features):
                    um_features[ufeat] = self.val2ud_um(ufeat, val)

        if self.verbose:
            print("UM Features:", um_features)

        return um_features

    def val2ud_um(self, feat, val) -> Union[str, List[str]]:
        return to_um_value(feat, val)

    def form2rows(self, form, um_features: Dict[str, str], lemma: Optional[str] = None):
        sub_df = (self.partial_df_match(self.form_groups, form, {**um_features, "lemma": lemma})
                  if lemma else None)
        if sub_df is None or len(sub_df) == 0:
            sub_df = self.partial_df_match(self.form_groups, form, um_features)

        if len(sub_df) > 1:
            sub_df = sub_df.drop_duplicates()

        if self.verbose:
            print("Matched Rows:", sub_df)

        return sub_df

    def lemma2form(self, row_features: Dict[str, str], return_bundle: bool = False):
        lemma = row_features.pop("lemma")
        sub_df = self.partial_df_match(self.lemma_groups, lemma, row_features)
        sub_df = sub_df[sub_df.form.map(lambda f: isinstance(f, str)).astype(bool)]
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
        """Every feature column's values across `rows_df` in UD notation (um_val_to_ud), for
        the "after_<role>_<Feat>" display columns. UNDEFINED is excluded like NaN."""
        bundle = {}
        for col in rows_df.columns:
            if col in ("lemma", "form", "ufeat", "upos"):
                continue
            values = {
                um_val_to_ud(col, allval2um(val)) for val in set(rows_df[col])
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

        # masks run on cached numpy views: pandas overhead dominates on these tiny groups
        cache_key = (id(groups), group_index)
        entry = self._group_cache.get(cache_key)
        if entry is None:
            entry = (groups.get_group(group_index), {}, {})
            self._group_cache[cache_key] = entry
        sub_df, col_arrays, is_set_arrays = entry

        if len(sub_df) == 0:
            return sub_df

        def col_array(col):
            arr = col_arrays.get(col)
            if arr is None:
                arr = col_arrays[col] = sub_df[col].to_numpy(dtype=object)
            return arr

        mask = np.ones(len(sub_df), dtype=bool)

        for col, val in features.items():
            if isinstance(val, list):
                sub_mask = np.zeros_like(mask)
                arr = col_array(col)
                for subval in val:
                    sub_mask |= arr == subval
                if col != self.ufeat:  # unknown cells don't contradict, as for scalars
                    sub_mask |= (arr == UNDEFINED) | pd.isna(arr)
                mask &= sub_mask
            elif (val is not None) and (pd.notna(val)):
                arr = col_array(col)
                if val.startswith("-") and col != "lemma":
                    mask &= (arr != val[1:]) & (arr != UNDEFINED)
                elif col == self.ufeat:
                    mask &= arr == val
                else:
                    mask &= (arr == val) | (arr == UNDEFINED) | pd.isna(arr)

        candidate_rows = sub_df[mask]

        if prefer_tight_match is None:
            prefer_tight_match = self.prefer_tight_match
        if (not prefer_tight_match) or (len(candidate_rows) < 2):
            return candidate_rows

        # For each candidate row, count the columns whose cell is set (not
        # nan, or UNDEFINED) although `features` doesn't constrain it, plus
        # the columns whose cell is nan although `features` has a real value
        # for it; keep only the rows with the fewest such additions.
        if any(isinstance(v, list) for v in features.values()):
            is_set_df = candidate_rows.notna() | (candidate_rows == UNDEFINED)
            not_in_features = pd.Series(
                {col: (col not in features) for col in candidate_rows.columns}
            )
            wants_feature = pd.Series(
                {
                    col: (col in features) and _has_value(features.get(col))
                    for col in candidate_rows.columns
                }
            )
            penalty = is_set_df.mul(not_in_features, axis=1).astype(int) + (
                (~is_set_df).mul(wants_feature, axis=1).astype(int)
            )
            features_added = penalty.sum(axis=1).to_numpy()
        else:
            rows = np.flatnonzero(mask)
            features_added = np.zeros(len(rows), dtype=int)
            for col in sub_df.columns:
                is_set = is_set_arrays.get(col)
                if is_set is None:
                    arr = col_array(col)
                    is_set = is_set_arrays[col] = ~pd.isna(arr) | (arr == UNDEFINED)
                is_set = is_set[rows]
                if col not in features:
                    features_added += is_set
                elif pd.notna(features[col]):
                    features_added += ~is_set

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
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
    ):
        """Per matching row, yield the paradigm query (swap feature replaced) and the row's own
        swap value. swap_ufeat_override replaces inflection_map[0] for this call (Basque's
        erg/abs split is decided per clause, see resolve_layered_head_key)."""
        swap_ufeat, swap_map = self.inflection_map
        if swap_ufeat_override is not None:
            swap_ufeat = self.resolve_column(swap_ufeat_override, slot_aliases)

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

            # lemma2form needs a lemma; a matched row without one is unusable
            if ("lemma" not in um_features) and (
                ("lemma" not in row.keys()) or pd.isna(row["lemma"])
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
        fetch_all = False,
        slot_aliases: Sequence[str] = (),
    ) -> Set[str]:
        form = form.lower()
        # memoized like inflect(): the same combos recur constantly
        cache_key = (
            form, frozenset(features.items()), ufeat,
            only_try_ud_if_no_um, prefer_tight_match, fetch_all, tuple(slot_aliases),
        )
        if cache_key in self.prev_form_features:
            # a copy, so callers can't mutate the cached set
            return set(self.prev_form_features[cache_key])

        result = self._get_form_features_uncached(
            form, features, ufeat, only_try_ud_if_no_um, prefer_tight_match, fetch_all,
            slot_aliases,
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
        fetch_all = False,
        slot_aliases: Sequence[str] = (),
    ) -> Set[str]:
        if ufeat is not None:
            ufeat = self.resolve_column(ufeat, slot_aliases)
        um_features = self.ud2um_features(features, set_defaults=False, slot_aliases=slot_aliases)
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

        # not gated on has_unimorph_df: the UD table covers languages/POS with no UM rows
        if self.ud_inflector is not None:
            if only_try_ud_if_no_um and len(form_features) > 0:
                return form_features
            # pass ufeat on only if it is one of ud_inflector's columns (the two tables can
            # resolve different bracket conventions: Person[nom] vs Person[subj]), else its default
            ud_ufeat = None if ufeat is None else self.ud_inflector.resolve_column(ufeat, slot_aliases)
            if ud_ufeat is not None and ud_ufeat not in self.ud_inflector.columns:
                ud_ufeat = None
            ud_form_features = self.ud_inflector.get_form_features(
                form, features, ud_ufeat,
                only_try_ud_if_no_um=only_try_ud_if_no_um,
                prefer_tight_match=prefer_tight_match,
                fetch_all=fetch_all,
                slot_aliases=slot_aliases,
            )
            form_features.update(ud_form_features)

        return form_features