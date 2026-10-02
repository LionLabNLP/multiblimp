import os
import shutil
from typing import *
from urllib.parse import quote, unquote
from unidecode import unidecode

import numpy as np
import pandas as pd

from .inflection_maps import InflectionMap
from .languages import latin_to_cyrillic, remove_diacritics_langs, remove_multiples_langs

import sys
from functools import lru_cache
sys.path.append("../../")
from resources.um2ud_annotation.UM2UD_mapper import (
    UM2UD_values, shortened_vals as SHORTENED_UM_VALS, map_um_value_to_ud,
    fix_typos as FIX_TYPOS, val2feat as PKG_VAL2FEAT, feat2val as PKG_FEAT2VAL,
    feature_for, um_tag_upos,
)
from resources.um2ud_annotation.UD2UM_mapper import UD2UM_values


# Tables above this many rows are pickled per UPOS (see pickle_unimorph_df).
SPLIT_PICKLE_MIN_ROWS = 5_000_000


class PosLookup(dict):
    """{upos: DataFrame} whose slices may be loaded on first .get() (a split
    pickle) instead of up front."""

    def __init__(self, loader=None, names=()):
        super().__init__()
        self._loader = loader
        self._names = frozenset(names)

    def get(self, key, default=None):
        if key not in self and key in self._names:
            self[key] = self._loader(key)
        return super().get(key, default)


@lru_cache(maxsize=None)
def _cached_map_um_value_to_ud(val: str):
    """lru_cache wrapper around the vendored map_um_value_to_ud (resources/
    um2ud_annotation): its ARG*/PSS* handling does an O(vocabulary) regex
    scan per call (parse_complex_um_value's nested loop over UM2UD_values/
    shortened_vals), and a single UM lexicon (e.g. Georgian's real UniMorph
    file, ~92k rows) repeats the same handful of ARG-tag strings across
    nearly every row -- caching by the literal input string turns that into
    a few hundred unique evaluations instead of tens of thousands. Safe to
    cache unbounded: the domain is UM tag strings, a small closed vocabulary
    per language, not user data.
    """
    return map_um_value_to_ud(val)


# multiblimp.ud2um's own encoding for a UD layered/argument-marking feature
# (e.g. Number[obj]=Plur) that survives into a UM tag string as a single
# component: "<um_code><LAYERED_FEAT_SEP><suffix>" (e.g. "PL$obj"). Decoded
# back into a "{feat}[{suffix}]" entry by UnimorphInflector.ufeats2dict
# below. Not a real UniMorph tag syntax -- genuine UniMorph argument-marking/
# possessor data uses ARG*/PSS* tags instead (see ufeats2dict's other
# branch, which routes those through the vendored UM2UD_mapper). "$" doesn't
# collide with anything in real UM tag vocabulary (uppercase alnum, plus the
# ";+/.,{}" this module's own preprocessing already treats specially).
LAYERED_FEAT_SEP = "$"

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
    # Mirrors process_treebank.expand_anno's own "if no VerbForm, assume
    # Fin" rule for the source token's feature completion -- without this,
    # a language whose treebank rarely annotates VerbForm at all (e.g.
    # Korean: unset on ~99% of VERB tokens) loses that constraint here
    # specifically for swap-target generation, letting the upos/VERB ->
    # "V" lookup (deliberately left ambiguous between finite and every
    # non-finite reading, see UD2UM_mapper.UD2UM_values) go unconstrained
    # and pull in participle/converb/masdar forms as equally-valid
    # candidates for what should be a plain finite swap.
    "VerbForm": "FIN",
    # An adjective with no Degree feature at all in UD is positive degree
    # by convention (Pos is UD's own default when the feature is left
    # unannotated) -- resolves to BASE via the ("Degree", "Pos") entry
    # added to UD2UM above, matching how unmarked_features fills the same
    # BASE value into the lexicon for Degree-unmarked UM entries.
    "Degree": "Pos",
}
# raw UM upos codes (not UD tags) for the two verbal POS
VERB_UPOS_VALUES = {"V", "AUX"}

# Every raw UM upos code this pipeline recognizes as "this token IS the
# part of speech", checked before feature_for()'s generic value->feature
# lookup in ufeats2dict. Necessary because several of these codes are
# ALSO legitimate values of other features in the vendored UM2UD_mapper's
# own tables (e.g. "DET"/"ART"/"PRO" appear there as PronType values,
# "PROPN" as a NounType value, "NUM" as a NumType value) -- feature_for()
# has no notion of tag position, so without this an upos token silently
# gets classified as one of those other features instead of "upos",
# leaving the upos column empty for every entry tagged with it. Restored
# from the pre-refactor val2feat table (deleted unimorph_features.py's own
# "upos" bucket), which correctly special-cased these before this
# function was rewritten to route through feature_for().
UPOS_VALUES = {
    "N", "PROPN", "ADJ", "PRO", "CLF", "ART", "DET", "V", "ADV", "AUX",
    "ADP", "COMP", "CONJ", "NUM", "PART", "INTJ", "AJD", "PRE", "ADJ.CVB",
    "PRON",
}

# UM2UD[tag] is the tag's full {UD_feature: UD_value} dict, e.g.
# UM2UD["V.PTCP"] == {"upos": "VERB", "VerbForm": "Part"}.
UM2UD = UM2UD_values
# (feature, value) -> UM tag, e.g. UD2UM[("Number", "Plur")] == "PL". Built
# in resources.um2ud_annotation.UD2UM_mapper (the reverse index of
# UM2UD_values above) -- re-exported under this name here since every
# existing caller in this codebase imports UD2UM from multiblimp.unimorph.
UD2UM = {
    **UD2UM_values,
    # UniMorph has no tag for plain/positive adjective degree -- it's the
    # unmarked form (see unmarked_features' Degree entry below), so the
    # vendored reverse index has no ("Degree", "Pos") pair at all. Without
    # this, ud_value_to_um/val2ud_um fall through to their generic value
    # lookups for "Pos" and resolve it as Polarity's POS tag instead (the
    # only real UM tag spelled "POS"), so a UD Degree=Pos adjective's
    # candidate search required a nonexistent "POS" Degree value and never
    # matched the BASE-tagged lexicon rows it should.
    ("Degree", "Pos"): BASE,
}



def ud_value_to_um(feat: str, val):
    """UD value -> UM code for `feat` (bracket suffix ignored). A comma value
    ("Masc,Neut": the form is ambiguous) becomes the list of its readings'
    codes, so a lexicon lookup retrieves each reading; None if any reading
    has no UM code (the constraint is then dropped, as for any unmappable
    value). A single reading's own code can itself be a "/"-joined
    disjunction (UD2UM_mapper.UD2UM_values' own convention for a UD value
    that several distinct UM tags collapse onto, e.g. Aspect/Perf ->
    "PFV/PRF") -- expanded into the same kind of list, for the same reason:
    a lexicon lookup by exact form+lemma concretizes to whichever one the
    real entry is, so offering every candidate here only widens that
    search rather than guessing."""
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
    # real UM tags now, see multiblimp.ud2um), so just normalize casing.
    return val.upper()


def um_val_to_ud(feat: str, val: str) -> str:
    """A UM-format value (e.g. "PL", "NOM") for UD feature `feat` (e.g.
    "Number", "Case") -> its UD-format value ("Plur", "Nom"). Same
    (SHORTENED_UM_VALS-preferred, falling back to UM2UD) lookup precedence
    resources.um2ud_annotation.UM2UD_mapper.parse_ufeat_value itself uses,
    since a handful of short codes (the 1-2 letter argument-marking/
    possession forms, e.g. "F"/"M"/"P"/"S") only exist in SHORTENED_UM_VALS,
    not UM2UD_values.

    Returns `val` unchanged if neither table has an entry for it under
    `feat` specifically -- e.g. `val` is UNDEFINED, already UD-format, or a
    tag whose UD2UM_values dict doesn't happen to set this particular
    feature (a single UM tag can carry several UD features at once, e.g.
    "V.PTCP" -> {"upos": "VERB", "VerbForm": "Part"}; only look up the one
    `feat` names).

    `feat` may be a bracketed column name (e.g. "Number[abs]", as
    _rows_to_bundle passes straight from a wide-format lexicon dataframe's
    own columns) -- UM2UD/SHORTENED_UM_VALS entries are always keyed by the
    plain feature name only (a tag never sets "Number[abs]" specifically,
    just "Number"), so the bracket is stripped before the lookup. Without
    this, a bracketed column's value silently never converted at all --
    "PL" stayed "PL" instead of becoming "Plur" -- while the exact same
    value under a plain column converted correctly, a display
    inconsistency between the two that's exactly backwards from the point
    of this function.
    """
    base_feat = feat.partition("[")[0]
    source = SHORTENED_UM_VALS if val in SHORTENED_UM_VALS else UM2UD
    return source.get(val, {}).get(base_feat, val)


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
        morphological features that are computed using `ud2um.py`.
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
        self.resource_dir = resource_dir or "."

        # ud_unimorph/ now stores real UM tag syntax too (multiblimp.ud2um),
        # so use_ud_inflections only changes which file gets read, not the vocabulary.
        # feat2val/val2feat come straight from the vendored package -- the
        # single source of truth for what a bare UM tag/value means -- and
        # are never mutated in place anywhere in this class, so sharing its
        # module-level dicts directly (instead of copying per instance) is
        # safe.
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
            # update_inflection_map() above resolved self.inflection_map against
            # self.unimorph_df alone -- for a list-valued ufeat (e.g. Number vs.
            # Number[subj]), an empty/POS-filtered-to-nothing self.unimorph_df
            # (e.g. a language with zero UniMorph-proper verb entries) leaves it
            # unresolved (None), even when self.ud_inflector's own copy, run
            # against the UD-derived table, resolved it fine. Callers (e.g.
            # sva_trees.create_pairs) read ufeat off this outer instance, so an
            # unresolved None here silently breaks every get_form_features call
            # for the whole language: the original (pre-swap) value never gets
            # stripped from the match constraints, so the lookup for the
            # swapped form (which necessarily differs) matches nothing.
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
        named and case-named, see word_order.process_treebank.slot_suffixes)
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

        row_ufeats = [self.ufeats2dict(ufeat) for ufeat in df.ufeat]

        # self.feat2val's fixed plain-feature vocabulary won't have entries
        # for the "{feat}[{suffix}]" argument-marking keys ufeats2dict can
        # also produce (from real UM ARG*/PSS* tags or multiblimp.ud2um's
        # LAYERED_FEAT_SEP encoding, e.g. "Number[obj]") -- discovered from
        # the rows themselves instead, so those columns aren't silently
        # dropped here.
        bracketed_cols = {k for d in row_ufeats for k in d if "[" in k}
        ufeat_cols = {x: [] for x in set(self.feat2val) | bracketed_cols}

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

        # Plain object dtype, not "category": partial_df_match's equality
        # lookups compare these columns against literal strings on every
        # inflection attempt, and category dtype pays a conversion cost on
        # each such comparison (profiled: ~50s of a 94s create_pairs run).
        for column in df.columns:
            df[column] = df[column].astype(object)

        return df

    def _finalize_pickle_df(self, df, filter: Dict[str, List[str]]):
        df = self.filter_entries(df, filter)

        # Convert to plain object dtype (same reason as load_unimorph above)
        for column in df.columns:
            df[column] = df[column].astype(object)

        return self.set_unk_values(df)

    def _split_pickle_dir(self, pickle_path: str) -> str:
        return os.path.join(self.resource_dir, pickle_path, f"{self.langcode}.split")

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
        path = os.path.join(self.resource_dir, pickle_path, f"{self.langcode}.pickle")
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
        """{upos: DataFrame slice} lookup over a pickled table. A split pickle
        (see pickle_unimorph_df) is read one UPOS at a time, on first use."""
        split_dir = self._split_pickle_dir(pickle_path)
        names = self._split_pickle_upos(split_dir)
        if len(names) > 0 and not os.path.isfile(
            os.path.join(self.resource_dir, pickle_path, f"{self.langcode}.pickle")
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
            self.resource_dir, f"unimorph/{self.langcode}/{self.langcode}"
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


    def stored_upos(self, tag: str) -> str:
        """The value stored in the upos column for a raw UM upos `tag`: its
        UD upos (um_tag_upos) run through val2ud_um, the same translation
        filter_entries applies to a requested upos -- so both sides agree
        (e.g. ART -> DET, PRO -> PRON, PRE -> ADP, COMP -> CONJ, CLF -> N)."""
        cache = self.__dict__.setdefault("_stored_upos_cache", {})
        if tag not in cache:
            ud_upos = um_tag_upos(tag)
            cache[tag] = tag if ud_upos is None else self.val2ud_um("upos", ud_upos)
        return cache[tag]

    def ufeats2dict(self, ufeats: str) -> Dict[str, str]: # replace with um2ud_mapper
        """Translates the unimorph X;Y;Z format to a dictionary.

        Every value is first normalized through the vendored package's own
        fix_typos table (resources/um2ud_annotation/UM2UD_mapper.py) -- the
        same table map_um_value_to_ud already applies for ARG*/PSS* tags,
        but until this normalization was added here too, a plain typo'd tag
        (e.g. Irish "MASV", a documented alias for "MASC") reached the val2feat
        lookup below unfixed and silently landed in an _EXTRA_FEAT2VAL patch
        entry for the misspelling instead of the correct value -- or, for a
        typo'd ARG*-prefixed tag ("ARBAB1S" for "ARGAB1S"), didn't even reach
        the ARG*/PSS* branch below, since it fails that literal prefix check.

        Two argument-marking encodings are handled ahead of the plain-tag
        lookup below (both produce "{feat}[{suffix}]" keys, e.g.
        "Number[obj]"):
          - LAYERED_FEAT_SEP-marked tags ("PL$obj"): multiblimp.ud2um's own
            round-trip encoding for a UD layered feature that survived into
            the UD-derived fallback lexicon (see LAYERED_FEAT_SEP's
            docstring).
          - Real UniMorph ARG*/PSS* tags ("ARGABS1", "PSS3S", ...): genuine
            argument-marking/possessor UM tags, e.g. from Basque/Georgian UM
            paradigm data. This module's own val2feat lookup below only
            understands plain (non-compound) UM tags, so these are routed
            through the vendored UM2UD_mapper (resources/um2ud_annotation)
            instead, which already decomposes them correctly -- then
            translated back to this module's raw-UM-code convention (val2feat's
            values, e.g. "PL"/"3", not UD-format "Plur"/"3") via UD2UM, so
            bracketed and plain columns stay directly comparable.
        """
        ufeats = (
            str(ufeats)
            .replace("/", "+")
            .replace(",", "+")
            .replace("{", "")
            .replace("}", "")
            .replace("V.PTCP", "V;V.PTCP")
            .replace("V.PCTP", "V;V.PTCP")
            .replace(":", ";")
        )
        ufeats = ufeats.strip()
        ufeat_vals = ufeats.split(";")
        ufeat_dict = {}

        for val in ufeat_vals:
            if len(val) == 0:
                continue
            val = FIX_TYPOS.get(val, val)
            if val in UPOS_VALUES:
                ufeat_dict["upos"] = self.stored_upos(val)
                continue
            elif LAYERED_FEAT_SEP in val:
                code, _, suffix = val.partition(LAYERED_FEAT_SEP)
                base_feat = self.val2feat.get(code) or self.val2feat.get(code.upper())
                if base_feat is None:
                    continue
                key = f"{base_feat}[{suffix}]"
                ufeat_dict[key] = f"{ufeat_dict[key]}+{code}" if key in ufeat_dict else code
                continue
            elif val.startswith("ARG") or val.startswith("PSS"):
                for bracketed_feat, ud_val in _cached_map_um_value_to_ud(val)["morpho"].items():
                    base_feat, _, suffix = bracketed_feat.partition("[")
                    suffix = suffix.rstrip("]")
                    um_code = UD2UM.get((base_feat, ud_val), ud_val)
                    key = f"{base_feat}[{suffix}]" if suffix else base_feat
                    ufeat_dict[key] = f"{ufeat_dict[key]}+{um_code}" if key in ufeat_dict else um_code
                continue
            elif (resolved := feature_for(val)) is not None:
                # feature_for (resources.um2ud_annotation) is the single
                # authoritative "what feature bucket does this UM value
                # belong to" lookup -- covers a real UD mapping (val2feat),
                # or a value the package knows is real but has deliberately
                # chosen not to map (LGSPEC*, blacklist, unk_values), so
                # e.g. Basque's HYP mood stays distinguishable from
                # genuinely-unrecognized input instead of collapsing into
                # "UNK". No local duplicate of that decision here.
                ufeat = resolved
            elif (resolved := feature_for(val.strip())) is not None:
                ufeat = resolved  # Livvi
            elif "." in val:
                val1, val2 = val.split(".")
                ufeat1 = feature_for(val1) or "UNK"
                ufeat2 = feature_for(val2) or "UNK"
                ufeat_dict[ufeat1] = val1
                ufeat_dict[ufeat2] = val2
                continue
            elif "+" in val:
                subval = val.split("+")[0]
                ufeat = feature_for(subval) or "UNK"
            elif (resolved := feature_for(val.upper())) is not None:
                ufeat = resolved
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
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
    ) -> Union[
        Tuple[Union[None, str, List[str]], Optional[Set[str]]],
        Tuple[Union[None, str, List[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]],
    ]:
        """return_swap_feats: also return {swap_form: {feature: {values}}},
        e.g. for flowchart display -- captured directly from the rows that
        produced each swap_form, instead of a separate get_form_feature_bundle
        lookup by result form. Default False keeps the old 2-tuple return
        shape so existing callers (e.g. multiblimp.pipeline) don't break.

        swap_ufeat_override: use this column instead of self.inflection_map[0]
        for this call only -- see yield_row_features' docstring for why
        (ergative-split languages need this resolved per row, not once per
        language). Passed through to self.ud_inflector.inflect() too, so the
        UD-derived fallback lexicon honors the same per-row choice.
        """
        prev_inflect_key = (
            form, frozenset(ud_features.items()), return_swap_feats, swap_ufeat_override,
            tuple(slot_aliases),
        )
        if prev_inflect_key in self.prev_inflections:
            return self.prev_inflections[prev_inflect_key]

        if self.unimorph_df is None or len(self.unimorph_df) == 0:
            swap_forms = None
            feature_vals = None
            swap_feats = {}
        else:
            swap_forms, feature_vals, swap_feats = self.inflect_features(
                form, ud_features, strategies, return_swap_feats,
                swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases,
            )

            if not self.form_found(swap_forms) and self.inflect_wo_ud_features:
                swap_forms, feature_vals, swap_feats = self.inflect_features(
                    form, {}, strategies, return_swap_feats,
                    swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases,
                )

        # Always consult the UD-derived fallback and merge its candidates in,
        # even when the primary (real-UM) lexicon already found exactly one
        # match: each lexicon resolves its own inflection_map key
        # independently (e.g. primary "Person[nom]" vs. fallback
        # "Person[subj]" for Georgian -- see swap_features.py), and a single
        # primary match can still be an ambiguous/syncretic paradigm form
        # (lands in create_pairs.process_item's "undefined_features" bucket)
        # where the fallback's corpus-annotated form would have been usable.
        # create_pairs already classifies every candidate form independently
        # (one process_item call per swap_form), so widening the candidate
        # set here can only add opportunities for "correct_swaps", never
        # remove one -- unlike the old exactly-one-match short circuit, which
        # could silently starve a row of its only usable candidate.
        if self.ud_inflector is not None:
            ud_result = self.ud_inflector.inflect(
                form, ud_features, strategies=strategies,
                return_swap_feats=return_swap_feats,
                swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases,
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
            # sorted: set order changes with every process (string hashing)
            swap_forms = sorted(swap_forms)

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
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]]:
        """Inflect form based on provided `ud_features`."""
        swap_forms, feature_vals, swap_feats = self.inflect_strategy(
            form, ud_features, return_swap_feats=return_swap_feats,
            swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases,
        )
        if not self.form_found(swap_forms):
            for strat in strategies:
                swap_forms, feature_vals, swap_feats = self.inflect_strategy(
                    form, ud_features, strategy=strat, return_swap_feats=return_swap_feats,
                    swap_ufeat_override=swap_ufeat_override, slot_aliases=slot_aliases,
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
        swap_ufeat_override: Optional[str] = None,
        slot_aliases: Sequence[str] = (),
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]], Dict[str, Dict[str, Set[str]]]]:
        um_features = self.ud2um_features(ud_features, strategy, slot_aliases=slot_aliases)
        form_rows = self.form2rows(form, um_features)

        # We skip inflection if no matching rows were found
        if len(form_rows) == 0:
            return None, None, {}

        forms = set()
        feature_vals = set()
        swap_feats = {}

        for row_features, feature_val in self.yield_row_features(
            um_features, form_rows, strategy, swap_ufeat_override=swap_ufeat_override,
            slot_aliases=slot_aliases,
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
        self, ud_features: Dict[str, str], strategy={}, set_defaults=True,
        slot_aliases: Sequence[str] = (),
    ) -> Dict[str, str]:
        um_features = {}
        columns = self.columns

        for src_ufeat, val in ud_features.items():
            # a slot-alias spelling ([dat] vs this lexicon's [io]) is matched
            # against the lexicon's own column, first spelling seen wins
            ufeat = self.resolve_column(src_ufeat, slot_aliases, columns)
            if ufeat not in columns or um_features.get(ufeat) is not None:
                continue
            elif ufeat == "lemma":
                um_features[ufeat] = val
            elif ufeat in strategy:
                um_features[ufeat] = strategy[ufeat]
            else:
                # UD2UM is keyed by plain feature names only (a UM tag never
                # sets "Number[abs]" specifically, just "Number" -- see
                # UD2UM_mapper.ud_feats_to_um_tags' own docstring), so a
                # bracketed ufeat (e.g. "Number[obj]") always missed here
                # and fell back to None -- silently excluding it from
                # candidate matching below (partial_df_match skips any
                # None-valued constraint entirely), for every bracketed
                # feature indiscriminately, not just the one actually being
                # swapped. Keyed under the original (possibly bracketed)
                # `ufeat` still, so partial_df_match still matches it
                # against the right column -- only the value lookup itself
                # needs the base name.
                um_features[ufeat] = ud_value_to_um(ufeat, val)

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
        # val can be a list here (see the isinstance(val, list) branch
        # below) -- FIX_TYPOS.get would try to hash it and crash.
        if isinstance(val, str):
            val = FIX_TYPOS.get(val, val)
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
        elif feature_for(val) is not None:
            # Covers both a real val2feat entry and a value the package
            # knows is real but deliberately unmapped (blacklist/
            # unk_values/LGSPEC*) -- see ufeats2dict's own use of
            # feature_for for why both matter here.
            return val
        elif (feat, val) in UD2UM:
            # Must come before val.upper() below: e.g. "PART" is also a valid
            # val2feat entry (upos Particle), which would misresolve VerbForm=Part.
            return UD2UM[(feat, val)]
        elif val in UM2UD and feat in UM2UD[val]:
            # val is already a UM tag carrying a value for this feature.
            return self.val2ud_um(feat, UM2UD[val][feat])
        elif feature_for(val.upper()) is not None:
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
        """Every UM feature column's value(s) across `rows_df`, converted to
        UD-format values (um_val_to_ud) -- e.g. for displaying a reinflected
        form's other features (flowchart's "after_<role>_<Feat>" columns,
        see sva_trees.create_pairs.create_pairs/npa.agreement.
        create_npa_pairs), which should read like the rest of a row's
        feature columns (all UD-format, e.g. "Plur"/"Nom") rather than
        switching to UM's own short codes ("PL"/"NOM") just for this one
        display. Column names are already UD feature names (Case, Gender,
        Number, ...) regardless -- only the values were ever UM-coded, see
        um_val_to_ud. UNDEFINED means "no info", same as NaN -- excluded."""
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

        # Every call slices a tiny per-form/per-lemma group, where pandas'
        # fixed per-operation overhead dominates, so the masks below run on
        # cached numpy views of the group's columns instead of Series.
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
                if val.startswith("-"):
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
        """
        Based on all matching rows, set and yield the (swapped) features
        we use for finding the inflected form.

        swap_ufeat_override: use this column instead of self.inflection_map[0]
        (the language-wide default resolved once in update_inflection_map).
        Needed for languages whose argument-marking bracket depends on the
        specific row being processed, not just the language -- e.g. Basque's
        ergative-absolutive split, where whether the verb's SUBJECT marking
        lives under Number[erg] or Number[abs] depends on that clause's own
        transitivity, not a language-wide constant. See
        word_order.process_treebank.resolve_layered_head_key, which callers
        (sva_trees.create_pairs) use to compute this per row.
        """
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

            # lemma2form() always needs a "lemma" key in row_features (it
            # pops it to look up lemma_groups) -- for closed-class POS
            # tables sourced from the UD-derived fallback (e.g. DET, whose
            # lemma annotation in UD is sparser/less consistent than for
            # open-class POS like verbs), a matched row's own "lemma" can
            # itself be missing. Previously that silently produced a
            # row_features dict with no "lemma" key at all (since the loop
            # below skips any NaN value, "lemma" included), and lemma2form's
            # row_features.pop("lemma") raised a bare KeyError instead of
            # this row just being skipped like any other unusable match.
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
        # Same form/features/ufeat combos recur constantly (common subjects, child
        # forms, agreement targets) and each uncached call does a full UniMorph
        # lookup — including a recursive self.ud_inflector.get_form_features call —
        # so this is memoized the same way inflect() already is.
        cache_key = (
            form, frozenset(features.items()), ufeat,
            only_try_ud_if_no_um, prefer_tight_match, fetch_all, tuple(slot_aliases),
        )
        if cache_key in self.prev_form_features:
            # Return a copy — callers must not mutate the cached set in place.
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

        # UD fallback: not gated behind has_unimorph_df, since it's exactly
        # meant to cover languages/POS for which the UM dataframe is empty
        # (e.g. Ancient Greek has zero UM verb entries) -- mirrors inflect()'s
        # unconditional fallback to self.ud_inflector.inflect().
        if self.ud_inflector is not None:
            if only_try_ud_if_no_um and len(form_features) > 0:
                return form_features
            # A caller-supplied ufeat (e.g. create_pairs.process_item passing
            # this instance's own resolved key, "Person[nom]" for Georgian)
            # is only meaningful against this instance's own columns. Forcing
            # it onto self.ud_inflector too silently breaks the readback
            # whenever the two lexicons resolved to different bracket
            # conventions (ud_inflector's own key is "Person[subj]" here) --
            # the column just doesn't exist there, so the lookup finds
            # nothing even though ud_inflector's own default key would have
            # read the value cleanly. Only pass it through when it's actually
            # one of ud_inflector's own columns; otherwise let it fall back
            # to its own resolved key (ufeat=None -> self.ufeat below).
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