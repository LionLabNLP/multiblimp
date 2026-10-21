"""The UD-derived lexicon in UD notation: one row per (lemma, form, UPOS, features) seen in
the UD treebanks, with its frequency, stored as parquet (resources/ud_tables/<langcode>.parquet).

File: columns lemma, form, upos, frequency and one string column per UD feature (comma
values as written in the corpus, e.g. "Masc,Neut"; bracketed names like "Number[obj]" are
columns of their own); an empty cell means the row does not state the feature. Only
features the UniMorph<->UD mapping knows by name are stored (their values are kept as they
are); lexical labels such as InflClass or Derivation are not.

In memory (UDSlice, one per UPOS): each cell is a bit mask over that feature's values, 0 =
not stated, so a comma value is several bits and "row agrees with the token" is
`cell == 0 or cell & token_bits`. Rows are sorted by form, so a form is a contiguous
slice of the mask matrix.
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import REPO_ROOT, RESOURCE_DIR

sys.path.append(REPO_ROOT)
from resources.um2ud_annotation.UM2UD_mapper import LGSPEC_UD_FEATURES, UM2UD_values

UD_TABLE_DIR = os.path.join(RESOURCE_DIR, "ud_tables")
META_COLUMNS = ("lemma", "form", "upos", "frequency")
UNMAPPED_PREFIX = "UM:"   # value: a UniMorph code with no UD value, see multiblimp.um2ud
SHADOW_PREFIX = "UM-code:"   # feature: the original UniMorph codes of a feature whose values UD merges
MAX_VALUES = 63   # values per feature that fit one uint64 mask
STORED_FEATURES = ({feat for ud in UM2UD_values.values() for feat in ud} | LGSPEC_UD_FEATURES) - {"upos"}
# Features whose being stated or not does not make one row a closer match for a token without
# them (the UniMorph tables fill these in as UNDEFINED/BASE, which counts as stated)
UNSCORED_UNLESS_QUERIED = {"Number", "Gender", "Person", "Degree"}


def is_stored_feature(feat: str) -> bool:
    return feat.partition("[")[0] in STORED_FEATURES


def table_path(langcode: str) -> str:
    return os.path.join(UD_TABLE_DIR, f"{langcode}.parquet")


def write_ud_table(rows: list, path: str) -> int:
    """rows: (lemma, form, upos, {feature: value}, frequency). Returns the row count."""
    features = sorted({feat for _, _, _, feats, _ in rows for feat in feats})
    columns = {
        "lemma": [r[0] for r in rows], "form": [r[1] for r in rows], "upos": [r[2] for r in rows],
        "frequency": [r[4] for r in rows],
        **{feat: [r[3].get(feat) for r in rows] for feat in features},
    }
    return write_ud_dataframe(pd.DataFrame(columns), path)


def write_ud_dataframe(df: pd.DataFrame, path: str) -> int:
    """Same file from a frame with the table's columns."""
    columns = {col: df[col] if col == "frequency" or isinstance(df[col].dtype, pd.CategoricalDtype)
               else df[col].astype("category") for col in df.columns}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(columns).to_parquet(path, index=False)
    return len(df)


class UDTable:
    """All UPOS slices of one language's table, built on first use."""

    def __init__(self, df: pd.DataFrame, expand_values: bool = False):
        self._df = df
        self._slices: Dict[str, Optional["UDSlice"]] = {}
        # True for a table derived from UniMorph tags: queries to it get UD_VALUE_EXPANSIONS
        # (Tense=Imp -> Past + Aspect=Imp); a table of the treebank's own UD values is queried as is
        self.expand_values = expand_values

    @classmethod
    def load(cls, langcode: str, upos: Optional[List[str]] = None) -> Optional["UDTable"]:
        return cls.from_file(table_path(langcode), upos=upos)

    @classmethod
    def from_file(cls, path: str, expand_values: bool = False, upos: Optional[List[str]] = None) -> Optional["UDTable"]:
        """The table at `path`. With `upos`, only those rows, read through pyarrow with the categorical columns
        decoded for the kept rows alone: their dictionaries hold the whole language's forms."""
        if not os.path.isfile(path):
            return None
        if not upos:
            return cls(pd.read_parquet(path), expand_values)
        table = pq.read_table(path, filters=[("upos", "in", list(upos))])
        categorical = [name for name, column in zip(table.column_names, table.columns) if pa.types.is_dictionary(column.type)]
        table = pa.table({name: column.cast(column.type.value_type) if name in categorical else column
                          for name, column in zip(table.column_names, table.columns)})
        return cls(table.to_pandas(categories=categorical), expand_values)

    def upos_names(self) -> List[str]:
        return sorted(self._df["upos"].astype(str).unique())

    def get(self, upos: str) -> Optional["UDSlice"]:
        if upos not in self._slices:
            part = self._df[self._df["upos"] == upos]
            piece = UDSlice(part) if len(part) else None
            if piece is not None:
                piece.expand_values = self.expand_values
            self._slices[upos] = piece
        return self._slices[upos]


class UDSlice:
    def __init__(self, df: pd.DataFrame):
        self.expand_values = False   # set by the owning UDTable
        df = df.assign(form=df["form"].astype(object), lemma=df["lemma"].astype(object)).sort_values(
            "form", kind="stable").reset_index(drop=True)
        self.features: List[str] = [
            c for c in df.columns if c not in META_COLUMNS and df[c].notna().any()]
        self.values: Dict[str, List[str]] = {}
        self._bit: Dict[str, Dict[str, int]] = {}
        self.masks = np.zeros((len(df), len(self.features)), dtype=np.uint64)
        for j, feat in enumerate(self.features):
            counts = {}
            for cell, n in df[feat].value_counts().items():
                for value in str(cell).split(","):
                    counts[value] = counts.get(value, 0) + n
            names = sorted(counts, key=lambda v: (-counts[v], v))[:MAX_VALUES]
            self.values[feat] = names
            self._bit[feat] = {v: i for i, v in enumerate(names)}
            category = df[feat].astype("category")
            cell_masks = np.array([self._cell_mask(feat, c) for c in category.cat.categories], dtype=np.uint64)
            codes = category.cat.codes.to_numpy()
            self.masks[:, j] = np.where(codes >= 0, cell_masks[np.maximum(codes, 0)], 0)
        self.feature_index = {feat: j for j, feat in enumerate(self.features)}
        self.ud_features = [f for f in self.features if not f.startswith(SHADOW_PREFIX)]
        self._unscored = np.array([feat in UNSCORED_UNLESS_QUERIED for feat in self.features], dtype=bool)
        self.frequency = df["frequency"].to_numpy()
        self.lemmas = df["lemma"].to_numpy(dtype=object)
        forms = self._forms = df["form"].to_numpy(dtype=object)
        unique, first = np.unique(forms, return_index=True)
        last = np.append(first[1:], len(forms))
        self._span = {form: (int(a), int(b)) for form, a, b in zip(unique, first, last)}
        lemma_ids, lemma_names = pd.factorize(self.lemmas)
        self._lemma_order = np.argsort(lemma_ids, kind="stable")
        bounds = np.searchsorted(lemma_ids[self._lemma_order], np.arange(len(lemma_names) + 1))
        self._lemma_span = {lemma: (int(bounds[i]), int(bounds[i + 1])) for i, lemma in enumerate(lemma_names)}

    def _cell_mask(self, feat: str, cell: str) -> int:
        mask = 0
        for value in str(cell).split(","):
            bit = self._bit[feat].get(value)
            if bit is not None:
                mask |= 1 << bit
        return mask

    def form_rows(self, form: str) -> np.ndarray:
        span = self._span.get(form)
        return np.arange(*span) if span else np.empty(0, dtype=int)

    def lemma_rows(self, lemma: str) -> np.ndarray:
        span = self._lemma_span.get(lemma)
        return self._lemma_order[span[0]:span[1]] if span else np.empty(0, dtype=int)

    def value_mask(self, feat: str, value: str) -> Optional[int]:
        """Bits of the readings of `value` (comma readings allowed) that `feat` has in the table;
        None if it has none of them."""
        bits = [b for b in (self._bit[feat].get(v) for v in str(value).split(",")) if b is not None]
        return sum(1 << b for b in bits) if bits else None

    def constraints(self, query: Dict[str, str], strict: bool = False):
        """(wanted masks, asked, ignored) arrays over the features for {feature: UD value}; a feature
        this slice never has is ignored. A value the table has none of is ignored too, or with
        `strict` asked as a constraint no stated cell satisfies (only rows not stating the feature
        agree)."""
        wanted = np.zeros(len(self.features), dtype=np.uint64)
        asked = np.zeros(len(self.features), dtype=bool)
        ignored = np.zeros(len(self.features), dtype=bool)
        for feat, value in query.items():
            j = self.feature_index.get(feat)
            if j is None:
                continue
            mask = self.value_mask(feat, value)
            if mask is None and not strict:
                ignored[j] = True
            else:
                asked[j], wanted[j] = True, mask or 0
        return wanted, asked, ignored

    def agreeing(self, rows: np.ndarray, wanted, asked, exact=None, negated=None) -> np.ndarray:
        """The `rows` that agree with the constraints: an unstated cell agrees, except for the
        `exact` features (must state it) and the `negated` ones (must not state `wanted`)."""
        cells = self.masks[rows]
        ok = (cells == 0) | ((cells & wanted) != 0)
        if exact is not None:
            ok = np.where(exact, (cells & wanted) != 0, ok)
        if negated is not None:
            ok = np.where(negated, (cells & wanted) == 0, ok)
        return rows[(ok | ~asked).all(axis=1)]

    def differences(self, rows: np.ndarray, asked, ignored) -> np.ndarray:
        """Per row: features it states that the query does not ask for, plus asked features it
        does not state (Number, Gender, Person and Degree only count when asked)."""
        stated = self.masks[rows] != 0
        return np.where(asked, ~stated, stated & ~ignored & ~self._unscored).sum(axis=1)

    def form_of(self, i: int) -> str:
        return self._forms[i]

    def decode_row(self, i: int) -> Dict[str, List[str]]:
        return {feat: self.decode(feat, int(self.masks[i, j])) for j, feat in enumerate(self.features)
                if self.masks[i, j] and not feat.startswith(SHADOW_PREFIX)}

    def decode(self, feat: str, mask: int) -> List[str]:
        names = self.values[feat]
        return sorted(names[i] for i in range(len(names)) if (mask >> i) & 1)

    def decode_ud(self, feat: str, mask: int) -> List[str]:
        """decode() without the UniMorph codes that have no UD value ("UM:..." values)."""
        return [v for v in self.decode(feat, mask) if not v.startswith(UNMAPPED_PREFIX)]

    def unify(self, form: str, lemma: Optional[str], query: Dict[str, str], tight: bool = True
              ) -> Optional[Dict[str, List[str]]]:
        """{feature: sorted values} stated by the rows for `form` (and `lemma`, if given) that agree
        with `query` ({feature: UD value}); only the rows with the fewest differences from the query
        are used (`tight`), and rows that do not state a feature say nothing about it. A query
        feature the table never has a value for is ignored. None if no row matches."""
        rows = self.form_rows(form)
        if lemma is not None:
            rows = rows[self.lemmas[rows] == lemma]
        if not len(rows):
            return None
        wanted, asked, ignored = self.constraints(query)
        rows = self.agreeing(rows, wanted, asked)
        if not len(rows):
            return None
        if tight and len(rows) > 1:
            differences = self.differences(rows, asked, ignored)
            rows = rows[differences == differences.min()]
        cells = self.masks[rows]
        merged = np.bitwise_or.reduce(cells, axis=0)
        unified = {feat: self.decode_ud(feat, int(merged[j])) for j, feat in enumerate(self.features)
                   if merged[j] and not feat.startswith(SHADOW_PREFIX)}
        return {feat: values for feat, values in unified.items() if values}


def um_lookup_table(inflector):
    """The UniMorph-derived UD-notation table (multiblimp.um2ud) for annotation filling
    (expand_anno); None without one. `inflector` is anything with a langcode."""
    from .um2ud import um_table_path
    return UDTable.from_file(um_table_path(inflector.langcode), expand_values=True)


def ud_lookup_table(inflector):
    """The UD-derived table, the fallback for features the UniMorph pass could not fill. None
    without a ud_inflector/fallback or without a table file."""
    has_ud = getattr(inflector, "ud_inflector", None) is not None or getattr(inflector, "fallback", None) is not None
    return UDTable.load(inflector.langcode) if has_ud else None


# ---- lemma hygiene shared by table building and reinflection queries ----
# placeholder lemmas some treebanks use for tokens they could not lemmatize
PLACEHOLDER_LEMMAS = {"unknown", "-", "null"}


def clean_lemma(lemma, form: Optional[str] = None) -> Optional[str]:
    """The lemma in lower case, or None if it is missing or a placeholder ("_", "unknown", "-",
    "null") that is not simply the form itself."""
    if not isinstance(lemma, str) or lemma == "_":
        return None
    low = lemma.lower()
    if low in PLACEHOLDER_LEMMAS and (form is None or form.lower() != low):
        return None
    return low
