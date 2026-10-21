"""Reinflection on UD-notation tables (multiblimp.ud_table): features and values are UD's (Number=Sing,
not SG) and a row is a bit-mask vector, so matching a lemma's paradigm is a few array operations.

inflect(form, ud_features):
  1. rows for `form` that agree with the token's features (tightest only);
  2. per row: its own stated features, with the swap feature replaced by the swap map's target
     (or "any other value" when the map is None), are looked up among the rows of the same lemma
     and UPOS; the matching rows' forms are the swapped forms.
The first try leaves out the Mood/Voice/VerbForm/Degree defaults, the second adds them if that did
not give exactly one form. The fallback inflector (the UD-derived table) adds its forms.
"""
import copy
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .ud_table import UNMAPPED_PREFIX, UDSlice, UDTable   # first: puts the repo root on sys.path
from resources.um2ud_annotation.UD2UM_mapper import UD2UM, expand_ud_feats, widen_ud_value
from resources.um2ud_annotation.UM2UD_mapper import um_tag_upos
from .um2ud import um_table_path

def _shown(value: str) -> str:
    """A UniMorph code with no UD value is shown as the code itself."""
    return value[len(UNMAPPED_PREFIX):] if value.startswith(UNMAPPED_PREFIX) else value


DEFAULTS = {"Mood": "Ind", "Voice": "Act", "VerbForm": "Fin", "Degree": "Pos"}
VERBAL_UPOS = {"VERB", "AUX"}


def _assume_finite(piece: UDSlice) -> UDSlice:
    """A copy of a verbal slice whose rows with a Mood but no VerbForm are finite."""
    mood, form = piece.feature_index.get("Mood"), piece.feature_index.get("VerbForm")
    fin = piece._bit.get("VerbForm", {}).get("Fin")
    if mood is None or form is None or fin is None:
        return piece
    rows = (piece.masks[:, mood] != 0) & (piece.masks[:, form] == 0)
    if not rows.any():
        return piece
    adjusted = copy.copy(piece)
    adjusted.masks = piece.masks.copy()
    adjusted.masks[rows, form] = np.uint64(1 << fin)
    return adjusted


class UDInflector:
    def __init__(self, table: Optional[UDTable], inflection_map, fallback: Optional["UDInflector"] = None,
                 upos: Optional[Sequence[str]] = None, inflect_wo_ud_features: bool = False,
                 prefer_tight_match: bool = True, remove_multiword_forms: bool = False,
                 expand_values: bool = True):
        self.fallback = fallback
        self.inflect_wo_ud_features = inflect_wo_ud_features
        self.prefer_tight_match = prefer_tight_match
        self.remove_multiword_forms = remove_multiword_forms
        # UD_VALUE_EXPANSIONS bridge (Tense=Imp -> Past + Aspect=Imp) on the query: for a table derived
        # from UniMorph tags; off for the treebank's own UD values
        self.expand_values = expand_values
        self._multiword: Dict[int, np.ndarray] = {}
        self.slices: List[UDSlice] = []
        self.upos_of: Dict[int, str] = {}
        for name in (table.upos_names() if table is not None else []):
            if upos is not None and name not in upos:
                continue
            piece = table.get(name)
            if piece is not None:
                if name in VERBAL_UPOS:
                    piece = _assume_finite(piece)
                self._multiword[id(piece)] = np.array([" " in form for form in piece._forms], dtype=bool)
                self.slices.append(piece)
                self.upos_of[id(piece)] = name
        self.inflection_map = self._resolve_map(inflection_map)
        if self.inflection_map[0] is None and fallback is not None and fallback.inflection_map[0] is not None:
            self.inflection_map = (fallback.inflection_map[0], self.inflection_map[1])   # callers read ufeat off the outer one
        self._cache: dict = {}
        self._vocabulary: dict = {}
        self.dropped_values: Counter = Counter()   # query values with no UniMorph counterpart in the mapping

    # ---- columns

    @property
    def columns(self) -> Set[str]:
        return {feat for piece in self.slices for feat in piece.features}

    @property
    def ufeat(self):
        return self.inflection_map[0]

    def resolve_column(self, key: str, slot_aliases: Sequence[str] = (), columns: Optional[Set[str]] = None) -> str:
        columns = self.columns if columns is None else columns
        base, bracket, suffix = key.partition("[")
        if key in columns or not bracket or suffix.rstrip("]") not in slot_aliases:
            return key
        return next((f"{base}[{s}]" for s in slot_aliases if f"{base}[{s}]" in columns), key)

    def _resolve_map(self, inflection_map):
        """A list of candidate features is resolved to the one with the most evenly attested values.
        Bracketed candidates (Number[abs]) go first; the plain one (Number) only if none qualifies,
        since in a language with argument-marked agreement the plain feature belongs to nominal forms."""
        feat, swap_map = inflection_map
        if not isinstance(feat, list):
            return feat, swap_map
        for group in ([c for c in feat if "[" in c], [c for c in feat if "[" not in c]):
            best, best_min = None, 0
            for candidate in group:
                per_value: Counter = Counter()   # pooled over the slices
                for piece in self.slices:
                    j = piece.feature_index.get(candidate)
                    if j is None:
                        continue
                    column = piece.masks[:, j]
                    for i, value in enumerate(piece.values[candidate]):
                        per_value[value] += int(((column >> np.uint64(i)) & np.uint64(1)).sum())
                counts = [c for c in per_value.values() if c]
                if len(counts) > 1 and min(counts) > best_min:
                    best, best_min = candidate, min(counts)
            if best is not None:
                return best, swap_map
        return None, swap_map

    @property
    def can_feature_swap(self) -> bool:
        return isinstance(self.ufeat, str) or (self.fallback is not None and self.fallback.can_feature_swap)

    def __len__(self) -> int:
        return sum(len(piece.masks) for piece in self.slices) + (len(self.fallback) if self.fallback else 0)

    # ---- vocabulary (coverage statistics)

    def _own_strings(self, attr: str) -> Set[str]:
        out: Set[str] = set()
        for piece in self.slices:
            values = piece._forms if attr == "forms" else piece.lemmas
            if self.remove_multiword_forms:
                values = values[~self._multiword[id(piece)]]
            out.update(v for v in values if isinstance(v, str))
        return out

    @property
    def own_forms(self) -> Set[str]:
        """The forms of this inflector's own table, without the fallback's."""
        if "forms" not in self._vocabulary:
            self._vocabulary["forms"] = self._own_strings("forms")
        return self._vocabulary["forms"]

    @property
    def own_lemmas(self) -> Set[str]:
        """The lemmas of this inflector's own table, without the fallback's."""
        if "lemmas" not in self._vocabulary:
            self._vocabulary["lemmas"] = self._own_strings("lemmas")
        return self._vocabulary["lemmas"]

    @property
    def unique_forms(self) -> Set[str]:
        return self.own_forms | (self.fallback.unique_forms if self.fallback else set())

    @property
    def unique_lemmas(self) -> Set[str]:
        return self.own_lemmas | (self.fallback.unique_lemmas if self.fallback else set())

    @property
    def num_forms(self) -> int:
        return len(self.unique_forms)

    @property
    def num_lemmas(self) -> int:
        return len(self.unique_lemmas)

    # ---- query

    def _single_word(self, piece: UDSlice, rows: np.ndarray) -> np.ndarray:
        return rows[~self._multiword[id(piece)][rows]] if self.remove_multiword_forms else rows

    def _has_value(self, feat: str, value: str) -> bool:
        return any(piece.value_mask(feat, value) is not None for piece in self.slices if feat in piece.feature_index)

    def _query(self, ud_features: Dict[str, str], slot_aliases, set_defaults: bool) -> Dict[str, str]:
        columns = self.columns
        query: Dict[str, str] = {}
        for source, value in (expand_ud_feats(ud_features) if self.expand_values else ud_features).items():
            feat = self.resolve_column(source, slot_aliases, columns)
            if feat in columns and query.get(feat) is None and isinstance(value, str):
                base = feat.partition("[")[0]
                # a UniMorph-derived table holds only values with a UniMorph code (or ones rewritten into it,
                # Tense=Pqp), so a query for another value would match nothing: it is dropped. A UD-derived
                # table holds what the treebank attests, so an unattested value rightly matches nothing.
                if self.expand_values and not any(
                        (base, v) in UD2UM or self._has_value(feat, v) for v in value.split(",")):
                    self.dropped_values[f"{feat}={value}"] += 1
                    continue
                query[feat] = widen_ud_value(feat, value)
        if set_defaults:
            for feat, value in DEFAULTS.items():
                if feat in columns and feat not in query:
                    query[feat] = value
        return query

    def _form_rows(self, form: str, query: Dict[str, str], lemma: Optional[str] = None, upos: Optional[str] = None):
        """[(slice, row indices)] of the tightest rows for `form` and `lemma` that agree with `query`;
        the lemma is ignored only if the token has none or this table holds no lemmas. `upos` limits the
        search to that part of speech."""
        use_lemma = bool(lemma) and bool(self.own_lemmas)
        found = self._agreeing_rows(form, query, lemma if use_lemma else None, upos)
        if not found:
            return []
        if not self.prefer_tight_match:
            return [(piece, rows) for piece, rows, _ in found]
        best = min(int(d.min()) for _, _, d in found)
        return [(piece, rows[d == best]) for piece, rows, d in found if (d == best).any()]

    def _agreeing_rows(self, form: str, query: Dict[str, str], lemma: Optional[str], upos: Optional[str] = None):
        found = []
        for piece in self.slices:
            if upos is not None and self.upos_of[id(piece)] != upos:
                continue
            rows = piece.form_rows(form)
            if lemma is not None:
                rows = rows[piece.lemmas[rows] == lemma]
            rows = self._single_word(piece, rows)
            if not len(rows):
                continue
            wanted, asked, ignored = piece.constraints(query, strict=True)
            rows = piece.agreeing(rows, wanted, asked)
            if len(rows):
                found.append((piece, rows, piece.differences(rows, asked, ignored)))
        return found

    # ---- reinflection

    def _inflect_once(self, form, ud_features, swap_ufeat_override, slot_aliases, set_defaults, lemma=None):
        query = self._query(ud_features, slot_aliases, set_defaults)
        swap_feat = self.ufeat
        if swap_ufeat_override is not None:
            swap_feat = self.resolve_column(swap_ufeat_override, slot_aliases)
        swap_map = self.inflection_map[1]
        matched = self._form_rows(form, query, lemma)
        if not matched:
            return None, None, {}, set()
        forms: Set[str] = set()
        feature_vals: Set[str] = set()
        bundles: Dict[str, Dict[str, Set[str]]] = {}
        readings: Set[frozenset] = set()
        for piece, rows in matched:
            j = piece.feature_index.get(swap_feat)
            if j is None:
                continue
            base_wanted, base_asked, _ = piece.constraints(query, strict=True)
            for i in rows:
                cell = int(piece.masks[i, j])
                if not cell:
                    continue
                own = piece.decode(swap_feat, cell)
                wanted, asked = base_wanted.copy(), base_asked.copy()
                stated = piece.masks[i] != 0
                wanted[stated], asked[stated] = piece.masks[i][stated], True
                exact = np.zeros(len(piece.features), dtype=bool)
                negated = np.zeros(len(piece.features), dtype=bool)
                if swap_map is None:
                    negated[j] = True
                else:
                    if any(v not in swap_map for v in own):
                        continue
                    target = piece.value_mask(swap_feat, ",".join(swap_map[v] for v in own))
                    if target is None:
                        continue
                    wanted[j], exact[j] = target, True
                candidates = self._single_word(piece, piece.lemma_rows(piece.lemmas[i]))
                candidates = piece.agreeing(candidates, wanted, asked, exact=exact, negated=negated)
                if swap_map is None:
                    candidates = candidates[piece.masks[candidates, j] != 0]   # must state the swap feature
                if self.prefer_tight_match and len(candidates) > 1:
                    d = piece.differences(candidates, asked, np.zeros_like(asked))
                    candidates = candidates[d == d.min()]
                if not len(candidates):
                    continue
                feature_vals.update(_shown(v) for v in own)
                readings.add(frozenset(piece.form_of(k) for k in candidates))
                for k in candidates:
                    form_k = piece.form_of(k)
                    forms.add(form_k)
                    bundle = bundles.setdefault(form_k, {})
                    for feat, values in piece.decode_row(k).items():
                        bundle.setdefault(feat, set()).update(_shown(v) for v in values)
        return forms, feature_vals, bundles, readings

    @staticmethod
    def form_found(forms) -> bool:
        return forms is not None and len(forms) == 1

    def inflect(self, form: str, ud_features: Dict[str, str], strategies=(), return_swap_feats: bool = False,
                swap_ufeat_override: Optional[str] = None, slot_aliases: Sequence[str] = (),
                lemma: Optional[str] = None, return_readings: bool = False):
        form = form.lower()
        lemma = lemma.lower() if isinstance(lemma, str) else lemma
        key = (form, frozenset(ud_features.items()), return_swap_feats, swap_ufeat_override, tuple(slot_aliases), lemma, return_readings)
        if key in self._cache:
            return self._cache[key]
        forms, vals, bundles, readings = self._inflect_once(form, ud_features, swap_ufeat_override, slot_aliases, False, lemma)
        if not self.form_found(forms):
            forms, vals, bundles, readings = self._inflect_once(form, ud_features, swap_ufeat_override, slot_aliases, True, lemma)
        if not self.form_found(forms) and self.inflect_wo_ud_features:
            forms, vals, bundles, readings = self._inflect_once(form, {}, swap_ufeat_override, slot_aliases, True, lemma)
        if self.fallback is not None:
            fb_forms, fb_vals, fb_bundles, fb_readings = self.fallback.inflect(
                form, ud_features, strategies, True, swap_ufeat_override, slot_aliases, lemma, True)
            readings = readings | fb_readings
            if forms is None:
                forms, vals, bundles = (set(fb_forms) if fb_forms is not None else None), fb_vals, fb_bundles
            elif fb_forms is not None:
                forms = set(forms) | set(fb_forms)
                vals = set(vals) | set(fb_vals)
                for swap_form, bundle in fb_bundles.items():
                    merged = bundles.setdefault(swap_form, {})
                    for feat, values in bundle.items():
                        merged.setdefault(feat, set()).update(values)
        if isinstance(forms, set):
            forms = sorted(forms)
        result = (forms, vals, bundles) if return_swap_feats else (forms, vals)
        if return_readings:
            result = (forms, vals, bundles, readings)
        self._cache[key] = result
        return result

    def get_form_features(self, form: str, features: Dict[str, str], ufeat: Optional[str] = None,
                          only_try_ud_if_no_um: bool = False, prefer_tight_match: bool = False,
                          fetch_all: bool = False, slot_aliases: Sequence[str] = (),
                          upos: Optional[str] = None) -> Set[str]:
        """The values of `ufeat` (default: the swap feature) among the rows of `form` (of part of speech
        `upos`, if given) that agree with `features`, plus the fallback table's."""
        form = form.lower()
        ufeat = self.resolve_column(ufeat, slot_aliases) if ufeat is not None else self.ufeat
        values: Set[str] = set()
        query = self._query(features, slot_aliases, False)
        query.pop(ufeat, None)
        if not fetch_all and ufeat is not None:
            tight, self.prefer_tight_match = self.prefer_tight_match, prefer_tight_match
            try:
                for piece, rows in self._form_rows(form, query, upos=upos):
                    j = piece.feature_index.get(ufeat)
                    if j is not None:
                        for i in rows:
                            values.update(_shown(v) for v in piece.decode(ufeat, int(piece.masks[i, j])))
            finally:
                self.prefer_tight_match = tight
        if self.fallback is not None and not (only_try_ud_if_no_um and values):
            values |= self.fallback.get_form_features(form, features, ufeat, only_try_ud_if_no_um,
                                                      prefer_tight_match, fetch_all, slot_aliases, upos)
        return values


def _ud_upos(values: Sequence[str]) -> List[str]:
    """UD tags for a UniMorph-or-UD upos filter ("V" -> VERB; a UD tag stays)."""
    return list(dict.fromkeys(um_tag_upos(v) or v for v in values))


def load_ud_inflector(lang: str, langcode: str, unimorph_args: dict, inflection_map, resource_dir=None,
                      also_upos=None):
    """(inflector, number of lemmas, number of forms), the counterpart of unimorph.load_inflector.
    unimorph_args: filter_entries {"upos": [...]}, combine_um_ud, remove_multiword_forms.
    also_upos: further UPOS whose rows are read with the tables, for nominal_inflector.
    The UniMorph-derived table is the inflector, with the UD-derived table as its fallback when
    combine_um_ud is set; a language without UniMorph rows runs on the UD-derived table alone,
    and one without either gets an empty inflector."""
    upos = unimorph_args.get("filter_entries", {}).get("upos")
    ud_upos = _ud_upos(upos) if upos else None
    read_upos = list(dict.fromkeys(ud_upos + _ud_upos(also_upos))) if ud_upos and also_upos else ud_upos
    kwargs = dict(upos=ud_upos, remove_multiword_forms=unimorph_args.get("remove_multiword_forms", False))
    um_table = UDTable.from_file(um_table_path(langcode), expand_values=True, upos=read_upos)
    ud_table = UDTable.load(langcode, upos=read_upos) if unimorph_args.get("combine_um_ud", False) else None
    # the UniMorph-derived table is the inflector itself (empty if the language has none)
    fallback = UDInflector(ud_table, inflection_map, expand_values=False, **kwargs) if ud_table is not None else None
    inflector = UDInflector(um_table, inflection_map, fallback=fallback, **kwargs)
    inflector.langcode, inflector.resource_dir = langcode, resource_dir or "."
    inflector.um_table, inflector.ud_table = um_table, ud_table
    return inflector, inflector.num_lemmas, inflector.num_forms


def nominal_inflector(inflector: UDInflector, upos: Sequence[str], inflection_map) -> UDInflector:
    """An inflector over the tables `inflector` was loaded with, limited to `upos` (which load_ud_inflector
    must have been given as also_upos): for form lookups on the nominals, without reading the tables again."""
    ud_upos = _ud_upos(upos)
    fallback = (UDInflector(inflector.ud_table, inflection_map, expand_values=False, upos=ud_upos,
                            remove_multiword_forms=True) if inflector.ud_table is not None else None)
    nominal = UDInflector(inflector.um_table, inflection_map, fallback=fallback, upos=ud_upos,
                          remove_multiword_forms=True)
    nominal.langcode, nominal.resource_dir = inflector.langcode, inflector.resource_dir
    return nominal
