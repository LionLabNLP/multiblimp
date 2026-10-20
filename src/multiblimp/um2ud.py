"""Real UniMorph table -> UD notation (the same file layout as multiblimp.ud_table):
resources/um_tables/<langcode>.parquet.

The raw file is read by _raw_frames and _feature_columns (typos, argument-marking tags, compounds kept
whole), each column value is turned into its UD value by the mapping module, and the
one completion is applied here: a verb row without VerbForm is Fin (plain UniMorph verb rows
are finite; non-finite ones carry V.PTCP, NFIN, ...). Mood and Voice stay unstated where
UniMorph leaves them out; the inflector's second try adds Mood=Ind / Voice=Act as query
defaults, which excludes subjunctive and passive rows without writing values the file lacks.
"""
import os
import re
import sys
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from unidecode import unidecode

from .config import REPO_ROOT, RESOURCE_DIR
from .languages import (
    convert_unimorph_script, latin_to_cyrillic, lang2langcode, lang2unimorph_lang, remove_diacritics_langs,
    remove_multiples_langs, strip_stress, strip_um_stress_langs, unimorph_code,
)
from .ud_table import SHADOW_PREFIX, UNMAPPED_PREFIX, is_stored_feature, table_path, write_ud_dataframe

sys.path.append(REPO_ROOT)
from resources.um2ud_annotation.UM2UD_mapper import (
    COMPOUND_TAGS, LGSPEC_UD, SHORTENED_UM_VALS, UM2UD, UPOS_VALUES, cached_map_um_value_to_ud, feat2val, feature_for,
    fix_typos, negation_code, negation_feature, parse_negation, pos_is_degree, um_tag_upos, val2feat,
)
from resources.um2ud_annotation.UD2UM_mapper import UD2UM

UM_TABLE_DIR = os.path.join(RESOURCE_DIR, "um_tables")
META = ("lemma", "form", "ufeat", "upos")


def um_table_path(langcode: str, directory: str = UM_TABLE_DIR) -> str:
    return os.path.join(directory, f"{langcode}.parquet")


LGSPEC_COLUMN = "Language_Specific"   # ufeats_to_dict's key for LGSPEC tags
UNKNOWN_COLUMN = "UNK"   # where ufeats_to_dict puts values whose feature it cannot tell


def to_um_value(feat, val):
    """`val` (a UD or UM value, or a list/"a,b" disjunction of them) as the UM code stored for
    UD feature `feat`; a list for disjunctions. ValueError for an unknown value."""
    if isinstance(val, str):
        val = fix_typos.get(val, val)
    if ("," in val) or ("|" in val) or ("/" in val):
        val = val.replace(",", "+").replace("|", "+").replace("/", "+")
        return [to_um_value(feat, subval) for subval in val.split("+")]

    if f"{feat}_{val}" in val2feat:
        return f"{feat}_{val}"
    elif feat == "lemma":
        return val
    elif isinstance(val, list):
        return [to_um_value(feat, subval) for subval in val]
    elif feature_for(val) is not None:
        return val
    elif (feat, val) in UD2UM:
        # before val.upper(): "PART" is also a upos tag and would misresolve VerbForm=Part
        return UD2UM[(feat, val)]
    elif val in UM2UD and feat in UM2UD[val]:
        # val is already a UM tag carrying a value for this feature.
        return to_um_value(feat, UM2UD[val][feat])
    elif feature_for(val.upper()) is not None:
        return val.upper()
    elif val.startswith("-"):
        return "-" + to_um_value(feat, val[1:])
    elif f"{feat}_{val.title()}" in val2feat:
        return f"{feat}_{val.title()}"
    else:
        raise ValueError(f"Unknown {feat}: {val}")


@lru_cache(maxsize=None)
def stored_upos(tag: str) -> str:
    """The upos value stored for a raw UM upos `tag`: its UD upos run through to_um_value, the
    translation a requested upos goes through too (ART -> DET, PRO -> PRON, PRE -> ADP, ...)."""
    ud_upos = um_tag_upos(tag)
    return tag if ud_upos is None else to_um_value("upos", ud_upos)


def ufeats_to_dict(ufeats: str) -> Dict[str, str]:
    """A UniMorph X;Y;Z feature string as {feature: UM code}, values typo-fixed first. Argument
    marking becomes "{feat}[{suffix}]" keys (Number[obj]): ARG*/PSS* tags are decomposed by the mapping
    module and mapped back to raw UM codes via UD2UM."""
    # a braced negation (non{NOM/ACC}) goes first: its "/" is not a disjunction of the row
    ufeats = re.sub(
        r"(?i)(?:\(non\)|non|not)\{[^{}]*\}",
        lambda m: negation_code(parse_negation(m.group(0))) if parse_negation(m.group(0)) else m.group(0),
        str(ufeats),
    )
    ufeats = (
        ufeats
        .replace("/", "+")
        .replace(",", "+")
        .replace("{", "")
        .replace("}", "")
        .replace("V.PCTP", "V.PTCP")
        .replace(":", ";")
    )
    # a verbal dotted tag ("V.CVB", "V.MSDR") implies its part of speech: "V;V.CVB"
    ufeats = re.sub(r"(?<![\w.])(V\.[A-Z]+)(?![\w.])", r"V;\1", ufeats)
    ufeats = ufeats.strip()
    ufeat_vals = ufeats.split(";")
    ufeat_dict = {}

    for val in ufeat_vals:
        val = val.strip()
        if len(val) == 0:
            continue
        val = fix_typos.get(val, val)
        upos_parts = [part.strip() for part in val.split("+")]   # "ADP + ADJ": a fused form (ckb)
        if all(part in UPOS_VALUES for part in upos_parts):
            ufeat_dict["upos"] = "+".join(stored_upos(part) for part in upos_parts)
            continue
        elif (val.startswith("ARG") or val.startswith("PSS")) and val not in UM2UD:
            for bracketed_feat, ud_val in cached_map_um_value_to_ud(val)["morpho"].items():
                base_feat, _, suffix = bracketed_feat.partition("[")
                suffix = suffix.rstrip("]")
                um_code = "+".join(UD2UM.get((base_feat, part), part) for part in ud_val.split(","))   # "Plur,Sing": PSS3S/PSS3P
                key = f"{base_feat}[{suffix}]" if suffix else base_feat
                ufeat_dict[key] = f"{ufeat_dict[key]}+{um_code}" if key in ufeat_dict else um_code
            continue
        elif (inner := parse_negation(val)) is not None:
            key = negation_feature(val)
            code = negation_code(inner)
            ufeat_dict[key] = f"{ufeat_dict[key]}+{code}" if key in ufeat_dict else code
            continue
        elif (resolved := feature_for(val)) is not None:
            # also covers values the package knows but deliberately leaves unmapped
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
            # each part is read on its own (FEM+NEUT+PL, ALN+PSS3S, SG+FOO); parts of one feature join with "+"
            for part in val.split("+"):
                for key, code in ufeats_to_dict(part).items():
                    ufeat_dict[key] = f"{ufeat_dict[key]}+{code}" if key in ufeat_dict else code
            continue
        elif (resolved := feature_for(val.upper())) is not None:
            ufeat = resolved
        else:
            ufeat = "UNK"

        if ufeat in ufeat_dict:
            ufeat_dict[ufeat] += f"+{val}"
        else:
            ufeat_dict[ufeat] = val

    if ufeat_dict.get("Polarity") == "POS" and pos_is_degree(
        ufeat_dict.get("upos"), [f for f in ufeat_dict if f not in ("upos", "Polarity")]
    ):
        del ufeat_dict["Polarity"]
        ufeat_dict.setdefault("Degree", "BASE")

    return ufeat_dict


def _expand_multiple_values(df: pd.DataFrame) -> pd.DataFrame:
    """One row per reading of an "a+b" value (compound tags stay whole)."""
    df_rows = df.to_dict(orient="records")
    idx = 0
    delete_rows = []
    while idx < len(df_rows):
        row = df_rows[idx]
        for ufeat, val in row.items():
            if ufeat not in {"lemma", "form", "ufeat"} and isinstance(val, str) and "+" in val and val not in COMPOUND_TAGS:
                for extra_val in val.split("+"):
                    extra_row = row.copy()
                    extra_row[ufeat] = to_um_value(ufeat, extra_val)
                    df_rows.append(extra_row)
                # one feature per pass; a row with several disjunctions is split again when its copies come up
                delete_rows.append(idx)
                break
        idx += 1
    return pd.DataFrame(df_rows).drop(index=delete_rows)


def _unimorph_path(langcode: str, resource_dir: str) -> Optional[str]:
    um_langcode = unimorph_code(langcode)
    path = os.path.join(resource_dir, f"unimorph/{um_langcode}/{um_langcode}")
    return path if os.path.isfile(path) else None


def _chunk_size(path: str, chunk_rows: Optional[int]) -> Optional[int]:
    """Rows per chunk for this file, None to read it whole: an explicit `chunk_rows`, else CHUNK_ROWS for a
    file above CHUNK_MIN_BYTES. A file with a .segmentations file is always read whole."""
    if os.path.isfile(path + ".segmentations"):
        return None
    return chunk_rows or (CHUNK_ROWS if os.path.getsize(path) > CHUNK_MIN_BYTES else None)


def _raw_frames(path: str, chunk_rows: Optional[int], usecols=None):
    """The file's lemma, form and ufeat columns (plus its .segmentations rows): one frame, or `chunk_rows` rows
    at a time."""
    # only empty cells are missing: words like "nan", "null", "None" are real forms
    read = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    names = ["lemma", "form", "ufeat"]
    if chunk_rows:
        yield from pd.read_csv(path, names=names, usecols=usecols, chunksize=chunk_rows, **read)
        return
    df = pd.read_csv(path, names=names, **read)
    if os.path.isfile(path + ".segmentations"):
        df_seg = pd.read_csv(path + ".segmentations", names=names + ["segmentation"], **read)
        df_seg = df_seg.drop("segmentation", axis=1)
        df_seg["ufeat"] = [str(ufeat).replace("|", ";") for ufeat in df_seg.ufeat]
        df = pd.concat([df, df_seg], ignore_index=True).drop_duplicates()
    yield df


def _normalise_text(df: pd.DataFrame, langcode: str, remove_diacritics: bool, remove_multiples: bool,
                    remove_stress: bool) -> pd.DataFrame:
    if remove_diacritics:
        df.lemma = [unidecode(lemma) for lemma in df.lemma]
        df.form = [unidecode(form) for form in df.form]
    if remove_stress:
        df.lemma = [strip_stress(lemma) for lemma in df.lemma]
        df.form = [strip_stress(form) for form in df.form]
    to_treebank_script = convert_unimorph_script(langcode)
    if to_treebank_script is not None:
        df.lemma = [to_treebank_script(lemma) for lemma in df.lemma]
        df.form = [to_treebank_script(form) for form in df.form]
    if langcode == "tat":
        df.lemma = [latin_to_cyrillic(lemma) for lemma in df.lemma]
        df.form = [latin_to_cyrillic(form) for form in df.form]
    if remove_multiples:
        df.form = [form.split(", ")[0] for form in df.form]
    for column in ("lemma", "form"):
        df[column] = df[column].map(lambda v: v.lower() if isinstance(v, str) else v)
    return df


def _feature_columns(df: pd.DataFrame, um_langcode: str) -> pd.DataFrame:
    """`df` plus one column of UniMorph codes per feature its ufeat strings state, and upos; "a+b" values are
    split into rows. Depends on a row's ufeat string only."""
    keys = [str(ufeat) for ufeat in df.ufeat]
    parsed = {key: ufeats_to_dict(key) for key in set(keys)}
    row_ufeats = [parsed[key] for key in keys]
    # "{feat}[{suffix}]" columns (ARG*/PSS* tags) are not in feat2val
    bracketed_cols = {k for d in parsed.values() for k in d if "[" in k}
    lgspec = LGSPEC_UD.get(um_langcode)   # language-specific tags with a UD value, applied in convert_language
    for col in sorted(set(feat2val) | bracketed_cols | ({LGSPEC_COLUMN} if lgspec else set())):
        values = [d.get(col) for d in row_ufeats]
        if any(values):
            df[col] = values

    df = _expand_multiple_values(df)
    return df.astype(object)


def _distinct_ufeat_rows(path: str, chunk_rows: Optional[int], um_langcode: str) -> pd.DataFrame:
    """One row per distinct ufeat string of the file, through the same feature step as the table's rows: the
    columns and the codes in them are those of the whole table."""
    distinct = pd.concat([df[["ufeat"]].drop_duplicates()
                          for df in _raw_frames(path, chunk_rows, usecols=["ufeat"])]).drop_duplicates()
    return _feature_columns(distinct.reset_index(drop=True), um_langcode)


def _ud_value(feat: str, code) -> Optional[str]:
    """UD value of a UniMorph code in column `feat`. A code with no UD value stays as
    "UM:<code>": the row still states something there, so it does not match a UD value."""
    if not isinstance(code, str):
        return None
    # the one/two-letter codes are only unambiguous in argument-marking columns ("Number[obj]")
    source = SHORTENED_UM_VALS if "[" in feat and code in SHORTENED_UM_VALS else UM2UD
    value = source.get(code, {}).get(feat.partition("[")[0])
    return value if isinstance(value, str) else UNMAPPED_PREFIX + code


def _ud_table_values(langcode: str, col: str) -> set:
    """The values of one feature in the language's UD-derived table."""
    path = table_path(langcode)
    if not os.path.isfile(path) or col not in pd.read_parquet(path).columns:
        return set()
    cells = pd.read_parquet(path, columns=[col])[col].dropna().unique()
    return {v for cell in cells for v in str(cell).split(",")}


CHUNK_MIN_BYTES = 250_000_000   # a UniMorph file above this (Slovak, 1.5 GB; the rest < 100 MB) is converted in chunks
CHUNK_ROWS = 250_000
DEDUPE_BUCKETS = 64

PQP_MIN_MATCHED = 30       # UD Pqp rows that must be found in the UniMorph table
PQP_MIN_SHARE = 0.8        # of them, the share with a PRF reading
PQP_MAX_PAST_SHARE = 0.05  # UD past rows allowed to have only PRF readings


def _pluperfect_is_prf(out: pd.DataFrame, langcode: str) -> bool:
    """Whether this language's UniMorph past + PRF rows are the pluperfect UD calls Tense=Pqp. UniMorph has
    no pluperfect tag, so the rows are Tense=Past + Aspect=Perf, like the preterite (PFV). Checked on the
    UD-derived table by (lemma, form): enough UD Pqp rows (PQP_MIN_MATCHED) must be found among the
    UniMorph past rows, nearly all (PQP_MIN_SHARE) with a PRF reading, and UD past rows must almost never
    have only PRF readings (PQP_MAX_PAST_SHARE). Languages where PRF is the perfect fail the check."""
    shadow = SHADOW_PREFIX + "Aspect"
    path = table_path(langcode)
    if shadow not in out.columns or "Tense" not in out.columns or not os.path.isfile(path):
        return False
    ud = pd.read_parquet(path)
    if "Tense" not in ud.columns:
        return False
    tense = ud["Tense"].astype(str)
    if (tense == "Pqp").sum() < PQP_MIN_MATCHED:
        return False
    past = out[out["Tense"] == "Past"]
    past_codes = past[["lemma", "form"]].assign(code=past[shadow].astype(object))
    codes = past_codes.groupby(["lemma", "form"])["code"].agg(lambda s: set(s.dropna()))

    def readings(rows):
        return [codes[key] for key in zip(rows["lemma"], rows["form"]) if key in codes.index and codes[key]]

    pqp, past_rows = readings(ud[tense == "Pqp"]), readings(ud[tense == "Past"])
    if len(pqp) < PQP_MIN_MATCHED or not past_rows:
        return False
    return (sum("PRF" in c for c in pqp) / len(pqp) >= PQP_MIN_SHARE
            and sum(c == {"PRF"} for c in past_rows) / len(past_rows) <= PQP_MAX_PAST_SHARE)


def _value_plan(ref: pd.DataFrame, langcode: str) -> Dict[str, tuple]:
    """Per stored feature column of the table: (UniMorph code -> UD value, whether a shadow column keeps the
    codes). Built from the distinct codes, so every chunk of a table is converted alike."""
    plan = {}
    for col in ref.columns:
        if col in META or not (is_stored_feature(col) or col == UNKNOWN_COLUMN):
            continue
        if col == UNKNOWN_COLUMN:
            plan[col] = (None, False)
            continue
        cache = {code: _ud_value(col, code) for code in ref[col].dropna().unique()}
        # "not NOM" states the language's other values of the feature, as a comma value: those of
        # its UniMorph table and its UD-derived table
        negated = {code: parse_negation(code) for code in cache if parse_negation(code) is not None}
        for code, excluded in negated.items():
            others = {v for c, v in cache.items() if c not in negated and not v.startswith(UNMAPPED_PREFIX)}
            others |= _ud_table_values(langcode, col)
            others -= {_ud_value(col, target) for target in excluded}
            if others:
                cache[code] = ",".join(sorted(others))
        # UD merges some UniMorph values (PFV and PRF are both Aspect=Perf): keep the code in a
        # shadow column so a paradigm lookup does not return the other tense's forms
        by_value = {}
        for code, value in cache.items():
            by_value.setdefault(value, set()).add(code)
        plan[col] = (cache, any(len(codes) > 1 for codes in by_value.values()))
    return plan


def _column(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(None, index=df.index, dtype=object)


def _convert_chunk(df: pd.DataFrame, plan: Dict[str, tuple], langcode: str) -> pd.DataFrame:
    out = pd.DataFrame({"lemma": df["lemma"], "form": df["form"], "upos": _column(df, "upos").map(um_tag_upos)})
    for col, (cache, shadow) in plan.items():
        codes = _column(df, col)
        if cache is None:
            # a UniMorph value the mapper does not know: kept as UM:<code> (several joined by "+" become
            # a comma value) so the row still states something, and reported with the unmapped codes
            out[col] = codes.map(lambda v: ",".join(UNMAPPED_PREFIX + c for c in str(v).split("+")) if isinstance(v, str) else None)
            continue
        out[col] = codes.map(cache)
        if shadow:
            out[SHADOW_PREFIX + col] = codes
    for tag, ud_feats in LGSPEC_UD.get(unimorph_code(langcode), {}).items():
        for feat, value in ud_feats.items():
            out.loc[_column(df, LGSPEC_COLUMN) == tag, feat] = value
    out = out[out["upos"].notna() & out["lemma"].notna() & out["form"].notna()]
    # a cell holding several readings ("A+B" splits earlier) never reaches here, so values are single
    if "VerbForm" not in out.columns:
        out["VerbForm"] = pd.Series(None, index=out.index, dtype=object)
    verb = out["upos"].isin(["VERB", "AUX"])
    out.loc[verb & out["VerbForm"].isna(), "VerbForm"] = "Fin"
    return out


def _compact(piece: pd.DataFrame) -> pd.DataFrame:
    """A converted chunk in cheap dtypes while the rest of the table is converted: arrow strings for lemma and
    form (nearly all distinct), categories for the rest."""
    piece = piece.drop_duplicates()
    for col in piece.columns:
        piece[col] = piece[col].astype(pd.StringDtype("pyarrow") if col in ("lemma", "form") else "category")
    return piece


def _concat_pieces(pieces: list) -> pd.DataFrame:
    if len(pieces) == 1:
        return pieces[0]
    columns = {}
    for col in pieces[0].columns:
        parts = [piece[col] for piece in pieces]
        if isinstance(parts[0].dtype, pd.CategoricalDtype):
            # a chunk with no value in the column has object categories, the others str ones
            dtype = next((p.dtype for p in parts if len(p.dtype.categories)), parts[0].dtype)
            columns[col] = union_categoricals([part.astype(dtype).array if not len(part.dtype.categories) else part.array
                                               for part in parts])
        else:
            columns[col] = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns)


def _drop_duplicate_rows(out: pd.DataFrame) -> pd.DataFrame:
    """`out` without duplicate rows, index reset. A chunked table (categorical columns) is checked in buckets
    of its category codes, which duplicates share, so the group index of `duplicated` stays small."""
    categorical = [col for col in out.columns if isinstance(out[col].dtype, pd.CategoricalDtype)]
    if not categorical:
        return out.drop_duplicates().reset_index(drop=True)
    bucket = np.zeros(len(out), dtype=np.uint64)
    for col in categorical:
        bucket = bucket * np.uint64(1000003) + out[col].cat.codes.to_numpy().astype(np.uint64)
    bucket %= np.uint64(DEDUPE_BUCKETS)
    keep = np.ones(len(out), dtype=bool)
    for value in range(DEDUPE_BUCKETS):
        rows = np.flatnonzero(bucket == value)
        keep[rows] = ~out.iloc[rows].duplicated().to_numpy()
    out = out.take(np.flatnonzero(keep))
    out.index = pd.RangeIndex(len(out))
    return out


def convert_language(lang: str, resource_dir: str = RESOURCE_DIR, chunk_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """The language's UniMorph table in UD notation (columns lemma, form, upos, frequency and the
    stored features); None if there is no UniMorph file. A file above CHUNK_MIN_BYTES (or any file, with
    `chunk_rows`) is converted that many rows at a time."""
    langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
    path = _unimorph_path(langcode, resource_dir)
    if path is None:
        return None
    um_langcode = unimorph_code(langcode)
    chunk_rows = _chunk_size(path, chunk_rows)
    ref = _distinct_ufeat_rows(path, chunk_rows, um_langcode)
    if not len(ref) or "upos" not in ref.columns:
        return None
    plan = _value_plan(ref, langcode)
    text_args = (langcode, lang in remove_diacritics_langs, lang in remove_multiples_langs, lang in strip_um_stress_langs)
    pieces = []
    for df in _raw_frames(path, chunk_rows):
        piece = _convert_chunk(_feature_columns(_normalise_text(df, *text_args), um_langcode), plan, langcode)
        pieces.append(_compact(piece) if chunk_rows else piece)
    out = _concat_pieces(pieces)
    del pieces
    out = out.drop(columns=[col for col in out.columns if not out[col].notna().any()])
    if "VerbForm" not in out.columns:
        out["VerbForm"] = pd.Series(None, index=out.index, dtype=object)
    # where UniMorph's past + PRF is the pluperfect, state it as UD does (Tense=Pqp) so a Tense=Past token
    # does not match both the preterite and the pluperfect forms
    pluperfect = False
    if _pluperfect_is_prf(out, langcode):
        if isinstance(out["Tense"].dtype, pd.CategoricalDtype):
            out["Tense"] = out["Tense"].cat.add_categories(["Pqp"])
        mask = (out["Tense"] == "Past") & (out[SHADOW_PREFIX + "Aspect"] == "PRF")
        out.loc[mask, "Tense"] = "Pqp"
        out.loc[mask, ["Aspect", SHADOW_PREFIX + "Aspect"]] = None
        pluperfect = True
    out = _drop_duplicate_rows(out)
    out["frequency"] = 1
    out.attrs["pluperfect_rows"] = int((out["Tense"] == "Pqp").sum()) if pluperfect else 0
    return out


def write_language(lang: str, directory: str = UM_TABLE_DIR, chunk_rows: Optional[int] = None) -> dict:
    """Convert and write one language. {"rows": row count (0 if it has no UniMorph file),
    "unmapped": {"Feature=CODE": rows} for the codes with no UD value, "pluperfect": rows rewritten to Tense=Pqp}."""
    out = convert_language(lang, chunk_rows=chunk_rows)
    if out is None:
        return {"rows": 0, "unmapped": {}, "pluperfect": 0}
    unmapped = {}
    for col in out.columns:
        if col in ("lemma", "form", "upos", "frequency") or col.startswith(SHADOW_PREFIX):
            continue
        for code, n in out[col].value_counts().items():
            if str(code).startswith(UNMAPPED_PREFIX):
                unmapped[f"{col}={code[len(UNMAPPED_PREFIX):]}"] = int(n)
    langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
    return {"rows": write_ud_dataframe(out, um_table_path(langcode, directory)), "unmapped": unmapped,
            "pluperfect": out.attrs.get("pluperfect_rows", 0)}
