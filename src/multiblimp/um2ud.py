"""Real UniMorph table -> UD notation (the same file layout as multiblimp.ud_table):
resources/um_tables/<langcode>.parquet.

The raw file is read by read_unimorph (typos, argument-marking tags, compounds kept
whole), each column value is turned into its UD value by the mapping module, and the
one completion is applied here: a verb row without VerbForm is Fin (plain UniMorph verb rows
are finite; non-finite ones carry V.PTCP, NFIN, ...). Mood and Voice stay unstated where
UniMorph leaves them out; the inflector's second try adds Mood=Ind / Voice=Act as query
defaults, which excludes subjunctive and passive rows without writing values the file lacks.
"""
import os
import sys
from typing import Optional

import pandas as pd
from unidecode import unidecode

from .config import REPO_ROOT, RESOURCE_DIR
from .languages import (
    latin_to_cyrillic, lang2langcode, lang2unimorph_lang, remove_diacritics_langs,
    remove_multiples_langs, unimorph_code,
)
from .ud_table import SHADOW_PREFIX, UNMAPPED_PREFIX, is_stored_feature, table_path, write_ud_dataframe

sys.path.append(REPO_ROOT)
from resources.um2ud_annotation.UM2UD_mapper import (
    COMPOUND_TAGS, NEGATED_UM_VALUES, SHORTENED_UM_VALS, UM2UD, feat2val, um_tag_upos,
)
from resources.um2ud_annotation.UD2UM_mapper import to_um_value, ufeats_to_dict

UM_TABLE_DIR = os.path.join(RESOURCE_DIR, "um_tables")
META = ("lemma", "form", "ufeat", "upos")


def um_table_path(langcode: str, directory: str = UM_TABLE_DIR) -> str:
    return os.path.join(directory, f"{langcode}.parquet")


UNKNOWN_COLUMN = "UNK"   # where ufeats2dict puts values whose feature it cannot tell


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


def read_unimorph(langcode: str, resource_dir: str = RESOURCE_DIR, remove_diacritics: bool = False,
                  remove_multiples: bool = False) -> Optional[pd.DataFrame]:
    """The language's raw UniMorph file (plus its .segmentations file) as a DataFrame: lemma, form,
    the feature columns holding UniMorph codes, and upos. Lemma and form are lowercased; "a+b"
    values are split into rows. None if there is no file."""
    um_langcode = unimorph_code(langcode)
    path = os.path.join(resource_dir, f"unimorph/{um_langcode}/{um_langcode}")
    if not os.path.isfile(path):
        return None
    # only empty cells are missing: words like "nan", "null", "None" are real forms
    read = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    df = pd.read_csv(path, names=["lemma", "form", "ufeat"], **read)
    if os.path.isfile(path + ".segmentations"):
        df_seg = pd.read_csv(path + ".segmentations", names=["lemma", "form", "ufeat", "segmentation"], **read)
        df_seg = df_seg.drop("segmentation", axis=1)
        df_seg["ufeat"] = [str(ufeat).replace("|", ";") for ufeat in df_seg.ufeat]
        df = pd.concat([df, df_seg], ignore_index=True).drop_duplicates()

    if remove_diacritics:
        df.lemma = [unidecode(lemma) for lemma in df.lemma]
        df.form = [unidecode(form) for form in df.form]
    if langcode == "tat":
        df.lemma = [latin_to_cyrillic(lemma) for lemma in df.lemma]
        df.form = [latin_to_cyrillic(form) for form in df.form]
    if remove_multiples:
        df.form = [form.split(", ")[0] for form in df.form]
    for column in ("lemma", "form"):
        df[column] = df[column].map(lambda v: v.lower() if isinstance(v, str) else v)

    row_ufeats = [ufeats_to_dict(ufeat) for ufeat in df.ufeat]
    # "{feat}[{suffix}]" columns (ARG*/PSS* tags) are not in feat2val
    bracketed_cols = {k for d in row_ufeats for k in d if "[" in k}
    for col in sorted(set(feat2val) | bracketed_cols):
        values = [d.get(col) for d in row_ufeats]
        if any(values):
            df[col] = values

    df = _expand_multiple_values(df)
    return df.astype(object)


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
    codes = past.groupby(["lemma", "form"])[shadow].agg(lambda s: set(s.dropna()))

    def readings(rows):
        return [codes[key] for key in zip(rows["lemma"], rows["form"]) if key in codes.index and codes[key]]

    pqp, past_rows = readings(ud[tense == "Pqp"]), readings(ud[tense == "Past"])
    if len(pqp) < PQP_MIN_MATCHED or not past_rows:
        return False
    return (sum("PRF" in c for c in pqp) / len(pqp) >= PQP_MIN_SHARE
            and sum(c == {"PRF"} for c in past_rows) / len(past_rows) <= PQP_MAX_PAST_SHARE)


def convert_language(lang: str, resource_dir: str = RESOURCE_DIR) -> Optional[pd.DataFrame]:
    """The language's UniMorph table in UD notation (columns lemma, form, upos, frequency and the
    stored features); None if there is no UniMorph file."""
    langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
    df = read_unimorph(langcode, resource_dir, lang in remove_diacritics_langs, lang in remove_multiples_langs)
    if df is None or not len(df) or "upos" not in df.columns:
        return None
    out = pd.DataFrame({"lemma": df["lemma"], "form": df["form"],
                        "upos": df["upos"].map(um_tag_upos)})
    for col in df.columns:
        if col in META or not (is_stored_feature(col) or col == UNKNOWN_COLUMN):
            continue
        if col == UNKNOWN_COLUMN:
            # a UniMorph value the mapper does not know: kept as UM:<code> (several joined by "+" become
            # a comma value) so the row still states something, and reported with the unmapped codes
            out[col] = df[col].map(lambda v: ",".join(UNMAPPED_PREFIX + c for c in str(v).split("+")) if isinstance(v, str) else None)
            continue
        cache = {code: _ud_value(col, code) for code in df[col].dropna().unique()}
        # "not NOM" states the language's other values of the feature, as a comma value: those of
        # its UniMorph table and its UD-derived table
        for code, target in NEGATED_UM_VALUES.items():
            if code in cache:
                others = {v for c, v in cache.items() if c not in NEGATED_UM_VALUES and not v.startswith(UNMAPPED_PREFIX)}
                others |= _ud_table_values(langcode, col)
                others -= {_ud_value(col, target)}
                if others:
                    cache[code] = ",".join(sorted(others))
        out[col] = df[col].map(cache)
        # UD merges some UniMorph values (PFV and PRF are both Aspect=Perf): keep the code in a
        # shadow column so a paradigm lookup does not return the other tense's forms
        by_value = {}
        for code, value in cache.items():
            by_value.setdefault(value, set()).add(code)
        if any(len(codes) > 1 for codes in by_value.values()):
            out[SHADOW_PREFIX + col] = df[col]
    out = out[out["upos"].notna() & out["lemma"].notna() & out["form"].notna()]
    out = out.loc[:, out.notna().any(axis=0)]
    if "VerbForm" not in out.columns:
        out["VerbForm"] = pd.Series(None, index=out.index, dtype=object)
    # a cell holding several readings ("A+B" splits earlier) never reaches here, so values are single
    verb = out["upos"].isin(["VERB", "AUX"])
    out.loc[verb & out["VerbForm"].isna(), "VerbForm"] = "Fin"
    # where UniMorph's past + PRF is the pluperfect, state it as UD does (Tense=Pqp) so a Tense=Past token
    # does not match both the preterite and the pluperfect forms
    pluperfect = False
    if _pluperfect_is_prf(out, langcode):
        mask = (out["Tense"] == "Past") & (out[SHADOW_PREFIX + "Aspect"] == "PRF")
        out.loc[mask, "Tense"] = "Pqp"
        out.loc[mask, ["Aspect", SHADOW_PREFIX + "Aspect"]] = None
        pluperfect = True
    out = out.drop_duplicates().reset_index(drop=True)
    out["frequency"] = 1
    out.attrs["pluperfect_rows"] = int((out["Tense"] == "Pqp").sum()) if pluperfect else 0
    return out


def write_language(lang: str, directory: str = UM_TABLE_DIR) -> dict:
    """Convert and write one language. {"rows": row count (0 if it has no UniMorph file),
    "unmapped": {"Feature=CODE": rows} for the codes with no UD value, "pluperfect": rows rewritten to Tense=Pqp}."""
    out = convert_language(lang)
    if out is None:
        return {"rows": 0, "unmapped": {}, "pluperfect": 0}
    unmapped = {}
    for col in out.columns:
        if col in ("lemma", "form", "upos", "frequency") or col.startswith(SHADOW_PREFIX):
            continue
        counts = out[col].dropna().astype(str).str.startswith(UNMAPPED_PREFIX)
        for code, n in out.loc[counts[counts].index, col].value_counts().items():
            unmapped[f"{col}={code[len(UNMAPPED_PREFIX):]}"] = int(n)
    langcode = lang2langcode(lang2unimorph_lang.get(lang, lang))
    return {"rows": write_ud_dataframe(out, um_table_path(langcode, directory)), "unmapped": unmapped,
            "pluperfect": out.attrs.get("pluperfect_rows", 0)}
