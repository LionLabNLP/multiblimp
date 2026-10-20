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
import re
import sys
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd
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


def read_unimorph(langcode: str, resource_dir: str = RESOURCE_DIR, remove_diacritics: bool = False,
                  remove_multiples: bool = False, remove_stress: bool = False) -> Optional[pd.DataFrame]:
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

    row_ufeats = [ufeats_to_dict(ufeat) for ufeat in df.ufeat]
    # "{feat}[{suffix}]" columns (ARG*/PSS* tags) are not in feat2val
    bracketed_cols = {k for d in row_ufeats for k in d if "[" in k}
    lgspec = LGSPEC_UD.get(um_langcode)   # language-specific tags with a UD value, applied in convert_language
    for col in sorted(set(feat2val) | bracketed_cols | ({LGSPEC_COLUMN} if lgspec else set())):
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
    df = read_unimorph(langcode, resource_dir, lang in remove_diacritics_langs, lang in remove_multiples_langs,
                       lang in strip_um_stress_langs)
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
        negated = {code: parse_negation(code) for code in cache if parse_negation(code) is not None}
        for code, excluded in negated.items():
            others = {v for c, v in cache.items() if c not in negated and not v.startswith(UNMAPPED_PREFIX)}
            others |= _ud_table_values(langcode, col)
            others -= {_ud_value(col, target) for target in excluded}
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
    for tag, ud_feats in LGSPEC_UD.get(unimorph_code(langcode), {}).items():
        for feat, value in ud_feats.items():
            if LGSPEC_COLUMN in df.columns:
                out.loc[df[LGSPEC_COLUMN] == tag, feat] = value
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
