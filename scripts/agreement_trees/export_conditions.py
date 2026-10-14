"""Per-condition debugging export: one TSV per agreement condition listing every
instance with its label (agree / disagree / unk_head / unk_dep / unk_both),
sentence id, treebank link, the marked sentence, and separate columns for the two
agreement-relevant tokens.

Reads the treebank feature caches (verb-phrase families and the NP instances), so run
cache_treebank_features.py first. Every condition with at least one agree or disagree
row is exported, with all of its rows; no minimum count. The features in
scan.exclude_features (resources/agreement_config.json) are never conditions.

Columns: label, treebank, sent_id, conllu_file, sent_line (first line of the sentence's block),
link, sentence, then per token role: <role>_id (the CoNLL-U ID), <role>_line (its line in the
file), form, lemma, upos, deprel, xpos, the agreement feature, the full feats, raw_feats (the file's FEATS
field verbatim) and <Feature>_keys (the key(s) the file uses for the tested feature, e.g. Number or
Number[subj]; the pipeline folds [subj]/[nsubj] into the plain feature before comparing). Caches built
before these columns existed give empty file/line cells.

Output: {output dir}/export/{lang}/{condition}.tsv, plus _summary.tsv (rows per label) and
_not_exported.tsv (the conditions that were tried but had no agree or disagree row, with their label counts).
Condition names are the fit pipelines' ids ("svGa", "svCase", "HEAD-DET_Gender").

Then, unless --no_bundle, one shareable bundle per language: {run dir}/dist/multiblimp-{lang}-debug/
(export/*.tsv, site/ = the generated html, serve.py, README.md) and the same folder zipped as
{run dir}/dist/multiblimp-{lang}-debug.zip. The README gets {run dir}/readme_extra.md appended, if
it exists (or --readme_extra); the web server's port is --port.

  MULTIBLIMP_DEBUG_MODE=1 python3 export_conditions.py --langs Swedish --treebanks Talbanken
"""
import argparse
import glob
import os
import re
import shutil
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from agreement_trees.config import AGREEMENT_CONFIG_PATH, lexical_features, load_agreement_config, vp_target_id
from multiblimp.config import HTML_DECISION_TREES_DIR, NP_INSTANCES_DIR, OUTPUT_DIR, RESOURCE_DIR, RUN_DIR, UD_PATH
from multiblimp.languages import gblang2udlang
from agreement_trees.vp.conditions import CONDITIONS
from agreement_trees.utils import build_grew_url

LABEL_ORDER = ["disagree", "agree", "unk_head", "unk_dep", "unk_both", "unk_other"]
FEAT_RE = r"[A-Z][A-Za-z]*(?:\[[^\]]+\])?"
NOT_AGREEMENT = {"SubjAuxCount"}
TOKEN_FIELDS = (("form_orig", "form"), ("lemma", "lemma"), ("pos", "upos"), ("deprel", "deprel"), ("xpos", "xpos"))
RAW_COLUMN = (1, 2, 3, 7, 4)   # where TOKEN_FIELDS sit in a CoNLL-U line


def read_columns(path: str, wanted) -> pd.DataFrame:
    present = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in wanted if c in present])


def role_feat_columns(names, role: str) -> list[str]:
    return [c for c in names if re.fullmatch(rf"{re.escape(role)}_{FEAT_RE}", c)]


def unk_reason(df: pd.DataFrame, labels: pd.Series, col_a: str, col_b: str) -> pd.Series:
    """agree / disagree for yes / no; for the rest, which side lacks the feature. The verb-phrase
    labels say it ("--" both, "+-" one; the dependent's own column tells which), because for objects
    the verb's value is a layered one (Number[obj]), not the plain column. Noun-phrase labels are
    "unk" and go by the two role columns. unk_other is a defensive fallback that should stay empty."""
    a_missing = df[col_a].isna() if col_a in df else pd.Series(True, index=df.index)
    b_missing = df[col_b].isna() if col_b in df else pd.Series(True, index=df.index)
    out = pd.Series("unk_other", index=df.index)
    out[~a_missing & b_missing] = "unk_dep"
    out[a_missing & ~b_missing] = "unk_head"
    out[a_missing & b_missing] = "unk_both"
    out[labels == "--"] = "unk_both"
    out[(labels == "+-") & b_missing] = "unk_dep"
    out[(labels == "+-") & ~b_missing] = "unk_head"
    out[labels == "yes"] = "agree"
    out[labels == "no"] = "disagree"
    return out


def marked_sentence(sen, idx_by_role: dict) -> str:
    marks = {}
    for role, idx in idx_by_role.items():
        if pd.notna(idx):
            marks.setdefault(int(idx) - 1, []).append(role)
    return " ".join(f"[{w}|{'+'.join(marks[i])}]" if i in marks else str(w) for i, w in enumerate(sen))


_FILE_LINES = {}


def raw_token(conllu_file, line):
    """The ten CoNLL-U columns of one token line exactly as the file has them (None when the file or
    line is unknown)."""
    if conllu_file is None or pd.isna(conllu_file) or line is None or pd.isna(line):
        return None
    if conllu_file not in _FILE_LINES:
        path = os.path.join(RESOURCE_DIR, UD_PATH, conllu_file)
        _FILE_LINES[conllu_file] = open(path, encoding="utf-8").read().split("\n") if os.path.exists(path) else []
    lines = _FILE_LINES[conllu_file]
    n = int(line) - 1
    cols = lines[n].split("\t") if 0 <= n < len(lines) else []
    return cols if len(cols) >= 8 else None


def raw_feats_field(conllu_file, line) -> str:
    """The FEATS column of one token line exactly as the .conllu file has it ("" for "_" or when the
    file or line is unknown)."""
    cols = raw_token(conllu_file, line)
    return cols[5] if cols and cols[5] != "_" else ""


def raw_keys_of(feats_field: str, feat: str) -> str:
    """The keys the file uses for `feat` on this token: "Number", "Number[subj]", or both."""
    keys = [kv.split("=", 1)[0] for kv in feats_field.split("|") if kv]
    return ",".join(k for k in keys if k == feat or k.startswith(feat + "["))


def feats_string(row_values: dict) -> str:
    items = [(k, v) for k, v in row_values.items() if pd.notna(v) and v not in ("", "_")]
    return "|".join(f"{k}={v}" for k, v in sorted(items, key=lambda kv: kv[0].lower()))


def build_table(path: str, label_col: str, role_a: str, role_b: str, feat: str, names: dict, treebanks) -> pd.DataFrame:
    """names: {"a": prefix in the cache for the first role, "b": likewise, "label_a"/"label_b": names in the output}."""
    schema = pq.read_schema(path).names
    roles = {"a": names["a"], "b": names["b"]}
    wanted = {label_col, "treebank", "sent_id", "tree_idx", "sen", "no_space_after", "conllu_file", "conllu_sent_line"}
    feat_cols = {r: role_feat_columns(schema, p) for r, p in roles.items()}
    for r, p in roles.items():
        wanted |= {f"{p}_idx", f"{p}_idx_line", *(f"{p}_{src}" for src, _ in TOKEN_FIELDS), *feat_cols[r], f"{p}_{feat}"}
    df = read_columns(path, wanted)
    if treebanks:
        keep = df["treebank"].map(lambda t: any(t == n or t.endswith("-" + n) for n in treebanks)).astype(bool)
        df = df[keep]
    df = df[df[label_col].notna()]
    if not len(df):
        return df
    out = pd.DataFrame(index=df.index)
    out["label"] = unk_reason(df, df[label_col], f"{roles['a']}_{feat}", f"{roles['b']}_{feat}")
    out["treebank"] = df["treebank"]
    out["sent_id"] = df["sent_id"]
    out["conllu_file"] = df["conllu_file"] if "conllu_file" in df else None
    out["sent_line"] = df["conllu_sent_line"].astype("Int64") if "conllu_sent_line" in df else pd.array([None] * len(df), dtype="Int64")
    form_a, form_b = (df.get(f"{roles[r]}_form_orig", pd.Series(None, index=df.index)) for r in "ab")
    out["link"] = [build_grew_url(t, s, [fa, fb]) for t, s, fa, fb in zip(df["treebank"], df["sent_id"], form_a, form_b)]
    out["sentence"] = [
        marked_sentence(sen, {names["label_a"]: ia, names["label_b"]: ib})
        for sen, ia, ib in zip(df["sen"], df[f"{roles['a']}_idx"], df[f"{roles['b']}_idx"])]
    for r, p in roles.items():
        label = names[f"label_{r}"]
        out[f"{label}_id"] = pd.to_numeric(df[f"{p}_idx"]).astype("Int64")
        line_col = f"{p}_idx_line"
        out[f"{label}_line"] = (df[line_col].astype("Int64") if line_col in df
                                else pd.array([None] * len(df), dtype="Int64"))
        # token fields come from the file itself when the line is known; a column the cache dropped
        # because it was constant (e.g. every subject nsubj:pass) would otherwise come out empty
        files = df["conllu_file"] if "conllu_file" in df else [None] * len(df)
        raw_cols = [raw_token(f, ln) for f, ln in zip(files, out[f"{label}_line"])]
        for (src, name), idx in zip(TOKEN_FIELDS, RAW_COLUMN):
            col = f"{p}_{src}"
            cached = df[col].tolist() if col in df else [None] * len(df)
            out[f"{label}_{name}"] = [
                (rc[idx] if rc[idx] != "_" or idx == 1 else "") if rc else c for rc, c in zip(raw_cols, cached)]
        out[f"{label}_{feat}"] = df[f"{p}_{feat}"] if f"{p}_{feat}" in df else None
        cols = feat_cols[r]
        out[f"{label}_feats"] = [feats_string(dict(zip((c[len(p) + 1:] for c in cols), vals)))
                                 for vals in zip(*(df[c] for c in cols))] if cols else ""
        raw = [rc[5] if rc and rc[5] != "_" else "" for rc in raw_cols]
        out[f"{label}_raw_feats"] = raw
        out[f"{label}_{feat}_keys"] = [raw_keys_of(r, feat) for r in raw]
    out["_order"] = out["label"].map(LABEL_ORDER.index)
    out = out.sort_values(["_order", "treebank", "sent_id"], kind="stable").drop(columns="_order")
    return out


def write(out: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, sep="\t", index=False)


def export_vp(lang, cached_lang, config, treebanks, out_dir, exclude):
    summary = []
    for family in config["vp"]["families"]:
        cond = CONDITIONS[f"{family}Na"]
        path = os.path.join(cond.word_order_dir(), f"{cached_lang}.parquet")
        if not os.path.exists(path):
            print(f"{family}: no cache at {path}")
            continue
        deprel = cond.deprel
        pattern = re.compile(rf"head_{deprel}_({FEAT_RE})_agreement")
        feats = sorted({m.group(1) for c in pq.read_schema(path).names if (m := pattern.fullmatch(c))}
                       - exclude - NOT_AGREEMENT)
        names = {"a": "head", "b": deprel, "label_a": "head", "label_b": deprel}
        for feat in feats:
            table = build_table(path, f"head_{deprel}_{feat}_agreement", "head", deprel, feat, names, treebanks)
            summary.append(finish(table, vp_target_id(family, feat), lang, out_dir))
    return summary


def export_np(lang, cached_lang, treebanks, out_dir, exclude):
    path = os.path.join(NP_INSTANCES_DIR, f"{cached_lang}.parquet")
    if not os.path.exists(path):
        print(f"NP: no cache at {path}")
        return []
    summary = []
    pattern = re.compile(rf"([A-Z]+)-([A-Z]+)_({FEAT_RE})")
    for col in sorted(pq.read_schema(path).names):
        m = pattern.fullmatch(col)
        if not m or m.group(3) in exclude:
            continue
        a, b, feat = m.groups()
        names = {"a": a, "b": b, "label_a": a.lower(), "label_b": b.lower()}
        table = build_table(path, col, a, b, feat, names, treebanks)
        summary.append(finish(table, col, lang, out_dir))
    return summary


def finish(table, condition, lang, out_dir):
    labeled = int(table["label"].isin(["agree", "disagree"]).sum()) if len(table) else 0
    row = {"condition": condition, "rows": len(table), **{k: int((table["label"] == k).sum()) if len(table) else 0 for k in LABEL_ORDER},
           "exported": bool(labeled)}
    if labeled:
        write(table, os.path.join(out_dir, lang, f"{condition}.tsv"))
        print(f"{condition}: {len(table)} rows ({row['agree']} agree, {row['disagree']} disagree)")
    return row


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle_template")


def make_bundle(lang, tsv_dir, n_conditions, dist_dir, port, treebanks, extra_path):
    """dist_dir/multiblimp-{lang}-debug/ (tsvs, html site, serve.py, README) and its zip."""
    if not os.path.isdir(HTML_DECISION_TREES_DIR):
        print(f"{lang}: no html at {HTML_DECISION_TREES_DIR}; not bundling")
        return None
    name = f"multiblimp-{lang.lower().replace(' ', '_')}-debug"
    root = os.path.join(dist_dir, name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "export"))
    for fn in sorted(glob.glob(os.path.join(tsv_dir, "*.tsv"))):
        shutil.copy2(fn, os.path.join(root, "export"))
    shutil.copytree(HTML_DECISION_TREES_DIR, os.path.join(root, "site", "decision_trees"))
    with open(os.path.join(root, "site", "index.html"), "w", encoding="utf-8") as f:
        f.write('<!doctype html>\n<meta charset="utf-8">\n<title>Decision trees</title>\n'
                '<meta http-equiv="refresh" content="0; url=decision_trees/">\n'
                '<a href="decision_trees/">Decision trees overview</a>\n')
    with open(os.path.join(TEMPLATE_DIR, "serve.py"), encoding="utf-8") as f:
        serve = f.read().replace("__PORT__", str(port))
    with open(os.path.join(root, "serve.py"), "w", encoding="utf-8") as f:
        f.write(serve)
    extra = ""
    if extra_path and os.path.exists(extra_path):
        with open(extra_path, encoding="utf-8") as f:
            extra = "\n" + f.read().strip() + "\n"
    with open(os.path.join(TEMPLATE_DIR, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    for key, value in {
        "{lang}": lang, "{port}": str(port), "{n_conditions}": str(n_conditions), "{extra}": extra,
        "{treebanks_title}": f", {' + '.join(treebanks)} only" if treebanks else "",
        "{treebanks_sentence}": f", {' + '.join(treebanks)} only" if treebanks else "",
    }.items():
        readme = readme.replace(key, value)
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    archive = shutil.make_archive(root, "zip", dist_dir, name)
    print(f"{lang}: bundle {root}\n{lang}: zip    {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=AGREEMENT_CONFIG_PATH)
    parser.add_argument("--langs", "-l", nargs="*", default=None)
    parser.add_argument("--treebanks", nargs="*", default=[], help="Only these treebanks ('Talbanken').")
    parser.add_argument("--vp", action="store_true")
    parser.add_argument("--np", action="store_true")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "export"))
    parser.add_argument("--no_bundle", action="store_true", help="Only write the TSVs.")
    parser.add_argument("--dist", default=os.path.join(RUN_DIR, "dist"), help="Where the bundle and its zip go.")
    parser.add_argument("--port", type=int, default=8746, help="Default port of the bundle's serve.py.")
    parser.add_argument("--readme_extra", default=os.path.join(RUN_DIR, "readme_extra.md"),
                        help="Markdown appended to the bundle's README (default: {run dir}/readme_extra.md if present).")
    args = parser.parse_args()
    config = load_agreement_config(args.config)
    langs = args.langs or config["langs"]
    if not langs:
        sys.exit("--langs is required (or set langs in the config)")
    do_vp, do_np = (args.vp or not args.np), (args.np or not args.vp)
    exclude = set(lexical_features())
    for lang in langs:
        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        summary = []
        if do_vp:
            summary += export_vp(lang, cached_lang, config, args.treebanks, args.out, exclude)
        if do_np:
            summary += export_np(lang, cached_lang, args.treebanks, args.out, exclude)
        everything = pd.DataFrame([r for r in summary if r["rows"]])
        for stale in glob.glob(os.path.join(args.out, lang, "*.tsv.gz")):
            os.remove(stale)
        if len(everything) and not everything["unk_other"].any():
            everything = everything.drop(columns="unk_other")
        summary = everything[everything["exported"]].drop(columns="exported") if len(everything) else everything
        tried = everything[~everything["exported"]].drop(columns="exported") if len(everything) else everything
        os.makedirs(os.path.join(args.out, lang), exist_ok=True)
        for name, frame in (("_summary.tsv", summary), ("_not_exported.tsv", tried)):
            path = os.path.join(args.out, lang, name)
            if len(frame):
                frame.to_csv(path, sep="\t", index=False)
            elif os.path.exists(path):
                os.remove(path)
        print(f"{lang}: {len(summary)} condition(s) exported to {os.path.join(args.out, lang)}")
        if len(summary) and not args.no_bundle:
            make_bundle(lang, os.path.join(args.out, lang), len(summary), args.dist, args.port,
                        args.treebanks, args.readme_extra)


if __name__ == "__main__":
    main()
