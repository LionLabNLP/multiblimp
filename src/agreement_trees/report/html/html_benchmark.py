"""The benchmark page (html/index.html, the main page of the site): how many pairs of the final benchmark each language and condition
holds, how the sampling changed the marker distributions, and links back to the decision-tree and overview pages.
Rendered from the stats.json and manifest.json that agreement_trees.eval.benchmark.build_benchmark writes; the command
line is scripts/benchmark/render_page.py."""
import difflib
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multiblimp.downsample import condition_roles
from agreement_trees.taxonomy import (
    FEATURE_SUFFIXES, GROUP_PREFIXES, NPA_FEATURE_NAMES, OTHER_VERB_FORMS, other_verb_form, NPA_ROLE_ORDER, NPA_ROLE_PROSE, classify_flat_condition, npa_subgroup_label,
)

OVERVIEW_HREF = "data.html"
TREES_HREF = "decision_trees/index.html"


ARG_ORDER = ["Subject", "Object", "Indirect Object", "Noun Phrase", "Other"]
LEGACY_COMBINED = "Other verb forms (combined)"   # tables from before the per-VerbForm families (sxNa, sxCase, ...)
HEAD_ORDER = ["Verb", "Participle", "Auxiliary", *(label for _, label in OTHER_VERB_FORMS.values()), LEGACY_COMBINED]
FEATURE_ORDER = ["Number", "Gender", "Person", "Case"]
NPA_FEATURE_RANK = {"N": 0, "G": 1, "P": 2, "C": 3}


def _rank(order: list, value: str) -> int:
    return order.index(value) if value in order else len(order)


def condition_info(cond: str) -> dict:
    """Header parts of a benchmark condition (a verb family or a noun-phrase role pair): group and label for the
    tables, tree_dir for the links, and the three header tiers category / head / feature with a sort key."""
    form = other_verb_form(cond) or (
        (LEGACY_COMBINED, FEATURE_SUFFIXES.get(cond[2:], cond[2:])) if cond.startswith("sx") and cond[2:3].isupper() else None)
    if form:
        head, feature = form
        fixed = feature in FEATURE_SUFFIXES.values()   # like svCase, a Case family is an ad-hoc candidate and sits under "Other"
        category = "Subject" if fixed else "Other"
        key = (_rank(ARG_ORDER, category), 0, _rank(HEAD_ORDER, head), _rank(FEATURE_ORDER, feature), feature)
        return {"group": "Subject–Other verb forms" if fixed else "Other", "label": f"Subject–Other verb forms · {head} ({feature})",
                "tree_dir": cond, "category": category, "head": head if fixed else f"Subject–{head}", "feature": feature, "sort_key": key}
    flat = classify_flat_condition(cond)
    if flat:
        group, label = flat[0], flat[1]
        if group != "Other":
            arg, head = group.split("\u2013")
            feature = label
            category = arg
        else:
            head_full = max((g for g in GROUP_PREFIXES.values() if label.startswith(g + " ")), key=len)
            arg, head = head_full.split("\u2013")
            feature, category = label[len(head_full) + 1:], "Other"
        key = (_rank(ARG_ORDER, category), _rank(ARG_ORDER, arg), _rank(HEAD_ORDER, head), _rank(FEATURE_ORDER, feature), feature)
        shown = head_full if category == "Other" else head
        return {"group": group, "label": f"{group} \u00b7 {label}" if group != "Other" else label, "tree_dir": cond,
                "category": category, "head": shown, "feature": feature, "sort_key": key}
    roles, _, suffix = cond.partition("_")
    first, _, second = roles.partition("-")
    feature = NPA_FEATURE_NAMES.get(suffix, suffix)
    key = (ARG_ORDER.index("Noun Phrase"), NPA_ROLE_ORDER.get(first, 9), NPA_ROLE_ORDER.get(second, 9), NPA_FEATURE_RANK.get(suffix, 9), feature)
    pair = f"{NPA_ROLE_PROSE.get(first, first)}\u2013{NPA_ROLE_PROSE.get(second, second)}"
    return {"group": "Noun Phrase", "label": f"Noun Phrase \u00b7 {npa_subgroup_label(cond)}", "tree_dir": f"npa/{cond}",
            "category": "Noun Phrase", "head": pair, "feature": feature, "sort_key": key}




def _bare(text: str) -> str:
    """Text without marks (vowel points, cantillation), punctuation and spaces, casefolded: a token and its written
    word compare equal when only such marks differ."""
    return "".join(ch for ch in unicodedata.normalize("NFD", str(text)).casefold() if unicodedata.category(ch)[0] not in "MPZ")


def _align_by_text(tokens, words):
    """Word index of every token when the words are the tokens' texts run together (marks and punctuation aside:
    Ancient Hebrew writes cantillation and maqqef that the tokens lack); None if they are not."""
    bare = [_bare(t) for t in tokens]
    mapping, pos = [None] * len(tokens), 0
    for wi, word in enumerate(words):
        target, acc = _bare(word), ""
        if not target:
            if pos < len(tokens) and not bare[pos]:
                mapping[pos] = wi
                pos += 1
                continue
            return None
        while pos < len(tokens) and len(acc) < len(target):
            acc += bare[pos]
            mapping[pos] = wi
            pos += 1
            if not target.startswith(acc):
                return None
        if acc != target:
            return None
    return mapping if all(not b for b in bare[pos:]) else None


def controller_word(tokens, words, token_idx):
    """Position in `words` (the written sentence split on spaces) of the 1-based token `token_idx` of `tokens`: the same
    position when the two agree; else the aligned word, or, when the token sits in a stretch the written form merged or
    respelled (in das -> ins, a glued quote), the words that stretch became (an int, or a list for several words);
    None if there is none."""
    if token_idx is None or pd.isna(token_idx):
        return None
    t = int(token_idx) - 1
    tokens = list(tokens)
    if not 0 <= t < len(tokens):
        return None
    if tokens == words:
        return t
    mapping = _align_by_text(tokens, words)
    if mapping is not None and mapping[t] is not None:
        return mapping[t]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, tokens, words, autojunk=False).get_opcodes():
        if i1 <= t < i2:
            if tag == "equal":
                return j1 + (t - i1)
            if tag == "replace" and 0 < j2 - j1 <= 3:
                return j1 if j2 - j1 == 1 else list(range(j1, j2))
            return None
    return None


def write_pair_files(bench_dir: str, out_dir: str) -> dict:
    """One JSON file per language (out_dir/<Language>.json) with every benchmark pair of the language, for the
    pair browser of the page: {"language", "conditions": {cond: {label, dir, roles, n}}, "pairs": [...]}. Pair keys
    are short: c condition, id pair_id, s/w the sentence and its swapped version, fv original -> swap value,
    uf swapped feature, sk swapped role, tb treebank, sid sent_id, sc second chance, ss written-form status,
    at attractor class, d distance in tokens, la/lb the two lemmas, ci the controller's word position(s) in s. Returns {language: number of pairs}."""
    os.makedirs(out_dir, exist_ok=True)
    columns = ["pair_id", "phenomenon", "deprel", "treebank", "sent_id", "sen_tokens", "sen", "wrong_sen", "feature_vals",
               "swap_ufeat", "swap_kind", "second_chance", "surface_status", "attractors", "idx_a", "idx_b", "lemma_a", "lemma_b"]
    counts = {}
    for path in sorted(Path(bench_dir).glob("*.parquet")):
        lang = path.stem
        df = pq.read_table(path, columns=[c for c in columns if c in set(pq.ParquetFile(path).schema_arrow.names)]).to_pandas()
        conditions, pairs = {}, []
        for (cond, deprel), group in df.groupby(["phenomenon", "deprel"], sort=True):
            roles = condition_roles(cond, deprel)
            info = condition_info(cond)
            conditions[cond] = {"label": info["label"], "dir": f"features/{lang}.parquet", "roles": list(roles), "n": int(len(group))}
            for _, row in group.iterrows():
                idx_a, idx_b = (pd.to_numeric(row.get(c), errors="coerce") for c in ("idx_a", "idx_b"))
                gap = abs(idx_a - idx_b)
                swapped = str(row.get("swap_kind"))
                other = [i for i, r in enumerate(roles) if r.lower() != swapped.lower()]
                controller_idx = (idx_a, idx_b)[other[0]] if len(other) == 1 else None
                ci = controller_word(row["sen_tokens"], str(row["sen"]).split(), controller_idx) if controller_idx is not None else None
                pairs.append({
                    "c": cond, "id": row["pair_id"], "s": row["sen"], "w": row["wrong_sen"], "fv": row["feature_vals"],
                    "uf": row.get("swap_ufeat"), "sk": row.get("swap_kind"), "tb": row["treebank"], "sid": str(row["sent_id"]),
                    "sc": int(bool(row.get("second_chance"))), "ss": row.get("surface_status"),
                    "at": row.get("attractors"), "d": None if pd.isna(gap) else int(gap),
                    "la": row.get("lemma_a"), "lb": row.get("lemma_b"), "ci": ci,
                })
        clean = lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        pairs = [{k: clean(v) for k, v in p.items()} for p in pairs]
        with open(os.path.join(out_dir, f"{lang}.json"), "w", encoding="utf-8") as f:
            json.dump({"language": lang, "conditions": conditions, "pairs": pairs}, f, ensure_ascii=False, separators=(",", ":"))
        counts[lang] = len(pairs)
    return counts


def render_benchmark_page(stats: dict, manifest: dict) -> str:
    infos = {cell["cond"]: condition_info(cell["cond"]) for cell in stats["cells"]}
    rank = {cond: i for i, cond in enumerate(sorted(infos, key=lambda c: infos[c]["sort_key"]))}
    cells = [{**cell, **{k: v for k, v in infos[cell["cond"]].items() if k != "sort_key"}, "order": rank[cell["cond"]]}
             for cell in stats["cells"]]
    data = {
        "n_items": stats["n_items"], "lemma_cap": stats["lemma_cap"], "seed": stats["seed"], "built": stats["built"],
        "method": manifest.get("method", ""), "languages": stats["languages"], "cells": cells,
        "min_pairs": manifest.get("min_pairs"), "source": manifest.get("source"), "left_out": manifest.get("languages_left_out", {}),
        "cleaning": {**manifest.get("cleaning", {}), "cross_condition": manifest.get("cross_condition_repeats_removed")}, "overview_href": OVERVIEW_HREF, "trees_href": TREES_HREF,
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    return TEMPLATE.replace("__DATA__", blob)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>
  try { var t = localStorage.getItem('sva-dt-theme'); if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t); } catch (e) {}
</script>
<title>MultiBLiMP – Agreement</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root { color-scheme: light; --bg:#f5f5f4; --card:#fff; --panel-bg:#fafaf9; --text:#1c1917; --text-muted:#78716c; --accent:#2563eb;
  --accent-soft:#eff6ff; --border:#e7e5e4; --hover:#f5f5f5; --bar-sel:#2563eb; --bar-avail:#cbd5e1; }
:root[data-theme="dark"] { color-scheme: dark; --bg:#16140f; --card:#221f19; --panel-bg:#1c1a15; --text:#f0ede6; --text-muted:#a39a8a;
  --accent:#6ea8ff; --accent-soft:#1e2b40; --border:#38332a; --hover:#2a261e; --bar-sel:#6ea8ff; --bar-avail:#4d4636; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#16140f; --card:#221f19; --panel-bg:#1c1a15;
  --text:#f0ede6; --text-muted:#a39a8a; --accent:#6ea8ff; --accent-soft:#1e2b40; --border:#38332a; --hover:#2a261e; --bar-sel:#6ea8ff; --bar-avail:#4d4636; } }
* { box-sizing: border-box; }
body { margin:0; font-family:'DM Sans',system-ui,sans-serif; background:var(--bg); color:var(--text); padding:2rem; }
.container { max-width:1400px; margin:0 auto; }
.card { background:var(--card); padding:1.75rem 2.25rem; border-radius:12px; border:1px solid var(--border); box-shadow:0 4px 24px rgba(0,0,0,.08); margin-bottom:1.5rem; }
header.card { position:relative; }
h1 { margin:0 0 .5rem; font-size:1.6rem; } h2 { margin:0 0 .75rem; font-size:1.2rem; }
p, li { line-height:1.55; } .muted { color:var(--text-muted); } .small { font-size:.85rem; }
a { color:var(--accent); } .nav { display:flex; gap:1.25rem; flex-wrap:wrap; margin-top:.75rem; font-weight:500; }
.theme-toggle { position:absolute; top:1rem; right:1rem; background:none; border:1px solid var(--border); color:var(--text); border-radius:8px; padding:.3rem .6rem; cursor:pointer; font-size:1rem; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; }
.tile { background:var(--panel-bg); border:1px solid var(--border); border-radius:10px; padding:1rem 1.1rem; }
.tile .v { font-size:1.7rem; font-weight:600; } .tile .l { color:var(--text-muted); font-size:.82rem; margin-top:.15rem; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--border); }
th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
th button { all:unset; cursor:pointer; font-weight:600; } th button:hover { color:var(--accent); }
tr.lang-row { cursor:pointer; } tr.lang-row:hover { background:var(--hover); }
tr.detail > td { background:var(--panel-bg); padding:.75rem 1rem 1rem 2rem; }
.scroll { max-height:38rem; overflow:auto; border:1px solid var(--border); border-radius:8px; }
.scroll thead th { position:sticky; top:0; background:var(--card); z-index:1; }
input[type=search], select { background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:.45rem .7rem; font:inherit; }
.controls { display:flex; gap:1rem; flex-wrap:wrap; align-items:center; margin-bottom:1rem; }
.markers { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:1.25rem; }
.marker { background:var(--panel-bg); border:1px solid var(--border); border-radius:10px; padding:1rem 1.2rem; }
.marker h3 { margin:0 0 .15rem; font-size:1rem; } .marker .hint { color:var(--text-muted); font-size:.8rem; margin-bottom:.6rem; }
.row { display:grid; grid-template-columns:9.5rem 1fr 6.5rem; gap:.6rem; align-items:center; margin:.28rem 0; font-size:.85rem; }
.row .name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bars { position:relative; height:1.15rem; } .bars i { position:absolute; left:0; display:block; height:.5rem; border-radius:3px; }
.bars .avail { top:0; background:var(--bar-avail); } .bars .sel { top:.62rem; background:var(--bar-sel); }
.row .pct { text-align:right; font-variant-numeric:tabular-nums; color:var(--text-muted); }
.legend { display:flex; gap:1.2rem; font-size:.82rem; color:var(--text-muted); margin-bottom:.8rem; }
.legend span::before { content:""; display:inline-block; width:.9rem; height:.5rem; border-radius:3px; margin-right:.4rem; background:var(--bar-sel); }
.legend span.avail::before { background:var(--bar-avail); }
.chip { display:inline-block; padding:.05rem .5rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-size:.78rem; }
.chips { display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.8rem; }
.chips .chip a { text-decoration:none; } .chips .chip b { font-weight:600; }
button#csvBtn { background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:.45rem .8rem; font:inherit; cursor:pointer; }
button#csvBtn:hover { border-color:var(--accent); color:var(--accent); }
.toggle { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.toggle button { all:unset; cursor:pointer; padding:.4rem .8rem; font-size:.85rem; } .toggle button.on { background:var(--accent); color:#fff; }
.mx { border-collapse:separate; border-spacing:0; font-size:.8rem; }
.mx th, .mx td { padding:.22rem .38rem; border-bottom:1px solid var(--border); white-space:nowrap; }
.mx thead th { background:var(--card); border-bottom:0; }
.mx thead th.band { white-space:normal; line-height:1.15; }
.mx thead tr:first-child th.band { background:var(--accent-soft); text-align:center; font-weight:600; border-left:2px solid var(--card); border-right:2px solid var(--card); }
.mx thead tr:nth-child(2) th.band { background:var(--accent-soft); text-align:center; font-weight:500; border-left:2px solid var(--card); border-right:2px solid var(--card); font-size:.74rem; }
.mx thead { position:sticky; top:0; z-index:3; background:var(--card); }
.mx thead th.feat { background:transparent; }
.scroll .mx thead th:not(.lang):not(.total):not(.feat) { position:static; }
.scroll .mx thead th.feat { position:relative; }
.mx thead th.feat { height:7.6rem; vertical-align:bottom; padding:0; width:1.9rem; min-width:1.9rem; position:relative; }
.mx thead th.feat > a { position:absolute; left:50%; bottom:.3rem; transform-origin:0 100%; transform:rotate(-58deg); white-space:nowrap; font-weight:400; font-size:.76rem; text-decoration:none; color:var(--text); }
.mx thead th.feat > a:hover { color:var(--accent); text-decoration:underline; }
.mx td.cell { text-align:right; font-variant-numeric:tabular-nums; min-width:1.9rem; }
.mx td.cell a { color:inherit; text-decoration:none; }
.mx tbody tr:hover td { filter:brightness(1.08); }
.mx td.lang, .mx th.lang { position:sticky; left:0; background:var(--card); z-index:2; text-align:left; font-weight:500; }
.mx td.total, .mx th.total { text-align:right; font-weight:600; position:sticky; right:0; background:var(--card); z-index:2; }
.pair { border:1px solid var(--border); border-radius:10px; padding:.7rem .95rem; margin:.6rem 0; background:var(--panel-bg); }
.pair .meta { color:var(--text-muted); font-size:.78rem; margin-bottom:.35rem; display:flex; flex-wrap:wrap; gap:.3rem .8rem; align-items:center; }
.pair .sent { margin:.15rem 0; line-height:1.5; } .pair .sent b.tag { display:inline-block; width:1.4rem; font-weight:700; }
.pair .good b.tag { color:#16a34a; } .pair .bad b.tag { color:#dc2626; }
mark.ctl, mark.ok, mark.bad { color:inherit; border-radius:3px; padding:0 .12rem; }
mark.ctl { background:rgba(250,204,21,.4); } mark.ok { background:rgba(34,197,94,.38); } mark.bad { background:rgba(239,68,68,.4); }
.legend2 { display:flex; gap:1rem; flex-wrap:wrap; font-size:.8rem; color:var(--text-muted); margin:.2rem 0 .6rem; }
.pair .foot { color:var(--text-muted); font-size:.74rem; margin-top:.3rem; }
.pager { margin-left:auto; display:inline-flex; gap:.5rem; align-items:center; }
.pager button, button#pCsv { background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:.35rem .7rem; font:inherit; cursor:pointer; }
.pager button:disabled { opacity:.4; cursor:default; }
.warn { color:#b45309; }
.topnav { position:sticky; top:0; z-index:10; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:.55rem 1rem; margin-bottom:1.5rem;
  display:flex; gap:.4rem 1.1rem; flex-wrap:wrap; align-items:center; box-shadow:0 2px 12px rgba(0,0,0,.06); font-size:.88rem; }
.topnav a { text-decoration:none; color:var(--text-muted); font-weight:500; } .topnav a:hover { color:var(--accent); }
.topnav .sep { flex:1; } .topnav a.strong { color:var(--accent); }
section.card { scroll-margin-top:4.5rem; }
.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:.9rem; counter-reset:step; margin:.4rem 0 1rem; }
.step { background:var(--panel-bg); border:1px solid var(--border); border-radius:10px; padding:.85rem 1rem; font-size:.88rem; counter-increment:step; }
.step::before { content:counter(step); display:inline-block; width:1.5rem; height:1.5rem; line-height:1.5rem; text-align:center; border-radius:50%; background:var(--accent); color:#fff; font-weight:600; font-size:.8rem; margin-bottom:.4rem; }
.step b { display:block; margin-bottom:.15rem; }
.linkrow { display:flex; gap:.7rem; flex-wrap:wrap; }
.linkrow a { display:inline-block; padding:.45rem .9rem; border:1px solid var(--border); border-radius:8px; text-decoration:none; font-weight:500; background:var(--card); }
.linkrow a:hover { border-color:var(--accent); }
.notes { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; }
.note { background:var(--panel-bg); border:1px solid var(--border); border-radius:10px; padding:.9rem 1.1rem; font-size:.88rem; }
.note b { display:block; font-size:1.25rem; font-weight:600; } .note p { margin:.3rem 0 0; color:var(--text-muted); line-height:1.45; }
pre.code { background:var(--panel-bg); border:1px solid var(--border); border-radius:8px; padding:.8rem 1rem; overflow:auto; font-size:.8rem; line-height:1.45; }
code { font-size:.85em; background:var(--panel-bg); padding:.05rem .3rem; border-radius:4px; }
table.cols td:first-child { white-space:nowrap; font-family:ui-monospace,monospace; font-size:.82rem; }
@media (max-width:640px) { body { padding:1rem; } .card { padding:1.25rem; } .row { grid-template-columns:7rem 1fr 5.5rem; } .markers { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="container">
<header class="card">
  <h1>MultiBLiMP – Agreement</h1>
  <p class="muted" id="intro"></p>
  <div class="linkrow" style="margin-top:.9rem"><a id="navOverview" href="#">Data pre-downsampling &rarr;</a><a id="navTrees" href="#">Decision trees &rarr;</a></div>
  <button type="button" class="theme-toggle" id="themeBtn" title="Toggle light/dark theme">&#9790;</button>
</header>
<nav class="topnav" aria-label="Sections"><a href="#glance">At a glance</a><a href="#matrix">Pairs per language &amp; condition</a><a href="#languages">Languages</a><a href="#conditions">Conditions</a><a href="#pairs">Browse pairs</a><a href="#markers">Marker distributions</a><a href="#quality">Quality notes</a><a href="#use">Using the data</a><a href="#method">How the pairs were chosen</a></nav>

<section class="card" id="glance"><h2>At a glance</h2><div class="tiles" id="tiles"></div><p class="muted small" id="tilesNote" style="margin:.9rem 0 0"></p></section>

<section class="card" id="origin"><h2>Where the pairs come from</h2>
  <div class="steps">
    <div class="step"><b>Treebanks</b>Universal Dependencies sentences, with the written text and any multiword tokens.</div>
    <div class="step"><b>Agreement instances</b>Every head–dependent pair of a condition (verb and subject, noun and adjective, …) with its morphological features.</div>
    <div class="step"><b>Decision trees</b>Per language and condition a tree learns when the two tokens agree; only agreeing configurations are kept, with their leaf recorded.</div>
    <div class="step"><b>Minimal pairs</b>The dependent is reinflected for another feature value, and the written forms of both sentences are built.</div>
    <div class="step"><b>This benchmark</b>A balanced sample of those pairs, up to <span id="stepN"></span> per language.</div>
  </div>
  <div class="linkrow"><a id="navOverview3" href="#">Data overview: samples, pairs, diagnostics &rarr;</a><a id="navTrees3" href="#">Decision trees per language and condition &rarr;</a></div>
</section>

<section class="card" id="matrix"><h2>Pairs per language and condition</h2>
  <p class="muted small" id="matrixHint"></p>
  <div class="controls"><label>Category <select id="matrixGroup"></select></label>
    <div class="toggle" id="viewToggle" role="radiogroup" aria-label="View"><button type="button" class="on" data-view="matrix">Matrix</button><button type="button" data-view="list">List</button></div>
    <div class="toggle" id="modeToggle" role="radiogroup" aria-label="Count"><button type="button" class="on" data-mode="selected">In benchmark</button><button type="button" data-mode="available">Available</button></div>
    <input type="search" id="matrixFilter" placeholder="Filter languages…" aria-label="Filter languages">
    <button type="button" id="csvBtn">Download all counts (CSV)</button></div>
  <div class="scroll" id="matrixWrap"></div>
</section>

<section class="card" id="languages"><h2>Languages</h2>
  <div class="controls"><input type="search" id="langFilter" placeholder="Filter languages…" aria-label="Filter languages"><span class="muted small">Click a row for its conditions and links to the decision trees.</span></div>
  <div class="scroll"><table id="langTable"><thead><tr id="langHead"></tr></thead><tbody></tbody></table></div>
</section>

<section class="card" id="conditions"><h2>Conditions</h2>
  <div class="scroll"><table id="condTable"><thead><tr id="condHead"></tr></thead><tbody></tbody></table></div>
</section>

<section class="card" id="pairs"><h2>Browse the pairs</h2>
  <p class="muted small">Every pair of the benchmark, per language and condition: the original sentence and its swapped version, with the swapped word (green originally, red reinflected) and the token it agrees with (yellow) marked. Cells of the matrix and the "pairs" links open it here.</p>
  <div class="controls"><label>Language <select id="pLang"></select></label><label>Condition <select id="pCond"></select></label><label>Treebank <select id="pTb"></select></label>
    <label><input type="checkbox" id="pSecond"> second chance only</label><input type="search" id="pSearch" placeholder="Search sentence or lemma…" aria-label="Search pairs"></div>
  <div class="controls"><span class="chip" id="pCount"></span><span id="pLinks" class="small"></span>
    <button type="button" id="pCsv">Download shown pairs (CSV)</button><span class="pager"><button type="button" id="pPrev">&larr; Prev</button> <span id="pPage" class="small"></span> <button type="button" id="pNext">Next &rarr;</button></span></div>
  <div class="legend2"><span><mark class="ctl">controller</mark> the token the swapped word agrees with</span><span><mark class="ok">correct</mark> original token</span><span><mark class="bad">reinflected</mark> swapped token</span></div>
  <div id="pList"></div>
  <p class="muted small" id="pFoot"></p>
</section>

<section class="card" id="markers"><h2>Marker distributions</h2>
  <p class="muted small">The sampler balances on these markers. Each bar shows the share of the unique pairs available before downsampling (grey) and in the benchmark (blue), summed over the chosen scope; bars are scaled to the longest one in their panel.</p>
  <div class="controls"><label>Language <select id="selLang"></select></label><label>Condition <select id="selCond"></select></label><span class="chip" id="scopeCount"></span></div>
  <div class="legend"><span class="avail">available</span><span>in the benchmark</span></div>
  <div class="markers" id="markerGrid"></div>
</section>

<section class="card" id="quality"><h2>Quality notes</h2>
  <p class="muted small">What to keep in mind when reading results on this benchmark. All numbers are computed from the build.</p>
  <div class="notes" id="notes"></div>
</section>

<section class="card" id="use"><h2>Using the data</h2>
  <p class="small">The benchmark is a directory with two parquet files per language, joined by <code>pair_id</code>, next to a <code>manifest.json</code> (settings, treebank counts before and after, languages left out) and a <code>stats.json</code> (the counts and distributions on this page).</p>
  <pre class="code" id="layout"></pre>
  <table class="cols"><thead><tr><th>Column</th><th>Meaning</th></tr></thead><tbody>
    <tr><td>phenomenon, deprel</td><td>The condition of the pair (such as <code>svNa</code>) and its relation (<code>nsubj</code>; empty for noun-phrase conditions).</td></tr>
    <tr><td>pair_id</td><td>Stable id of the pair (language, condition, relation and the swapped token), the same across rebuilds with the same source.</td></tr>
    <tr><td>sen / wrong_sen</td><td>The original sentence and the sentence with the swapped word (the pair a model is scored on), as written (contractions such as <i>du</i> or <i>ins</i> restored).</td></tr>
    <tr><td>sen_hash</td><td>Hash of the original sentence, to spot edits to the treebank.</td></tr>
    <tr><td>feature_vals</td><td>Original and swapped value, such as <code>Sing -&gt; Plur</code>; <code>|</code> lists values the swapped form also spells.</td></tr>
    <tr><td>swap_kind, swap_idx, swap_&lt;kind&gt;</td><td>Which token was reinflected, its position in the sentence, and the new form.</td></tr>
    <tr><td>leaf_id, second_chance</td><td>The decision-tree leaf the pair stems from, and whether its fit was accepted at the laxer second-chance accuracy.</td></tr>
    <tr><td>surface_status</td><td>How the written form was built (plain, contracted, merged, uncontracted, …).</td></tr>
    <tr><td>alt_phenomena</td><td>Other phenomena in which the same sentence pair was found; the pair is kept only under the rarest of them.</td></tr>
    <tr><td>treebank, sent_id</td><td>Where the sentence comes from.</td></tr>
  </tbody></table>
  <p class="small"><code>&lt;Language&gt;.parquet</code> holds these columns plus the attractor flags, the lemmas, POS tags, order and distance of both tokens (<code>idx_a</code>/<code>idx_b</code>, <code>word_order</code>, <code>distance</code>). <code>features/&lt;Language&gt;.parquet</code> holds the same pairs with every tree and feature column of the source tables.</p>
  <pre class="code">import pandas as pd
pairs = pd.read_parquet("output/benchmark/German.parquet")
pairs[["sen", "wrong_sen", "feature_vals", "phenomenon"]].head()</pre>
  <p class="small muted" id="provenance"></p>
</section>
<section class="card" id="method"><h2>How the pairs were chosen</h2>
  <ul id="methodList"></ul>
</section>
</div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const pct = (n, d) => d ? (100 * n / d).toFixed(1) + '%' : '–';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

for (const s of ['', '3']) { $('navOverview' + s).href = DATA.overview_href; $('navTrees' + s).href = DATA.trees_href; }
$('stepN').textContent = DATA.n_items;
$('themeBtn').addEventListener('click', () => {
  const dark = (document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  const next = dark ? 'light' : 'dark'; document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('sva-dt-theme', next); } catch (e) {}
});

const cells = DATA.cells, langs = DATA.languages;
const langNames = Object.keys(langs).sort();
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const totalAvail = sum(cells.map(c => c.available)), totalSel = sum(cells.map(c => c.selected));
const condIds = [...new Set(cells.map(c => c.cond))];
const atCap = langNames.filter(l => !langs[l].kept_all).length;
const relaxed = langNames.filter(l => langs[l].lemma_cap_count && langs[l].lemma_cap_count > Math.ceil(DATA.lemma_cap * DATA.n_items)).length;
const ambiguousSel = sum(cells.map(c => c.ambiguous_selected || 0)), ambiguousAvail = sum(cells.map(c => c.ambiguous || 0));
const secondSel = sum(cells.map(c => (c.selected_markers.second_chance || {})['True'] || 0));

$('intro').textContent = `The final benchmark: up to ${DATA.n_items} minimal pairs per language, drawn across all of the language's verb and noun-phrase agreement conditions. Built ${DATA.built}, seed ${DATA.seed}.`;
const tiles = [[langNames.length, 'languages'], [condIds.length, 'conditions with pairs'], [fmt(totalSel), 'pairs in the benchmark'],
  [fmt(totalAvail), 'unique pairs available before downsampling'], [atCap, `languages sampled down to ${DATA.n_items}`],
  [langNames.length - atCap, 'languages kept in full'], [pct(secondSel, totalSel), 'pairs from second-chance fits'], [pct(ambiguousSel, totalSel), 'pairs with an ambiguous swap value']];
$('tiles').innerHTML = tiles.map(([v, l]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
const cl = DATA.cleaning || {};
const cleaningText = cl.written ? `The fits wrote ${fmt(cl.written)} pairs, which the data overview counts. Each sentence pair is counted once here, so ${fmt(cl.written - totalAvail)} are not unique pairs available for the benchmark: ${[
  [cl.identical, 'pairs whose two written sentences are identical (a contraction absorbed the swap, so they cannot be scored)'],
  [cl.repeated_in_condition, 'repeats of a sentence pair within a condition (the same sentence under several ids, or in overlapping treebanks)'],
  [cl.cross_condition, 'sentence pairs found in several conditions of a language, kept only in the condition with the fewest pairs (the others are listed in <code>alt_phenomena</code>)'],
  [cl.no_written_form, 'pairs without a valid written form'],
  [cl.left_out_languages, `pairs of languages with fewer than ${DATA.min_pairs} pairs in total`],
].filter(([n]) => n).map(([n, t]) => `${fmt(n)} ${t}`).join('; ')}.` : '';
if (cleaningText) $('tilesNote').innerHTML = cleaningText;
const capCount = Math.ceil(DATA.lemma_cap * DATA.n_items);
$('methodList').innerHTML = [
  ...(cleaningText ? ['<b>Unique pairs.</b> ' + cleaningText] : []),
  `A language with ${DATA.n_items} pairs or fewer keeps all of them; larger ones are sampled down to ${DATA.n_items}.`,
  `Pairs whose swap value is ambiguous (for example <code>Masc → Fem|Neut</code>, where the swapped form also spells another value) are used last, only once a condition has no other pair left, and count for each of their values in equal parts, so one such pair is never counted twice (${pct(ambiguousAvail, totalAvail)} of the available pairs, ${pct(ambiguousSel, totalSel)} of the benchmark).`,
  `The conditions share the sample equally: the condition with the fewest pairs so far goes next, so a small condition simply runs out and the others take over its share.`,
  'Within a condition each pair is drawn with a weight that falls with every time its value is already in the sample (original → swap value, role order, attractor class, part of speech of both tokens, distance, sentence length, decision-tree leaf, treebank, both lemmas), so the sample flattens on all markers at once.',
  `A lemma may appear in at most ${(100 * DATA.lemma_cap).toFixed(0)}% of a language's sample (${capCount} of ${DATA.n_items}) and a sentence at most once; if the quota cannot be filled the lemma limit rises ${(100 * DATA.lemma_cap).toFixed(0)}% at a time` + (relaxed ? ` (${relaxed} languages needed it)` : '') + ', and only then may sentences repeat.',
  ...(DATA.min_pairs && Object.keys(DATA.left_out).length ? [`Languages with fewer than ${DATA.min_pairs} pairs in total are left out (${Object.keys(DATA.left_out).length}: ${Object.entries(DATA.left_out).map(([l, n]) => l.replace(/_/g, ' ') + ' ' + n).join(', ')}).`] : []),
  'Every column of the source tables is kept, including <code>second_chance</code> (fits accepted at the laxer second-chance accuracy), and each pair gets a stable <code>pair_id</code> and a <code>sen_hash</code> of its sentence.',
].map(t => `<li>${t}</li>`).join('');
(function () {
  const fewer = langNames.filter(l => langs[l].kept_all);
  const relaxedL = langNames.filter(l => langs[l].lemma_cap_count && langs[l].lemma_cap_count > Math.ceil(DATA.lemma_cap * DATA.n_items));
  const repeatL = langNames.filter(l => (langs[l].sentence_limit || 1) > 1);
  const oneCond = langNames.filter(l => langs[l].conditions === 1);
  const secondL = langNames.filter(l => sum(cells.filter(c => c.lang === l).map(c => (c.selected_markers.second_chance || {})['True'] || 0)) > 0);
  const med = (xs) => { const a = xs.slice().sort((x, y) => x - y); return a.length ? a[Math.floor(a.length / 2)] : 0; };
  const list = (ls) => ls.length > 12 ? ls.slice(0, 12).map(l => l.replace(/_/g, ' ')).join(', ') + ', …' : ls.map(l => l.replace(/_/g, ' ')).join(', ');
  const note = (v, title, text) => `<div class="note"><b>${v}</b>${title}<p>${text}</p></div>`;
  $('notes').innerHTML = [
    note(`${fewer.length} of ${langNames.length}`, 'languages kept in full', `They have ${DATA.n_items} usable pairs or fewer, so nothing was sampled away and their distributions are the natural ones${fewer.length ? ': ' + list(fewer) : ''}.`),
    note(`${oneCond.length}`, 'languages with a single condition', `Their sample covers one agreement relation only${oneCond.length ? ': ' + list(oneCond) : ''}. Median across languages: ${med(langNames.map(l => langs[l].conditions))} conditions.`),
    note(pct(ambiguousSel, totalSel), 'pairs with an ambiguous swap value', `The swapped form also spells another value (<code>Masc → Fem|Neut</code>). They are used only when a condition has nothing else left, so what the model sees is a small, deliberate minority (${pct(ambiguousAvail, totalAvail)} of all available).`),
    note(pct(secondSel, totalSel), 'pairs from second-chance fits', `Trees accepted at the laxer second-chance accuracy, in ${secondL.length} language${secondL.length === 1 ? '' : 's'}. Filter on <code>second_chance</code> to leave them out.`),
    note(`${relaxedL.length}`, 'languages with a relaxed lemma limit', `Their sample could not be filled with at most ${Math.ceil(DATA.lemma_cap * DATA.n_items)} pairs per lemma${relaxedL.length ? ': ' + list(relaxedL) : ''}. Sentences repeat in ${repeatL.length} language${repeatL.length === 1 ? '' : 's'}.`),
    note(`${DATA.left_out ? Object.keys(DATA.left_out).length : 0}`, 'languages left out', `Fewer than ${DATA.min_pairs} pairs in total${DATA.left_out && Object.keys(DATA.left_out).length ? ': ' + Object.keys(DATA.left_out).map(l => l.replace(/_/g, ' ')).join(', ') : ''}.`),
  ].join('');
  $('layout').textContent = 'output/benchmark/\n  manifest.json  stats.json\n  <Language>.parquet            one row per pair: sentences, phenomenon, balance markers\n  features/<Language>.parquet   the same pairs with all tree and feature columns (join on pair_id)\n  meta/<Language>.json          coverage figures of each condition';
  $('provenance').textContent = `Built ${DATA.built} from ${DATA.source || 'the correct_swaps tables'}, seed ${DATA.seed}.`;
})();

function treeLink(c, lang) { return `${DATA.trees_href.replace('index.html', '')}${c.tree_dir}/${lang}.html`; }
function pairHash(lang, cond) { return `#pairs/${encodeURIComponent(lang)}` + (cond ? `/${encodeURIComponent(cond)}` : ''); }
function condLink(c) { return `${DATA.trees_href.replace('index.html', '')}${c.tree_dir}/index.html`; }

// ---- sortable tables
function makeTable(headId, bodyId, columns, rows, rowHtml, detailHtml) {
  const head = $(headId), body = document.querySelector('#' + bodyId + ' tbody') || $(bodyId).querySelector('tbody');
  let key = columns.find(c => c.sort).key, dir = -1, filter = '';
  head.innerHTML = columns.map(c => `<th scope="col" class="${c.num ? 'num' : ''}"><button type="button" data-key="${c.key}">${c.label}</button></th>`).join('');
  head.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return;
    if (key === b.dataset.key) dir = -dir; else { key = b.dataset.key; dir = columns.find(c => c.key === key).num ? -1 : 1; } draw(); });
  function draw() {
    const shown = rows.filter(r => !filter || r.name.toLowerCase().includes(filter))
      .sort((a, b) => (typeof a[key] === 'string' ? a[key].localeCompare(b[key]) : (a[key] ?? -1) - (b[key] ?? -1)) * dir);
    body.innerHTML = shown.map(r => rowHtml(r) + (detailHtml ? `<tr class="detail" hidden data-for="${esc(r.name)}"><td colspan="${columns.length}">${detailHtml(r)}</td></tr>` : '')).join('');
  }
  body.addEventListener('click', (e) => { if (e.target.closest('a')) return; const tr = e.target.closest('tr.lang-row'); if (!tr) return;
    const d = tr.nextElementSibling; if (d && d.classList.contains('detail')) d.hidden = !d.hidden; });
  draw();
  return (f) => { filter = f.toLowerCase(); draw(); };
}

const langRows = langNames.map(l => ({ name: l, ...langs[l], cap: langs[l].kept_all ? 0 : langs[l].lemma_cap_count,
  second: sum(cells.filter(c => c.lang === l).map(c => (c.selected_markers.second_chance || {})['True'] || 0)) }));
const setLangFilter = makeTable('langHead', 'langTable', [
  { key: 'name', label: 'Language' }, { key: 'available', label: 'Available', num: 1 }, { key: 'selected', label: 'In benchmark', num: 1, sort: 1 },
  { key: 'conditions', label: 'Conditions', num: 1 }, { key: 'treebanks', label: 'Treebanks', num: 1 }, { key: 'head_lemmas', label: 'Head lemmas', num: 1 },
  { key: 'child_lemmas', label: 'Child lemmas', num: 1 }, { key: 'cap', label: 'Lemma limit', num: 1 }, { key: 'second', label: '2nd chance', num: 1 }],
  langRows,
  (r) => `<tr class="lang-row"><td>${esc(r.name.replace(/_/g, ' '))}</td><td class="num">${fmt(r.available)}</td><td class="num">${fmt(r.selected)}</td><td class="num">${r.conditions}</td><td class="num">${r.treebanks}</td><td class="num">${r.head_lemmas}</td><td class="num">${r.child_lemmas}</td><td class="num">${r.kept_all ? 'all kept' : r.cap}</td><td class="num">${r.second || ''}</td></tr>`,
  (r) => { const mine = cells.filter(c => c.lang === r.name && c.selected).sort((a, b) => b.selected - a.selected);
    return `<div class="chips">` + mine.map(c => `<span class="chip" title="${esc(c.label)}">${esc(c.cond)} <b>${c.selected}</b></span>`).join('') + `</div>` +
      `<table><thead><tr><th>Condition</th><th class="num">Available</th><th class="num">In benchmark</th><th>Pairs and decision tree</th></tr></thead><tbody>` +
      mine.map(c => `<tr><td>${esc(c.label)}</td><td class="num">${fmt(c.available)}</td><td class="num">${fmt(c.selected)}</td><td><a href="${pairHash(r.name, c.cond)}">pairs</a> · <a href="${treeLink(c, r.name)}">decision tree</a> · <a href="${condLink(c)}">all languages</a></td></tr>`).join('') + '</tbody></table>'; });
$('langFilter').addEventListener('input', (e) => setLangFilter(e.target.value));

const condRows = condIds.map(id => { const mine = cells.filter(c => c.cond === id); const c0 = mine[0];
  return { name: c0.label, id, group: c0.group, link: condLink(c0), languages: new Set(mine.filter(c => c.selected).map(c => c.lang)).size,
    available: sum(mine.map(c => c.available)), selected: sum(mine.map(c => c.selected)) }; });
makeTable('condHead', 'condTable', [{ key: 'name', label: 'Condition' }, { key: 'languages', label: 'Languages', num: 1 },
  { key: 'available', label: 'Available', num: 1 }, { key: 'selected', label: 'In benchmark', num: 1, sort: 1 }, { key: 'link', label: 'Decision trees' }],
  condRows, (r) => `<tr><td>${esc(r.name)}</td><td class="num">${r.languages}</td><td class="num">${fmt(r.available)}</td><td class="num">${fmt(r.selected)}</td><td><a href="${r.link}">overview</a></td></tr>`);

// ---- pairs per language x condition: matrix (numbers under category / head / feature headers) and list
const CATS = ['Subject', 'Object', 'Indirect Object', 'Noun Phrase', 'Other'];
const view = { view: 'matrix', mode: 'selected', sortKey: 'selected', sortDir: -1 };
const nCond = (g) => new Set(cells.filter(c => c.category === g).map(c => c.cond)).size;
$('matrixGroup').innerHTML = `<option value="">All categories (${condIds.length} conditions)</option>` +
  CATS.filter(g => nCond(g)).map(g => `<option value="${esc(g)}">${esc(g)} (${nCond(g)} condition${nCond(g) === 1 ? '' : 's'})</option>`).join('');
const tierBands = (cols, key) => { const out = []; for (const c of cols) { const last = out[out.length - 1];
  if (last && last.key === key(c)) last.n++; else out.push({ key: key(c), label: c.__label || '', n: 1, c }); } return out; };

function drawMatrix() {
  const group = $('matrixGroup').value, mode = view.mode, filter = $('matrixFilter').value.toLowerCase().replace(/ /g, '_');
  const inScope = cells.filter(c => !group || c.category === group);
  const byCond = {}; for (const c of inScope) byCond[c.cond] = c;
  const conds = Object.values(byCond).sort((a, b) => a.order - b.order);
  const value = {}; for (const c of inScope) value[c.lang + '|' + c.cond] = c;
  const rows = langNames.filter(l => l.toLowerCase().includes(filter) && inScope.some(c => c.lang === l && (mode === 'available' ? c.available : c.selected)));
  const colMax = {}; for (const c of conds) colMax[c.cond] = Math.max(1, ...rows.map(l => (value[l + '|' + c.cond] || {})[mode] || 0));
  const bands = (key, label) => { const out = []; for (const c of conds) { const k = key(c), last = out[out.length - 1];
    if (last && last.k === k) last.n++; else out.push({ k, n: 1, label: label(c) }); } return out; };
  const SHORT = { 'Participle': 'Part.', 'Auxiliary': 'Aux' };
  const tier = (list) => list.map(b => `<th class="band" colspan="${b.n}">${esc(b.n < 3 && SHORT[b.label] ? SHORT[b.label] : b.label).replace('–', '–\u200b')}</th>`).join('');
  const head = `<tr><th class="lang" rowspan="3">Language</th>${tier(bands(c => c.category, c => c.category))}<th class="total" rowspan="3">Total</th></tr>` +
    `<tr>${tier(bands(c => c.category + '|' + c.head, c => c.head))}</tr>` +
    `<tr>${conds.map(c => `<th class="feat" title="${esc(c.label)}"><a href="${condLink(c)}">${esc(c.feature)}</a></th>`).join('')}</tr>`;
  const body = rows.map(l => { let total = 0;
    const tds = conds.map(cd => { const c = value[l + '|' + cd.cond]; const n = c ? c[mode] : 0; total += n;
      if (!n) return '<td class="cell"></td>';
      const a = 0.1 + 0.45 * Math.sqrt(n / colMax[cd.cond]);
      return `<td class="cell" style="background:color-mix(in srgb, var(--accent) ${(a * 100).toFixed(0)}%, transparent)" title="${esc(l.replace(/_/g, ' '))} · ${esc(c.label)}: ${c.selected} in the benchmark of ${c.available} available (click to browse the pairs)"><a href="${pairHash(l, c.cond)}">${fmt(n)}</a></td>`; }).join('');
    return `<tr><td class="lang">${esc(l.replace(/_/g, ' '))}</td>${tds}<td class="total">${fmt(total)}</td></tr>`; }).join('');
  $('matrixWrap').innerHTML = `<table class="mx"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

const LIST_COLUMNS = [['lang', 'Language'], ['label', 'Condition'], ['selected', 'In benchmark', 1], ['available', 'Available', 1], ['kept', '% kept', 1], ['second', '2nd chance', 1], ['link', 'Pairs · decision tree']];
function drawList() {
  const group = $('matrixGroup').value, filter = $('matrixFilter').value.toLowerCase().replace(/ /g, '_');
  const rows = cells.filter(c => (!group || c.category === group) && c.lang.toLowerCase().includes(filter) && (view.mode === 'available' ? c.available : c.selected))
    .map(c => ({ ...c, kept: c.available ? c.selected / c.available : 0, second: (c.selected_markers.second_chance || {})['True'] || 0 }));
  const key = view.sortKey, dir = view.sortDir;
  rows.sort((a, b) => (typeof a[key] === 'string' ? a[key].localeCompare(b[key]) : (a[key] ?? 0) - (b[key] ?? 0)) * dir || a.lang.localeCompare(b.lang) || a.order - b.order);
  $('matrixWrap').innerHTML = `<table><thead><tr>${LIST_COLUMNS.map(([k, label, num]) => `<th scope="col" class="${num ? 'num' : ''}"><button type="button" data-key="${k}">${label}</button></th>`).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.lang.replace(/_/g, ' '))}</td><td>${esc(r.label)}</td><td class="num">${fmt(r.selected)}</td><td class="num">${fmt(r.available)}</td><td class="num">${(100 * r.kept).toFixed(r.kept < 0.01 ? 2 : 1)}%</td><td class="num">${r.second || ''}</td><td><a href="${pairHash(r.lang, r.cond)}">pairs</a> · <a href="${treeLink(r, r.lang)}">decision tree</a> · <a href="${condLink(r)}">all languages</a></td></tr>`).join('') + '</tbody></table>';
  $('matrixWrap').querySelector('thead').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return;
    if (view.sortKey === b.dataset.key) view.sortDir = -view.sortDir; else { view.sortKey = b.dataset.key; view.sortDir = ['lang', 'label', 'link'].includes(b.dataset.key) ? 1 : -1; } drawList(); });
}
function drawPairs() {
  $('matrixHint').textContent = view.view === 'matrix'
    ? 'Pairs per language (rows) and condition (columns, grouped by category and head; the slanted names are the agreement feature, hover for the full condition). Hover a cell for the unique pairs available before downsampling, click it to browse that language\'s pairs. Blank cells have no pairs.'
    : 'One row per language and condition with pairs; sort by any column.';
  (view.view === 'matrix' ? drawMatrix : drawList)();
}
for (const [id, key, attr] of [['viewToggle', 'view', 'view'], ['modeToggle', 'mode', 'mode']]) {
  $(id).addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; view[key] = b.dataset[attr];
    $(id).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); drawPairs(); });
}
$('matrixGroup').addEventListener('change', drawPairs); $('matrixFilter').addEventListener('input', drawPairs); drawPairs();
$('csvBtn').addEventListener('click', () => {
  const lines = ['language,condition,category,available,in_benchmark,ambiguous_available,ambiguous_in_benchmark,second_chance_in_benchmark'].concat(cells.slice().sort((a, b) => a.lang.localeCompare(b.lang) || a.cond.localeCompare(b.cond)).map(c =>
    [c.lang, c.cond, c.category, c.available, c.selected, c.ambiguous || 0, c.ambiguous_selected || 0, (c.selected_markers.second_chance || {})['True'] || 0].map(v => /[",]/.test(String(v)) ? `"${v}"` : v).join(',')));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' })); a.download = 'benchmark_pairs_per_language_condition.csv'; a.click();
});

// ---- pair browser (one JSON per language under pairs/, loaded on demand)
const PAGE = 50;
const pairs = { lang: null, data: null, shown: [], page: 0, cache: {} };
const langsWithPairs = langNames.filter(l => langs[l].selected);
$('pLang').innerHTML = langsWithPairs.map(l => `<option value="${esc(l)}">${esc(l.replace(/_/g, ' '))} (${langs[l].selected})</option>`).join('');

function wordDiff(a, b, ci) {
  const x = a.split(/\s+/), y = b.split(/\s+/);
  let i = 0; while (i < x.length && i < y.length && x[i] === y[i]) i++;
  let j = 0; while (j < x.length - i && j < y.length - i && x[x.length - 1 - j] === y[y.length - 1 - j]) j++;
  const cx = new Set(ci == null ? [] : [].concat(ci)), cy = new Set();
  for (const c of cx) { if (c < i) cy.add(c); else if (c >= x.length - j) cy.add(c - x.length + y.length); }
  const mark = (t, from, to, cls, ctl) => t.map((w, k) => ctl.has(k) ? `<mark class="ctl">${esc(w)}</mark>` : k >= from && k < to ? `<mark class="${cls}">${esc(w)}</mark>` : esc(w)).join(' ');
  return [mark(x, i, x.length - j, 'ok', cx), mark(y, i, y.length - j, 'bad', cy)];
}
function pairCard(p) {
  const info = pairs.data.conditions[p.c] || {}; const [g, bad] = wordDiff(p.s || '', p.w || '', p.ci);
  const chips = [p.sc ? '<span class="chip">second chance</span>' : '', p.at && p.at !== 'none' && p.at !== 'unknown' ? `<span class="chip">attractors: ${esc(p.at.replace('_', ' '))}</span>` : '',
    p.ss && !['plain', 'mwt_elsewhere'].includes(p.ss) ? `<span class="chip">written form: ${esc(p.ss)}</span>` : ''].join(' ');
  return `<div class="pair"><div class="meta"><b>${esc(info.label || p.c)}</b><span>${esc(p.fv || '')}${p.uf ? ' (' + esc(p.uf) + ')' : ''}</span><span>${esc(p.tb)} · ${esc(p.sid)}</span>${chips}</div>` +
    `<div class="sent good"><b class="tag">✓</b>${g}</div><div class="sent bad"><b class="tag">✗</b>${bad}</div>` +
    `<div class="foot">${esc((info.roles || ['a', 'b'])[0])}: ${esc(p.la || '–')} · ${esc((info.roles || ['a', 'b'])[1])}: ${esc(p.lb || '–')}${p.d != null ? ' · distance ' + p.d : ''} · swapped ${esc(p.sk || '')} · <span title="pair_id">${esc(p.id)}</span></div></div>`;
}
function drawPairs2() {
  if (!pairs.data) return;
  const cond = $('pCond').value, tb = $('pTb').value, second = $('pSecond').checked, q = $('pSearch').value.toLowerCase();
  pairs.shown = pairs.data.pairs.filter(p => (!cond || p.c === cond) && (!tb || p.tb === tb) && (!second || p.sc) &&
    (!q || (p.s || '').toLowerCase().includes(q) || (p.w || '').toLowerCase().includes(q) || (p.la || '').includes(q) || (p.lb || '').includes(q)));
  const pages = Math.max(1, Math.ceil(pairs.shown.length / PAGE)); pairs.page = Math.min(pairs.page, pages - 1);
  const from = pairs.page * PAGE;
  $('pList').innerHTML = pairs.shown.slice(from, from + PAGE).map(pairCard).join('') || '<p class="muted">No pairs match.</p>';
  $('pCount').textContent = `${fmt(pairs.shown.length)} of ${fmt(pairs.data.pairs.length)} pairs`;
  $('pPage').textContent = `page ${pairs.page + 1} of ${pages}`; $('pPrev').disabled = pairs.page === 0; $('pNext').disabled = pairs.page >= pages - 1;
  const c = cond ? pairs.data.conditions[cond] : null, cell = cond ? cells.find(x => x.lang === pairs.lang && x.cond === cond) : null;
  $('pLinks').innerHTML = `<a href="pairs/${encodeURIComponent(pairs.lang)}.json" download>language JSON</a>` + (cell ? ` · <a href="${treeLink(cell, pairs.lang)}">decision tree</a> · <a href="${condLink(cell)}">condition overview</a>` : '');
  $('pFoot').textContent = c ? `All columns of these pairs: output/benchmark/${c.dir} (phenomenon ${cond})` : 'All columns of the pairs: output/benchmark/features/<Language>.parquet';
}
async function loadLanguage(lang, cond) {
  pairs.lang = lang; $('pLang').value = lang;
  try {
    if (!pairs.cache[lang]) { const r = await fetch(`pairs/${encodeURIComponent(lang)}.json`); if (!r.ok) throw new Error(r.status); pairs.cache[lang] = await r.json(); }
  } catch (e) { $('pList').innerHTML = `<p class="warn">The pair file pairs/${esc(lang)}.json could not be loaded (${esc(e.message)}). Serve the page over http (the files sit next to it), a file:// page cannot fetch them.</p>`; return; }
  pairs.data = pairs.cache[lang]; pairs.page = 0;
  const conds = Object.entries(pairs.data.conditions).sort((a, b) => b[1].n - a[1].n);
  $('pCond').innerHTML = '<option value="">All conditions</option>' + conds.map(([k, v]) => `<option value="${esc(k)}">${esc(v.label)} (${v.n})</option>`).join('');
  $('pCond').value = cond && pairs.data.conditions[cond] ? cond : '';
  const tbs = [...new Set(pairs.data.pairs.map(p => p.tb))].sort();
  $('pTb').innerHTML = '<option value="">All treebanks</option>' + tbs.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  drawPairs2();
}
function routePairs() {
  const m = decodeURIComponent(location.hash).match(/^#pairs\/([^/]+)(?:\/(.+))?$/); if (!m || !langs[m[1]]) return;
  loadLanguage(m[1], m[2]).then(() => $('pairs').scrollIntoView());
}
window.addEventListener('hashchange', routePairs);
$('pLang').addEventListener('change', () => { location.hash = pairHash($('pLang').value); });
$('pCond').addEventListener('change', () => { pairs.page = 0; drawPairs2(); });
$('pTb').addEventListener('change', () => { pairs.page = 0; drawPairs2(); });
$('pSecond').addEventListener('change', () => { pairs.page = 0; drawPairs2(); });
$('pSearch').addEventListener('input', () => { pairs.page = 0; drawPairs2(); });
$('pPrev').addEventListener('click', () => { pairs.page--; drawPairs2(); }); $('pNext').addEventListener('click', () => { pairs.page++; drawPairs2(); });
$('pCsv').addEventListener('click', () => {
  const cols = ['c', 'id', 's', 'w', 'fv', 'uf', 'sk', 'tb', 'sid', 'sc', 'ss', 'at', 'd', 'la', 'lb'];
  const head = ['condition', 'pair_id', 'sentence', 'swapped_sentence', 'feature_vals', 'swap_ufeat', 'swapped_role', 'treebank', 'sent_id', 'second_chance', 'written_form_status', 'attractors', 'distance', 'lemma_a', 'lemma_b'];
  const q = (v) => v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v;
  const text = [head.join(',')].concat(pairs.shown.map(p => cols.map(c => q(p[c])).join(','))).join('\n') + '\n';
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' })); a.download = `benchmark_pairs_${pairs.lang}.csv`; a.click();
});
if (location.hash.startsWith('#pairs/')) routePairs(); else if (langsWithPairs.length) loadLanguage(langsWithPairs[0]).then(() => {});

// ---- marker distributions
const MARKERS = [
  ['feature_vals', 'Original → swap value', 'the feature value of the pair and the one it was swapped to; the ten most frequent, the rest grouped'],
  ['word_order', 'Role order', 'which of the pair\'s two tokens comes first (verb families: the verb or the subject; noun phrases: the first or second role in the pair name)'],
  ['swap_kind', 'Reinflected token', 'which token of the pair was reinflected (only noun-phrase pairs of two dependents, such as determiner and adjective, can go either way)'],
  ['attractors', 'Attractors', 'tokens between the two that carry the feature, against the controller\'s value'],
  ['pos_a', 'UPOS of the first role', 'the head for verb families'],
  ['pos_b', 'UPOS of the second role', 'the controller (subject) for verb families'],
  ['distance', 'Distance in tokens', 'between the two tokens'],
  ['length', 'Sentence length', 'characters of the written sentence'],
  ['treebank', 'Treebank', 'the largest ones; the rest are grouped'],
  ['second_chance', 'Second-chance fit', 'True: the tree\'s leaf accuracy is 0.90–0.95 instead of at least 0.95'],
  ['surface_status', 'Written form', 'how the sentence was turned into its written form (contractions and the like)'],
];
const ROLE_LABELS = { a_first: 'first role first', b_first: 'second role first' };
const ORDER = { distance: ['1', '2', '3-4', '5-8', '9+'], length: ['up to 40', '41-80', '81-140', '141-240', '241+'], attractors: ['none', 'congruent', 'weakly_congruent', 'incongruent', 'mixed', 'unknown'] };
$('selLang').innerHTML = '<option value="">All languages</option>' + langNames.map(l => `<option value="${esc(l)}">${esc(l.replace(/_/g, ' '))}</option>`).join('');
const condOptions = [...condRows].sort((a, b) => a.name.localeCompare(b.name));
$('selCond').innerHTML = '<option value="">All conditions</option>' + condOptions.map(r => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');

function drawMarkers() {
  const lang = $('selLang').value, cond = $('selCond').value;
  const scope = cells.filter(c => (!lang || c.lang === lang) && (!cond || c.cond === cond));
  $('scopeCount').textContent = `${fmt(sum(scope.map(c => c.selected)))} of ${fmt(sum(scope.map(c => c.available)))} pairs`;
  $('markerGrid').innerHTML = MARKERS.map(([key, title, hint]) => {
    const avail = {}, sel = {};
    for (const c of scope) { for (const [k, v] of Object.entries(c.available_markers[key] || {})) avail[k] = (avail[k] || 0) + v;
      for (const [k, v] of Object.entries(c.selected_markers[key] || {})) sel[k] = (sel[k] || 0) + v; }
    let names = Object.keys(avail);
    if (!names.length) return '';
    const ranked = ORDER[key] ? ORDER[key].filter(n => names.includes(n)) : names.sort((a, b) => (avail[b] || 0) - (avail[a] || 0));
    let shown = ranked, rest = null;
    if ((key === 'treebank' || key === 'feature_vals') && ranked.length > 10) { shown = ranked.slice(0, 10); rest = ranked.slice(10); }
    const aT = sum(Object.values(avail)), sT = sum(Object.values(sel));
    const lines = shown.map(n => [n, avail[n] || 0, sel[n] || 0]);
    if (rest) lines.push([`${rest.length} others`, sum(rest.map(n => avail[n] || 0)), sum(rest.map(n => sel[n] || 0))]);
    const top = Math.max(...lines.map(([, a, s]) => Math.max(aT ? a / aT : 0, sT ? s / sT : 0)), 1e-9);
    return `<div class="marker"><h3>${title}</h3><div class="hint">${hint}</div>` + lines.map(([n, a, s]) =>
      `<div class="row"><span class="name" title="${esc(n)}">${esc(key === 'word_order' ? (ROLE_LABELS[n] || n) : n)}</span><span class="bars"><i class="avail" style="width:${aT ? 100 * a / aT / top : 0}%"></i><i class="sel" style="width:${sT ? 100 * s / sT / top : 0}%"></i></span><span class="pct">${pct(s, sT)} <span title="share before sampling">(${pct(a, aT)})</span></span></div>`).join('') + '</div>';
  }).join('');
}
$('selLang').addEventListener('change', drawMarkers); $('selCond').addEventListener('change', drawMarkers); drawMarkers();
</script>
</body>
</html>
"""
