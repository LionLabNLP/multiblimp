"""Per-treebank decision trees for languages with several UD treebanks.

The pooled tree (word_order.decision_tree.fit_dt on every treebank at once)
can hide annotation differences between treebanks. Here each treebank gets
its own tree, and a treebank is tagged as diverging when its own tree
predicts its held-out rows clearly better than the pooled tree does.
Analysis only -- no minimal pairs are generated, so these pages carry no
keep-threshold/kept-pairs annotations (leaf_threshold=None).

Each treebank also gets a second, debugging-oriented tree fit on its unk rows
too (when it has any and the run normally drops them), saved and linked
alongside the as-is one as {treebank}__unk. The pooled tree gets the same
incl.-unk view ({lang}__unk).

Excluded treebanks (multiblimp.languages.excluded_treebanks) can get the same
pages from a separate feature df (extra_df): they never touch the pooled
tree, its comparison numbers, or the minimal pairs, and their pages are
tagged with the normalised exclusion reason.
"""

import json
import os
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from multiblimp.languages import is_treebank_row_excluded

from .decision_tree import UNK_LABELS, fit_dt, split_unk_reasons
from .viz_tree import tree2html, _display_predictor_var


def split_excluded(df: pd.DataFrame):
    """(included_df, excluded_df) halves of a unified all-treebanks df, split
    by whether each row's "treebank" is excluded RIGHT NOW (the live
    excluded_treebanks config), so an exclusion change needs no
    re-extraction. excluded_df is None when df has no "treebank" column or
    nothing in it is excluded.

    is_treebank_row_excluded checks the treebank's OWN embedded language
    (e.g. "Turkish_German" out of "UD_Turkish_German-SAGT"), not the
    language whose cache the row was read from -- see multiblimp.languages.
    _split_treebank_row. astype(bool): "treebank" may be categorical dtype,
    where Series.map returns a categorical and "~" on it raises TypeError."""
    if "treebank" not in df.columns or not len(df):
        return df, None
    mask = df["treebank"].map(is_treebank_row_excluded).astype(bool)
    excluded_df = df[mask]
    return df[~mask], (excluded_df if len(excluded_df) else None)

TREEBANK_COL = "treebank"
UNK_SUFFIX = "__unk"
# 7: incl.-unk trees can now split "unk" into why (unk_split) rather than
# fitting/rendering it as one merged class -- old cached fits are stale.
CACHE_VERSION = 7


def _unk_split_col(predictor_var: str) -> str:
    return f"{predictor_var}__unk_split"


def _with_unk_split(df: pd.DataFrame, predictor_var: str, unk_split: dict) -> pd.DataFrame:
    """`df` plus its unk-split label column (see decision_tree.split_unk_reasons),
    under a fixed, deterministic name -- computed identically at fit time
    (fit_treebank_trees) and render time (render_treebank_pages), so the two
    never need to agree via the JSON cache. `unk_split`'s own "palette" key
    (consumed by render_treebank_pages, not split_unk_reasons) is dropped
    before the call."""
    df = df.copy()
    split_kwargs = {k: v for k, v in unk_split.items() if k != "palette"}
    df[_unk_split_col(predictor_var)] = split_unk_reasons(df, predictor_var, **split_kwargs)
    return df


def _short(tb, lang):
    name = tb[3:] if tb.startswith("UD_") else tb
    prefix = f"{lang.replace(' ', '_')}-"
    return name[len(prefix):] if name.startswith(prefix) else name


def _unk_rows(entry):
    return sum(n for lab, n in entry["label_counts"].items() if lab in UNK_LABELS)


def _acc(eval_df):
    return float((eval_df["y"] == eval_df["pred"]).mean()) if len(eval_df) else None


def fit_treebank_trees(
    full_df, lang, pooled_model, predictor_var, target, cache_dir,
    fit_kwargs, impurity_fn, pooled_impurity, drop_unk=True,
    min_test=30, delta_threshold=0.05, alpha=0.05, never_skip=False,
    omit_feats_fn=None, extra_df=None, reason_fn=None, include_excluded=False,
    unk_split=None, incl_unk=False,
):
    """Fit (or load from cache) one tree per treebank and compare each with
    the pooled tree on the same held-out rows. A treebank diverges when its own
    tree is at least delta_threshold more accurate AND an exact sign test on
    the rows where exactly one of the two trees is right gives p < alpha (so a
    couple of rows on a tiny test set don't count). Returns (summary, fresh),
    cached at {cache_dir}/{lang}.json. summary is
    {"version", "include_excluded", "pooled": {"acc", "test_n"}, "pooled_unk": {"status", "test_acc",
    "test_n"} | None, "treebanks": [per-treebank entries]}; an entry's optional
    "unk" (like "pooled_unk") holds the incl.-unk tree's status and
    held-out accuracy (not compared with the pooled tree -- it predicts a
    different label set).

    unk_split: when given, every incl.-unk fit (per-treebank and pooled)
    predicts *why* a row is unk (see decision_tree.split_unk_reasons) --
    e.g. "Head unknown"/"Subject unknown"/"Both unknown" -- instead of
    fitting/rendering it as one merged class. A dict with keys col_a/
    label_a/col_b/label_b/both_label/other_label(/palette, read only by
    render_treebank_pages); None (default) keeps the old single-"unk"
    behavior.

    incl_unk: also fit the debugging-oriented incl.-unk tree (per-treebank
    and pooled) described above. Default False -- the caller opts in."""
    summary_fn = os.path.join(cache_dir, f"{lang}.json")
    if not never_skip and os.path.exists(summary_fn):
        with open(summary_fn) as f:
            cached = json.load(f)
        if (isinstance(cached, dict) and cached.get("version") == CACHE_VERSION
                and cached.get("include_excluded") == include_excluded
                and cached.get("incl_unk") == incl_unk
                and cached.get("unk_split") == (unk_split is not None)):
            return cached, False

    tb_dir = os.path.join(cache_dir, lang)
    entries, models = [], {}
    groups = [(tb, sub, False) for tb, sub in full_df.groupby(TREEBANK_COL, observed=True)]
    if extra_df is not None:
        groups += [(tb, sub, True) for tb, sub in extra_df.groupby(TREEBANK_COL, observed=True)]
    for tb, sub, excluded in groups:
        omit = {"omit_feats": omit_feats_fn(sub)} if omit_feats_fn else {}
        labels = sub[predictor_var]
        if drop_unk:
            labels = labels[~labels.isin(UNK_LABELS)]
        entry = {"treebank": tb, "short": _short(tb, lang), "rows": int(len(sub)),
                 "excluded": excluded, "reason": reason_fn(tb) if excluded and reason_fn else None,
                 "label_counts": {str(k): int(v) for k, v in sub[predictor_var].value_counts().items()}}
        # Same gate as the pooled tree (sva_trees.pipeline): a tree is attempted
        # whenever the label column has more than one value (unk counts as
        # one), and fit_dt itself refuses when too few usable rows remain
        # (its min_df_len). No separate size cutoff here.
        if sub[predictor_var].nunique() < 2:
            entry["status"] = "single label"
        else:
            model, _, _, _ = fit_dt(
                full_df=sub, model_type="decision_tree", target=target, verbose=0,
                predictor_var=predictor_var, drop_unk=drop_unk,
                min_impurity_decrease=impurity_fn(len(sub)),
                save_to=os.path.join(tb_dir, tb), **fit_kwargs, **omit)
            if model is None:
                entry["status"] = "too few samples"
            elif not hasattr(model, "test_eval_"):
                entry["status"] = "no tree"
            else:
                entry["status"] = "ok"
                entry["test_n"] = int(len(model.test_eval_))
                entry["test_acc"] = _acc(model.test_eval_)
                if not excluded:
                    models[tb] = model
        if incl_unk and drop_unk and sub[predictor_var].isin(UNK_LABELS).any() \
                and sub[predictor_var].nunique() >= 2:
            unk_sub = _with_unk_split(sub, predictor_var, unk_split) if unk_split else sub
            unk_predictor_var = _unk_split_col(predictor_var) if unk_split else predictor_var
            # unk_split's new column is DERIVED from predictor_var (relabels
            # its own unk rows), so the original column must be kept out of
            # the fit too -- otherwise it's a feature that trivially reveals
            # "unk or not" (fit_dt only ever excludes predictor_var itself
            # from being its own feature, not a *different* column this one
            # was built from).
            unk_omit = {"omit_feats": (omit.get("omit_feats") or set()) | {predictor_var}} if unk_split else omit
            unk_model, _, _, _ = fit_dt(
                full_df=unk_sub, model_type="decision_tree", target=target, verbose=0,
                predictor_var=unk_predictor_var, drop_unk=False,
                min_impurity_decrease=impurity_fn(len(sub)),
                save_to=os.path.join(tb_dir, tb + UNK_SUFFIX), **fit_kwargs, **unk_omit)
            if unk_model is None or not hasattr(unk_model, "test_eval_"):
                entry["unk"] = {"status": "no tree"}
            else:
                entry["unk"] = {"status": "ok", "test_n": int(len(unk_model.test_eval_)),
                                "test_acc": _acc(unk_model.test_eval_)}
        entries.append(entry)

    pooled = None
    if models and pooled_model is not None:
        held_out = np.concatenate([m.test_eval_.index.values for m in models.values()])
        pooled_holdout, _, _, _ = fit_dt(
            full_df=full_df, model_type="decision_tree", target=target, verbose=0,
            predictor_var=predictor_var, drop_unk=drop_unk,
            min_impurity_decrease=pooled_impurity, save_to=None,
            test_index=held_out, **fit_kwargs,
            **({"omit_feats": omit_feats_fn(full_df)} if omit_feats_fn else {}))
        pooled_eval = getattr(pooled_holdout, "test_eval_", None)
        if pooled_eval is not None:
            pooled = {"acc": _acc(pooled_eval), "test_n": int(len(pooled_eval))}
        for e in entries:
            if e["status"] != "ok" or e.get("excluded") or pooled_eval is None:
                continue
            rows = pooled_eval.index.intersection(models[e["treebank"]].test_eval_.index)
            e["pooled_acc"] = _acc(pooled_eval.loc[rows])
            e["delta"] = e["test_acc"] - e["pooled_acc"]
            own = models[e["treebank"]].test_eval_.loc[rows]
            own_right = (own["y"] == own["pred"]).values
            pooled_right = (pooled_eval.loc[rows, "y"] == pooled_eval.loc[rows, "pred"]).values
            own_only, pooled_only = int((own_right & ~pooled_right).sum()), int((~own_right & pooled_right).sum())
            p = binomtest(own_only, own_only + pooled_only, 0.5).pvalue if own_only + pooled_only else 1.0
            e["own_only"], e["pooled_only"], e["p"] = own_only, pooled_only, p
            e["diverges"] = bool(e["test_n"] >= min_test and e["delta"] >= delta_threshold and p < alpha)

    pooled_unk = None
    if incl_unk and drop_unk and full_df[predictor_var].isin(UNK_LABELS).any() \
            and full_df[predictor_var].nunique() >= 2:
        pooled_unk_df = _with_unk_split(full_df, predictor_var, unk_split) if unk_split else full_df
        pooled_unk_predictor_var = _unk_split_col(predictor_var) if unk_split else predictor_var
        base_omit_feats = omit_feats_fn(full_df) if omit_feats_fn else None
        pooled_unk_omit_feats = (
            (base_omit_feats or set()) | {predictor_var} if unk_split else base_omit_feats
        )
        m, _, _, _ = fit_dt(
            full_df=pooled_unk_df, model_type="decision_tree", target=target, verbose=0,
            predictor_var=pooled_unk_predictor_var, drop_unk=False,
            min_impurity_decrease=pooled_impurity,
            save_to=os.path.join(cache_dir, lang + UNK_SUFFIX), **fit_kwargs,
            **({"omit_feats": pooled_unk_omit_feats} if pooled_unk_omit_feats is not None else {}))
        pooled_unk = ({"status": "ok", "test_n": int(len(m.test_eval_)), "test_acc": _acc(m.test_eval_)}
                      if m is not None and hasattr(m, "test_eval_") else {"status": "no tree"})

    summary = {"version": CACHE_VERSION, "include_excluded": include_excluded,
               "incl_unk": incl_unk, "unk_split": unk_split is not None,
               "pooled": pooled, "pooled_unk": pooled_unk, "treebanks": entries}
    os.makedirs(cache_dir, exist_ok=True)
    with open(summary_fn, "w") as f:
        json.dump(summary, f, indent=1)
    return summary, True


def _page(tb, variant):
    return f"{quote(tb + UNK_SUFFIX if variant else tb)}.html"


def build_nav(summary, lang, current=None, variant=None):
    """Pill-row data for the tree page: the pooled tree first, then every
    treebank. current=None means the pooled page itself; variant="unk" is the
    incl.-unk debugging view of that treebank (its pills stay in that view)."""
    enc = quote(lang)
    prefix = f"{enc}/" if current is None else ""
    pooled_unk = summary.get("pooled_unk") or {}
    pooled_unk_ok = pooled_unk.get("status") == "ok"
    pooled_href = None if current is None else (
        f"../{enc}{UNK_SUFFIX}.html" if variant and pooled_unk_ok else f"../{enc}.html")
    items = []
    for e in summary["treebanks"]:
        unk = e.get("unk") or {}
        as_is_ok, unk_ok = e["status"] == "ok", unk.get("status") == "ok"
        items.append({
            "name": e["short"], "rows": e["rows"] - _unk_rows(e), "unkRows": _unk_rows(e),
            "status": e["status"], "excluded": e.get("excluded", False), "reason": e.get("reason"),
            "acc": e.get("test_acc"), "pooledAcc": e.get("pooled_acc"),
            "delta": e.get("delta"), "diverges": e.get("diverges", False),
            "href": prefix + _page(e["treebank"], None),
            "placeholder": not as_is_ok,
            "unkAcc": unk.get("test_acc"),
            "unkHref": prefix + _page(e["treebank"], "unk") if unk_ok else None,
            "current": e["treebank"] == current,
        })
    if current is None:
        counterpart = (f"{enc}.html" if variant else
                       f"{enc}{UNK_SUFFIX}.html" if pooled_unk_ok else None)
    else:
        cur = next(i for i in items if i["current"])
        counterpart = cur["href"] if variant else cur["unkHref"]
    pooled = summary.get("pooled") or {}
    return {"pooledAcc": pooled.get("acc"), "pooledTestN": pooled.get("test_n"),
            "pooledUnkAcc": pooled_unk.get("test_acc"), "pooledUnkTestN": pooled_unk.get("test_n"),
            "pooledRows": sum(e["rows"] - _unk_rows(e) for e in summary["treebanks"] if not e.get("excluded")),
            "pooledUnkRows": sum(_unk_rows(e) for e in summary["treebanks"] if not e.get("excluded")),
            "pooledHref": pooled_href, "pooledCurrent": current is None,
            "items": items, "variant": variant, "counterpartHref": counterpart,
            "overviewHref": "index.html" if current is None else "../index.html"}


def _unk_view_extra(df: pd.DataFrame, tree_html_kwargs: dict, unk_split: dict | None) -> tuple[dict, dict]:
    """(kwargs, extra_meta) overriding tree2html's defaults for one incl.-unk
    render, when unk_split is active: predictor_var switched to the split
    column (dt_df, loaded from the matching fit's cache, already carries it
    -- see _with_unk_split/fit_treebank_trees), full_label_distribution
    recomputed over that column's own classes, palette_map swapped to
    unk_split's own (omitted, so tree_html_kwargs' plain one applies, if
    unk_split sets none). extra_meta (a separate dict, NOT nested in kwargs
    -- the caller's own `meta` dict has Language/Treebank/titleSuffix/
    Excluded already and a plain kwargs.update would clobber it wholesale)
    pins Predictor to the ORIGINAL predictor_var's display string, so the
    title keeps reading e.g. "AUX_NSUBJ_PERSON", not the internal split
    column's own name.

    (kwargs, {}) when unk_split is None -- always safe to merge in.
    """
    if not unk_split:
        return {}, {}
    predictor_var = tree_html_kwargs["predictor_var"]
    split_col = _unk_split_col(predictor_var)
    split_df = _with_unk_split(df, predictor_var, unk_split)
    kwargs = {
        "predictor_var": split_col,
        "full_df": split_df,
        "full_label_distribution": {
            str(k): int(v) for k, v in split_df[split_col].value_counts().items()
        },
    }
    if unk_split.get("palette"):
        kwargs["palette_map"] = unk_split["palette"]
    extra_meta = {"Predictor": _display_predictor_var(predictor_var, tree_html_kwargs.get("head_label", "head"))}
    return kwargs, extra_meta


def render_treebank_pages(summary, lang, full_df, cache_dir, html_dir, tree_html_kwargs, extra_df=None,
                           unk_split=None):
    tree_html_kwargs = {**tree_html_kwargs, "leaf_threshold": None}
    for e in summary["treebanks"]:
        tb = e["treebank"]
        sub = (extra_df if e.get("excluded") else full_df).pipe(lambda d: d[d[TREEBANK_COL] == tb])
        if e["status"] != "ok":
            # No tree for this treebank: a placeholder page (label counts, sample
            # rows) so its tab is still a link.
            kwargs = dict(tree_html_kwargs)
            kwargs.update(
                pipeline_model=None, dt_df=sub.copy(), full_df=sub.copy(), full_tree_html=False,
                out_file=os.path.join(html_dir, lang, f"{tb}.html"),
                meta={"Language": lang, "Treebank": e["short"], "Status": e["status"],
                      "Excluded": e.get("reason") if e.get("excluded") else None},
                treebank_nav=build_nav(summary, lang, current=tb, variant=None),
            )
            tree2html(**kwargs)
        variants = [(None, e["status"] == "ok"), ("unk", (e.get("unk") or {}).get("status") == "ok")]
        for variant, ok in variants:
            if not ok:
                continue
            name = tb + UNK_SUFFIX if variant else tb
            base = os.path.join(cache_dir, lang, name)
            unk_extra, unk_extra_meta = (
                _unk_view_extra(sub, tree_html_kwargs, unk_split) if variant == "unk" else ({}, {})
            )
            kwargs = dict(tree_html_kwargs)
            kwargs.update(
                pipeline_model=joblib.load(base + ".joblib"),
                dt_df=pd.read_parquet(base + ".parquet"),
                full_df=sub,
                out_file=os.path.join(html_dir, lang, f"{name}.html"),
                meta={"Language": lang, "Treebank": e["short"], "titleSuffix": "incl. unk" if variant else None,
                      "Excluded": e.get("reason") if e.get("excluded") else None, **unk_extra_meta},
                full_label_distribution=e["label_counts"],
                correct_swaps_df=None,
                treebank_nav=build_nav(summary, lang, current=tb, variant=variant),
            )
            kwargs.update(unk_extra)
            tree2html(**kwargs)

    if (summary.get("pooled_unk") or {}).get("status") == "ok":
        base = os.path.join(cache_dir, lang + UNK_SUFFIX)
        unk_extra, unk_extra_meta = _unk_view_extra(full_df, tree_html_kwargs, unk_split)
        kwargs = dict(tree_html_kwargs)
        kwargs.update(
            pipeline_model=joblib.load(base + ".joblib"),
            dt_df=pd.read_parquet(base + ".parquet"),
            full_df=full_df,
            out_file=os.path.join(html_dir, f"{lang}{UNK_SUFFIX}.html"),
            meta={"Language": lang, "titleSuffix": "incl. unk", **unk_extra_meta},
            correct_swaps_df=None,
            treebank_nav=build_nav(summary, lang, current=None, variant="unk"),
        )
        kwargs.update(unk_extra)
        tree2html(**kwargs)
