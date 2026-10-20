"""Per-treebank decision trees for languages with several UD treebanks.

Each treebank gets its own tree; it diverges when that tree predicts its held-out rows clearly better than
the pooled tree (agreement_trees.fit.decision_tree.fit_dt on all treebanks) does. Analysis only: no minimal pairs,
so these pages carry no keep-threshold annotations (leaf_threshold=None). Each treebank (and the pooled
tree) can also get an incl.-unk tree, saved as {treebank}__unk / {lang}__unk. Excluded treebanks
(multiblimp.excluded_treebanks) get the same pages from a separate frame (extra_df); they never touch the
pooled tree, its comparison numbers or the pairs, and are tagged with their exclusion reason.
"""

import json
import os
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from multiblimp.excluded_treebanks import is_treebank_row_excluded

from agreement_trees.fit.decision_tree import UNK_LABELS, fit_dt
from agreement_trees.report.viz_tree import tree2html


def split_excluded(df: pd.DataFrame):
    """(included_df, excluded_df) halves of an all-treebanks frame, split by whether each row's treebank is
    excluded now (live config, so an exclusion change needs no re-extraction). excluded_df is None when
    there is no "treebank" column or nothing is excluded. The treebank's own embedded language decides
    (e.g. "Turkish_German" in Turkish's cache); astype(bool) because a categorical column's map is categorical."""
    if "treebank" not in df.columns or not len(df):
        return df, None
    mask = df["treebank"].map(is_treebank_row_excluded).astype(bool)
    excluded_df = df[mask]
    return df[~mask], (excluded_df if len(excluded_df) else None)


def restrict_treebanks(df: pd.DataFrame, names):
    """Rows of the named treebanks only ("UD_Swedish-Talbanken" or just "Talbanken");
    all rows when names is empty."""
    if not names or "treebank" not in df.columns:
        return df
    keep = df["treebank"].map(lambda t: any(t == n or t.endswith("-" + n) for n in names)).astype(bool)
    return df[keep]


TREEBANK_COL = "treebank"
UNK_SUFFIX = "__unk"
CACHE_VERSION = 7   # bump when the cached fits' layout or meaning changes


def _short(tb, lang):
    name = tb[3:] if tb.startswith("UD_") else tb
    prefix = f"{lang.replace(' ', '_')}-"
    return name[len(prefix):] if name.startswith(prefix) else name


def _unk_rows(entry):
    return sum(n for lab, n in entry["label_counts"].items() if lab in UNK_LABELS)


def _acc(eval_df):
    return float((eval_df["y"] == eval_df["pred"]).mean()) if len(eval_df) else None


def treebanks_done(summary_fn: str, include_excluded: bool, incl_unk: bool) -> bool:
    """The per-treebank summary (or the marker for a language without one)
    exists and was built with these settings and the current CACHE_VERSION."""
    if not os.path.exists(summary_fn):
        return False
    with open(summary_fn) as f:
        cached = json.load(f)
    return (isinstance(cached, dict)
            and cached.get("include_excluded", False) == include_excluded
            and cached.get("incl_unk", False) == incl_unk
            and cached.get("version") == CACHE_VERSION)


def write_treebanks_marker(summary_fn: str, include_excluded: bool, incl_unk: bool) -> None:
    """Mark a language with no per-treebank trees as handled (see treebanks_done)."""
    os.makedirs(os.path.dirname(summary_fn), exist_ok=True)
    with open(summary_fn, "w") as f:
        json.dump({"version": CACHE_VERSION, "include_excluded": include_excluded,
                   "incl_unk": incl_unk, "treebanks": []}, f)


def fit_treebank_trees(
    full_df, lang, pooled_model, predictor_var, target, cache_dir,
    fit_kwargs, impurity_fn, pooled_impurity, drop_unk=True,
    min_test=30, delta_threshold=0.05, alpha=0.05, never_skip=False,
    omit_feats_fn=None, extra_df=None, reason_fn=None, include_excluded=False,
    incl_unk=False,
):
    """Fit (or load from cache) one tree per treebank and compare each with the pooled tree on the same
    held-out rows. A treebank diverges when its tree is at least delta_threshold more accurate AND an exact sign
    test on the rows where exactly one tree is right gives p < alpha. Returns (summary, fresh), cached at
    {cache_dir}/{lang}.json; summary is {"version", "include_excluded", "incl_unk", "pooled": {"acc",
    "test_n"}, "pooled_unk": {...} | None, "treebanks": [entries]}. An entry's "unk" (like "pooled_unk") holds
    the incl.-unk tree's status and held-out accuracy (not compared with the pooled tree: other label set).

    incl_unk: also fit the incl.-unk trees (opt-in)."""
    summary_fn = os.path.join(cache_dir, f"{lang}.json")
    if not never_skip and os.path.exists(summary_fn):
        with open(summary_fn) as f:
            cached = json.load(f)
        if (isinstance(cached, dict) and cached.get("version") == CACHE_VERSION
                and cached.get("include_excluded") == include_excluded
                and cached.get("incl_unk") == incl_unk):
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
        # same gate as the pooled tree: attempted when the label column has more than one value (unk counts as
        # one); fit_dt refuses when too few usable rows remain
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
            unk_model, _, _, _ = fit_dt(
                full_df=sub, model_type="decision_tree", target=target, verbose=0,
                predictor_var=predictor_var, drop_unk=False,
                min_impurity_decrease=impurity_fn(len(sub)),
                save_to=os.path.join(tb_dir, tb + UNK_SUFFIX), **fit_kwargs, **omit)
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
        base_omit_feats = omit_feats_fn(full_df) if omit_feats_fn else None
        m, _, _, _ = fit_dt(
            full_df=full_df, model_type="decision_tree", target=target, verbose=0,
            predictor_var=predictor_var, drop_unk=False,
            min_impurity_decrease=pooled_impurity,
            save_to=os.path.join(cache_dir, lang + UNK_SUFFIX), **fit_kwargs,
            **({"omit_feats": base_omit_feats} if base_omit_feats is not None else {}))
        pooled_unk = ({"status": "ok", "test_n": int(len(m.test_eval_)), "test_acc": _acc(m.test_eval_)}
                      if m is not None and hasattr(m, "test_eval_") else {"status": "no tree"})

    summary = {"version": CACHE_VERSION, "include_excluded": include_excluded,
               "incl_unk": incl_unk,
               "pooled": pooled, "pooled_unk": pooled_unk, "treebanks": entries}
    os.makedirs(cache_dir, exist_ok=True)
    with open(summary_fn, "w") as f:
        json.dump(summary, f, indent=1)
    return summary, True



def page_stem(base, variant):
    """File stem of the as-is (variant None) or incl.-unk ("unk") page of a pooled tree or a treebank."""
    return base + UNK_SUFFIX if variant else base


def pooled_page_stem(lang):
    """Stem of the pooled as-is page (the one the pipeline itself writes)."""
    return page_stem(lang, None)


def _page(tb, variant):
    return f"{quote(page_stem(tb, variant))}.html"


def build_nav(summary, lang, current=None, variant=None):
    """Pill-row data for the tree page: the pooled tree first, then every
    treebank. current=None means the pooled page itself; variant="unk" is the
    incl.-unk view of that treebank (its pills stay in that view)."""
    enc = quote(lang)
    prefix = f"{enc}/" if current is None else ""
    pooled_unk = summary.get("pooled_unk") or {}
    pooled_unk_ok = pooled_unk.get("status") == "ok"
    pooled_href = None if current is None else (
        f"../{quote(page_stem(lang, 'unk'))}.html" if variant and pooled_unk_ok
        else f"../{quote(page_stem(lang, None))}.html")
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
        counterpart = (f"{quote(page_stem(lang, None))}.html" if variant else
                       f"{quote(page_stem(lang, 'unk'))}.html" if pooled_unk_ok else None)
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


def render_treebank_pages(summary, lang, full_df, cache_dir, html_dir, tree_html_kwargs, extra_df=None):
    tree_html_kwargs = {**tree_html_kwargs, "leaf_threshold": None, "full_tree_html": True}
    for e in summary["treebanks"]:
        tb = e["treebank"]
        sub = (extra_df if e.get("excluded") else full_df).pipe(lambda d: d[d[TREEBANK_COL] == tb])
        if e["status"] != "ok":
            # No tree for this treebank: a placeholder page (label counts, sample
            # rows) so its tab is still a link.
            kwargs = dict(tree_html_kwargs)
            placeholder_df = sub.copy()
            kwargs.update(
                pipeline_model=None, dt_df=placeholder_df, full_df=sub.copy(), full_tree_html=False,
                out_file=os.path.join(html_dir, lang, f"{page_stem(tb, None)}.html"),
                meta={"Language": lang, "Treebank": e["short"], "Status": e["status"],
                      "Excluded": e.get("reason") if e.get("excluded") else None},
                treebank_nav=build_nav(summary, lang, current=tb, variant=None),
            )
            tree2html(**kwargs)
        variants = [(None, e["status"] == "ok"), ("unk", (e.get("unk") or {}).get("status") == "ok")]
        for variant, ok in variants:
            if not ok:
                continue
            name = page_stem(tb, variant)
            base = os.path.join(cache_dir, lang, tb + UNK_SUFFIX if variant else tb)
            unk_extra = {"discounted_classes": None} if variant == "unk" else {}
            kwargs = dict(tree_html_kwargs)
            kwargs.update(
                pipeline_model=joblib.load(base + ".joblib"),
                dt_df=pd.read_parquet(base + ".parquet"),
                full_df=sub,
                out_file=os.path.join(html_dir, lang, f"{name}.html"),
                meta={"Language": lang, "Treebank": e["short"], "titleSuffix": "incl. unk" if variant else None,
                      "Excluded": e.get("reason") if e.get("excluded") else None},
                full_label_distribution=e["label_counts"],
                correct_swaps_df=None,
                treebank_nav=build_nav(summary, lang, current=tb, variant=variant),
            )
            kwargs.update(unk_extra)
            tree2html(**kwargs)

    if (summary.get("pooled_unk") or {}).get("status") == "ok":
        base = os.path.join(cache_dir, lang + UNK_SUFFIX)
        unk_extra = {"discounted_classes": None}
        kwargs = dict(tree_html_kwargs)
        kwargs.update(
            pipeline_model=joblib.load(base + ".joblib"),
            dt_df=pd.read_parquet(base + ".parquet"),
            full_df=full_df,
            out_file=os.path.join(html_dir, f"{page_stem(lang, 'unk')}.html"),
            meta={"Language": lang, "titleSuffix": "incl. unk"},
            correct_swaps_df=None,
            treebank_nav=build_nav(summary, lang, current=None, variant="unk"),
        )
        kwargs.update(unk_extra)
        tree2html(**kwargs)
