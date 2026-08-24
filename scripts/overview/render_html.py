"""Render output/overview/stats.json (see build_stats.py) into a self-
contained html/index.html — samples/minimal-pair counts per language, per
condition, and per language x condition, for the sva-dt project.

Run from the repo root: python scripts/overview/render_html.py
(after build_stats.py, or use scripts/overview/run_all.sh for both)
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO_ROOT, "src"))
from multiblimp.config import HTML_DIR, OUTPUT_OVERVIEW_DIR  # noqa: E402

STATS_PATH = os.path.join(OUTPUT_OVERVIEW_DIR, "stats.json")
OUT_PATH = os.path.join(HTML_DIR, "index.html")

# Category tabs, in display order. Matches the sv/sp/sa/ov/npa grouping
# CONDITION_META (build_stats.py) already assigns each condition.
GROUP_ORDER = ["Subject–Verb", "Subject–Participle", "Subject–Auxiliary", "Noun Phrase"]
GROUP_SLUGS = {"Subject–Verb": "sv", "Subject–Participle": "sp", "Subject–Auxiliary": "sa", "Noun Phrase": "npa"}


def _category_items(data: dict, group: str) -> list[dict]:
    """Pickable conditions for one category tab, pairs-descending.

    Every group picks among its own top-level conditions (data["conditions"])
    except Noun Phrase: the matrix only tracks NPA down to the aggregate
    "npa" condition (summed over all role pairs), so a meaningful
    "pick a specific condition, e.g. Number" experience there means picking
    among data["npa_subgroups"] instead (each backed by its own
    language x role-pair breakdown in data["npa_matrix"]) -- "npa" itself is
    left out of the picker since its subgroups replace it.
    """
    if group == "Noun Phrase":
        items = [dict(s, isNpaSub=True) for s in data["npa_subgroups"]]
    else:
        items = [dict(c, isNpaSub=False) for c in data["conditions"] if c["group"] == group]
    items.sort(key=lambda it: -it["pairs"])
    return items


def _union_language_count(data: dict, cond_ids: list[str], npa_sub_ids: list[str]) -> int:
    seen = set()
    if cond_ids:
        conds = data["matrix"]["conditions"]
        idxs = [conds.index(c) for c in cond_ids]
        for i, lang in enumerate(data["matrix"]["languages"]):
            if any(data["matrix"]["pairs"][i][j] or data["matrix"]["samples"][i][j] for j in idxs):
                seen.add(lang)
    if npa_sub_ids:
        subs = data["npa_matrix"]["subgroups"]
        idxs = [subs.index(s) for s in npa_sub_ids]
        for i, lang in enumerate(data["npa_matrix"]["languages"]):
            if any(data["npa_matrix"]["pairs"][i][j] or data["npa_matrix"]["samples"][i][j] for j in idxs):
                seen.add(lang)
    return len(seen)


COVERAGE_THRESHOLDS = (10, 30, 50, 100, 500)


def _coverage_union(data: dict, items: list[dict], lang_idx: dict, npa_lang_idx: dict) -> dict:
    """Same shape as build_stats.py's per-condition coverage stats
    (tried/with_samples/ge10.../ge500), but summed across every item
    (condition or NPA subgroup) in one category tab first -- a language
    only clears "ge100" here if its combined total across the whole
    category does, not any single condition in it alone.
    """
    tried: set[str] = set()
    lang_samples: dict[str, int] = {}
    lang_pairs: dict[str, int] = {}
    for it in items:
        matrix = data["npa_matrix"] if it["isNpaSub"] else data["matrix"]
        idx_map = npa_lang_idx if it["isNpaSub"] else lang_idx
        col_key = "subgroups" if it["isNpaSub"] else "conditions"
        j = matrix[col_key].index(it["id"])
        for lang in it["tried_languages"]:
            tried.add(lang)
            i = idx_map.get(lang)
            if i is not None:
                lang_samples[lang] = lang_samples.get(lang, 0) + matrix["samples"][i][j]
                lang_pairs[lang] = lang_pairs.get(lang, 0) + matrix["pairs"][i][j]
    with_samples = sum(1 for lang in tried if lang_samples.get(lang, 0) > 0)
    ge = {t: sum(1 for lang in tried if lang_pairs.get(lang, 0) >= t) for t in COVERAGE_THRESHOLDS}
    zero_langs = sorted(lang for lang in tried if lang_samples.get(lang, 0) == 0)
    return {
        "tried": len(tried),
        "with_samples": with_samples,
        **{f"ge{t}": ge[t] for t in COVERAGE_THRESHOLDS},
        "zero_languages": zero_langs,
    }


def _category_totals(data: dict, group: str, items: list[dict], lang_idx: dict, npa_lang_idx: dict) -> dict:
    cond_ids = [it["id"] for it in items if not it["isNpaSub"]]
    npa_sub_ids = [it["id"] for it in items if it["isNpaSub"]]
    # Samples-weighted mean of each item's own already-weighted accuracy --
    # consistent with how build_stats.py rolls per-language accuracy up to
    # per-condition in the first place, just one level higher.
    acc_num = sum(it["acc"] * it["samples"] for it in items if it["acc"] is not None)
    acc_den = sum(it["samples"] for it in items if it["acc"] is not None)
    dropped = {}
    for it in items:
        for label, n in it["dropped_before_fitting"].items():
            dropped[label] = dropped.get(label, 0) + n
    return {
        "conditions": len(items),
        "languages": _union_language_count(data, cond_ids, npa_sub_ids),
        "samples": sum(it["samples"] for it in items),
        "pairs": sum(it["pairs"] for it in items),
        "acc": round(acc_num / acc_den, 3) if acc_den else None,
        "dropped_before_fitting": dropped,
        **_coverage_union(data, items, lang_idx, npa_lang_idx),
    }


def _sortable_th(key: str, label: str, num: bool = False, extra_class: str = "") -> str:
    """A WAI-ARIA sortable-column-header: aria-sort lives on the <th> itself
    (updated by JS as sort state changes), the click/keyboard target is a
    real <button> nested inside it -- a bare <th data-key=...> with a click
    listener is invisible to keyboard and screen-reader users, since a <th>
    isn't a focusable, activatable element on its own.

    extra_class, when given, is added alongside "num" (e.g. a toggle-hidden
    column shown only once its checkbox is checked -- see the CSS
    `.show-{{key}} .col-{{key}} {{ display: ... }}` pattern in _category_tab_html).
    """
    classes = " ".join(c for c in ["num" if num else "", extra_class] if c)
    cls = f' class="{classes}"' if classes else ""
    return f'<th scope="col"{cls} aria-sort="none"><button type="button" class="th-sort-btn" data-key="{key}">{label}</button></th>'


def _mini_stat_html(value: str, label: str) -> str:
    return f'<div class="mini-stat"><div class="mini-stat-value">{value}</div><div class="mini-stat-label">{label}</div></div>'


def _zero_languages_html(zero_langs: list[str]) -> str:
    """The tried-but-nothing gap between the ladder's first two steps names
    no one -- 'tried' and 'any samples' just differ by a count. This makes
    that gap inspectable: which specific languages were attempted here and
    came back with zero candidate samples in every condition, collapsed
    behind a <details> since the list can run to several dozen names and
    most readers won't need it.
    """
    if not zero_langs:
        return ""
    n = len(zero_langs)
    return f"""<details class="zero-langs">
                    <summary>{n} language{"s" if n != 1 else ""} tried, zero samples anywhere</summary>
                    <p class="zero-lang-list">{", ".join(zero_langs)}</p>
                </details>"""


def _coverage_ladder_html(cov: dict, chart_id: str) -> str:
    """A funnel chart -- how many languages were even attempted, down to how
    many cleared each minimal-pair bar. The chart itself is drawn client-side
    (renderCoverageFunnel, called for this divId once the page/tab is
    actually visible -- Plotly into a hidden container lays out at 0x0); this
    just emits the values as a data attribute so JS never has to re-derive a
    category's own coverage union from scratch.
    """
    payload = json.dumps({
        "tried": cov["tried"], "with_samples": cov["with_samples"],
        "ge10": cov["ge10"], "ge30": cov["ge30"], "ge50": cov["ge50"],
        "ge100": cov["ge100"], "ge500": cov["ge500"],
    })
    chart = f"<div class=\"ladder-chart\" id=\"{chart_id}\" data-cov='{payload}' style=\"height: 200px;\"></div>"
    return chart + _zero_languages_html(cov.get("zero_languages", []))


def _category_tab_html(data: dict, group: str, lang_idx: dict, npa_lang_idx: dict) -> str:
    slug = GROUP_SLUGS[group]
    items = _category_items(data, group)
    totals = _category_totals(data, group, items, lang_idx, npa_lang_idx)

    # Sparkbar width is each item's pairs relative to the category's biggest
    # picker item -- a tiny small-multiples comparison (Number vs. Gender vs.
    # Person, at a glance) riding along the picker itself, rather than a
    # separate chart grid competing with the detail view below it.
    max_pairs = max((it["pairs"] for it in items), default=0) or 1
    picker_html = "\n                    ".join(
        f'<button class="picker-btn{" active" if i == 0 else ""}" role="radio" '
        f'aria-checked="{"true" if i == 0 else "false"}" '
        f'data-cond="{it["id"]}" data-npasub="{"true" if it["isNpaSub"] else "false"}">'
        f'<span>{it["label"]}</span>'
        f'<span class="picker-spark"><span class="picker-spark-fill" '
        f'style="width:{round(100 * it["pairs"] / max_pairs)}%"></span></span>'
        f'</button>'
        for i, it in enumerate(items)
    )
    n = totals["conditions"]

    return f"""
        <div id="tab-{slug}" role="tabpanel" aria-labelledby="tabbtn-{slug}" tabindex="0" data-group="{group}" hidden>
            <div class="card">
                <h2>{group}</h2>
                <p class="section-desc">{n} condition{"s" if n != 1 else ""} tracked in this category, {totals["languages"]} languages combined.
                    Pick one below for its own breakdown &mdash; the chart shows its top 25 languages by pairs; the table has all of them.</p>
                <div class="mini-stats">
                    {_mini_stat_html(str(totals["languages"]), "Languages")}
                    {_mini_stat_html(f'{totals["samples"]:,}', "Samples")}
                    {_mini_stat_html(f'{totals["pairs"]:,}', "Minimal pairs")}
                    {_mini_stat_html(f'{totals["acc"] * 100:.1f}%' if totals["acc"] is not None else "&mdash;", "Decision-tree accuracy")}
                </div>
                {_coverage_ladder_html(totals, f"ladder-cat-{slug}-chart")}

                <details class="table-view" id="cat-{slug}-dropped-details">
                    <summary>Dropped before fitting in this category</summary>
                    <p class="section-desc" style="margin-top: 0.6rem;">
                        Rows across {group}'s {n} condition{"s" if n != 1 else ""} where the agreement label itself
                        was missing (head, subject, or both), so they never reached the swap-candidate pipeline at
                        all &mdash; see the same chart on the Overview tab for the all-conditions total this is part of.
                    </p>
                    <div id="cat-{slug}-dropped-chart" style="height: 160px;"></div>
                </details>

                <div class="picker-bar" id="cat-{slug}-picker" role="radiogroup" aria-label="Condition within {group}">
                    {picker_html}
                </div>

                <div id="cat-{slug}-ministats"></div>

                <div class="extra-toggle" id="cat-{slug}-extra-toggle" aria-label="Additional data for the picked condition">
                    <label><input type="checkbox" class="extra-check" data-extra="head_unk"> Head unknown (per language)</label>
                    <label><input type="checkbox" class="extra-check" data-extra="nsubj_unk"> Subject unknown (per language)</label>
                    <label><input type="checkbox" class="extra-check" data-extra="both_unk"> Both unknown (per language)</label>
                    <label><input type="checkbox" id="cat-{slug}-include-zero"> Zero-sample languages (table only)</label>
                </div>

                <div id="cat-{slug}-chart" style="height: 480px;"></div>

                <div class="toolbar" style="margin-top: 1.5rem;">
                    <label for="cat-{slug}-search" class="sr-only">Filter languages by name</label>
                    <input class="search" id="cat-{slug}-search" type="text" placeholder="Filter languages&hellip;">
                    <span class="count-note" id="cat-{slug}-count" aria-live="polite"></span>
                </div>
                <div class="table-scroll">
                    <table id="cat-{slug}-table">
                        <thead>
                            <tr>
                                {_sortable_th("name", "Language")}
                                {_sortable_th("samples", "Samples", num=True)}
                                {_sortable_th("pairs", "Pairs", num=True)}
                                {_sortable_th("ratio", "Pairs/sample", num=True)}
                                {_sortable_th("acc", "Accuracy", num=True)}
                                {_sortable_th("n_lemma", "Lemmas", num=True)}
                                {_sortable_th("head_unk", "Head unknown", num=True, extra_class="col-head_unk")}
                                {_sortable_th("nsubj_unk", "Subject unknown", num=True, extra_class="col-nsubj_unk")}
                                {_sortable_th("both_unk", "Both unknown", num=True, extra_class="col-both_unk")}
                            </tr>
                        </thead>
                        <tbody id="cat-{slug}-tbody"></tbody>
                    </table>
                </div>
            </div>
        </div>"""


def render(data: dict) -> str:
    totals = data["totals"]
    data_json = json.dumps(data, ensure_ascii=False)

    lang_idx = {lang: i for i, lang in enumerate(data["matrix"]["languages"])}
    npa_lang_idx = {lang: i for i, lang in enumerate(data["npa_matrix"]["languages"])}

    groups_present = [g for g in GROUP_ORDER if _category_items(data, g)]
    tab_buttons_html = (
        '<button class="tab-btn active" id="tabbtn-overall" role="tab" aria-selected="true" '
        'aria-controls="tab-overall" data-tab="overall">Overall</button>\n            '
    ) + "\n            ".join(
        f'<button class="tab-btn" id="tabbtn-{GROUP_SLUGS[g]}" role="tab" aria-selected="false" '
        f'aria-controls="tab-{GROUP_SLUGS[g]}" data-tab="{GROUP_SLUGS[g]}">{g}</button>'
        for g in groups_present
    )
    category_tabs_html = "".join(_category_tab_html(data, g, lang_idx, npa_lang_idx) for g in groups_present)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>sva-dt &mdash; Dataset Overview</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f5f5f4;
            --card: #ffffff;
            --panel-bg: #fafaf9;
            --text: #1c1917;
            --text-muted: #78716c;
            --text-subtle: #57534e;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --danger: #dc2626;
            --border: #e7e5e4;
            --border-hover: #c7c4c0;
            --grid-line: #e7e5e4;
            --hover: #f5f5f5;
        }}
        /* Same palette as decision_trees/index.html (word_order.html.html_overview),
           for a consistent look across every page in the site. */
        :root[data-theme="dark"] {{
            color-scheme: dark;
            --bg: #16140f; --card: #221f19; --panel-bg: #1c1a15;
            --text: #f0ede6; --text-muted: #a39a8a; --text-subtle: #a39a8a;
            --accent: #6ea8ff; --accent-soft: #1e2b40; --danger: #f87171;
            --border: #38332a; --border-hover: #4d4636;
            --grid-line: #38332a; --hover: #2a261e;
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                color-scheme: dark;
                --bg: #16140f; --card: #221f19; --panel-bg: #1c1a15;
                --text: #f0ede6; --text-muted: #a39a8a; --text-subtle: #a39a8a;
                --accent: #6ea8ff; --accent-soft: #1e2b40; --danger: #f87171;
                --border: #38332a; --border-hover: #4d4636;
                --grid-line: #38332a; --hover: #2a261e;
            }}
        }}
        * {{ box-sizing: border-box; }}
        :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
        .sr-only {{
            position: absolute;
            width: 1px; height: 1px;
            padding: 0; margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }}
        .skip-link {{
            position: absolute;
            top: -3rem;
            left: 1rem;
            z-index: 100;
            background: var(--accent);
            color: #fff;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            transition: top 0.15s;
        }}
        .skip-link:focus {{ top: 1rem; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: 'DM Sans', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--card);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1rem 2rem;
        }}

        .modal-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.55);
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 5vh 1.5rem;
            z-index: 100;
            overflow-y: auto;
        }}
        .modal-overlay[hidden] {{ display: none; }}
        .modal {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: 0 12px 48px rgba(0,0,0,0.35);
            width: 100%;
            max-width: 720px;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
        }}
        .modal-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .modal-header h2 {{ font-size: 1.2rem; }}
        .modal-close {{
            font: inherit;
            font-size: 1.4rem;
            line-height: 1;
            color: var(--text-muted);
            background: none;
            border: none;
            cursor: pointer;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
        }}
        .modal-close:hover {{ color: var(--text); background: var(--hover); }}
        .modal-body {{ padding: 1.5rem; overflow-y: auto; }}
        .modal-body table {{ width: 100%; }}
        .modal-body .ext-link {{
            color: inherit;
            text-decoration: none;
            border-bottom: 1px dotted var(--text-muted);
        }}
        .modal-body .ext-link:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}
        h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.75rem;
            font-weight: 600;
        }}
        h2 {{
            font-size: 1.15rem;
            font-weight: 600;
            margin: 0 0 0.35rem;
        }}
        .section-desc {{
            color: var(--text-muted);
            font-size: 0.875rem;
            line-height: 1.6;
            max-width: 780px;
            margin: 0 0 1.25rem;
        }}
        .description {{
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 720px;
            margin: 0.5rem 0 0;
        }}
        .meta-line {{
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 0.75rem;
        }}
        .nav-link {{
            font-size: 0.85rem;
            color: var(--accent);
            text-decoration: none;
            font-weight: 500;
            white-space: nowrap;
            padding: 0.5rem 0.9rem;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}
        .nav-link:hover {{ border-color: var(--accent); background: var(--accent-soft); }}

        .lang-link {{
            font: inherit;
            color: inherit;
            background: none;
            border: none;
            border-bottom: 1px dotted var(--text-muted);
            padding: 0;
            cursor: pointer;
            text-align: left;
        }}
        .lang-link:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}

        .tab-bar {{
            display: flex;
            gap: 0.6rem;
            margin: 0 0 1rem;
            padding: 0.75rem 0;
            flex-wrap: wrap;
            position: sticky;
            top: 0;
            z-index: 20;
            background: var(--bg);
        }}
        .jump-nav {{
            display: flex;
            gap: 0.5rem 1.25rem;
            flex-wrap: wrap;
            font-size: 0.82rem;
            margin: -0.5rem 0 1.25rem;
        }}
        .jump-nav a {{
            color: var(--text-muted);
            text-decoration: none;
            padding: 0.2rem 0;
            border-bottom: 2px solid transparent;
        }}
        .jump-nav a:hover {{ color: var(--accent); border-color: var(--accent); }}
        .card[id^="sec-"] {{ scroll-margin-top: 5rem; }}
        .tab-btn {{
            padding: 0.6rem 1.25rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--card);
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: border-color 0.15s, color 0.15s, background 0.15s;
        }}
        .tab-btn:hover {{ color: var(--text); border-color: var(--border-hover); }}
        .tab-btn.active {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }}

        .picker-bar {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1.25rem;
        }}
        .picker-btn {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 0.3rem;
            min-width: 96px;
            padding: 0.4rem 0.85rem 0.5rem;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--panel-bg);
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.82rem;
            font-weight: 600;
            text-align: left;
            cursor: pointer;
            transition: border-color 0.15s, color 0.15s, background 0.15s;
        }}
        .picker-btn:hover {{ color: var(--text); border-color: var(--border-hover); }}
        .picker-btn.active {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }}
        .picker-spark {{
            display: block;
            height: 4px;
            border-radius: 2px;
            background: var(--border);
            overflow: hidden;
        }}
        .picker-spark-fill {{
            display: block;
            height: 100%;
            border-radius: 2px;
            background: var(--accent);
            opacity: 0.55;
        }}
        .picker-btn.active .picker-spark-fill {{ opacity: 1; }}

        .ratio-high {{ color: var(--accent); font-weight: 700; }}

        .mini-stats {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 0.75rem 0 1.25rem;
        }}
        .mini-stat {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.55rem 0.9rem;
            min-width: 100px;
        }}
        .mini-stat-value {{
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        .mini-stat-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .kpi-tile {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        .kpi-label {{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.83rem;
        }}
        thead th {{
            text-align: left;
            font-weight: 600;
            color: var(--text-subtle);
            border-bottom: 1px solid var(--border);
            padding: 0.5rem 0.7rem;
            white-space: nowrap;
        }}
        thead th.num {{ text-align: right; }}
        .th-sort-btn {{
            background: none;
            border: none;
            padding: 0;
            margin: 0;
            font: inherit;
            font-weight: inherit;
            color: inherit;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        thead th.num .th-sort-btn {{ width: 100%; text-align: right; }}
        .th-sort-btn:hover {{ color: var(--accent); }}
        thead th.sorted .th-sort-btn {{ color: var(--accent); }}
        thead th.sorted::after {{ content: " \\25BE"; font-size: 0.7em; color: var(--accent); }}
        thead th.sorted.asc::after {{ content: " \\25B4"; font-size: 0.7em; color: var(--accent); }}

        .table-view {{ margin-top: 0.85rem; }}
        .table-view summary {{
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--accent);
            user-select: none;
        }}
        .table-view summary:hover {{ text-decoration: underline; }}
        tbody td {{
            padding: 0.42rem 0.7rem;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        tbody tr:hover {{ background: var(--hover); }}
        td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        tbody tr.row-zero {{ color: var(--text-muted); }}
        td.treebank-list {{
            white-space: normal;
            color: var(--text-subtle);
            font-size: 0.82rem;
            min-width: 260px;
        }}
        tbody tr.row-zero td:first-child::after {{
            content: "0 pairs";
            display: inline-block;
            margin-left: 0.5rem;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: var(--text-muted);
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.05rem 0.35rem;
            vertical-align: middle;
        }}
        .table-scroll {{
            max-height: 520px;
            overflow: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        .table-scroll thead th {{ position: sticky; top: 0; background: var(--panel-bg); }}

        .toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem 1rem;
            margin-bottom: 0.85rem;
        }}
        input.search {{
            padding: 0.5rem 0.875rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--card);
            color: var(--text);
            font-family: inherit;
            font-size: 0.85rem;
            width: 100%;
            max-width: 280px;
            flex: 1 1 200px;
        }}
        input.search:focus {{ outline: none; border-color: var(--accent); }}
        .count-note {{ font-size: 0.8rem; color: var(--text-muted); }}

        .ladder-chart {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 0.5rem 0;
            margin: 0.75rem 0 1.25rem;
        }}

        .zero-langs {{ margin: 0 0 1.25rem; }}
        .zero-langs summary {{
            cursor: pointer;
            font-size: 0.78rem;
            color: var(--text-muted);
            user-select: none;
        }}
        .zero-langs summary:hover {{ color: var(--text); }}
        .zero-lang-list {{
            font-size: 0.8rem;
            color: var(--text-subtle);
            line-height: 1.6;
            margin-top: 0.5rem;
            padding: 0.65rem 0.85rem;
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
        }}

        .toggle-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.85rem;
        }}
        .toggle-row .metric-toggle {{ margin-bottom: 0; }}

        .extra-toggle {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem 1.1rem;
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin: 0.85rem 0;
        }}
        .extra-toggle label {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            cursor: pointer;
        }}
        .extra-toggle label:has(input:checked) {{ color: var(--text); }}
        .extra-toggle input {{ accent-color: var(--accent); }}
        /* Toggle-hidden extra columns (per-language unk counts) -- shown
           only once their checkbox is checked, via a matching .show-{{key}}
           class JS adds to the <table> itself (see wireExtraToggles). */
        .col-head_unk, .col-nsubj_unk, .col-both_unk {{ display: none; }}
        table.show-head_unk .col-head_unk {{ display: table-cell; }}
        table.show-nsubj_unk .col-nsubj_unk {{ display: table-cell; }}
        table.show-both_unk .col-both_unk {{ display: table-cell; }}

        .metric-toggle {{
            display: inline-flex;
            gap: 0.3rem;
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.25rem;
            margin-bottom: 0.85rem;
        }}
        .metric-toggle-btn {{
            font: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-muted);
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 0.35rem 0.85rem;
            cursor: pointer;
        }}
        .metric-toggle-btn:hover {{ color: var(--text); }}
        .metric-toggle-btn.active {{ background: var(--card); color: var(--accent); box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}

        .heatmap-wrap {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: auto;
            max-height: 720px;
        }}

        .legend-note {{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.6rem;
        }}
        .preview-badge {{
            display: inline-block;
            vertical-align: middle;
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--danger);
            border: 1px solid var(--danger);
            border-radius: 4px;
            padding: 0.12rem 0.4rem;
            margin-left: 0.5rem;
        }}
        code {{
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.05rem 0.35rem;
            font-size: 0.85em;
        }}

        @media (max-width: 640px) {{
            body {{ padding: 1rem; }}
            .card {{ padding: 1.25rem; }}
            h1 {{ font-size: 1.4rem; }}
            .nav-link {{ white-space: normal; }}
        }}
    </style>
</head>
<body>
    <a class="skip-link" href="#main-content">Skip to main content</a>
    <div class="container">
        <header class="card">
            <div class="header">
                <div>
                    <h1>sva-dt &mdash; Dataset Overview</h1>
                    <p class="description">
                        How many candidate samples were considered and how many minimal pairs were actually
                        generated from them &mdash; broken down per language, per grammatical condition (subject
                        agreement with the verb/participle/auxiliary, and noun-phrase agreement across its role
                        pairs), and per language&times;condition. "Samples" are the raw UD-derived candidate
                        instances a condition considered; "minimal pairs" are the rows that came out the other end
                        as usable contrastive pairs &mdash; one sample can yield more than one pair when several
                        inflected forms are valid contrasts, so pairs isn't a subset of samples and the ratio
                        between them can exceed 1&times;.
                    </p>
                    <p class="meta-line">Generated {data["generated_at"]} &middot; from the published MultiBLiMP v2 site</p>
                </div>
                <a class="nav-link" href="https://jshrdt.github.io/multiblimp/" target="_blank" rel="noopener">Decision tree diagnostics &rarr;</a>
            </div>
        </header>

        <div class="tab-bar" role="tablist" aria-label="View">
            {tab_buttons_html}
        </div>

        <main id="main-content">
        <div id="tab-overall" role="tabpanel" aria-labelledby="tabbtn-overall" tabindex="0">
            <nav class="jump-nav" aria-label="Jump to section">
                <a href="#sec-kpi">Overview</a>
                <a href="#sec-languages">Languages</a>
                <a href="#sec-treebanks">Treebanks</a>
                <a href="#sec-treebank-pairs">Samples by treebank</a>
                <a href="#sec-scatter">Samples vs. pairs</a>
                <a href="#sec-heatmap">Heatmap</a>
                <a href="#sec-funnel">Outcomes</a>
            </nav>

            <div class="card" id="sec-kpi">
                <div class="kpi-grid">
                    <div class="kpi-tile">
                        <div class="kpi-value">{totals["languages"]}</div>
                        <div class="kpi-label">Languages covered</div>
                    </div>
                    <div class="kpi-tile">
                        <div class="kpi-value">{totals["conditions"]}</div>
                        <div class="kpi-label">Conditions tracked</div>
                    </div>
                    <div class="kpi-tile">
                        <div class="kpi-value">{totals["samples"]:,}</div>
                        <div class="kpi-label">Candidate samples</div>
                    </div>
                    <div class="kpi-tile">
                        <div class="kpi-value">{totals["pairs"]:,}</div>
                        <div class="kpi-label">Minimal pairs generated</div>
                    </div>
                    <div class="kpi-tile">
                        <div class="kpi-value">{totals["ratio"]}&times;</div>
                        <div class="kpi-label">Pairs per sample, overall</div>
                    </div>
                </div>
                {_coverage_ladder_html(totals["coverage"], "ladder-overall-chart")}
            </div>

            <div class="card" id="sec-languages">
                <h2>Samples &amp; pairs per language</h2>
                <p class="section-desc">
                    Totals summed across every condition a language appears in. The chart shows the top 25 languages
                    by minimal pairs generated; the table below covers all {totals["languages"]}.
                </p>
                <div id="language-chart" style="height: 560px;"></div>

                <div class="toolbar" style="margin-top: 1.5rem;">
                    <label for="lang-search" class="sr-only">Filter languages by name</label>
                    <input class="search" id="lang-search" type="text" placeholder="Filter languages&hellip;">
                    <span class="count-note" id="lang-count" aria-live="polite"></span>
                </div>
                <div class="table-scroll">
                    <table id="language-table">
                        <thead>
                            <tr>
                                {_sortable_th("name", "Language")}
                                {_sortable_th("samples", "Samples", num=True)}
                                {_sortable_th("pairs", "Pairs", num=True)}
                                {_sortable_th("ratio", "Pairs/sample", num=True)}
                                {_sortable_th("acc", "Accuracy", num=True)}
                                {_sortable_th("n_lemma", "Lemmas", num=True)}
                                {_sortable_th("n_conditions", "# Conditions", num=True)}
                            </tr>
                        </thead>
                        <tbody id="language-tbody"></tbody>
                    </table>
                </div>
            </div>

            <div class="card" id="sec-treebanks">
                <h2>Treebanks per language</h2>
                <p class="section-desc">
                    Every UD treebank this pipeline is eligible to draw samples from for each attempted language
                    (some languages restrict to a specific subset; most use every treebank the UD release has for
                    them) &mdash; an inventory of possible sources, not a per-sample or per-pair count by treebank.
                    Includes the {len(data["totals"]["coverage"]["zero_languages"])} zero-yield languages above,
                    since which treebank(s) a language drew from can be part of why it came back empty. For the
                    finer, per-sample breakdown, see the preview below.
                </p>
                <div class="toolbar">
                    <label for="treebank-search" class="sr-only">Filter languages by name</label>
                    <input class="search" id="treebank-search" type="text" placeholder="Filter languages&hellip;">
                    <span class="count-note" id="treebank-count" aria-live="polite"></span>
                </div>
                <div class="table-scroll">
                    <table id="treebank-table">
                        <thead>
                            <tr>
                                {_sortable_th("name", "Language")}
                                {_sortable_th("n_treebanks", "# Treebanks", num=True)}
                                <th scope="col">Treebanks</th>
                            </tr>
                        </thead>
                        <tbody id="treebank-tbody"></tbody>
                    </table>
                </div>
            </div>

            <div class="card" id="sec-treebank-pairs">
                <h2>Samples by treebank <span class="preview-badge">preview</span></h2>
                <p class="section-desc">
                    Which treebank each language's raw candidate samples actually came from, for every minimal pair
                    those samples could produce &mdash; not an estimate, real per-row counts from this repo's own
                    local pipeline runs. This is genuinely partial and evolving, not a scoped-down version of the
                    rest of this page: local processing is comprehensive for Noun Phrase but has barely started on
                    Subject&ndash;Verb/Participle/Auxiliary (currently {len(data["local_treebank_pairs"])} language
                    &times; condition combinations across {len(set(e["language"] for e in data["local_treebank_pairs"]))}
                    languages, out of this page's full {data["totals"]["languages"]}). It also isn't the same pipeline
                    run as the rest of this page (a separate, newer local pass) &mdash; its own sample counts can
                    differ from the "published" samples/pairs shown everywhere else, which is exactly why both are
                    shown side by side below rather than only one. Samples, not pairs, by treebank: nothing here
                    flags which individual rows became accepted pairs, so a treebank's share of samples is used as
                    the closest honest proxy for its share of that combination's pairs.
                </p>
                <div class="toolbar">
                    <label for="treebank-pairs-search" class="sr-only">Filter languages by name</label>
                    <input class="search" id="treebank-pairs-search" type="text" placeholder="Filter languages&hellip;">
                    <span class="count-note" id="treebank-pairs-count" aria-live="polite"></span>
                </div>
                <div class="table-scroll">
                    <table id="treebank-pairs-table">
                        <thead>
                            <tr>
                                {_sortable_th("name", "Language")}
                                {_sortable_th("condition", "Condition")}
                                {_sortable_th("treebank", "Treebank")}
                                {_sortable_th("samples", "Local samples", num=True)}
                                {_sortable_th("share", "Share", num=True)}
                                {_sortable_th("published_pairs", "Published pairs", num=True)}
                                {_sortable_th("published_samples", "Published samples", num=True)}
                            </tr>
                        </thead>
                        <tbody id="treebank-pairs-tbody"></tbody>
                    </table>
                </div>
            </div>

            <div class="card" id="sec-scatter">
                <h2>Samples vs. minimal pairs, per language</h2>
                <p class="section-desc">
                    Each dot is one language, totals across every condition it appears in. The dashed line marks
                    where pairs equal samples (1&times;) &mdash; languages well above it produced unusually many
                    contrastive pairs per candidate; languages near the bottom found samples but little usable
                    contrast. Both axes are log-scaled given the range from a handful of samples to hundreds of
                    thousands; languages with zero pairs are nudged just above zero so they stay visible rather
                    than dropping off a log axis.
                </p>
                <div id="scatter-chart" style="height: 460px;"></div>
            </div>

            <div class="card" id="sec-heatmap">
                <h2>Language &times; condition matrix</h2>
                <p class="section-desc">
                    Minimal pairs generated for every language/condition combination. Sort by total pairs or
                    alphabetically; hover a cell for exact samples, pairs, ratio, and decision-tree accuracy, or
                    click it to open that language/condition's own diagnostics page. Blank cells are combinations
                    that were never run, not zero-pair combinations.
                </p>
                <div class="toggle-row">
                    <div class="metric-toggle" role="radiogroup" aria-label="Heatmap color metric" id="heatmap-metric-toggle">
                        <button type="button" class="metric-toggle-btn active" role="radio" aria-checked="true" data-metric="pairs">Pairs</button>
                        <button type="button" class="metric-toggle-btn" role="radio" aria-checked="false" data-metric="acc">Accuracy</button>
                    </div>
                    <div class="metric-toggle" role="radiogroup" aria-label="Heatmap language order" id="heatmap-sort-toggle">
                        <button type="button" class="metric-toggle-btn active" role="radio" aria-checked="true" data-sort="pairs">By pairs</button>
                        <button type="button" class="metric-toggle-btn" role="radio" aria-checked="false" data-sort="alpha">A&ndash;Z</button>
                    </div>
                </div>
                <div class="heatmap-wrap">
                    <div id="matrix-heatmap"></div>
                </div>
                <p class="legend-note">{len(data["matrix"]["languages"])} languages &times; {len(data["matrix"]["conditions"])} conditions.</p>

                <details class="table-view">
                    <summary>View as table</summary>
                    <div class="table-scroll" style="margin-top: 0.75rem; max-height: 480px;">
                        <table id="heatmap-table">
                            <thead><tr id="heatmap-table-head"></tr></thead>
                            <tbody id="heatmap-table-body"></tbody>
                        </table>
                    </div>
                </details>
            </div>

            <div class="card" id="sec-funnel">
                <h2>Where samples don't become pairs</h2>
                <p class="section-desc">
                    Aggregated outcome buckets across every language/condition (from each run's own diagnostic
                    buckets). Not a strict partition of the sample count above &mdash; one sample can land in more
                    than one outcome across its accepted swap candidates &mdash; but it shows where the corpus loses
                    candidates: no matching swap candidate, an inflector that couldn't produce a contrastive form,
                    a reinflection that came out identical, or a genuinely ambiguous subject.
                </p>
                <div id="funnel-chart" style="height: 300px;"></div>

                <details class="table-view">
                    <summary>View as table</summary>
                    <table style="margin-top: 0.75rem;">
                        <thead><tr><th scope="col">Outcome</th><th scope="col" class="num">Instances</th></tr></thead>
                        <tbody id="funnel-table-body"></tbody>
                    </table>
                </details>

                <h2 style="margin-top: 2rem;">Dropped before fitting</h2>
                <p class="section-desc">
                    A different, earlier population than the outcome buckets above: rows where the agreement label
                    itself was missing (not "Yes"/"No" but undefined), so they never entered the swap-candidate
                    pipeline at all &mdash; split by whether the head (verb/participle/auxiliary), the subject, or
                    both were missing their feature annotation. From each condition's decision-tree fit (see the
                    "Dropped before fitting" stats on each language's own diagnostics page).
                </p>
                <div id="dropped-chart" style="height: 220px;"></div>
            </div>
        </div>
        {category_tabs_html}
        </main>
    </div>

    <div class="modal-overlay" id="lang-modal" hidden>
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="lang-modal-title">
            <div class="modal-header">
                <h2 id="lang-modal-title">Language</h2>
                <button type="button" class="modal-close" id="lang-modal-close" aria-label="Close">&times;</button>
            </div>
            <div class="modal-body" id="lang-modal-body"></div>
        </div>
    </div>

    <script>
        const DATA = {data_json};

        const THEME = (() => {{
            const root = getComputedStyle(document.documentElement);
            const v = (name) => root.getPropertyValue(name).trim();
            return {{
                text: v('--text'), muted: v('--text-muted'), grid: v('--grid-line'),
                accent: v('--accent'), border: v('--border'), danger: v('--danger'),
                panelBg: v('--panel-bg'), card: v('--card'),
            }};
        }})();

        const BASE_LAYOUT = {{
            font: {{ family: 'DM Sans, system-ui, sans-serif', color: THEME.text, size: 11 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            // Plotly's default hover box derives its background from
            // whatever mark got hovered and auto-picks black/white text for
            // it -- fine most of the time, but this page's marks range from
            // near-panel-dark to full accent, and leaving hover styling
            // implicit means it's only ever accidentally readable. Fixed,
            // theme-correct colors here make every hover box the same
            // known-good contrast regardless of the mark underneath it.
            hoverlabel: {{
                bgcolor: THEME.card,
                bordercolor: THEME.border,
                font: {{ family: 'DM Sans, system-ui, sans-serif', color: THEME.text, size: 12 }},
            }},
            bargap: 0.28,
            bargroupgap: 0.1,
        }};
        const CONFIG = {{ responsive: true, displayModeBar: false }};

        // Linear RGB mix between two hex colors -- used to build sequential
        // ramps (heatmap, funnel) that derive from the page's own theme
        // tokens instead of a hardcoded light-mode hex, so they adapt
        // automatically between light and dark.
        function hexToRgb(hex) {{
            hex = hex.replace('#', '');
            if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
            const num = parseInt(hex, 16);
            return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
        }}
        function mixHex(hexA, hexB, t) {{
            const a = hexToRgb(hexA), b = hexToRgb(hexB);
            const m = a.map((c, i) => Math.round(c + (b[i] - c) * t));
            return `rgb(${{m[0]}},${{m[1]}},${{m[2]}})`;
        }}

        // One direct label on the extreme bar only (never every bar -- see
        // dataviz skill's mark spec), so the headline value reads without a
        // hover. xshift nudges it clear of the bar end.
        function endLabelAnnotation(x, y, text) {{
            return {{
                x, y, text, xanchor: 'left', yanchor: 'middle', showarrow: false,
                font: {{ size: 10, color: THEME.text }}, xshift: 8,
            }};
        }}

        // Samples vs. minimal pairs is the one series distinction every bar
        // chart on this page makes, so it gets one fixed, high-contrast
        // color pair everywhere rather than an opacity variant of a single
        // hue (too subtle to read in the legend swatches, per user report)
        // or a per-category hue (a second job competing for the same
        // channel -- the category/condition name is already in the axis
        // label, so color stays reserved for this one distinction). Muted
        // neutral for the context measure, the page's own accent for the
        // headline one -- a near-neutral-vs-saturated pair stays reliably
        // distinct under color vision deficiency since it differs mainly in
        // chroma, not hue angle.
        const SERIES_COLORS = {{ samples: THEME.muted, pairs: THEME.accent }};
        // Shared red-purple-blue diverging scale -- used for the heatmap
        // (both Pairs and Accuracy color modes) and the scatter chart's
        // accuracy-colored points. A true midpoint hue (not a fade to gray)
        // so the "mid-field" band reads as its own distinct color instead
        // of the low/high ends just looking like paler versions of each
        // other -- a straight red<->blue fade desaturates through gray at
        // 0.5 (that was the previous version of this scale), which is
        // exactly the "hard to see the middle" complaint this replaces.
        const PURPLE = mixHex(THEME.danger, THEME.accent, 0.5);
        const DIVERGING_SCALE = [
            [0, THEME.danger],
            [0.25, mixHex(THEME.danger, PURPLE, 0.5)],
            [0.5, PURPLE],
            [0.75, mixHex(PURPLE, THEME.accent, 0.5)],
            [1, THEME.accent],
        ];

        // Ratios above 1x are the notable case (more pairs came out than
        // there were raw candidate samples) -- flagged in the accent color
        // so they stand out in a table of otherwise-neutral numbers, same
        // as a signed delta would.
        // Shared by every sortable table (language + each category table):
        // updates both the visual sorted/asc classes (the little arrow) and
        // the WAI-ARIA aria-sort state screen readers rely on, kept on the
        // <th> itself even though the click/keyboard target is the nested
        // button (a <th> isn't focusable on its own).
        function updateSortIndicators(tableId, sortKey, sortDir) {{
            document.querySelectorAll(`#${{tableId}} thead th`).forEach(th => {{
                const btn = th.querySelector('.th-sort-btn');
                const key = btn ? btn.dataset.key : null;
                const isSorted = key === sortKey;
                th.classList.toggle('sorted', isSorted);
                th.classList.toggle('asc', isSorted && sortDir === 1);
                th.setAttribute('aria-sort', isSorted ? (sortDir === 1 ? 'ascending' : 'descending') : 'none');
            }});
        }}

        function fmtRatio(r) {{
            if (r === null || r === undefined) return '&mdash;';
            const cls = r > 1 ? ' class="ratio-high"' : '';
            return `<span${{cls}}>${{r.toFixed(2)}}&times;</span>`;
        }}

        function fmtAcc(a) {{
            if (a === null || a === undefined) return '&mdash;';
            return `${{(a * 100).toFixed(1)}}%`;
        }}

        // Language name as a trigger for the cross-condition language modal
        // (openLangModal) -- always clickable, even for a row whose own
        // url is null (e.g. a zero-sample-*for-this-condition* row added by
        // the "include zero" toggle): that language can still have real data
        // in other conditions, which the modal looks up fresh from
        // DATA.matrix rather than trusting this one row's own url field.
        function langLinkHtml(name) {{
            const escaped = name.replace(/&/g, '&amp;').replace(/</g, '&lt;');
            return `<button type="button" class="lang-link" data-lang="${{escaped.replace(/"/g, '&quot;')}}">${{escaped}}</button>`;
        }}

        function miniStatHtml(value, label) {{
            return `<div class="mini-stat"><div class="mini-stat-value">${{value}}</div><div class="mini-stat-label">${{label}}</div></div>`;
        }}

        // The cross-condition language overview modal -- one language's
        // results across every condition it was tried in, in one place,
        // instead of only ever linking straight to a single condition's
        // external page (see build_stats.py's matrix_url comment: that
        // single link was always just "the highest-pairs condition", never
        // the full picture). Pulled together entirely from DATA already
        // embedded in the page -- matrix (per condition), language_treebanks,
        // and local_treebank_pairs (the local-run preview, when it covers
        // this language) -- no extra network request.
        function openLangModal(lang) {{
            document.getElementById('lang-modal-title').textContent = lang;
            const body = document.getElementById('lang-modal-body');
            const idx = DATA.matrix.languages.indexOf(lang);
            const treebankEntry = DATA.language_treebanks.find(r => r.name === lang);
            const treebanks = treebankEntry ? treebankEntry.treebanks : [];
            const treebankNote = treebanks.length
                ? `<p class="section-desc">Eligible treebanks: ${{treebanks.join(', ')}}</p>` : '';

            // Same local-run preview as the "Samples by treebank" section --
            // covers a language here whenever that section covers it, which
            // as of writing is comprehensive for Noun Phrase and sparse
            // elsewhere (see that section's own caveat copy).
            const localEntries = DATA.local_treebank_pairs.filter(e => e.language === lang);
            const localNote = localEntries.length ? `
                <p class="section-desc" style="margin-top:1rem;">
                    <span class="preview-badge">preview</span> Local per-treebank sample counts:
                </p>
                <ul class="zero-lang-list" style="margin:0.5rem 0 0; padding-left: 1.1rem;">
                    ${{localEntries.map(e => `<li>${{e.condition_label}}: ${{
                        Object.entries(e.samples_by_treebank).filter(([, n]) => n)
                            .map(([tb, n]) => `${{tb}} (${{n.toLocaleString()}})`).join(', ')
                    }}</li>`).join('')}}
                </ul>
            ` : '';

            if (idx === -1) {{
                body.innerHTML = `
                    <p class="section-desc">No samples were found for <b>${{lang}}</b> in any of this page's
                        ${{DATA.totals.conditions}} tracked conditions.</p>
                    ${{treebankNote}}
                    ${{localNote}}
                `;
            }} else {{
                const condRows = DATA.matrix.conditions.map((condId, j) => {{
                    const samples = DATA.matrix.samples[idx][j];
                    const pairs = DATA.matrix.pairs[idx][j];
                    if (!samples && !pairs) return null;
                    const c = DATA.conditions.find(c => c.id === condId);
                    const url = DATA.matrix.urls[idx][j];
                    return {{
                        label: c ? `${{c.label}} (${{c.group}})` : condId,
                        samples, pairs, ratio: samples ? pairs / samples : null,
                        acc: DATA.matrix.acc[idx][j], url,
                    }};
                }}).filter(Boolean).sort((a, b) => b.pairs - a.pairs);

                const totalSamples = condRows.reduce((s, r) => s + r.samples, 0);
                const totalPairs = condRows.reduce((s, r) => s + r.pairs, 0);

                body.innerHTML = `
                    <div class="mini-stats">
                        ${{miniStatHtml(condRows.length, 'Conditions with data')}}
                        ${{miniStatHtml(totalSamples.toLocaleString(), 'Samples')}}
                        ${{miniStatHtml(totalPairs.toLocaleString(), 'Minimal pairs')}}
                    </div>
                    <div class="table-scroll">
                        <table>
                            <thead>
                                <tr>
                                    <th scope="col">Condition</th>
                                    <th scope="col" class="num">Samples</th>
                                    <th scope="col" class="num">Pairs</th>
                                    <th scope="col" class="num">Pairs/sample</th>
                                    <th scope="col" class="num">Accuracy</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{condRows.map(r => `
                                    <tr>
                                        <td>${{r.url
                                            ? `<a class="ext-link" href="${{r.url}}" target="_blank" rel="noopener" title="Open ${{lang}}'s diagnostics page for this condition">${{r.label}} &#8599;</a>`
                                            : r.label}}</td>
                                        <td class="num">${{r.samples.toLocaleString()}}</td>
                                        <td class="num">${{r.pairs.toLocaleString()}}</td>
                                        <td class="num">${{fmtRatio(r.ratio)}}</td>
                                        <td class="num">${{fmtAcc(r.acc)}}</td>
                                    </tr>`).join('')}}
                            </tbody>
                        </table>
                    </div>
                    ${{treebankNote}}
                    ${{localNote}}
                `;
            }}

            document.getElementById('lang-modal').hidden = false;
            document.getElementById('lang-modal-close').focus();
        }}

        function closeLangModal() {{
            document.getElementById('lang-modal').hidden = true;
        }}

        document.addEventListener('click', (e) => {{
            const trigger = e.target.closest('.lang-link');
            if (trigger) {{ openLangModal(trigger.dataset.lang); return; }}
            if (e.target.id === 'lang-modal-close' || e.target.id === 'lang-modal') closeLangModal();
        }});
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && !document.getElementById('lang-modal').hidden) closeLangModal();
        }});

        // Space- and/or comma-separated terms, case-insensitive, OR'd
        // together -- "portuguese, spanish" or "Portuguese Spanish" both
        // match either language.
        function matchesLangFilter(name, filterText) {{
            const terms = filterText.toLowerCase().split(/[\\s,]+/).filter(Boolean);
            if (!terms.length) return true;
            const lower = name.toLowerCase();
            return terms.some(t => lower.includes(t));
        }}

        // Opens a point's linked deep-link (build_stats.py's per-language/
        // per-cell url) on click -- getUrl(point) returns null for points
        // with no page to link to (e.g. a tried-but-zero-samples language),
        // which this just silently ignores rather than erroring.
        //
        // Charts that redraw in place on user interaction (the heatmap's
        // metric toggle, a category tab's condition picker) call this again
        // on every redraw with a fresh getUrl closure over that redraw's own
        // data. Whether the old listener needs clearing depends on which
        // Plotly call did the redraw -- newPlot purges the graph div's event
        // emitter (so the old listener is already gone, but so is the div's
        // ability to fire it correctly by the time a later click lands),
        // react does not (so skipping this would stack a second listener
        // and open every click's url twice). removeAllListeners before
        // re-binding handles both the same way: exactly one listener, always
        // pointed at the redraw's own data, regardless of which call made it.
        function wireClickThrough(divId, getUrl) {{
            const gd = document.getElementById(divId);
            gd.removeAllListeners('plotly_click');
            gd.on('plotly_click', (ev) => {{
                const pt = ev.points && ev.points[0];
                const url = pt && getUrl(pt);
                if (url) window.open(url, '_blank', 'noopener');
            }});
        }}

        // Overall's own global coverage funnel -- visible by default (it's
        // the default-shown tab), so this can render eagerly like every
        // other Overall-tab chart below instead of waiting for a tab-switch
        // the way each category tab's own funnel does.
        renderVisibleLadderCharts(document.getElementById('tab-overall'));

        // ---------- Language chart (top 25, overall) ----------
        (function renderLanguageChart() {{
            const top = [...DATA.languages].slice(0, 25).sort((a, b) => a.pairs - b.pairs);
            const traceSamples = {{
                y: top.map(l => l.name), x: top.map(l => l.samples), name: 'Samples',
                type: 'bar', orientation: 'h', marker: {{ color: SERIES_COLORS.samples, cornerradius: 3 }},
                hovertemplate: '<b>%{{y}}</b><br>Samples: %{{x:,}}<br><i>click to open &#8599;</i><extra></extra>',
            }};
            const tracePairs = {{
                y: top.map(l => l.name), x: top.map(l => l.pairs), name: 'Minimal pairs',
                type: 'bar', orientation: 'h', marker: {{ color: SERIES_COLORS.pairs, cornerradius: 3 }},
                hovertemplate: '<b>%{{y}}</b><br>Pairs: %{{x:,}}<br><i>click to open &#8599;</i><extra></extra>',
            }};
            const topEntry = top[top.length - 1];
            const maxX = Math.max(...top.map(l => Math.max(l.samples, l.pairs)));
            Plotly.newPlot('language-chart', [traceSamples, tracePairs], {{
                ...BASE_LAYOUT,
                barmode: 'group',
                margin: {{ t: 10, r: 70, b: 40, l: 140 }},
                xaxis: {{ title: {{ text: 'Count' }}, gridcolor: THEME.grid, range: [0, maxX * 1.18] }},
                yaxis: {{ automargin: true }},
                legend: {{ orientation: 'h', y: 1.05, x: 0 }},
                annotations: [endLabelAnnotation(topEntry.pairs, topEntry.name, topEntry.pairs.toLocaleString())],
            }}, CONFIG);
            wireClickThrough('language-chart', pt => {{
                const l = top[pt.pointIndex];
                return l && l.url;
            }});
        }})();

        // ---------- Language table (sortable + searchable, overall) ----------
        (function renderLanguageTable() {{
            let sortKey = 'pairs', sortDir = -1;
            let filterText = '';
            const tbody = document.getElementById('language-tbody');
            const countNote = document.getElementById('lang-count');

            function draw() {{
                let rows = DATA.languages.filter(l => matchesLangFilter(l.name, filterText));
                rows = [...rows].sort((a, b) => {{
                    const av = a[sortKey], bv = b[sortKey];
                    if (av === null) return 1;
                    if (bv === null) return -1;
                    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
                    return sortDir * (av - bv);
                }});
                tbody.innerHTML = rows.map(l => `
                    <tr${{l.pairs === 0 ? ' class="row-zero"' : ''}}>
                        <td>${{langLinkHtml(l.name)}}</td>
                        <td class="num">${{l.samples.toLocaleString()}}</td>
                        <td class="num">${{l.pairs.toLocaleString()}}</td>
                        <td class="num">${{fmtRatio(l.ratio)}}</td>
                        <td class="num">${{fmtAcc(l.acc)}}</td>
                        <td class="num">${{l.n_lemma.toLocaleString()}}</td>
                        <td class="num">${{l.n_conditions}}</td>
                    </tr>`).join('');
                countNote.textContent = `${{rows.length}} of ${{DATA.languages.length}} languages`;
                updateSortIndicators('language-table', sortKey, sortDir);
            }}
            document.querySelectorAll('#language-table thead .th-sort-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const key = btn.dataset.key;
                    sortDir = (key === sortKey) ? -sortDir : (key === 'name' ? 1 : -1);
                    sortKey = key;
                    draw();
                }});
            }});
            document.getElementById('lang-search').addEventListener('input', (e) => {{
                filterText = e.target.value;
                draw();
            }});
            draw();
        }})();

        // ---------- Treebanks per language (sortable + searchable, overall) ----------
        (function renderTreebankTable() {{
            let sortKey = 'n_treebanks', sortDir = -1;
            let filterText = '';
            const rows = DATA.language_treebanks.map(r => ({{ ...r, n_treebanks: r.treebanks.length }}));
            const tbody = document.getElementById('treebank-tbody');
            const countNote = document.getElementById('treebank-count');

            function draw() {{
                let filtered = rows.filter(r => matchesLangFilter(r.name, filterText));
                filtered = [...filtered].sort((a, b) => {{
                    const av = a[sortKey], bv = b[sortKey];
                    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
                    return sortDir * (av - bv);
                }});
                tbody.innerHTML = filtered.map(r => `
                    <tr${{r.n_treebanks === 0 ? ' class="row-zero"' : ''}}>
                        <td>${{r.name}}</td>
                        <td class="num">${{r.n_treebanks}}</td>
                        <td class="treebank-list">${{r.treebanks.join(', ') || '&mdash;'}}</td>
                    </tr>`).join('');
                countNote.textContent = `${{filtered.length}} of ${{rows.length}} languages`;
                updateSortIndicators('treebank-table', sortKey, sortDir);
            }}
            document.querySelectorAll('#treebank-table thead .th-sort-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const key = btn.dataset.key;
                    sortDir = (key === sortKey) ? -sortDir : (key === 'name' ? 1 : -1);
                    sortKey = key;
                    draw();
                }});
            }});
            document.getElementById('treebank-search').addEventListener('input', (e) => {{
                filterText = e.target.value;
                draw();
            }});
            draw();
        }})();

        // ---------- Samples by treebank, preview (sortable + searchable, overall) ----------
        (function renderTreebankPairsTable() {{
            let sortKey = 'samples', sortDir = -1;
            let filterText = '';
            // One row per (language, condition, treebank) -- build_stats.py
            // hands over one entry per language x condition with a nested
            // treebank->count map; flattened here since the table itself is
            // the finest-grained view this page has anywhere.
            const rows = [];
            DATA.local_treebank_pairs.forEach(entry => {{
                Object.entries(entry.samples_by_treebank).forEach(([treebank, samples]) => {{
                    if (!samples) return;
                    rows.push({{
                        name: entry.language, condition: entry.condition_label, treebank, samples,
                        share: entry.total_samples ? samples / entry.total_samples : 0,
                        published_pairs: entry.published_pairs, published_samples: entry.published_samples,
                    }});
                }});
            }});
            const tbody = document.getElementById('treebank-pairs-tbody');
            const countNote = document.getElementById('treebank-pairs-count');

            function draw() {{
                let filtered = rows.filter(r => matchesLangFilter(r.name, filterText));
                filtered = [...filtered].sort((a, b) => {{
                    const av = a[sortKey], bv = b[sortKey];
                    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
                    return sortDir * (av - bv);
                }});
                tbody.innerHTML = filtered.map(r => `
                    <tr>
                        <td>${{r.name}}</td>
                        <td>${{r.condition}}</td>
                        <td class="treebank-list">${{r.treebank}}</td>
                        <td class="num">${{r.samples.toLocaleString()}}</td>
                        <td class="num">${{(r.share * 100).toFixed(0)}}%</td>
                        <td class="num">${{r.published_pairs.toLocaleString()}}</td>
                        <td class="num">${{r.published_samples.toLocaleString()}}</td>
                    </tr>`).join('');
                countNote.textContent = `${{filtered.length}} of ${{rows.length}} rows`;
                updateSortIndicators('treebank-pairs-table', sortKey, sortDir);
            }}
            document.querySelectorAll('#treebank-pairs-table thead .th-sort-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const key = btn.dataset.key;
                    sortDir = (key === sortKey) ? -sortDir : (key === 'name' ? 1 : -1);
                    sortKey = key;
                    draw();
                }});
            }});
            document.getElementById('treebank-pairs-search').addEventListener('input', (e) => {{
                filterText = e.target.value;
                draw();
            }});
            draw();
        }})();

        // ---------- Language x condition heatmap (overall) ----------
        (function renderHeatmap() {{
            const langs = DATA.matrix.languages;
            const conds = DATA.matrix.conditions.map(id => {{
                const c = DATA.conditions.find(c => c.id === id);
                return c ? `${{c.label}} (${{c.group}})` : id;
            }});
            const pairsZ = DATA.matrix.pairs;
            const samplesZ = DATA.matrix.samples;
            const accZ = DATA.matrix.acc;
            // [pairs, samples, acc, url] per cell -- same regardless of
            // which metric is currently coloring the grid, so hover always
            // shows the full picture and the click-through url never changes.
            const customdata = pairsZ.map((row, i) => row.map((v, j) => [v, samplesZ[i][j], accZ[i][j], DATA.matrix.urls[i][j]]));

            const accVals = accZ.flat().filter(v => v !== null);
            const minAcc = Math.min(...accVals), maxAcc = Math.max(...accVals);

            // Row order is a permutation of indices into langs/pairsZ/etc,
            // applied after metricSpec builds z/customdata on the original
            // (by-pairs) order -- 'pairs' is the identity permutation since
            // build_stats.py already hands languages_sorted over in that
            // order; 'alpha' just resorts those same indices by name.
            const sortOrders = {{
                pairs: langs.map((_, i) => i),
                alpha: langs.map((_, i) => i).sort((a, b) => langs[a].localeCompare(langs[b])),
            }};

            const rowHeight = 13;
            const height = Math.max(300, langs.length * rowHeight + 120);
            document.getElementById('matrix-heatmap').style.height = height + 'px';

            const hovertemplate = '<b>%{{y}}</b> &times; <b>%{{x}}</b><br>Samples: %{{customdata[1]:,}}<br>Pairs: %{{customdata[0]:,}}' +
                '<br>Accuracy: %{{customdata[2]:.1%}}<br><i>click to open &#8599;</i><extra></extra>';

            function metricSpec(metric) {{
                if (metric === 'acc') {{
                    // Not log-scaled and not 0-anchored like pairs -- accuracy is
                    // already a bounded [0,1] ratio, and zooming the color domain
                    // to the range actually observed (not the full [0,1]) keeps
                    // real variation visible instead of everything reading as
                    // "uniformly high" against a 0 floor nothing here is near.
                    const z = accZ.map((row, i) => row.map((v, j) => (samplesZ[i][j] === 0) ? null : v));
                    const ticks = [minAcc, (minAcc + maxAcc) / 2, maxAcc];
                    return {{
                        z, zmin: minAcc, zmax: maxAcc,
                        // Anchored at the observed midpoint, not a fixed 0.5 --
                        // accuracy here rarely dips anywhere near chance, so
                        // anchoring at the real min-max midpoint is what actually
                        // makes the low end of this dataset's own range read as
                        // "red" and the high end "blue" instead of everything
                        // clustering in one narrow band of the scale.
                        colorscale: DIVERGING_SCALE,
                        colorbarTitle: 'Accuracy',
                        tickvals: ticks, ticktext: ticks.map(v => (v * 100).toFixed(0) + '%'),
                    }};
                }}
                // log1p so the huge NPA counts don't wash out everything else,
                // while 0 (a real "ran, produced nothing") stays distinct from
                // null (never run at all -- Plotly renders null as a gap).
                const z = pairsZ.map((row, i) => row.map((v, j) => (samplesZ[i][j] === 0 && v === 0) ? null : Math.log1p(v)));
                const maxPairs = Math.max(0, ...pairsZ.flat());
                const tickVals = [0, 10, 100, 1000, 10000, 100000].filter(v => v < maxPairs);
                tickVals.push(maxPairs);
                return {{
                    z, zmin: undefined, zmax: undefined,
                    // Same DIVERGING_SCALE as Accuracy, over the log1p domain
                    // (0 at the sparse/red end, the biggest count at the
                    // rich/blue end) -- the same shared scale everywhere on
                    // this page a color encodes a value, so "red" and "blue"
                    // mean the same visual thing regardless of which toggle
                    // put them there.
                    colorscale: DIVERGING_SCALE,
                    colorbarTitle: 'Pairs',
                    tickvals: tickVals.map(v => Math.log1p(v)), ticktext: tickVals.map(v => v.toLocaleString()),
                }};
            }}

            function draw(metric, sortMode) {{
                const spec = metricSpec(metric);
                const order = sortOrders[sortMode] || sortOrders.pairs;
                const langsOrdered = order.map(i => langs[i]);
                const zOrdered = order.map(i => spec.z[i]);
                const customdataOrdered = order.map(i => customdata[i]);
                Plotly.react('matrix-heatmap', [{{
                    type: 'heatmap',
                    x: conds, y: langsOrdered, z: zOrdered, customdata: customdataOrdered,
                    zmin: spec.zmin, zmax: spec.zmax,
                    // Derived from the theme's own panel/accent tokens (not a
                    // fixed light-mode hex) so the low end still adapts in dark
                    // mode. The z=0 stop is a faint tint, not pure panelBg: this
                    // heatmap sits on the card surface, and panelBg/card are
                    // close enough in the palette that a true-panelBg cell was
                    // nearly invisible against it -- exactly the "ran, produced
                    // one pair" case the copy below promises stays distinct
                    // from a genuine gap. The 0.12 stop keeps mid-low values
                    // visible too, instead of fading under log compression.
                    colorscale: spec.colorscale,
                    showscale: true,
                    colorbar: {{
                        title: {{ text: spec.colorbarTitle, side: 'right', font: {{ size: 10, color: THEME.muted }} }},
                        tickmode: 'array',
                        tickvals: spec.tickvals,
                        ticktext: spec.ticktext,
                        outlinewidth: 0,
                        thickness: 12,
                        len: 0.6,
                        tickfont: {{ size: 9, color: THEME.muted }},
                    }},
                    xgap: 1.5, ygap: 1,
                    hovertemplate,
                }}], {{
                    ...BASE_LAYOUT,
                    margin: {{ t: 90, r: 70, b: 10, l: 140 }},
                    height,
                    xaxis: {{ side: 'top', tickangle: -45, automargin: true }},
                    yaxis: {{ autorange: 'reversed', automargin: true, tickfont: {{ size: 9 }} }},
                }}, CONFIG);
            }}

            let currentMetric = 'pairs', currentSort = 'pairs';
            draw(currentMetric, currentSort);
            // Wired once, not inside draw() -- Plotly.react keeps the same
            // graph div across re-draws, so a listener added on every toggle
            // click would stack duplicates instead of replacing one.
            // customdata (and its url at [3]) is identical regardless of
            // metric/sort, so one wiring covers every combination.
            wireClickThrough('matrix-heatmap', pt => pt.customdata && pt.customdata[3]);

            // Two independent toggle groups (color metric, row order) share
            // the .metric-toggle-btn look but must clear/set "active" only
            // within their own group -- scoped to each group's own
            // container, not every .metric-toggle-btn on the page, so
            // clicking one group doesn't blow away the other's selection.
            function wireToggleGroup(containerId, onPick) {{
                const container = document.getElementById(containerId);
                container.querySelectorAll('.metric-toggle-btn').forEach(btn => {{
                    btn.addEventListener('click', () => {{
                        container.querySelectorAll('.metric-toggle-btn').forEach(b => {{
                            b.classList.toggle('active', b === btn);
                            b.setAttribute('aria-checked', b === btn ? 'true' : 'false');
                        }});
                        onPick(btn);
                        draw(currentMetric, currentSort);
                    }});
                }});
            }}
            wireToggleGroup('heatmap-metric-toggle', btn => {{ currentMetric = btn.dataset.metric; }});
            wireToggleGroup('heatmap-sort-toggle', btn => {{ currentSort = btn.dataset.sort; }});

            // Accessible twin for hover-only heat cells -- built lazily the
            // first time the <details> opens rather than unconditionally,
            // since it's a 148x10+ table nobody may ever expand. Always in
            // by-pairs order showing raw pair counts, independent of
            // whichever metric/sort the heatmap above it is currently
            // showing -- a stable full data dump, not a live mirror of it.
            const details = document.querySelector('#sec-heatmap .table-view');
            let populated = false;
            details.addEventListener('toggle', () => {{
                if (!details.open || populated) return;
                populated = true;
                document.getElementById('heatmap-table-head').innerHTML =
                    '<th scope="col">Language</th>' + conds.map(c => `<th scope="col" class="num">${{c}}</th>`).join('');
                document.getElementById('heatmap-table-body').innerHTML = langs.map((lang, i) => {{
                    const cells = pairsZ[i].map((v, j) => (samplesZ[i][j] === 0 && v === 0)
                        ? '<td class="num">&mdash;</td>'
                        : `<td class="num">${{v.toLocaleString()}}</td>`
                    ).join('');
                    return `<tr><td>${{lang}}</td>${{cells}}</tr>`;
                }}).join('');
            }});
        }})();

        // ---------- Funnel chart (overall) ----------
        (function renderFunnel() {{
            const entries = Object.entries(DATA.funnel).sort((a, b) => a[1] - b[1]);
            // These buckets are one measure (instance count) across
            // categories the axis already names, not separate series -- a
            // rainbow implied an identity distinction that wasn't there.
            // Tried a magnitude-mapped sequential ramp here first, but bar
            // *length* already carries magnitude on this linear axis, so
            // the color ramp was redundant -- and actively broke down for
            // extreme ratios (this corpus's smallest bucket is ~1/7000th of
            // its largest, so even the ramp's palest floor still rendered
            // as a bar nearly the same dark shade as the card behind it).
            // One flat accent color, same as every other bar chart on this
            // page, is simpler and stays readable at any ratio.
            const maxVal = Math.max(...entries.map(e => e[1]), 1);
            // This corpus's smallest bucket is ~1/7000th of its largest, so
            // on a linear axis its bar is sub-pixel -- invisible, not just
            // small. A log axis would fix the bar but break the "length
            // means magnitude" reading everywhere else on the page,  so
            // instead every bar gets its own value label (not just the
            // biggest, as the other bar charts here do): the number stays
            // legible even where the bar itself can't be seen.
            Plotly.newPlot('funnel-chart', [{{
                y: entries.map(e => e[0]), x: entries.map(e => e[1]),
                type: 'bar', orientation: 'h',
                marker: {{ color: THEME.accent, cornerradius: 3 }},
                text: entries.map(e => e[1].toLocaleString()),
                textposition: 'outside',
                cliponaxis: false,
                textfont: {{ size: 10, color: THEME.text }},
                hovertemplate: '<b>%{{y}}</b>: %{{x:,}}<extra></extra>',
            }}], {{
                ...BASE_LAYOUT,
                margin: {{ t: 10, r: 70, b: 40, l: 150 }},
                xaxis: {{ title: {{ text: 'Instances' }}, gridcolor: THEME.grid, range: [0, maxVal * 1.18] }},
            }}, CONFIG);

            document.getElementById('funnel-table-body').innerHTML = [...entries].reverse()
                .map(([label, n]) => `<tr><td>${{label}}</td><td class="num">${{n.toLocaleString()}}</td></tr>`)
                .join('');
        }})();

        // ---------- Dropped-before-fitting chart (overall) ----------
        (function renderDropped() {{
            const entries = Object.entries(DATA.dropped_before_fitting).sort((a, b) => a[1] - b[1]);
            const maxVal = Math.max(...entries.map(e => e[1]), 1);
            Plotly.newPlot('dropped-chart', [{{
                y: entries.map(e => e[0]), x: entries.map(e => e[1]),
                type: 'bar', orientation: 'h',
                marker: {{ color: SERIES_COLORS.samples, cornerradius: 3 }},
                text: entries.map(e => e[1].toLocaleString()),
                textposition: 'outside',
                cliponaxis: false,
                textfont: {{ size: 10, color: THEME.text }},
                hovertemplate: '<b>%{{y}}</b>: %{{x:,}}<extra></extra>',
            }}], {{
                ...BASE_LAYOUT,
                margin: {{ t: 10, r: 70, b: 40, l: 100 }},
                xaxis: {{ title: {{ text: 'Instances' }}, gridcolor: THEME.grid, range: [0, maxVal * 1.18] }},
            }}, CONFIG);
        }})();

        // ---------- Samples vs. pairs scatter (overall) ----------
        (function renderScatterChart() {{
            const rows = DATA.languages.filter(l => l.samples > 0);
            // A log axis can't plot 0, but a zero-pair language is exactly
            // the case worth seeing (samples found, nothing usable came of
            // them) -- nudge it to a small positive floor so it still shows,
            // near the bottom, distinct from the nonzero cluster above it.
            const xs = rows.map(l => l.samples);
            const ysPlot = rows.map(l => Math.max(l.pairs, 0.5));
            const lo = Math.min(...xs, ...ysPlot);
            const hi = Math.max(...xs, ...ysPlot);

            const refLine = {{
                x: [lo, hi], y: [lo, hi], mode: 'lines', type: 'scatter',
                line: {{ color: THEME.muted, width: 1.5, dash: 'dash' }},
                hoverinfo: 'skip', showlegend: false,
            }};

            const hovertemplate = '<b>%{{text}}</b><br>Samples: %{{x:,}}<br>Pairs: %{{customdata[0]:,}}<br>Pairs/sample: %{{customdata[1]}}' +
                '%{{customdata[2]}}<br><i>click to open &#8599;</i><extra></extra>';
            const customdataFor = (l) => [
                l.pairs, l.ratio === null ? '&mdash;' : l.ratio.toFixed(2) + '&times;',
                l.acc === null ? '' : `<br>Accuracy: ${{(l.acc * 100).toFixed(1)}}%`, l.url,
            ];

            // Split by whether a language has an accuracy figure at all --
            // a continuous Plotly colorscale can't represent "no data" for
            // some points inside one array, so those get their own flat-gray
            // trace instead of silently defaulting to some color on the scale.
            const withAcc = rows.filter(l => l.acc !== null);
            const noAcc = rows.filter(l => l.acc === null);
            const accVals = withAcc.map(l => l.acc);
            const minAcc = accVals.length ? Math.min(...accVals) : 0;
            const maxAcc = accVals.length ? Math.max(...accVals) : 1;

            const pointsWithAcc = {{
                x: withAcc.map(l => l.samples), y: withAcc.map(l => Math.max(l.pairs, 0.5)),
                mode: 'markers', type: 'scatter', text: withAcc.map(l => l.name),
                customdata: withAcc.map(customdataFor),
                marker: {{
                    size: 9, opacity: 0.8, color: withAcc.map(l => l.acc),
                    colorscale: DIVERGING_SCALE,
                    cmin: minAcc, cmax: maxAcc,
                    line: {{ width: 1, color: THEME.card }},
                    colorbar: {{
                        title: {{ text: 'Accuracy', side: 'right', font: {{ size: 10, color: THEME.muted }} }},
                        tickmode: 'array',
                        tickvals: [minAcc, (minAcc + maxAcc) / 2, maxAcc],
                        ticktext: [minAcc, (minAcc + maxAcc) / 2, maxAcc].map(v => (v * 100).toFixed(0) + '%'),
                        outlinewidth: 0, thickness: 12, len: 0.6, tickfont: {{ size: 9, color: THEME.muted }},
                    }},
                }},
                hovertemplate, showlegend: false,
            }};
            const pointsNoAcc = {{
                x: noAcc.map(l => l.samples), y: noAcc.map(l => Math.max(l.pairs, 0.5)),
                mode: 'markers', type: 'scatter', text: noAcc.map(l => l.name),
                customdata: noAcc.map(customdataFor),
                marker: {{ size: 9, opacity: 0.5, color: THEME.muted, line: {{ width: 1, color: THEME.card }} }},
                hovertemplate, showlegend: false,
            }};

            Plotly.newPlot('scatter-chart', [refLine, pointsWithAcc, pointsNoAcc], {{
                ...BASE_LAYOUT,
                margin: {{ t: 10, r: 20, b: 45, l: 55 }},
                xaxis: {{ type: 'log', title: {{ text: 'Samples' }}, gridcolor: THEME.grid }},
                yaxis: {{ type: 'log', title: {{ text: 'Minimal pairs' }}, gridcolor: THEME.grid }},
            }}, CONFIG);
            wireClickThrough('scatter-chart', pt => pt.customdata && pt.customdata[3]);
        }})();

        // ---------- Category tabs: condition lookup + per-condition drill-down ----------
        const CONDITION_LOOKUP = {{}};
        DATA.conditions.forEach(c => {{ CONDITION_LOOKUP[c.id] = {{ ...c, isNpaSub: false }}; }});
        DATA.npa_subgroups.forEach(s => {{ CONDITION_LOOKUP[s.id] = {{ ...s, group: 'Noun Phrase', isNpaSub: true }}; }});

        // includeZero appends this condition's tried-but-zero-samples
        // languages too (see CONDITION_LOOKUP[condId].zero_languages) --
        // real language names, but every metric on them is a flat 0/null:
        // a zero-yield language has no LANGUAGES.six entry at all for this
        // condition (see build_stats.py's _tried_languages comment), so
        // there's no head_unk/nsubj_unk/both_unk data for it either, not
        // even the "dropped before fitting" figures -- those need a real
        // entry to be attached to, same as every other per-language field.
        function getLangRows(condId, isNpaSub, includeZero) {{
            const src = isNpaSub ? DATA.npa_matrix : DATA.matrix;
            const colKey = isNpaSub ? 'subgroups' : 'conditions';
            const idx = src[colKey].indexOf(condId);
            const rows = [];
            const seen = new Set();
            src.languages.forEach((lang, i) => {{
                const samples = src.samples[i][idx] || 0;
                const pairs = src.pairs[i][idx] || 0;
                if (samples > 0 || pairs > 0) {{
                    seen.add(lang);
                    rows.push({{
                        name: lang, samples, pairs, ratio: samples ? pairs / samples : null,
                        acc: src.acc[i][idx], n_lemma: src.n_lemma[i][idx], url: src.urls[i][idx],
                        head_unk: src.head_unk[i][idx] || 0, nsubj_unk: src.nsubj_unk[i][idx] || 0,
                        both_unk: src.both_unk[i][idx] || 0,
                    }});
                }}
            }});
            if (includeZero) {{
                const info = CONDITION_LOOKUP[condId];
                (info.zero_languages || []).forEach(lang => {{
                    if (seen.has(lang)) return;
                    rows.push({{
                        name: lang, samples: 0, pairs: 0, ratio: null, acc: null, n_lemma: 0, url: null,
                        head_unk: 0, nsubj_unk: 0, both_unk: 0,
                    }});
                }});
            }}
            return rows;
        }}

        const CAT_STATE = {{}}; // slug -> {{condId, isNpaSub, sortKey, sortDir, filterText, rows}}

        function drawCategoryTable(slug) {{
            const state = CAT_STATE[slug];
            const tbody = document.getElementById(`cat-${{slug}}-tbody`);
            const countNote = document.getElementById(`cat-${{slug}}-count`);
            let filtered = state.rows.filter(r => matchesLangFilter(r.name, state.filterText));
            filtered = [...filtered].sort((a, b) => {{
                const av = a[state.sortKey], bv = b[state.sortKey];
                if (av === null) return 1;
                if (bv === null) return -1;
                if (typeof av === 'string') return state.sortDir * av.localeCompare(bv);
                return state.sortDir * (av - bv);
            }});
            tbody.innerHTML = filtered.map(r => `
                <tr${{r.pairs === 0 ? ' class="row-zero"' : ''}}>
                    <td>${{langLinkHtml(r.name)}}</td>
                    <td class="num">${{r.samples.toLocaleString()}}</td>
                    <td class="num">${{r.pairs.toLocaleString()}}</td>
                    <td class="num">${{fmtRatio(r.ratio)}}</td>
                    <td class="num">${{fmtAcc(r.acc)}}</td>
                    <td class="num">${{r.n_lemma.toLocaleString()}}</td>
                    <td class="num col-head_unk">${{r.head_unk.toLocaleString()}}</td>
                    <td class="num col-nsubj_unk">${{r.nsubj_unk.toLocaleString()}}</td>
                    <td class="num col-both_unk">${{r.both_unk.toLocaleString()}}</td>
                </tr>`).join('');
            countNote.textContent = `${{filtered.length}} of ${{state.rows.length}} languages`;
            updateSortIndicators(`cat-${{slug}}-table`, state.sortKey, state.sortDir);
        }}

        function zeroLanguagesHtml(zeroLangs) {{
            if (!zeroLangs || !zeroLangs.length) return '';
            const n = zeroLangs.length;
            return `<details class="zero-langs">
                <summary>${{n}} language${{n !== 1 ? 's' : ''}} tried, zero samples anywhere</summary>
                <p class="zero-lang-list">${{zeroLangs.join(', ')}}</p>
            </details>`;
        }}

        // Shared by every coverage funnel on the page (Overall's global one,
        // each category's own, each picked condition's own). A plain
        // left-aligned bar chart, not Plotly's own 'funnel' trace type --
        // that type centers every bar around the same midline, tapering
        // symmetrically inward (the classic funnel silhouette), which makes
        // bar *lengths* harder to compare at a glance than a standard
        // left-aligned bar chart does; this data's narrowing-pipeline shape
        // still reads fine without the taper. No click-through wiring here
        // (nothing to link to per step), so plain Plotly.newPlot on every
        // call is fine -- unlike the heatmap/category charts, there's no
        // listener that could stack from redrawing this one repeatedly.
        function renderCoverageFunnel(divId, cov) {{
            // Plotly renders category labels straight into SVG <text>, which
            // doesn't decode HTML entities the way real HTML does -- "&ge;"
            // would show up as the literal 4 characters, not "≥", so
            // this needs the actual Unicode character, unlike every other
            // "&ge;"/"&mdash;"/etc. string on this page that goes through
            // innerHTML instead.
            const steps = [
                ['tried', cov.tried], ['any samples', cov.with_samples],
                ['≥10 pairs', cov.ge10], ['≥30 pairs', cov.ge30], ['≥50 pairs', cov.ge50],
                ['≥100 pairs', cov.ge100], ['≥500 pairs', cov.ge500],
            ];
            const maxVal = Math.max(...steps.map(s => s[1]), 1);
            Plotly.newPlot(divId, [{{
                type: 'bar', orientation: 'h',
                y: steps.map(s => s[0]), x: steps.map(s => s[1]),
                text: steps.map(s => s[1].toLocaleString()),
                textposition: 'outside',
                cliponaxis: false,
                textfont: {{ color: THEME.text, size: 11 }},
                marker: {{ color: THEME.accent, cornerradius: 3 }},
                hovertemplate: '<b>%{{y}}</b>: %{{x:,}} languages<extra></extra>',
            }}], {{
                ...BASE_LAYOUT,
                margin: {{ t: 10, r: 50, b: 10, l: 90 }},
                xaxis: {{ range: [0, maxVal * 1.18] }},
                // No autorange override -- Plotly's default category order
                // for a horizontal bar puts the first array entry ("tried",
                // the largest count) at the bottom and later ones upward, so
                // "≥500 pairs" (the smallest, last in `steps`) lands at the
                // top: inverted from the funnel-shaped original, tried at
                // the bottom now instead of the top.
            }}, CONFIG);
        }}

        // Renders every SSR-emitted funnel that's actually visible right
        // now (a hidden category tab's own ladder-chart div exists in the
        // DOM already, data-cov and all, but Plotly into a display:none
        // container lays out at 0x0 -- same reasoning as the heatmap's
        // accessible-table twin) -- called once at load for the Overall
        // tab's chart, and again from initCategoryTab for each category's.
        function renderVisibleLadderCharts(root) {{
            root.querySelectorAll('.ladder-chart[data-cov]').forEach(el => {{
                renderCoverageFunnel(el.id, JSON.parse(el.dataset.cov));
            }});
        }}

        function coverageLadderHtml(info, chartId) {{
            return `<div class="ladder-chart" id="${{chartId}}" style="height: 200px;"></div>`
                + zeroLanguagesHtml(info.zero_languages);
        }}

        function decisionTreeUrl(condId, isNpaSub) {{
            return isNpaSub
                ? `https://jshrdt.github.io/multiblimp/npa/${{condId}}/index.html`
                : `https://jshrdt.github.io/multiblimp/${{condId}}/index.html`;
        }}

        // One color per unk category, reusing colors already established
        // elsewhere on the page (the diverging scale's red/purple, plus the
        // neutral "samples" gray) rather than inventing new ones just for
        // these three extra series.
        // THEME.muted was the original both_unk color -- identical to
        // Samples' own color, so the two were indistinguishable in both the
        // chart and its legend. A red/purple blend instead: distinct from
        // every other series here, and "both unknown" being a mix of
        // head_unk's red and nsubj_unk's purple also reads as "both of the
        // other two failure modes at once", which is what it actually is.
        const UNK_SERIES = {{
            head_unk: {{ label: 'Head unknown', color: () => THEME.danger }},
            nsubj_unk: {{ label: 'Subject unknown', color: () => PURPLE }},
            both_unk: {{ label: 'Both unknown', color: () => mixHex(THEME.danger, PURPLE, 0.6) }},
        }};

        function renderCategoryDetail(slug) {{
            const state = CAT_STATE[slug];
            const info = CONDITION_LOOKUP[state.condId];
            state.rows = getLangRows(state.condId, state.isNpaSub, state.includeZero);

            document.getElementById(`cat-${{slug}}-ministats`).innerHTML = `
                <div class="mini-stats">
                    <div class="mini-stat"><div class="mini-stat-value">${{info.languages}}</div><div class="mini-stat-label">Languages</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">${{info.samples.toLocaleString()}}</div><div class="mini-stat-label">Samples</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">${{info.pairs.toLocaleString()}}</div><div class="mini-stat-label">Minimal pairs</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">${{fmtRatio(info.ratio)}}</div><div class="mini-stat-label">Pairs/sample</div></div>
                    <div class="mini-stat"><div class="mini-stat-value">${{fmtAcc(info.acc)}}</div><div class="mini-stat-label">Decision-tree accuracy</div></div>
                </div>
                ${{coverageLadderHtml(info, `cat-${{slug}}-cond-ladder-chart`)}}
                <a class="nav-link" href="${{decisionTreeUrl(state.condId, state.isNpaSub)}}" target="_blank" rel="noopener"
                   style="display:inline-block;margin-bottom:1.25rem;">${{info.label}} decision tree &amp; entropy detail &rarr;</a>
            `;
            renderCoverageFunnel(`cat-${{slug}}-cond-ladder-chart`, info);

            // includeZero appends zero-sample rows at the end of state.rows
            // (see getLangRows), so they only displace real top-25 entries
            // when a condition has fewer than 25 languages with any data at
            // all -- consistent with the checkbox's own "(table only)" label.
            const top = [...state.rows].slice(0, 25).sort((a, b) => a.pairs - b.pairs);
            const extraKeys = [...state.extras];
            const chartHeight = Math.max(220, top.length * 22 + 90);
            document.getElementById(`cat-${{slug}}-chart`).style.height = chartHeight + 'px';
            const maxX = top.length
                ? Math.max(...top.map(r => Math.max(r.samples, r.pairs, ...extraKeys.map(k => r[k]))))
                : 1;
            const annotations = top.length
                ? [endLabelAnnotation(top[top.length - 1].pairs, top[top.length - 1].name, top[top.length - 1].pairs.toLocaleString())]
                : [];
            const traces = [
                {{
                    y: top.map(r => r.name), x: top.map(r => r.samples), name: 'Samples', type: 'bar', orientation: 'h',
                    marker: {{ color: SERIES_COLORS.samples, cornerradius: 3 }}, hovertemplate: '<b>%{{y}}</b><br>Samples: %{{x:,}}<br><i>click to open &#8599;</i><extra></extra>',
                }},
                {{
                    y: top.map(r => r.name), x: top.map(r => r.pairs), name: 'Minimal pairs', type: 'bar', orientation: 'h',
                    marker: {{ color: SERIES_COLORS.pairs, cornerradius: 3 }}, hovertemplate: '<b>%{{y}}</b><br>Pairs: %{{x:,}}<br><i>click to open &#8599;</i><extra></extra>',
                }},
                ...extraKeys.map(key => ({{
                    y: top.map(r => r.name), x: top.map(r => r[key]), name: UNK_SERIES[key].label, type: 'bar', orientation: 'h',
                    marker: {{ color: UNK_SERIES[key].color(), cornerradius: 3 }},
                    hovertemplate: `<b>%{{y}}</b><br>${{UNK_SERIES[key].label}}: %{{x:,}}<br><i>click to open &#8599;</i><extra></extra>`,
                }})),
            ];
            // The legend was sized for its original 2 fixed series (Samples,
            // Minimal pairs), which fit on one line at y:1.05 with a 10px
            // top margin. Each checked extra series adds another legend
            // entry, which wraps onto additional lines once they no longer
            // fit the chart's width -- without more headroom, those wrapped
            // lines render on top of the first data row instead of above
            // the plot. Not knowing the exact wrap point (it depends on the
            // container's actual pixel width), this errs generous once 2+
            // extras are on, rather than trying to predict the exact line
            // count.
            const legendLines = traces.length > 3 ? 2 : 1;
            Plotly.newPlot(`cat-${{slug}}-chart`, traces, {{
                ...BASE_LAYOUT,
                barmode: 'group',
                margin: {{ t: 10 + (legendLines - 1) * 34, r: 70, b: 40, l: 140 }},
                xaxis: {{ title: {{ text: 'Count' }}, gridcolor: THEME.grid, range: [0, maxX * 1.18] }},
                yaxis: {{ automargin: true }},
                legend: {{ orientation: 'h', y: 1.05 + (legendLines - 1) * 0.09, x: 0 }},
                annotations,
            }}, CONFIG);
            wireClickThrough(`cat-${{slug}}-chart`, pt => {{
                const r = top[pt.pointIndex];
                return r && r.url;
            }});

            drawCategoryTable(slug);
        }}

        // Same 3-bucket breakdown as the Overview tab's dropped-chart, just
        // summed over this category's own conditions instead of all 10 --
        // "Noun Phrase" uses its 4 role-pair subgroups (not the "npa"
        // aggregate condition, which would double-count against them, same
        // reasoning as the picker itself; see _category_items's docstring).
        function categoryDropped(group) {{
            const items = group === 'Noun Phrase'
                ? DATA.npa_subgroups
                : DATA.conditions.filter(c => c.group === group && c.id !== 'npa');
            const totals = {{}};
            items.forEach(it => {{
                Object.entries(it.dropped_before_fitting).forEach(([label, n]) => {{
                    totals[label] = (totals[label] || 0) + n;
                }});
            }});
            return totals;
        }}

        function renderCategoryDropped(slug, group) {{
            const entries = Object.entries(categoryDropped(group)).sort((a, b) => a[1] - b[1]);
            const maxVal = Math.max(...entries.map(e => e[1]), 1);
            Plotly.newPlot(`cat-${{slug}}-dropped-chart`, [{{
                y: entries.map(e => e[0]), x: entries.map(e => e[1]),
                type: 'bar', orientation: 'h',
                marker: {{ color: SERIES_COLORS.samples, cornerradius: 3 }},
                text: entries.map(e => e[1].toLocaleString()),
                textposition: 'outside', cliponaxis: false,
                textfont: {{ size: 10, color: THEME.text }},
                hovertemplate: '<b>%{{y}}</b>: %{{x:,}}<extra></extra>',
            }}], {{
                ...BASE_LAYOUT,
                margin: {{ t: 10, r: 70, b: 30, l: 100 }},
                xaxis: {{ gridcolor: THEME.grid, range: [0, maxVal * 1.18] }},
            }}, CONFIG);
        }}

        function initCategoryTab(slug) {{
            const tabEl = document.getElementById(`tab-${{slug}}`);
            const group = tabEl.dataset.group;
            // This category's own coverage funnel (the ladder-chart div
            // _category_tab_html already emitted with its data-cov) -- the
            // tab is visible by the time initCategoryTab runs (it's only
            // called from showTab), so no hidden-container 0x0 concern here.
            renderVisibleLadderCharts(tabEl);

            // Lazy like the heatmap's "View as table" twin -- this chart
            // lives inside a collapsed <details>, and Plotly.newPlot into a
            // closed (display:none) container lays out at 0x0, so it waits
            // for the first real open instead of drawing blind on tab init.
            const droppedDetails = document.getElementById(`cat-${{slug}}-dropped-details`);
            let droppedPopulated = false;
            droppedDetails.addEventListener('toggle', () => {{
                if (!droppedDetails.open || droppedPopulated) return;
                droppedPopulated = true;
                renderCategoryDropped(slug, group);
            }});

            const picker = document.getElementById(`cat-${{slug}}-picker`);
            const firstBtn = picker.querySelector('.picker-btn');
            CAT_STATE[slug] = {{
                condId: firstBtn.dataset.cond, isNpaSub: firstBtn.dataset.npasub === 'true',
                sortKey: 'pairs', sortDir: -1, filterText: '', rows: [],
                extras: new Set(), includeZero: false,
            }};

            const table = document.getElementById(`cat-${{slug}}-table`);
            document.querySelectorAll(`#cat-${{slug}}-extra-toggle .extra-check`).forEach(cb => {{
                cb.addEventListener('change', () => {{
                    const key = cb.dataset.extra;
                    const state = CAT_STATE[slug];
                    if (cb.checked) state.extras.add(key); else state.extras.delete(key);
                    table.classList.toggle(`show-${{key}}`, cb.checked);
                    renderCategoryDetail(slug);
                }});
            }});
            document.getElementById(`cat-${{slug}}-include-zero`).addEventListener('change', (e) => {{
                CAT_STATE[slug].includeZero = e.target.checked;
                renderCategoryDetail(slug);
            }});

            picker.querySelectorAll('.picker-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    picker.querySelectorAll('.picker-btn').forEach(b => {{
                        b.classList.remove('active');
                        b.setAttribute('aria-checked', 'false');
                    }});
                    btn.classList.add('active');
                    btn.setAttribute('aria-checked', 'true');
                    CAT_STATE[slug].condId = btn.dataset.cond;
                    CAT_STATE[slug].isNpaSub = btn.dataset.npasub === 'true';
                    renderCategoryDetail(slug);
                    setHash(slug, btn.dataset.cond);
                }});
            }});

            document.querySelectorAll(`#cat-${{slug}}-table thead .th-sort-btn`).forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const key = btn.dataset.key;
                    const state = CAT_STATE[slug];
                    state.sortDir = (key === state.sortKey) ? -state.sortDir : (key === 'name' ? 1 : -1);
                    state.sortKey = key;
                    drawCategoryTable(slug);
                }});
            }});

            document.getElementById(`cat-${{slug}}-search`).addEventListener('input', (e) => {{
                CAT_STATE[slug].filterText = e.target.value;
                drawCategoryTable(slug);
            }});

            renderCategoryDetail(slug);
        }}

        // ---------- Tabs ----------
        // Overall's charts render eagerly above (it's the default-visible
        // tab); each category tab is initialized lazily on its first visit
        // instead, both to avoid 5 unused Plotly charts on every page load
        // and because Plotly.newPlot on a display:none container lays out
        // at 0x0 -- a chart already built just needs an explicit resize
        // when its tab becomes visible again, not a rebuild.
        const INITIALIZED_TABS = new Set();
        function showTab(name) {{
            document.querySelectorAll('[id^="tab-"]').forEach(panel => {{
                panel.hidden = panel.id !== `tab-${{name}}`;
            }});
            document.querySelectorAll('.tab-btn').forEach(b => {{
                const active = b.dataset.tab === name;
                b.classList.toggle('active', active);
                b.setAttribute('aria-selected', active ? 'true' : 'false');
            }});

            if (name === 'overall') {{
                ['language-chart', 'matrix-heatmap', 'funnel-chart'].forEach(id => {{
                    const el = document.getElementById(id);
                    if (el && el.data) Plotly.Plots.resize(el);
                }});
                setHash('overall');
                return;
            }}
            if (!INITIALIZED_TABS.has(name)) {{
                initCategoryTab(name);
                INITIALIZED_TABS.add(name);
            }} else {{
                const el = document.getElementById(`cat-${{name}}-chart`);
                if (el && el.data) Plotly.Plots.resize(el);
            }}
            setHash(name, CAT_STATE[name] && CAT_STATE[name].condId);
        }}
        document.querySelectorAll('.tab-btn').forEach(b => {{
            b.addEventListener('click', () => showTab(b.dataset.tab));
        }});

        // ---------- Deep-linking: #slug or #slug/condId shares the exact
        // tab + picked condition, e.g. #sv/svGa or #npa/HEAD-DET_N.
        // history.replaceState (not location.hash =) so clicking around
        // doesn't spam browser back-button history, only the final state
        // is bookmarkable/shareable.
        function setHash(tab, condId) {{
            const newHash = (tab === 'overall' && !condId) ? '' : '#' + [tab, condId].filter(Boolean).join('/');
            if (location.hash !== newHash) {{
                history.replaceState(null, '', newHash || (location.pathname + location.search));
            }}
        }}
        function applyHash() {{
            const hash = location.hash.replace(/^#/, '');
            if (!hash || hash === 'overall') {{
                showTab('overall');
                return;
            }}
            const [tab, condId] = hash.split('/');
            if (!document.getElementById(`tab-${{tab}}`)) {{
                showTab('overall');
                return;
            }}
            showTab(tab);
            if (condId) {{
                const btn = document.querySelector(`#cat-${{tab}}-picker .picker-btn[data-cond="${{condId}}"]`);
                if (btn) btn.click();
            }}
        }}
        window.addEventListener('hashchange', applyHash);
        applyHash();

        // Jump-to-section nav: driven explicitly via scrollIntoView rather
        // than relying on the browser's native #hash-anchor scroll. Native
        // fragment navigation is blocked in at least one real environment
        // this page gets viewed in (a sandboxed preview pane serving local
        // files as a data: URL, where even a bare `location.hash = 'x'`
        // silently no-ops) -- doing it ourselves works the same everywhere
        // instead of depending on which contexts allow it. Hash is still
        // updated for bookmarking/sharing where that's permitted, but
        // wrapped in try/catch since it's the same operation that throws on
        // a data: URL (see setHash's history.replaceState).
        document.querySelectorAll('.jump-nav a[href^="#"]').forEach(a => {{
            a.addEventListener('click', (e) => {{
                const id = a.getAttribute('href').slice(1);
                const target = document.getElementById(id);
                if (!target) return;
                e.preventDefault();
                target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                try {{ history.replaceState(null, '', '#' + id); }} catch (err) {{ /* e.g. data: URL sandbox */ }}
            }});
        }});
    </script>
</body>
</html>"""


def main():
    with open(STATS_PATH) as f:
        data = json.load(f)
    html = render(data)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
