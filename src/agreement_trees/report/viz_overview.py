from pathlib import Path
import math
import re
import json
from urllib.parse import quote, unquote

from agreement_trees.report.html.html_overview import create_html
from agreement_trees.taxonomy import (
    GROUP_PREFIXES as _GROUP_PREFIXES,
    classify_flat_condition,
    NPA_ROLE_PROSE as _NPA_ROLE_PROSE,
    NPA_ROLE_ORDER as _NPA_ROLE_ORDER,
    NPA_FEATURE_NAMES as _NPA_FEATURE_NAMES,
    NPA_FEATURE_ORDER as _NPA_FEATURE_ORDER,
)


def _extract_plot_data(html_content: str) -> dict | None:
    """The per-language scatter points from a deprel index page: its "plotData = [...]" assignment (the
    diagnostics page also has a richer LANGUAGES blob, which must not match). [.*?], not [.+?]: a deprel with
    no processed language renders "plotData = []" and must still show as "no data yet". Returns {"acc": [...]},
    or None for a page that predates the accuracy switch (entropy units: regenerate it).
    """
    match = re.search(r"plotData\s*=\s*(\[.*?\])\s*;", html_content, flags=re.S)
    if not match:
        return None
    return {"acc": json.loads(match.group(1))}


def _safe_id(deprel: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", deprel)


def _point_url(raw_url: str | None, deprel: str, prefix: str = "") -> str | None:
    """Re-anchor a scatter point's URL for embedding outside the deprel's own directory: only the filename of
    plotData's "url" survives (stale absolute paths without ".html" exist on old pages), re-addressed as
    "{prefix}{deprel}/{filename}"."""
    if not raw_url:
        return raw_url
    filename = unquote(raw_url).rsplit("/", 1)[-1]
    if not filename.endswith(".html"):
        filename += ".html"
    return f"{prefix}{deprel}/{quote(filename)}"


def _localize_plot_data(data: dict, deprel: str, prefix: str = "") -> dict:
    return {
        measure: [
            {**point, "url": _point_url(point.get("url"), deprel, prefix)}
            for point in points
        ]
        for measure, points in data.items()
    }


# The condition naming vocabulary (group prefixes, feature suffixes, NPA roles and features) comes from
# agreement_trees.taxonomy, shared with scripts/overview/build_stats.py.
_NPA_LEAF_RE = re.compile(r"^([A-Z]+)-([A-Z]+)_([A-Za-z]+)$")


def _npa_axes(deprel: str) -> tuple | None:
    """(role_key, role_label, role_order, feat_key, feat_label, feat_order) for an "npa/{ROLE1}-{ROLE2}_{Abbrev}"
    deprel ("npa/HEAD-DET_N" -> ("HEAD-DET", "Head–Determiner", (0, 1), "N", "Number", 0)), else None. The one
    parser for both NPA grouping axes."""
    if not deprel.startswith("npa/"):
        return None
    leaf_match = _NPA_LEAF_RE.match(deprel[len("npa/"):])
    if not leaf_match:
        return None
    role1, role2, abbrev = leaf_match.groups()
    role_key = f"{role1}-{role2}"
    role_label = f"{_NPA_ROLE_PROSE.get(role1, role1)}–{_NPA_ROLE_PROSE.get(role2, role2)}"
    role_order = (_NPA_ROLE_ORDER.get(role1, 99), _NPA_ROLE_ORDER.get(role2, 99))
    feat_label = _NPA_FEATURE_NAMES.get(abbrev, abbrev)
    feat_order = _NPA_FEATURE_ORDER.get(abbrev, 99)
    return role_key, role_label, role_order, abbrev, feat_label, feat_order


def _classify_deprel(deprel: str) -> tuple[str, str, int, tuple | None]:
    """(group label, panel label, sort key, subgroup key) for a deprel / target id such as "svNa", "spGa_keepunk"
    or "npa/HEAD-DET_N"; anything else goes to "Other", sorted last. The subgroup key is None for VP conditions
    (one flat grid per group) and (role_order, label) for NPA; sorting by role_order instead of alphabetically."""
    flat = classify_flat_condition(deprel)
    if flat:
        group, label, order, variant = flat
        if variant:
            label += f" ({variant.replace('_', ' ')})"
        return group, label, order, None

    axes = _npa_axes(deprel)
    if axes:
        _, role_label, role_order, _, feat_label, feat_order = axes
        return "Noun Phrase", feat_label, feat_order, (role_order, role_label)

    return "Other", deprel, 99, None


def _panel_html(deprel: str, panel_label: str, prefix: str = "") -> str:
    # deprel is relative to html_directory; _write_npa_subpages embeds the same markup two levels deeper and
    # passes prefix="../../"
    url = f"{prefix}{deprel}/index.html"
    return (
        f'            <div class="panel">\n'
        f'                <div id="plot-{_safe_id(deprel)}" class="mini-plot"'
        f' data-url="{url}" data-deprel="{deprel}"></div>\n'
        f'                <div class="panel-label"><a href="{url}">{panel_label}</a></div>\n'
        f"            </div>"
    )


def _grid_html(entries: list, prefix: str = "") -> str:
    return (
        '            <div class="grid">\n'
        + "\n".join(_panel_html(deprel, panel_label, prefix) for _, panel_label, deprel in sorted(entries))
        + "\n            </div>"
    )


def _npa_group_by(npa_deprels: dict[str, dict], axis: int) -> dict[str, dict]:
    """{key: {"label", "order", "entries": [(order, panel_label, deprel)]}} for one NPA axis (0: role pair,
    each entry labelled by its feature; 1: feature, labelled by role pair)."""
    groups: dict[str, dict] = {}
    for deprel in npa_deprels:
        role_key, role_label, role_order, feat_key, feat_label, feat_order = _npa_axes(deprel)
        if axis == 0:
            key, label, order = role_key, role_label, role_order
            panel_label, panel_order = feat_label, feat_order
        else:
            key, label, order = feat_key, feat_label, feat_order
            panel_label, panel_order = role_label, role_order
        entry = groups.setdefault(key, {"label": label, "order": order, "entries": []})
        entry["entries"].append((panel_order, panel_label, deprel))
    return groups


def _npa_languages_covered(npa_deprels: dict[str, dict], entries: list) -> int:
    langs = set()
    for _, _, deprel in entries:
        for point in npa_deprels[deprel].get("acc", []):
            langs.add(point.get("name"))
    return len(langs)


def _npa_card_html(group: dict, url: str, n_languages: int, stat_noun: str) -> str:
    n = len(group["entries"])
    return (
        f'            <a class="npa-card" href="{url}">\n'
        f'                <div class="npa-card-title">{group["label"]}</div>\n'
        f'                <div class="npa-card-stat">{n} {stat_noun}{"s" if n != 1 else ""}'
        f' &middot; {n_languages} language{"s" if n_languages != 1 else ""}</div>\n'
        f"            </a>"
    )


def _npa_page_filename(axis: int, key: str, label: str) -> str:
    # role-pair keys are plain ASCII; feature keys are short abbreviations ("N"), so the full label ("Number")
    # names the page. Shared with the cards' hrefs.
    return key if axis == 0 else label


def _shared_axis_lo(deprels: dict[str, dict]) -> float:
    """Lower bound of every scatter axis: 0.44, or lower when any point of any condition needs it."""
    vals = [v for data in deprels.values() for d in data.get("acc", [])
            for v in (d.get("base"), d.get("tree")) if isinstance(v, (int, float)) and v == v]
    return min(0.44, math.floor((min(vals) - 0.03) * 20) / 20) if vals else 0.44


def _write_npa_subpages(npa_deprels: dict[str, dict], html_directory: Path, axis_lo: float | None = None) -> None:
    """One page per role pair (npa/by_pair/...) and per feature (npa/by_feature/...): a single grid of the
    scatter panels, with the embedded JSON scoped to that page's deprels."""
    for axis, subdir in ((0, "by_pair"), (1, "by_feature")):
        out_dir = html_directory / "npa" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, group in _npa_group_by(npa_deprels, axis).items():
            filename = _npa_page_filename(axis, key, group["label"])
            entries = group["entries"]
            page_deprels = {
                deprel: _localize_plot_data(npa_deprels[deprel], deprel, prefix="../../")
                for _, _, deprel in entries
            }
            sections_html = (
                f'        <div class="group-section">\n'
                f'            <h2 class="group-title">{group["label"]}</h2>\n'
                + _grid_html(entries, prefix="../../")
                + "\n        </div>"
            )
            # the overview is two levels up, the main page three
            html_content = create_html(
                sections_html, json.dumps(page_deprels),
                back_links=(("../../../index.html", "Main Page"), ("../../index.html", "Overview")),
                axis_lo=axis_lo,
            )
            (out_dir / f"{filename}.html").write_text(html_content, encoding="utf-8")


def _npa_toggle_group_html(npa_deprels: dict[str, dict]) -> str:
    """The "Noun Phrase" section body: a role-pair / feature toggle over two card grids, each card linking to
    the page _write_npa_subpages wrote."""
    by_pair = _npa_group_by(npa_deprels, axis=0)
    by_feature = _npa_group_by(npa_deprels, axis=1)

    pair_cards = "\n".join(
        _npa_card_html(
            group, f"npa/by_pair/{_npa_page_filename(0, key, group['label'])}.html",
            _npa_languages_covered(npa_deprels, group["entries"]), "feature",
        )
        for key, group in sorted(by_pair.items(), key=lambda kv: kv[1]["order"])
    )
    feature_cards = "\n".join(
        _npa_card_html(
            group, f"npa/by_feature/{_npa_page_filename(1, key, group['label'])}.html",
            _npa_languages_covered(npa_deprels, group["entries"]), "role pair",
        )
        for key, group in sorted(by_feature.items(), key=lambda kv: kv[1]["order"])
    )

    return (
        '            <div class="npa-view-toggle">\n'
        '                <button type="button" id="npaBtnPair" class="active">By role pair</button>\n'
        '                <button type="button" id="npaBtnFeat">By feature</button>\n'
        "            </div>\n"
        f'            <div id="npaGridPair" class="npa-card-grid">\n{pair_cards}\n            </div>\n'
        f'            <div id="npaGridFeat" class="npa-card-grid" style="display:none">\n{feature_cards}\n            </div>'
    )


def generate_html_overview_index(html_directory: str) -> None:
    """Write html_directory/index.html: scatter thumbnails of every condition, scraped from the deprel index pages
    found by a recursive glob (NPA's pages sit one level deeper; the overview's own index.html is skipped).
    Panels are grouped by _classify_deprel (Subject-Verb, ..., Other); Noun Phrase instead gets a role-pair /
    feature toggle of cards linking to npa/by_pair and npa/by_feature pages (_write_npa_subpages)."""
    html_directory = Path(html_directory)
    own_index = html_directory / "index.html"

    deprels: dict[str, dict] = {}
    for index_file in sorted(html_directory.glob("**/index.html")):
        if index_file == own_index:
            continue
        deprel = index_file.parent.relative_to(html_directory).as_posix()
        data = _extract_plot_data(index_file.read_text(encoding="utf-8"))
        if data:
            deprels[deprel] = data

    if not deprels:
        raise ValueError(
            f"No deprel index pages with plot data found in {html_directory}"
        )

    # NPA is split out of the normal grouping: it gets the toggle of cards
    npa_deprels = {d: data for d, data in deprels.items() if _npa_axes(d)}
    axis_lo = _shared_axis_lo(deprels)
    if npa_deprels:
        _write_npa_subpages(npa_deprels, html_directory, axis_lo)

    # group -> subgroup key (None, or (order_tuple, label)) -> [(order, panel_label, deprel), ...]
    groups: dict[str, dict] = {}
    for deprel in deprels:
        if deprel in npa_deprels:
            continue
        group, panel_label, order, subgroup = _classify_deprel(deprel)
        groups.setdefault(group, {}).setdefault(subgroup, []).append(
            (order, panel_label, deprel)
        )
    if npa_deprels:
        groups.setdefault("Noun Phrase", {})

    def _group_body_html(subgroups: dict) -> str:
        # every remaining group is flat (only NPA has subgroups): one plain grid
        return _grid_html(subgroups[None])

    # the VP groups in their fixed reading order, other named groups alphabetically, "Other" last
    group_order = [g for g in _GROUP_PREFIXES.values() if g in groups]
    group_order += sorted(g for g in groups if g not in group_order and g != "Other")
    if "Other" in groups:
        group_order.append("Other")

    sections_html = "\n".join(
        f'        <div class="group-section">\n'
        f'            <h2 class="group-title">{group}</h2>\n'
        + (_npa_toggle_group_html(npa_deprels) if group == "Noun Phrase" else _group_body_html(groups[group]))
        + "\n        </div>"
        for group in group_order
    )

    all_data_json = json.dumps(
        {deprel: _localize_plot_data(data, deprel) for deprel, data in deprels.items()}
    )

    html_content = create_html(sections_html, all_data_json, axis_lo=axis_lo)

    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
