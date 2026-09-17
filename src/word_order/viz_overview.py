from pathlib import Path
import re
import json

from .html.html_overview import create_html
from multiblimp.condition_taxonomy import (
    GROUP_PREFIXES as _GROUP_PREFIXES,
    FEATURE_SUFFIXES as _FEATURE_SUFFIXES,
    FEATURE_ORDER as _FEATURE_ORDER,
    NPA_ROLE_PROSE as _NPA_ROLE_PROSE,
    NPA_ROLE_ORDER as _NPA_ROLE_ORDER,
    NPA_FEATURE_NAMES as _NPA_FEATURE_NAMES,
    NPA_FEATURE_ORDER as _NPA_FEATURE_ORDER,
)


def _extract_plot_data(html_content: str) -> dict | None:
    """Extract plotData.six and plotData.binary from a deprel index HTML page.

    Anchored to the "plotData" assignment specifically -- the diagnostics-
    enabled page (_create_diagnostics_html, what every current sva_trees run
    produces) also declares a LANGUAGES blob with its own six:/binary: keys
    earlier in the page, holding much richer nested per-language dicts. A
    bare "six:...binary:..." search matches that one first, and its nested
    arrays (e.g. "buckets": {"no_match": [33, 19.3], ...}) break the
    non-greedy [.+?] used here, which relies on plotData's own six/binary
    arrays never containing a nested "[" (see word_order.viz_deprel.
    generate_plot_data -- each entry is a flat dict of scalars).

    \\s* alone (not a hardcoded \\n) between six/binary: the classic table page
    (_create_classic_html) spreads plotData across lines; the diagnostics
    page embeds it on one line, which a literal \\n requirement never matches.

    [.*?] not [.+?]: a deprel with zero successfully-processed languages
    renders "plotData = { six: [], binary: [] }" -- the "+" required at
    least one character inside the brackets, so an empty array never
    matched and the whole deprel silently vanished from the overview
    instead of showing up as a "no data yet" panel.
    """
    match = re.search(
        r"plotData\s*=\s*\{\s*six:\s*(\[.*?\]),\s*binary:\s*(\[.*?\])\s*\}",
        html_content,
    )
    if not match:
        return None
    return {
        "six": json.loads(match.group(1)),
        "binary": json.loads(match.group(2)),
    }


def _safe_id(deprel: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", deprel)


# _GROUP_PREFIXES/_FEATURE_SUFFIXES/_FEATURE_ORDER and the NPA vocabulary
# below now come from multiblimp.condition_taxonomy -- the single shared
# copy of this naming convention, also used by scripts/overview/
# build_stats.py's CONDITION_META, so the two overview pages can't drift
# apart on what a condition id means. See that module's own docstring.
# Prefix alternation built from _GROUP_PREFIXES itself (longest-first, so a
# shorter prefix that happens to be another's suffix -- none today, but
# nothing guarantees that stays true -- can never shadow it) rather than
# hardcoded here a second time, so a new prefix only ever needs adding in
# condition_taxonomy.py.
_DEPREL_RE = re.compile(
    rf"^({'|'.join(sorted(_GROUP_PREFIXES, key=len, reverse=True))})(Na|Ga|Pa)(?:_(.+))?$"
)
_NPA_LEAF_RE = re.compile(r"^([A-Z]+)-([A-Z]+)_([A-Za-z]+)$")


def _npa_axes(deprel: str) -> tuple | None:
    """(role_key, role_label, role_order, feat_key, feat_label, feat_order)
    for an "npa/{ROLE1}-{ROLE2}_{FeatAbbrev}" deprel (e.g. "npa/HEAD-DET_N"
    -> ("HEAD-DET", "Head–Determiner", (0, 1), "N", "Number", 0)), or None
    if it doesn't match that convention. Single source of truth for both
    axes NPA conditions can be grouped by -- _classify_deprel (role pair as
    subgroup) and the role-pair/feature toggle in generate_html_overview_
    index (either axis as the top-level grouping) both call this rather
    than re-parsing the deprel string themselves.
    """
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
    """(group label, panel label, sort key within its group/subgroup,
    subgroup key) for a deprel/target_id/url_path like "svNa", "spGa_
    keepunk", "saPa", or NPA's "npa/{ROLE1}-{ROLE2}_{FeatAbbrev}" (e.g.
    "npa/HEAD-DET_N"). Anything matching neither convention falls into a
    catch-all "Other" group instead of being dropped, sorted after the
    recognized ones.

    subgroup key is None for every non-NPA deprel (sv/sp/sa render as one
    flat grid per group, same as before subgroups existed) and
    (role_order_tuple, prose_label) for NPA (e.g. ((0, 1), "Head–
    Determiner")) -- generate_html_overview_index nests a group's panels
    one tier deeper, under their own subheader, whenever every entry in it
    shares a non-None subgroup key; role_order_tuple (not used for display)
    sorts those subheaders by role priority instead of alphabetically.
    """
    match = _DEPREL_RE.match(deprel)
    if match:
        prefix, feature, variant = match.groups()
        label = _FEATURE_SUFFIXES[feature]
        if variant:
            label += f" ({variant.replace('_', ' ')})"
        return _GROUP_PREFIXES[prefix], label, _FEATURE_ORDER[feature], None

    axes = _npa_axes(deprel)
    if axes:
        _, role_label, role_order, _, feat_label, feat_order = axes
        return "Noun Phrase", feat_label, feat_order, (role_order, role_label)

    return "Other", deprel, 99, None


def _panel_html(deprel: str, panel_label: str, prefix: str = "") -> str:
    # deprel is always a path relative to html_directory (e.g. "svNa" or
    # NPA's "npa/HEAD-DET_N"), so "{deprel}/index.html" is only a valid
    # href when this panel is embedded in a page that itself lives at
    # html_directory's own root. _write_npa_subpages embeds the same
    # markup two levels deeper (html_directory/npa/_by_pair|_by_feature/),
    # so it passes prefix="../../" to walk back up to that root first --
    # see its own call site.
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
    """{key: {"label": str, "order": tuple, "entries": [(order, panel_label,
    deprel), ...]}} for one axis of an NPA deprel (axis=0: role pair, its
    own key/label/order plus the FEATURE as each entry's panel_label; axis=1:
    feature, the reverse) -- the data _npa_axes already carries for every
    npa/ deprel, just picking which half is the grouping key vs. the panel
    label. Two calls (one per axis) over the same npa_deprels produce the
    role-pair-first and feature-first groupings the toggle switches between.
    """
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
        for point in npa_deprels[deprel].get("six", []):
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
    # Role-pair keys ("HEAD-DET") are already plain ASCII identifiers, same
    # as every other on-disk NPA path segment -- used as-is. Feature keys
    # are the short npa_id abbreviation ("N", "Deg", "PT", ...), which makes
    # a poor page name for something meant to be clicked into by a human;
    # the full feature label ("Number") is itself always a single plain
    # ASCII word, so it's used instead. Shared by _write_npa_subpages (the
    # actual filename) and _npa_toggle_group_html (the card's href) so the
    # two can never drift apart.
    return key if axis == 0 else label


def _write_npa_subpages(npa_deprels: dict[str, dict], html_directory: Path) -> None:
    """One page per role pair (npa/_by_pair/{ROLE1}-{ROLE2}.html) and one per
    feature (npa/_by_feature/{Feature}.html), each a plain single grid of the
    same live scatter-plot panels the main overview used to inline under a
    subheader -- reuses create_html unchanged (it only ever needed a body +
    a data blob, never assumed it was building the top-level page), scoped
    to just that page's own deprels so the embedded JSON stays small.
    """
    for axis, subdir in ((0, "_by_pair"), (1, "_by_feature")):
        out_dir = html_directory / "npa" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, group in _npa_group_by(npa_deprels, axis).items():
            filename = _npa_page_filename(axis, key, group["label"])
            entries = group["entries"]
            page_deprels = {deprel: npa_deprels[deprel] for _, _, deprel in entries}
            sections_html = (
                f'        <div class="group-section">\n'
                f'            <h2 class="group-title">{group["label"]}</h2>\n'
                + _grid_html(entries, prefix="../../")
                + "\n        </div>"
            )
            html_content = create_html(sections_html, json.dumps(page_deprels))
            (out_dir / f"{filename}.html").write_text(html_content, encoding="utf-8")


def _npa_toggle_group_html(npa_deprels: dict[str, dict]) -> str:
    """The main overview's "Noun Phrase" section body: a role-pair/feature
    toggle over two card grids (see html_overview.create_html's matching
    JS), each card linking out to the per-pair/per-feature page
    _write_npa_subpages just wrote -- replaces the old inline per-role-pair
    scatter-plot grids, which now live on those pages instead.
    """
    by_pair = _npa_group_by(npa_deprels, axis=0)
    by_feature = _npa_group_by(npa_deprels, axis=1)

    pair_cards = "\n".join(
        _npa_card_html(
            group, f"npa/_by_pair/{_npa_page_filename(0, key, group['label'])}.html",
            _npa_languages_covered(npa_deprels, group["entries"]), "feature",
        )
        for key, group in sorted(by_pair.items(), key=lambda kv: kv[1]["order"])
    )
    feature_cards = "\n".join(
        _npa_card_html(
            group, f"npa/_by_feature/{_npa_page_filename(1, key, group['label'])}.html",
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
    """Generate a top-level index page with scatter plot thumbnails per dependency relation.

    Globs (recursively -- see below) for every {...}/index.html page inside
    html_directory, scrapes their embedded scatter-plot data, and produces
    an index.html at the root of that directory with the panels grouped by
    agreement kind (Subject-Verb, Subject-Participle, Subject-Auxiliary,
    Verb-Object Agreement, Noun Phrase, ...) via _classify_deprel, each
    group its own 3-column grid -- except Noun Phrase, which instead gets a
    role-pair/feature toggle over two card grids (see
    _npa_toggle_group_html); each card links to its own page under
    npa/_by_pair/ or npa/_by_feature/ (see _write_npa_subpages) holding the
    3-column scatter-plot grid that used to be inlined here.

    The glob is recursive ("**/index.html", excluding html_directory's own
    index.html -- this function's own previous output, not a deprel page)
    rather than one level ("*/index.html") specifically so it still finds
    NPA's pages, nested one level deeper than sv/sp/sa's flat {target_id}/
    convention (decision_trees/npa/{ROLE1}-{ROLE2}_{Feat}/index.html, not
    decision_trees/{something}/index.html) -- every existing flat page is
    still found too, just via a 0-extra-directories match instead of
    exactly-one.

    Args:
        html_directory: Directory containing per-deprel subdirectories with index.html files.
    """
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

    # NPA gets its own role-pair/feature toggle (see _npa_toggle_group_html)
    # instead of the flat/subheaded grid every other group uses, so it's
    # split out of the normal classification loop entirely -- a plain
    # `groups.setdefault("Noun Phrase", {})` keeps it participating in
    # group_order below without ever populating a real subgroup dict for it.
    npa_deprels = {d: data for d, data in deprels.items() if _npa_axes(d)}
    if npa_deprels:
        _write_npa_subpages(npa_deprels, html_directory)

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
        # Every remaining group (NPA is split out above, into its own
        # toggle) is flat -- _classify_deprel only ever returns a non-None
        # subgroup for NPA -- so this is always exactly one plain grid.
        return _grid_html(subgroups[None])

    # sv/sp/sa in a fixed, deliberate reading order first; any other named
    # group _classify_deprel produces (e.g. "Noun Phrase") sorted
    # alphabetically after them -- these weren't in the original
    # 3-groups-plus-"Other" design (a group name not in
    # _GROUP_PREFIXES.values() and not literally "Other" used to just be
    # silently dropped from group_order, even though _classify_deprel
    # classified it and it's sitting right there in `groups`); "Other"
    # (anything that didn't match ANY naming convention) always last.
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

    all_data_json = json.dumps(deprels)

    html_content = create_html(sections_html, all_data_json)

    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
