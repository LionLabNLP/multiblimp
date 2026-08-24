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
_DEPREL_RE = re.compile(r"^(sv|sp|sa)(Na|Ga|Pa)(?:_(.+))?$")
_NPA_LEAF_RE = re.compile(r"^([A-Z]+)-([A-Z]+)_([A-Za-z]+)$")


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

    if deprel.startswith("npa/"):
        leaf_match = _NPA_LEAF_RE.match(deprel[len("npa/"):])
        if leaf_match:
            role1, role2, abbrev = leaf_match.groups()
            subgroup_label = (
                f"{_NPA_ROLE_PROSE.get(role1, role1)}–{_NPA_ROLE_PROSE.get(role2, role2)}"
            )
            subgroup_order = (
                _NPA_ROLE_ORDER.get(role1, 99), _NPA_ROLE_ORDER.get(role2, 99),
            )
            panel_label = _NPA_FEATURE_NAMES.get(abbrev, abbrev)
            order = _NPA_FEATURE_ORDER.get(abbrev, 99)
            return "Noun Phrase", panel_label, order, (subgroup_order, subgroup_label)

    return "Other", deprel, 99, None


def generate_html_overview_index(html_directory: str) -> None:
    """Generate a top-level index page with scatter plot thumbnails per dependency relation.

    Globs (recursively -- see below) for every {...}/index.html page inside
    html_directory, scrapes their embedded scatter-plot data, and produces
    an index.html at the root of that directory with the panels grouped by
    agreement kind (Subject-Verb, Subject-Participle, Subject-Auxiliary,
    Noun Phrase, ...) via _classify_deprel, each group its own 3-column
    grid -- or, for a group whose entries all carry a subgroup key (NPA),
    one 3-column grid per subgroup, each under its own subheader inside
    that group's section (see _classify_deprel).

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

    # group -> subgroup key (None, or (order_tuple, label)) -> [(order, panel_label, deprel), ...]
    groups: dict[str, dict] = {}
    for deprel in deprels:
        group, panel_label, order, subgroup = _classify_deprel(deprel)
        groups.setdefault(group, {}).setdefault(subgroup, []).append(
            (order, panel_label, deprel)
        )

    def _panel_html(deprel: str, panel_label: str) -> str:
        return (
            f'            <div class="panel">\n'
            f'                <div id="plot-{_safe_id(deprel)}" class="mini-plot"'
            f' data-url="/multiblimp/{deprel}" data-deprel="{deprel}"></div>\n'
            f'                <div class="panel-label"><a href="/multiblimp/{deprel}">{panel_label}</a></div>\n'
            f"            </div>"
        )

    def _grid_html(entries: list) -> str:
        return (
            '            <div class="grid">\n'
            + "\n".join(_panel_html(deprel, panel_label) for _, panel_label, deprel in sorted(entries))
            + "\n            </div>"
        )

    def _group_body_html(subgroups: dict) -> str:
        # {None: [...]}: a flat group (sv/sp/sa) -- one plain grid, no
        # subheaders, unchanged from before subgroups existed. Anything
        # else (NPA): every entry carries a real (order, label) subgroup
        # key (_classify_deprel never mixes None with a real key within
        # one group), so render one subheaded grid per subgroup, ordered
        # by that key's leading role_order_tuple.
        if None in subgroups:
            return _grid_html(subgroups[None])
        return "\n".join(
            f'            <div class="subgroup-section">\n'
            f'                <h3 class="subgroup-title">{subgroup_label}</h3>\n'
            + _grid_html(entries)
            + "\n            </div>"
            for (_, subgroup_label), entries in sorted(subgroups.items())
        )

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
        + _group_body_html(groups[group])
        + "\n        </div>"
        for group in group_order
    )

    all_data_json = json.dumps(deprels)

    html_content = create_html(sections_html, all_data_json)

    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
