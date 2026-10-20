"""CoNLL-U sentences as units (plain tokens and multiword tokens, UD ID ranges such as 3-4 "du" = "de le"), and
the statistics of a language's multiword tokens."""
import collections
import re

RANGE = re.compile(r"^(\d+)-(\d+)$")

Unit = collections.namedtuple("Unit", "mwt form lo hi comps space upos comp_upos")


def norm(form: str) -> str:
    """Lookup form: lower case, without the trailing "_" some treebanks put on a span's first component."""
    form = form.lower()
    return form[:-1] if len(form) > 1 and form.endswith("_") else form


def recase(original: str, form: str) -> str:
    """`form` in the casing of `original`: all caps for a multi-letter upper-case original, a leading capital for
    a capitalized one (a single capital such as "À" is title case, not all caps)."""
    if len(original) > 1 and original.isupper():
        return form.upper()
    if original[:1].isupper():
        return form[:1].upper() + form[1:]
    return form


def iter_sentences(path):
    """(line of '# sent_id', '# text', units) per sentence; a unit is a plain token or a multiword token
    (mwt=True, surface form in `form`, component forms in `comps`, their UPOS in `comp_upos`; a plain token's UPOS
    in `upos`). Empty nodes are skipped."""
    units, start, text, last = [], None, None, None
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line[0] == "#":
                if line.startswith("# sent_id"):
                    start = i
                elif line.startswith("# text ="):
                    text = line[len("# text ="):].strip()
                continue
            if not line.strip():
                if units:
                    yield start, text, units
                units, start, text, last = [], None, None, None
                continue
            cols = line.rstrip("\n").split("\t")
            if "." in cols[0]:
                continue
            space = len(cols) < 10 or "SpaceAfter=No" not in cols[9]
            upos = cols[3] if len(cols) > 3 else "_"
            m = RANGE.match(cols[0])
            if m:
                last = Unit(True, cols[1], int(m[1]), int(m[2]), [], space, None, [])
                units.append(last)
            elif last is not None and last.mwt and last.lo <= int(cols[0]) <= last.hi:
                last.comps.append(cols[1])
                last.comp_upos.append(upos)
            else:
                units.append(Unit(False, cols[1], int(cols[0]), int(cols[0]), [], space, upos, []))
    if units:
        yield start, text, units


def render(units) -> str:
    return "".join(u.form + (" " if u.space else "") for u in units)


def language_of(conllu_file: str) -> str:
    return conllu_file.split("/")[0][3:].split("-")[0]


def contraction_map(files):
    """(contracted, positions): {normalized components: {normalized surface: count}} over every multiword token of
    the given conllu files, and {UPOS: {(position in the span, span length)}} for where each UPOS occurs in them."""
    contracted = collections.defaultdict(collections.Counter)
    positions = collections.defaultdict(set)
    for path in files:
        for _, _, units in iter_sentences(path):
            for u in units:
                if u.mwt and len(u.comps) > 1:
                    contracted[tuple(norm(c) for c in u.comps)][norm(u.form)] += 1
                    for j, upos in enumerate(u.comp_upos):
                        positions[upos].add((j, len(u.comps)))
    return contracted, positions


def span_stats(files):
    """(stats, types) over the given conllu files: counts of sentences, sentences with a multiword token, multiword
    tokens, syntactic words and spans by component count (comp2, comp3, ...), and {(surface, components, UPOS
    pattern): count} with lower-cased forms."""
    stats, types = collections.Counter(), collections.Counter()
    for path in files:
        for _, _, units in iter_sentences(path):
            spans = [u for u in units if u.mwt]
            stats["sentences"] += 1
            stats["sent_with_mwt"] += bool(spans)
            stats["mwt_tokens"] += len(spans)
            stats["tokens"] += sum(len(u.comps) if u.mwt else 1 for u in units)
            for u in spans:
                if u.comps:
                    stats[f"comp{len(u.comps)}"] += 1
                    types[(u.form.lower(), " ".join(u.comps).lower(), "+".join(u.comp_upos))] += 1
    return stats, types
