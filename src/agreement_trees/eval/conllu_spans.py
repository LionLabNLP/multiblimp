"""CoNLL-U sentences as units (plain tokens and multiword tokens, UD ID ranges such as 3-4 "du" = "de le"), and
the statistics of a language's multiword tokens."""
import collections
import re
from pathlib import Path

from multiblimp.languages import HIEROGLYPH_LANGS, hiero_signs

RANGE = re.compile(r"^(\d+)-(\d+)$")
# a value starting with "=" ("Hiero==...") is malformed; the conllu parser reads it as no value
HIERO_FIELD = re.compile(r"(?:^|\|)Hiero=([^|=][^|]*)")

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


def iter_sentences(path, hieroglyphs=None):
    """(first line of the sentence block, '# text', units) per sentence, the line number conllu_sent_line holds; a unit
    is a plain token or a multiword token
    (mwt=True, surface form in `form`, component forms in `comps`, their UPOS in `comp_upos`; a plain token's UPOS
    in `upos`). Empty nodes are skipped.

    With `hieroglyphs` (default: the treebank folder belongs to a HIEROGLYPH_LANGS language) the forms are the signs
    of MISC Hiero, as Treebank.to_hieroglyphs reads them: punctuation keeps its form, sentences with a word whose signs
    are lost or unencoded (UC_No) are left out, and a multiword token without signs of its own is written as its
    components' signs."""
    path = Path(path)
    if hieroglyphs is None:
        hieroglyphs = language_of(f"{path.parent.name}/") in HIEROGLYPH_LANGS
    units, start, text, last, lost = [], None, None, None, False
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if start is None and line.strip():
                start = i
            if line[0] == "#":
                if line.startswith("# text ="):
                    text = line[len("# text ="):].strip()
                continue
            if not line.strip():
                if units and not lost:
                    yield start, text, _written_spans(units) if hieroglyphs else units
                units, start, text, last, lost = [], None, None, None, False
                continue
            cols = line.rstrip("\n").split("\t")
            if "." in cols[0]:
                continue
            space = len(cols) < 10 or "SpaceAfter=No" not in cols[9]
            upos = cols[3] if len(cols) > 3 else "_"
            form = cols[1]
            m = RANGE.match(cols[0])
            if hieroglyphs:
                hiero = HIERO_FIELD.search(cols[9]) if len(cols) > 9 else None
                hiero = hiero[1] if hiero else ""
                form = hiero_signs(hiero)
                if not m and upos == "PUNCT":
                    form = cols[1]
                elif not m and (not form or "UC_No" in hiero):
                    lost = True
            if m:
                last = Unit(True, form, int(m[1]), int(m[2]), [], space, None, [])
                units.append(last)
            elif last is not None and last.mwt and last.lo <= int(cols[0]) <= last.hi:
                last.comps.append(form)
                last.comp_upos.append(upos)
            else:
                units.append(Unit(False, form, int(cols[0]), int(cols[0]), [], space, upos, []))
    if units and not lost:
        yield start, text, _written_spans(units) if hieroglyphs else units


def _written_spans(units):
    return [u._replace(form="".join(u.comps)) if u.mwt and not u.form else u for u in units]


def render(units) -> str:
    return "".join(u.form + (" " if u.space else "") for u in units)


def language_of(conllu_file: str) -> str:
    return conllu_file.split("/")[0][3:].split("-")[0]


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
