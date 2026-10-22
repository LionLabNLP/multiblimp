"""Per-language multiword-token statistics, read once from the raw UD files and cached in resources/ud_mwt/.

The pickles drop the multiword range lines ("3-4 du"), so the written form of a sentence can only come from the
raw CoNLL-U. A language's cache holds, for its non-excluded treebanks:
  contracted   {component forms: {written form: count}}
  positions    {UPOS: {(position in the span, span length)}}
  patterns     {component forms: {component UPOS tuple: count}}
  sentences    {(treebank file, line of "# sent_id"): (text, unit rows)} for sentences with a multiword token
and a second file with the counts of plain word sequences, filled on demand (MwtStats.counts) and kept.
A cache is rebuilt when the treebank files, their selection (exclusion list) or CACHE_VERSION change.
"""
import collections
import os
import pickle
import sys
from pathlib import Path

from agreement_trees.eval.conllu_spans import iter_sentences, norm
from multiblimp.config import RESOURCE_DIR, UD_PATH
from multiblimp.excluded_treebanks import is_treebank_excluded

CACHE_VERSION = 2
MWT_DIR = os.path.join(RESOURCE_DIR, "ud_mwt")
UD_DIR = os.path.join(RESOURCE_DIR, UD_PATH)


def treebank_files(lang: str, ud_dir: str = UD_DIR, only_included: bool = True) -> list[Path]:
    files = sorted(Path(ud_dir).glob(f"UD_{lang}-*/*.conllu"))
    if only_included:
        files = [f for f in files if not is_treebank_excluded(lang, f.parent.name.rsplit("-", 1)[-1])]
    return files


def _stamp(files: list[Path], ud_dir: str) -> dict:
    return {"version": CACHE_VERSION, "ud": os.path.basename(os.path.normpath(ud_dir)),
            "files": [(f"{f.parent.name}/{f.name}", f.stat().st_size) for f in files]}


def _write(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _read(path: str, stamp: dict):
    """The payload at `path` if it was written for `stamp`, else None."""
    try:
        with open(path, "rb") as f:
            saved, payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        return None
    return payload if saved == stamp else None


class MwtStats:
    def __init__(self, lang: str, files: list[Path], data: dict, counts_path: str | None, stamp: dict):
        self.lang, self.files = lang, files
        self.contracted = {k: collections.Counter(v) for k, v in data["contracted"].items()}
        self.positions = data["positions"]
        self.patterns = data["patterns"]
        self.sentences = data["sentences"]
        self._counts_path, self._stamp = counts_path, stamp
        self._counts = (_read(counts_path, stamp) or {}) if counts_path else {}

    @property
    def has_mwt(self) -> bool:
        return bool(self.contracted or self.sentences)

    def counts(self, queries) -> dict:
        """{sequence: how often it occurs as plain words (not span components) in the treebanks} for each query
        (tuples of normalised forms). Sequences not seen before cost one pass over the files and are kept."""
        missing = {q for q in queries if q not in self._counts}
        if missing:
            found = collections.Counter()
            lengths = {len(q) for q in missing}
            for path in self.files:
                for _, _, units in iter_sentences(path):
                    forms = [None if u.mwt else norm(u.form) for u in units]
                    for n in lengths:
                        for j in range(len(units) - n + 1):
                            window = tuple(forms[j:j + n])
                            if window in missing:
                                found[window] += 1
            for q in missing:
                self._counts[q] = found[q]
            if self._counts_path:
                _write(self._counts_path, (self._stamp, self._counts))
        return {q: self._counts[q] for q in queries}


def _build(files: list[Path]) -> dict:
    contracted = collections.defaultdict(collections.Counter)
    patterns = collections.defaultdict(collections.Counter)
    positions = collections.defaultdict(set)
    sentences = {}
    for path in files:
        rel = f"{path.parent.name}/{path.name}"
        for start, text, units in iter_sentences(path):
            if not any(u.mwt for u in units):
                continue
            sentences[(rel, start)] = (text, tuple(
                (u.mwt, sys.intern(u.form), u.lo, u.hi, tuple(sys.intern(c) for c in u.comps), u.space,
                 u.upos and sys.intern(u.upos), tuple(sys.intern(c) for c in u.comp_upos)) for u in units))
            for u in units:
                if u.mwt and len(u.comps) > 1:
                    key = tuple(norm(c) for c in u.comps)
                    contracted[key][norm(u.form)] += 1
                    patterns[key][tuple(u.comp_upos)] += 1
                    for j, upos in enumerate(u.comp_upos):
                        positions[upos].add((j, len(u.comps)))
    return {"contracted": {k: dict(v) for k, v in contracted.items()},
            "patterns": {k: dict(v) for k, v in patterns.items()},
            "positions": dict(positions), "sentences": sentences}


_LOADED: dict = {}


def load_stats(lang: str, ud_dir: str = UD_DIR, only_included: bool = True, rebuild: bool = False,
               cache_dir: str = MWT_DIR) -> MwtStats:
    """The language's statistics: from the cache when it is current, else built from the raw files and cached.
    Without `only_included` (every treebank, as the first version of the validation read them) nothing is cached."""
    files = treebank_files(lang, ud_dir, only_included)
    stamp = _stamp(files, ud_dir)
    memo_key = (lang, ud_dir, only_included, cache_dir)
    if not rebuild and memo_key in _LOADED and _LOADED[memo_key]._stamp == stamp:
        return _LOADED[memo_key]
    path = os.path.join(cache_dir, f"{lang}.pickle") if only_included else None
    data = None if rebuild or path is None else _read(path, stamp)
    if data is None:
        data = _build(files)
        if path:
            _write(path, (stamp, data))
            counts_path = path[:-len(".pickle")] + ".counts.pickle"
            if os.path.exists(counts_path):
                os.remove(counts_path)
    counts_path = path[:-len(".pickle")] + ".counts.pickle" if path else None
    stats = MwtStats(lang, files, data, counts_path, stamp)
    _LOADED[memo_key] = stats
    return stats
