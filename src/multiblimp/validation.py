"""Run the official UD validator (UniversalDependencies/tools, udtools) over a UD treebank
folder and keep its incidents in one JSON sidecar per treebank
(resources/ud/validation/<UD_Lang-Treebank>.json). Measure-only: nothing here changes the
treebank data; the sidecars are read by scripts/data_caching/validate_treebanks.py and,
later, by whatever filters on them.

Sidecar layout:
  treebank, lang_code, level_requested, level_run, tools_commit, ud_release, seconds
  files       {basename: {size, sentences, tokens}}
  incidents   [[testid, level, "E"|"W", file, lineno, sent_id, node_id, sentline], ...]
              sentline is the first token line of the sentence: it identifies the sentence even
              where sent_id is missing or repeated, and lineno is the flagged token's own line
  examples    {testid: [message, ...]}   first few messages per test
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from glob import glob

from .config import RESOURCE_DIR, UD_PATH, UD_TOOLS_DIR, VALIDATION_DIR

MAX_EXAMPLES = 5
SCHEMA = 2


def treebank_files(treebank_dir: str) -> list[str]:
    return sorted(glob(os.path.join(treebank_dir, "*.conllu")))


def validation_path(folder: str) -> str:
    return os.path.join(VALIDATION_DIR, f"{folder}.json")


def lang_code(files: list[str]) -> str:
    """ISO code from the file names ("de_gsd-ud-dev.conllu" -> "de"), the one the validator keys on."""
    return Counter(os.path.basename(f).split("_")[0] for f in files).most_common(1)[0][0]


def tools_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", UD_TOOLS_DIR, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_stats(path: str) -> dict:
    """Sentence and token (not MWT range / empty node) counts, to turn incident counts into rates."""
    sentences = tokens = 0
    in_sentence = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line == "\n" or not line.strip():
                in_sentence = False
            elif line[0] != "#":
                if not in_sentence:
                    sentences += 1
                    in_sentence = True
                ident = line.split("\t", 1)[0]
                if ident.isdigit():
                    tokens += 1
    return {"size": os.path.getsize(path), "sentences": sentences, "tokens": tokens}


def is_up_to_date(folder: str, level: int) -> bool:
    path = validation_path(folder)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    files = treebank_files(os.path.join(_ud_root(), folder))
    sizes = {os.path.basename(p): os.path.getsize(p) for p in files}
    return (
        old.get("schema") == SCHEMA
        and old.get("level_requested") == level
        and old.get("tools_commit") == tools_commit()
        and {k: v["size"] for k, v in old["files"].items()} == sizes
    )


def _ud_root() -> str:
    return os.path.join(RESOURCE_DIR, UD_PATH)


def _load_validator_classes():
    # validate.py imports udtools through the tools checkout rather than a pip install
    if UD_TOOLS_DIR not in sys.path:
        sys.path.insert(0, UD_TOOLS_DIR)
    from udtools.src.udtools.validator import Validator
    from udtools.src.udtools.incident import Incident
    return Validator, Incident


def validate_treebank(folder: str, level: int = 5, redo: bool = False) -> dict:
    """Validate UD_<Lang>-<Treebank> as a whole (cross-file checks need all its files in one
    run), write its sidecar, and return a small summary. Skipped when the sidecar is current."""
    if not redo and is_up_to_date(folder, level):
        return {"folder": folder, "skipped": True}

    Validator, Incident = _load_validator_classes()
    files = treebank_files(os.path.join(_ud_root(), folder))
    if not files:
        return {"folder": folder, "skipped": True, "empty": True}
    code = lang_code(files)

    # a code the validator has no language data for would flag every feature and relation,
    # so those treebanks only get the language-independent levels
    probe = Validator(lang=code, level=level, output=None)
    level_run = level if (code in probe.data.feats and code in probe.data.deprel) else min(level, 3)
    del probe

    incidents, examples = [], {}

    def store(self):
        incidents.append([
            self.testid, self.level, "E" if self.is_error() else "W",
            self.filename, self.lineno, self.sentid, self.nodeid, self.state.sentence_line,
        ])
        shown = examples.setdefault(self.testid, [])
        if len(shown) < MAX_EXAMPLES:
            shown.append(str(self.message))

    # keep incidents as compact rows instead of Incident objects (some treebanks have 100k+)
    original_store = Incident._store_me
    Incident._store_me = store
    start = time.time()
    try:
        validator = Validator(lang=code, level=level_run, output=None)
        validator.validate_files(files)
    finally:
        Incident._store_me = original_store

    result = {
        "schema": SCHEMA,
        "treebank": folder,
        "lang_code": code,
        "level_requested": level,
        "level_run": level_run,
        "tools_commit": tools_commit(),
        "ud_release": UD_PATH,
        "seconds": round(time.time() - start, 1),
        "files": {os.path.basename(p): file_stats(p) for p in files},
        "incidents": incidents,
        "examples": examples,
    }
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    tmp = validation_path(folder) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, validation_path(folder))
    return {
        "folder": folder,
        "skipped": False,
        "level_run": level_run,
        "incidents": len(incidents),
        "seconds": result["seconds"],
    }


def load_validation(folder: str) -> dict | None:
    path = validation_path(folder)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_validators = {}


def sentence_messages(block_lines: list[str], code: str, level: int) -> list[tuple]:
    """(testid, node_id, message, line) for each incident the validator reports on one
    sentence block validated on its own; line is 1-based within the block. The sidecars keep
    messages only for a few examples per test; this recovers the message of a specific
    incident. Checks that need other sentences (duplicate ids, document-level state) do
    not fire here."""
    Validator, Incident = _load_validator_classes()
    if (code, level) not in _validators:
        _validators[code, level] = Validator(lang=code, level=level, output=None)
    found = []

    def store(self):
        found.append((self.testid, self.nodeid, str(self.message), self.lineno))

    original_store = Incident._store_me
    Incident._store_me = store
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".conllu", encoding="utf-8", delete=False) as f:
            f.write("\n".join(block_lines) + "\n\n")
        try:
            _validators[code, level].validate_files([f.name])
        finally:
            os.unlink(f.name)
    finally:
        Incident._store_me = original_store
    return found


def test_classes() -> dict[str, str]:
    """{testid: validator test class name (FORMAT, MORPHO, SYNTAX, ...)}, read from the tools
    source: a test's class is its `testclass=` argument, else the `default_testclass` set above it."""
    import re
    classes = {}
    for path in sorted(glob(os.path.join(UD_TOOLS_DIR, "udtools", "src", "udtools", "level*.py"))):
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for call in re.finditer(r"\b(?:Error|Warning)\(", src):
            segment = src[call.end():call.end() + 900]
            testid = re.search(r"testid=['\"]([a-z0-9-]+)['\"]", segment)
            following = re.search(r"\b(?:Error|Warning)\(", segment)
            if not testid or (following and following.start() < testid.start()):
                continue
            explicit = re.search(r"testclass=TestClass\.([A-Z]+)", segment)
            defaults = re.findall(r"default_testclass = TestClass\.([A-Z]+)", src[:call.start()])
            classes[testid.group(1)] = explicit.group(1) if explicit else (defaults[-1] if defaults else "?")
    # right-to-left-<relation> ids are built from the relation name in level3.py
    for relation in ("conj", "fixed", "flat", "goeswith", "appos"):
        classes[f"right-to-left-{relation}"] = "SYNTAX"
    return classes


# ---- the sentence filter applied when treebanks are loaded for pickling ----
def drop_testids() -> list[str]:
    """Validator tests whose flagged sentences are dropped (resources/data_prep_config.json, validation.drop_testids)."""
    with open(os.path.join(RESOURCE_DIR, "data_prep_config.json"), encoding="utf-8") as f:
        return sorted(json.load(f)["validation"]["drop_testids"])


def dropped_sentence_lines(folder: str) -> dict[str, set[int]]:
    """{file name: first token line of every sentence of the treebank folder flagged by a drop test}."""
    data = load_validation(folder)
    if data is None:
        raise FileNotFoundError(
            f"no validator results for {folder} ({validation_path(folder)}); "
            "run scripts/data_caching/validate_treebanks.py first"
        )
    tests = set(drop_testids())
    flagged: dict[str, set[int]] = {}
    for testid, _, _, filename, _, _, _, sentline in data["incidents"]:
        if testid in tests:
            flagged.setdefault(filename, set()).add(sentline)
    return flagged


def sentence_start_lines(path: str) -> list[int]:
    """First token line of each sentence block of a .conllu file, in file order (comment-only blocks skipped)."""
    starts, in_block, has_token = [], False, False
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                in_block = has_token = False
            elif not line.startswith("#") and not has_token:
                starts.append(number)
                has_token = True
    return starts


def validation_stamp(folders: list[str]) -> str:
    """Fingerprint of what the filter removes from these treebank folders: the drop tests and the sentences they flag.
    A pickle built under a different stamp is stale."""
    import hashlib
    digest = hashlib.sha1(json.dumps(drop_testids()).encode())
    for folder in sorted(folders):
        flagged = dropped_sentence_lines(folder)
        digest.update(json.dumps([folder, {k: sorted(v) for k, v in sorted(flagged.items())}]).encode())
    return digest.hexdigest()[:12]
