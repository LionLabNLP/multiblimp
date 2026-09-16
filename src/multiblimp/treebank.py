import os
import pickle
import re
from functools import lru_cache
from glob import glob
import urllib.request
from pathlib import Path

from arabic2latin import arabic_to_latin
from conllu import parse_incr
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
from unidecode import unidecode
from bs4 import BeautifulSoup

from .config import UD_PATH
from .languages import udlang2treebanks, convert_arabic_to_latin_langs, add_langs


def has_typo(item):
    is_reparandum = item["deprel"] == "reparandum"

    feats = item.get("feats") or {}
    feats_has_typo = feats.get("Typo", "No") == "Yes"
    feats_has_style = "Style" in feats
    feats_has_foreign = "Foreign" in feats

    misc = item.get("misc") or {}
    misc_has_correction = any("Correct" in misc_key for misc_key in misc.keys())
    # misc_has_lang = ("Lang" in misc) or ("OrigLang" in misc)

    if (
        is_reparandum
        or feats_has_typo
        or feats_has_style
        or feats_has_foreign
        or misc_has_correction
        # or misc_has_lang
    ):
        return True

    return False


def tree_is_malformed(tree):
    for item in tree:
        if has_typo(item):
            return True

    if all(item["form"] == "_" for item in tree):
        return True

    return False


def _split_treebank_folder(folder: str) -> tuple[str, str]:
    """"UD_Old_East_Slavic-TOROT" -> ("Old_East_Slavic", "TOROT")."""
    parts = folder.removeprefix("UD_").split("-")
    return "_".join(parts[:-1]), parts[-1]


def _iter_treebank_dirs(resource_dir: str, ud_dir: str):
    for path in sorted(glob(os.path.join(resource_dir, ud_dir, "UD_*"))):
        if not os.path.isdir(path):
            continue
        folder = os.path.basename(path)
        language, treebank_id = _split_treebank_folder(folder)
        yield path, folder, language, treebank_id


def _read_readme(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _read_genre(readme_text: str) -> set[str]:
    match = re.search(r"^genre:\s*(.*)$", readme_text, re.I | re.M)
    if not match:
        return set()
    return set(match.group(1).strip().rstrip(".").lower().split())


def _summary_section(readme_text: str) -> str:
    lines = readme_text.splitlines()
    header_idx = [i for i, line in enumerate(lines) if line.startswith("# ")]
    start = next(
        (i for i in header_idx if lines[i].removeprefix("#").strip().lower() == "summary"),
        header_idx[0] if header_idx else None,
    )
    if start is None:
        return readme_text
    end = next((i for i in header_idx if i > start), len(lines))
    return "\n".join(lines[start:end])


def _exclude_sections(readme_text: str, excluded_names: set[str]) -> str:
    lines = readme_text.splitlines()
    header_idx = [i for i, line in enumerate(lines) if line.startswith("# ")]
    kept = lines[: header_idx[0]] if header_idx else lines
    for pos, start in enumerate(header_idx):
        name = lines[start].removeprefix("#").strip().lower()
        end = header_idx[pos + 1] if pos + 1 < len(header_idx) else len(lines)
        if name not in excluded_names:
            kept += lines[start:end]
    return "\n".join(kept)


def flag_treebanks(
    flag_type: str,
    resource_dir: str | None = None,
    ud_dir: str | None = None,
) -> dict[str, list[str]]:
    """
    Flag treebanks matching a given category, either by scraping UD or by
    scanning a local UD release checkout.

    Args:
        flag_type: One of:
            "words removed" (scraped: treebanks where the underlying text
                has been removed, detected via a data-hint marker on the
                treebank header itself).
            "sign language" (scraped: treebanks whose language is a sign
                language, detected via the "Sign Language" text on the
                parent language header, since individual sign language
                treebank rows carry no distinguishing marker of their own).
            "old variant" (local: treebanks of a historical language variety
                that UD filed as a treebank *within* the modern language's
                folder -- e.g. UD_Swedish-Old, UD_Italian-Old,
                UD_Guarani-OldTuDeT -- rather than granting it its own
                top-level language, unlike e.g. UD_Old_English-Cairo, which
                is a genuine top-level "Old_English" language and is not
                flagged).
            "poetry only" (local: treebanks whose README "Genre:" field is
                exclusively "poetry", e.g. UD_Russian-Poetry).
            "learner" (local: treebanks whose README "Genre:" field is
                exclusively "learner-essays", e.g. UD_Swedish-SweLL; a
                multi-genre treebank that merely includes some learner
                essays among other genres, e.g. UD_Yiddish-YiTB, is not
                flagged).
            "twitter" (local: treebanks built from tweets, detected via
                "twitter"/"tweet" appearing in the README's opening summary
                section, e.g. UD_Italian-TWITTIRO).
            "code switching" (local: treebanks whose text is described as
                code-switched, detected via "code-switch(ing)" appearing in
                a non-negated sentence of the README (outside the
                Changelog section, and not paired with "removed"/"excluded"
                -- which marks code-switching as something filtered *out*
                of the corpus rather than a property of it), e.g.
                UD_Telugu_English-TECT).
        resource_dir: Directory containing the UD release checkout. Only
            used for the local flag types; defaults to ".".
        ud_dir: Path to the UD release folder relative to resource_dir.
            Only used for the local flag types; defaults to UD_PATH.

    Returns:
        Dict mapping language_name -> [treebank_names]
    """
    flag_type = flag_type.strip().lower()
    local_flag_types = ("old variant", "poetry only", "learner", "twitter", "code switching")
    if flag_type not in ("words removed", "sign language") + local_flag_types:
        raise ValueError(
            "flag_type must be one of "
            f'{("words removed", "sign language") + local_flag_types}, got {flag_type!r}'
        )

    results = {}

    if flag_type in local_flag_types:
        if resource_dir is None:
            resource_dir = "."
        if ud_dir is None:
            ud_dir = UD_PATH

        if flag_type == "old variant":
            for _, folder, language, treebank_id in _iter_treebank_dirs(resource_dir, ud_dir):
                if language.startswith("Old_"):
                    continue  # genuine top-level Old_X language, not nested under a modern one
                if treebank_id.startswith("Old"):
                    results.setdefault(language, []).append(folder)

        elif flag_type == "poetry only":
            for path, folder, language, _ in _iter_treebank_dirs(resource_dir, ud_dir):
                genre = _read_genre(_read_readme(os.path.join(path, "README.md")))
                if genre == {"poetry"}:
                    results.setdefault(language, []).append(folder)

        elif flag_type == "learner":
            for path, folder, language, _ in _iter_treebank_dirs(resource_dir, ud_dir):
                genre = _read_genre(_read_readme(os.path.join(path, "README.md")))
                if genre == {"learner-essays"}:
                    results.setdefault(language, []).append(folder)

        elif flag_type == "twitter":
            for path, folder, language, _ in _iter_treebank_dirs(resource_dir, ud_dir):
                readme_text = _read_readme(os.path.join(path, "README.md"))
                if re.search(r"twitter|tweet", _summary_section(readme_text), re.I):
                    results.setdefault(language, []).append(folder)

        elif flag_type == "code switching":
            for path, folder, language, _ in _iter_treebank_dirs(resource_dir, ud_dir):
                readme_text = _read_readme(os.path.join(path, "README.md"))
                text = _exclude_sections(readme_text, {"changelog"})
                cs_sentences = [
                    s for s in re.split(r"(?<=[.!?])\s+", text)
                    if re.search(r"code[- ]switch", s, re.I)
                ]
                if not cs_sentences:
                    continue
                if any(re.search(r"remov|exclud", s, re.I) for s in cs_sentences):
                    continue  # code-switching was filtered out of the corpus, not a property of it
                results.setdefault(language, []).append(folder)

        return results

    fp = urllib.request.urlopen("https://universaldependencies.org")
    html_str = fp.read().decode("utf8")
    fp.close()

    soup = BeautifulSoup(html_str, features="html.parser")

    def name_of(header) -> str:
        try:
            return header.select_one("span.doublewidespan").get_text(strip=True)
        except AttributeError:
            return "UNKNOWN"

    if flag_type == "words removed":
        marker = 'span[data-hint="Underlying text not included"]'

        for treebank_header in soup.select("div.ui-accordion-header"):
            if treebank_header.select_one("span.flagspan img") is not None:
                continue  # skip language-level headers
            if treebank_header.select_one(marker) is None:
                continue

            try:
                lang_header = (
                    treebank_header
                    .find_parent("div", class_="ui-accordion-content")
                    .find_previous_sibling("div", class_="ui-accordion-header")
                )
                language_name = name_of(lang_header)
            except AttributeError:
                language_name = None

            results.setdefault(language_name, []).append(name_of(treebank_header))

    elif flag_type=="sign language":  # "sign language"
        for lang_header in soup.select("div.ui-accordion-header"):
            if lang_header.select_one("span.flagspan img") is None:
                continue  # skip treebank-level headers

            is_sign_language = any(
                span.get_text(strip=True) == "Sign Language"
                for span in lang_header.select("span.triplewidespan")
            )
            if not is_sign_language:
                continue

            language_name = name_of(lang_header).replace(" Sign Language", "")

            try:
                treebank_headers = lang_header.find_next_sibling(
                    "div", class_="ui-accordion-content"
                ).select("div.ui-accordion-header")
            except AttributeError:
                treebank_headers = []

            results.setdefault(language_name, []).extend(
                name_of(tb) for tb in treebank_headers
            )
        for name in list(results.keys()):
            if name is None:
                continue
            alt = name.replace(" ", "_") if " " in name else name.replace("_", " ")
            if alt != name:
                results.setdefault(alt, results[name])

    return results


LOCAL_FLAG_TYPES = ("old variant", "poetry only", "learner", "twitter", "code switching")


@lru_cache(maxsize=None)
def get_excluded_treebanks(
    resource_dir: str = ".",
    ud_dir: str | None = None,
) -> frozenset[str]:
    """Union of every LOCAL_FLAG_TYPES category's treebank folders -- the
    treebanks Treebank() (and, via it, every UD data-pickling / ud2um /
    data-caching stage) skips loading. Cached, since scanning the full UD
    checkout and re-reading every README is expensive and every pipeline
    stage instantiates Treebank() once per language."""
    excluded = set()
    for flag_type in LOCAL_FLAG_TYPES:
        for treebanks in flag_treebanks(flag_type, resource_dir=resource_dir, ud_dir=ud_dir).values():
            excluded.update(treebanks)
    return frozenset(excluded)


class Treebank:
    def __new__(
        cls,
        lang: str,
        remove_diacritics: bool = False,
        verbose: bool = False,
        load_from_pickle: bool = False,
        resource_dir: str | None = None,
        test_files_only: bool = False,
        use_selected_treebanks: bool = True,
        remove_typo: bool = True,
        pickle_path: str = "ud/ud_pickles",
    ):
        if resource_dir is None:
            resource_dir = "."

        if load_from_pickle:
            pickle_path = os.path.join(resource_dir, pickle_path, f"{lang}.pickle")
            # TODO: if no pickle, run pickle script
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    return pickle.load(f)
            else:
                with open("error_log.txt", "a") as f:
                    f.write(f"Pickle not found for {lang} at {pickle_path}\n")

        match_on = f"UD_{lang}-*" if lang not in add_langs else add_langs[lang]
        if test_files_only:
            treebank_glob = os.path.join(UD_PATH, f"{match_on}/*test*.conllu")
        else:
            treebank_glob = os.path.join(UD_PATH, f"{match_on}/*.conllu")
        treebank_glob = os.path.join(resource_dir, treebank_glob)
        treebank_paths = glob(treebank_glob)

        selected_treebanks = udlang2treebanks.get(lang)
        if use_selected_treebanks and selected_treebanks is not None:
            # An explicit udlang2treebanks pin is a deliberate, reviewed
            # choice -- it overrides flag-based exclusion (sign language,
            # and the 5 local flag_treebanks categories) entirely, rather
            # than having those flags silently drop a pinned treebank.
            treebank_paths = [
                p for p in treebank_paths
                if Path(p).parent.name.split("-")[-1] in selected_treebanks
            ]
        else:
            skip_flagged = {t.lower() for t in flag_treebanks("sign language").get(lang, [])}
            treebank_paths = [
                p for p in treebank_paths
                if p.split("/")[-2].split("-")[-1].lower() not in skip_flagged
            ]

            excluded_treebanks = get_excluded_treebanks(resource_dir)
            treebank_paths = [p for p in treebank_paths if Path(p).parent.name not in excluded_treebanks]

        if verbose:
            print("Loading:\n", "\n".join(treebank_paths))

        treebank = []
        for filename in treebank_paths:
            with open(filename, encoding="utf-8") as f:
                for tree in parse_incr(f):
                    tree.metadata["treebank"] = "/".join(filename.split("/")[-2:])
                    treebank.append(tree)

        if remove_typo:
            treebank = [tree for tree in treebank if not tree_is_malformed(tree)]

        for tree in treebank:
            remove_items = [tok for tok in tree if isinstance(tok["id"], tuple)]
            for item in remove_items:
                tree.remove(item)

        # Some noisier treebanks (e.g. Runyankore) have real tokens with a
        # missing HEAD ("_", parsed as None) rather than a proper int --
        # invalid per the UD spec outside of MWT/empty-node lines, which are
        # already stripped above. Such tokens can't be placed in the
        # dependency tree at all, so drop the whole sentence rather than let
        # None reach downstream head/child comparisons.
        treebank = [tree for tree in treebank if all(tok["head"] is not None for tok in tree)]

        if lang in convert_arabic_to_latin_langs:
            for tree in treebank:
                for item in tree:
                    if item.get("misc", {}).get("Translit"):
                        item["form"] = item["misc"]["Translit"]
                    else:
                        item["form"] = arabic_to_latin(item["form"])

                    if item.get("misc", {}).get("LTranslit"):
                        item["lemma"] = item["misc"]["LTranslit"]
                    else:
                        item["lemma"] = arabic_to_latin(item["lemma"])

        if remove_diacritics:
            for tree in treebank:
                for item in tree:
                    item["form"] = unidecode(item["form"])
                    item["lemma"] = unidecode(item["lemma"])

        if lang == "Sanskrit":
            for tree in treebank:
                for item in tree:
                    item["form"] = transliterate(item["form"], IAST, DEVANAGARI)
                    item["lemma"] = transliterate(item["lemma"], IAST, DEVANAGARI)

        if lang == "Ancient_Hebrew":

            def remove_hebrew_cantillation(text):
                # https://stackoverflow.com/q/44479533/351197
                pattern = r"[\u0591-\u05AF\u05BE\u05C0\u05C3]"
                return re.sub(pattern, "", text)

            for tree in treebank:
                for item in tree:
                    item["form"] = remove_hebrew_cantillation(item["form"])
                    item["lemma"] = remove_hebrew_cantillation(item["lemma"])

        return treebank
