import os
import re
import unicodedata

from glob import glob

from iso639 import Lang

from .config import UD_PATH


def lang2langcode(name: str):
    lookup_name = udlang2iso639.get(name, name)
    # Try the mapped name as-is, then with underscores turned into spaces --
    # get_ud_langs joins multi-word UD folder names with "_" (e.g.
    # "Norwegian_Bokmål"), but iso639's Lang() expects real spaces and
    # rejects the underscored form outright. Without this second attempt,
    # any such language silently falls through to the static dict below and
    # gets treated as code-less even when iso639 knows it perfectly well
    # (e.g. "Norwegian_Bokmål"/"Low_Saxon"/"Upper_Sorbian" -- all resolve
    # fine as "Norwegian Bokmål"/"Low Saxon"/"Upper Sorbian", losing access
    # to real UniMorph data sitting under their correct codes).
    for candidate in (lookup_name, lookup_name.replace("_", " ")):
        try:
            return Lang(candidate).pt3
        except Exception:
            continue
    return {"Ancient_Greek": "grc",
            "Ancient_Hebrew": name,
            "Bokota": name,
            "Cappadocian": name,
            "Central_Kurdish": "ckb",
            "Chintang": name,
            "Classical_Armenian": "xcl",
            "Classical_Chinese": name,
            "Frisian_Dutch": name,
            "Haitian_Creole": name,
            "Highland_Puebla_Nahuatl": name,
            "Kadiweu": name,
            "Komi_Permyak": name,
            "Komi_Zyrian": name,
            "Maghrebi_Arabic_French": name,
            "Mbya_Guarani": name,
            "Middle_Armenian": name,
            "Middle_French": "frm",
            "Naga": name,
            "North_Sami": "sme",
            "Northern_Kurdish": "kmr",
            "Northwest_Gbaya": name,
            "Norwegian_Nynorsk": "nno",
            "Occitan": "oci",
            "Old_Church_Slavonic": "chu",
            "Old_East_Slavic": name,
            "Old_English": "ang",
            "Old_French": "fro",
            "Old_Georgian": name,
            "Old_Irish": "sga",
            "Old_Occitan": name,
            "Old_Turkish": name,
            "Ottoman_Turkish": name,
            "Pomak": "poma",
            "Scottish_Gaelic": "gla",
            "Shanghainese": name,
            "Skolt_Sami": name,
            "South_Levantine_Arabic": name,
            "Southern_Kurdish": "sdh",
            "Spanish_Sign_Language": name,
            "Swedish_Sign_Language": name,
            "Telugu_English": name,
            "Turkish_English": name,
            "Turkish_German": name,
            "Western_Armenian": name,
            "Western_Sierra_Puebla_Nahuatl": name,
            }.get(name, name)

udlang2iso639 = {
    "Abkhaz": "Abkhazian",
    "Ancient Greek": "Ancient Greek (to 1453)",
    "Apurina": "Apurinã",
    "Arabic": "Standard Arabic",
    "Assyrian": "Assyrian Neo-Aramaic",
    "Bororo": "Borôro",
    "Buryat": "Buriat",
    "Cantonese": "Yue Chinese",
    "Chukchi": "Chukot",
    "Classical Chinese": "Literary Chinese",
    "Egyptian": "Egyptian (Ancient)",
    "Gheg": "Gheg Albanian",
    "Greek": "Modern Greek (1453-)",
    "Guajajara": "Guajajára",
    "Gwichin": "Gwich'in",
    "Karo": "Karo (Ethiopia)",
    "Kiche": "K'iche'",
    "Komi Permyak": "Komi-Permyak",
    "Komi Zyrian": "Komi-Zyrian",
    "Kurmanji": "Northern Kurdish",
    "Makurap": "Makuráp",
    "Mbya Guarani": "Mbyá Guaraní",
    "Middle French": "Middle French (ca. 1400-1600)",
    "Munduruku": "Mundurukú",
    "Nheengatu": "Nhengatu",
    "Naija": "Nigerian Pidgin",
    "North Sami": "Northern Sami",
    "Old East Slavic": "Old Russian",
    "Old French": "Old French (842-ca. 1400)",
    "Old Irish": "Old Irish (to 900)",
    "Ottoman Turkish": "Ottoman Turkish (1500-1928)",
    "Paumari": "Paumarí",
    "Pesh": "Pech",
    "Serbian-Croatian-Bosnian": "Serbo-Croatian",
    "South Levantine Arabic": "Levantine Arabic",
    "Teko": "Tektiteko",
    "Tupinamba": "Tupinambá",
    "Western Sierra Puebla Nahuatl": "Zacatlán-Ahuacatlán-Tepetzintla Nahuatl",  # zaca1241, nhi
    "Xavante": "Xavánte",
    "Yupik": "Central Siberian Yupik",
    "Zaar": "Saya",
}

gblang2udlang = {
    "Modern Greek": "Greek",
    "Classical-Middle Armenian": "Classical Armenian",
    "Western Farsi": "Persian",
    "Northern Tosk Albanian": "Albanian",
    "North Azerbaijani": "Azerbaijani",
    "Sakha": "Yakut",
    "Standard Arabic": "Arabic",
    "Eastern Armenian": "Armenian",
    "Mandarin Chinese": "Chinese",
    "Northern Uzbek": "Uzbek",
    "Southern Pashto": "Pashto",
    "Northern Karelian": "Karelian",
    "Modern Hebrew": "Hebrew",
}

add_langs = {"Aromanian": "UD_Romanian-ArT"}

# TODO: rewrite this as exclusion list rather than inclusion
# Maps a language to all the treebanks that should be used for that language.
# If a language is not in this dictionary we take all available treebanks.
# Note: a language listed here overrides flag_treebanks -- a treebank pinned
# here is used even if it's flagged. Flag exclusion only applies to
# languages with no entry here.
udlang2treebanks = {
    "Arabic": ["PADT", "PUD"],
    "Classical_Chinese": ["Kyoto"],
    "Czech": ["CAC", "CLTT", "FicTree", "PDT", "PUD"],
    "English": ["EWT", "LinES", "ParTUT", "PUD"],
    "Faroese": ["OFT"],
    "French": ["FQB", "GSD", "ParTUT", "PUD", "Sequoia"],
    "Galician": ["PUD", "TreeGal"],
    "German": ["GSD", "PUD", "HDT"],
    "Gheg": ["GPS"],
    "Guarani": ["OldTuDeT"],
    "Hebrew": ["HTB"],
    "Icelandic": ["GC", "Modern", "PUD"],
    "Italian": [ "ISDT", "MarkIT", "ParlaMint", "ParTUT", "PUD",
              "VIT"],
    "Japanese": ["GSD", "PUD", "BCCWJ"],
    "Latvian": ["LVTB"],
    "Romanian": ["RRT", "SiMoNERo"],
    "Russian": ["GSD", "PUD", "SynTagRus", "Taiga"],
    "Sanskrit": ["Vedic"],
    "Slovenian": ["SSJ"],
    "Spanish": ["AnCora", "GSD", "PUD"],
    "Swedish": ["LinES", "PUD", "Talbanken"],
    "Vietnamese": ["VTB"],
}

lang2unimorph_lang = {
    "Standard Arabic": "Arabic",
    "Southern Pashto": "Pashto",
    "Guarani": "Mbyá Guaraní",
    "Northern Uzbek": "Uzbek",
    "North Azerbaijani": "Azerbaijani",
}

remove_diacritics_langs = {
    "Latin",
    "Slovenian",
    "Western Farsi",
    # "Northern Uzbek",
    # "Kazakh",
}

remove_multiples_langs = {
    "Modern Greek",
    "Ancient Greek",
}

convert_arabic_to_latin_langs = {
    "Uyghur",
}

skip_langs = {
    "Frisian_Dutch",
    "Turkish_German",
    "Maghrebi_Arabic_French",
    "Telugu_English",
    "Turkish_English",
    "Spanish_Sign_Language",
    "Swedish_Sign_Language",
}


def latin_to_cyrillic(text):
    """Specific mapping for Tatar.
    Source: https://suzlek.antat.ru/about/TAAT2019/8.pdf
    """
    text = unicodedata.normalize("NFC", text)

    mapping = dict(
        [
            ("Uwa", "Уa"),
            ("uwa", "уa"),
            ("Ts", "Ц"),
            ("ts", "ц"),
            ("Gö", "Гө"),
            ("gö", "гө"),
            ("Gä", "Гә"),
            ("gä", "гә"),
            ("Ge", "Гe"),
            ("ge", "гe"),
            ("Yı", "Е"),
            ("yı", "е"),
            ("Ye", "Е"),
            ("ye", "е"),
            ("Yo", "Й"),
            ("yo", "й"),
            ("Yö", "Й"),
            ("yö", "й"),
            ("Aw", "Ay"),
            ("aw", "ay"),
            ("Ya", "Я"),
            ("ya", "я"),
            ("Yä", "Я"),
            ("yä", "я"),
            ("Yu", "Ю"),
            ("yu", "ю"),
            ("Yü", "Ю"),
            ("yü", "ю"),
            ("Şç", "Щ"),
            ("şç", "щ"),
            ("A", "А"),
            ("a", "а"),
            ("Ä", "Ә"),
            ("ä", "ә"),
            ("B", "Б"),
            ("b", "б"),
            ("C", "Җ"),
            ("c", "җ"),
            ("Ç", "Ч"),
            ("ç", "ч"),
            ("D", "Д"),
            ("d", "д"),
            ("E", "Е"),
            ("e", "е"),
            ("F", "Ф"),
            ("f", "ф"),
            ("Ğ", "Г"),
            ("ğ", "г"),
            ("G", "Г"),
            ("g", "г"),
            ("H", "Һ"),
            ("h", "һ"),
            ("İ", "И"),
            ("i", "и"),
            ("I", "Ы"),
            ("ı", "ы"),
            ("J", "Ж"),
            ("j", "ж"),
            ("K", "К"),
            ("k", "к"),
            ("L", "Л"),
            ("l", "л"),
            ("M", "М"),
            ("m", "м"),
            ("N", "Н"),
            ("n", "н"),
            ("Ñ", "Ң"),
            ("ñ", "ң"),
            ("O", "О"),
            ("o", "о"),
            ("Ö", "Ө"),
            ("ö", "ө"),
            ("P", "П"),
            ("p", "п"),
            ("Q", "К"),
            ("q", "к"),
            ("R", "Р"),
            ("r", "р"),
            ("S", "С"),
            ("s", "с"),
            ("Ş", "Ш"),
            ("ş", "ш"),
            ("T", "Т"),
            ("t", "т"),
            ("Y", "Й"),
            ("y", "й"),
            ("U", "У"),
            ("u", "у"),
            ("Ü", "Ү"),
            ("ü", "ү"),
            ("V", "В"),
            ("v", "в"),
            ("W", "В"),
            ("w", "в"),
            ("X", "X"),
            ("x", "x"),
            ("Z", "З"),
            ("z", "з"),
        ]
    )

    text = text[0].replace("E", "Э").replace("e", "э") + text[1:]

    idx = 0
    new_text = ""
    while idx < len(text):
        for j in range(3, 0, -1):
            if text[idx : idx + j] in mapping:
                new_text += mapping[text[idx : idx + j]]
                idx += j
                break
            elif j == 1:
                new_text += text[idx]
                idx += 1
                break

    return new_text


def get_ud_langs(resource_dir, ud_dir=None, do_skip_langs=True):
    if ud_dir is None:
        ud_dir = UD_PATH

    # Local import: multiblimp.treebank imports from this module at load
    # time, so importing it back at module level here would be circular.
    from .treebank import get_excluded_treebanks
    excluded_treebanks = get_excluded_treebanks(resource_dir, ud_dir)

    def ud_dir2lang(x):
        return("_".join(x.split("/")[-1].replace("UD_", "").split("-")[:-1]))

    # A language with an explicit udlang2treebanks pin is exempt from
    # flag-based exclusion (see get_lang_treebanks docstring).
    treebank_dirs = [
        d for d in glob(os.path.join(resource_dir, ud_dir, "*"))
        if os.path.basename(d) not in excluded_treebanks or ud_dir2lang(d) in udlang2treebanks
    ]
    treebank_langs = map(ud_dir2lang, treebank_dirs)
    treebank_langs = sorted(set(treebank_langs).union(set(add_langs.keys())))

    if do_skip_langs:
        treebank_langs = [lang for lang in treebank_langs if lang not in skip_langs]

    return treebank_langs


def get_lang_treebanks(resource_dir, ud_dir=None):
    """Every UD treebank id (e.g. "UD_French-GSD") this pipeline could pull
    samples from for each language, keyed by the same underscore-joined
    language name get_ud_langs uses (e.g. "Norwegian_Bokmål",
    "Ancient_Greek") -- one language can span several treebanks (e.g.
    French: FQB/GSD/PUD/ParTUT/Sequoia), and udlang2treebanks/add_langs
    below are the same two overrides get_ud_langs and the actual pipeline
    already apply, so this reflects real treebank selection rather than
    "everything that happens to exist in the UD release for that language".
    Treebanks flagged by multiblimp.treebank.flag_treebanks (historical
    variants filed under a modern language, poetry-only, learner-essay,
    Twitter, and code-switching treebanks) are excluded, same as Treebank()
    itself -- a language left with no treebanks at all after that is
    dropped entirely. A language with an explicit udlang2treebanks pin is
    exempt from this exclusion: the pin is a deliberate, reviewed choice
    and overrides flag-based exclusion rather than having it silently drop
    a pinned treebank.
    """
    if ud_dir is None:
        ud_dir = UD_PATH

    from .treebank import get_excluded_treebanks
    excluded_treebanks = get_excluded_treebanks(resource_dir, ud_dir)

    def ud_dir2lang(x):
        return "_".join(os.path.basename(x).replace("UD_", "").split("-")[:-1])

    by_lang = {}
    for path in sorted(glob(os.path.join(resource_dir, ud_dir, "*"))):
        folder = os.path.basename(path)
        if not folder.startswith("UD_"):
            continue
        by_lang.setdefault(ud_dir2lang(path), []).append(folder)

    # Drop flagged treebanks, and any language left with none at all -- but
    # only for that reason, so a language that's merely absent from the
    # current udlang2treebanks selection below still ends up with an empty
    # list rather than disappearing, same as before this exclusion existed.
    # Languages with an explicit pin are exempt (see docstring).
    by_lang = {
        lang: (tbs if lang in udlang2treebanks else [tb for tb in tbs if tb not in excluded_treebanks])
        for lang, tbs in by_lang.items()
    }
    by_lang = {lang: tbs for lang, tbs in by_lang.items() if tbs}

    for lang, codes in udlang2treebanks.items():
        if lang in by_lang:
            by_lang[lang] = [tb for tb in by_lang[lang] if tb.rsplit("-", 1)[-1] in codes]

    for lang, treebank_id in add_langs.items():
        by_lang.setdefault(lang, []).append(treebank_id)

    return by_lang


if __name__=="__main__":
    print(get_ud_langs("../../resources"))