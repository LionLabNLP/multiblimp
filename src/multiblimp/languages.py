import os
import unicodedata

from glob import glob

from iso639 import Lang

from .config import UD_PATH
from .excluded_treebanks import is_treebank_excluded, SIGN_LANGUAGES


def lang2langcode(name: str):
    lookup_name = udlang2iso639.get(name, udlang2iso639.get(name.replace("_", " "), name))

    for candidate in (lookup_name, lookup_name.replace("_", " ")):
        try:
            return Lang(candidate).pt3
        except Exception:
            continue

    return {"Ancient_Greek": "grc",
            "Bokota": name,
            "Cappadocian": name,
            "Chintang": name,
            "Frisian_Dutch": name,
            "Kadiweu": name,
            "Maghrebi_Arabic_French": name,
            "Middle_French": "frm",
            "Naga": name,
            "North_Sami": "sme",
            "Occitan": "oci",
            "Old_English": "ang",
            "Old_French": "fro",
            "Old_Irish": "sga",
            "Old_Occitan": name,
            "Pomak": "poma",
            "Shanghainese": name,
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

# Some UD treebanks merge several originally-distinct sub-corpora into one
# release, where we only want a subset -- unlike excluded_treebanks, which
# drops a treebank entirely, this keeps only the sentences whose sent_id
# starts with one of the given prefixes (see Treebank.__new__, which reads
# the source apart by sent_id since the merged file's "newdoc id" comment
# isn't repeated on every sentence). A treebank not listed here is
# unfiltered.
#
# UD_Czech-PDTC bundles PDT (the original newspaper/weekly/magazine
# treebank we actually want) with PCEDT (translated Wall Street Journal,
# "wsj"), PDTSC (transcribed spoken dialogs, "pdtsc") and Faust (MT test
# translations, "faust") -- see its README's "Domains and Data Split"
# section for the source->prefix mapping. "lnd" is a second Lidové noviny
# section not named in that table but distinct from every PCEDT/PDTSC/
# Faust prefix, so by elimination it's also original PDT text.
SENT_ID_PREFIX_KEEP = {
    "Czech": {
        "PDTC": {"ln", "lnd", "mf", "cmpr", "vesm"},
    },
}


def sent_id_prefix_keep(lang: str, treebank_id: str) -> set[str] | None:
    """Sent_id prefixes to keep for (lang, treebank_id), or None if the
    treebank isn't in SENT_ID_PREFIX_KEEP (i.e. unfiltered, keep every
    sentence)."""
    return SENT_ID_PREFIX_KEEP.get(lang, {}).get(treebank_id)


lang2unimorph_lang = {
    "Standard Arabic": "Arabic",
    "Southern Pashto": "Pashto",
    "Guarani": "Mbyá Guaraní",
    "Northern Uzbek": "Uzbek",
    "North Azerbaijani": "Azerbaijani",
}

# UD-derived tables are keyed by the UD code; these UD codes name a different file in resources/unimorph
UNIMORPH_CODE_ALIASES = {
    "arb": "ara",  # Standard Arabic -> Arabic macrolanguage file
    "glg": "gal",  # Galician
    "hrv": "hbs",  # Croatian and Serbian share the Serbo-Croatian file
    "srp": "hbs",
    "swa": "swc",  # Standard Swahili -> Congo Swahili (verbs only)
}


def unimorph_code(langcode: str) -> str:
    """The resources/unimorph/<code> directory holding the UniMorph data for a UD language code."""
    return UNIMORPH_CODE_ALIASES.get(langcode, langcode)


remove_diacritics_langs = {
    "Latin", # is principled bc it's annotated for pronunciation
}

# UniMorph marks stress (acute, grave, circumflex) where UD does not; the letters č š ž stay
strip_um_stress_langs = {"Slovenian"}

STRESS_MARKS = {"\u0300", "\u0301", "\u0302", "\u030f", "\u0311"}


def strip_stress(text: str) -> str:
    """text without stress marks (acute, grave, circumflex), other letters and diacritics kept."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if ch not in STRESS_MARKS))

remove_multiples_langs = {
    "Modern Greek",
    "Greek",
    "Ancient Greek",
    "Ancient_Greek",
}

# UniMorph files whose script differs from the treebank's: the table is converted once, in um2ud.read_unimorph,
# so the whole pipeline (forms, lemmas, sentences, pairs) stays in the treebank's script.
# Uyghur: UniMorph is Latin (ULS), the treebank Arabic (UAS).
unimorph_script_converters = {"uig": ("ULS", "UAS")}


def convert_unimorph_script(langcode: str):
    """text -> text function turning a UniMorph word into the treebank's script, or None if the scripts agree."""
    pair = unimorph_script_converters.get(langcode)
    if pair is None:
        return None
    from umsc import UgMultiScriptConverter
    return UgMultiScriptConverter(*pair)

skip_langs = {
    "Frisian_Dutch",
    "Turkish_German",
    "Maghrebi_Arabic_French",
    "Telugu_English",
    "Turkish_English",
} | SIGN_LANGUAGES


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
    def ud_dir2lang(x):
        return("_".join(x.split("/")[-1].replace("UD_", "").split("-")[:-1]))

    treebank_dirs = [
        d for d in glob(os.path.join(resource_dir, ud_dir, "*"))
        if not is_treebank_excluded(ud_dir2lang(d), os.path.basename(d).rsplit("-", 1)[-1])
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
    "Ancient_Greek"). Treebanks in excluded_treebanks are dropped (a
    language left with none is dropped entirely) and add_langs entries are
    appended, the same two adjustments get_ud_langs and Treebank() apply.
    """
    if ud_dir is None:
        ud_dir = UD_PATH

    def ud_dir2lang(x):
        return "_".join(os.path.basename(x).replace("UD_", "").split("-")[:-1])

    by_lang = {}
    for path in sorted(glob(os.path.join(resource_dir, ud_dir, "*"))):
        folder = os.path.basename(path)
        if not folder.startswith("UD_"):
            continue
        lang = ud_dir2lang(path)
        if not is_treebank_excluded(lang, folder.rsplit("-", 1)[-1]):
            by_lang.setdefault(lang, []).append(folder)

    for lang, treebank_id in add_langs.items():
        by_lang.setdefault(lang, []).append(treebank_id)

    return by_lang

