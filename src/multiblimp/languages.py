import os
import re
import unicodedata

from glob import glob

from iso639 import Lang

from .config import UD_PATH


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
            "Telugu_English": name,
            "Turkish_English": name,
            "Turkish_German": name,
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

# Normalised exclusion reasons (the only values excluded_treebanks may use).
EXCLUSION_REASONS = (
    "spoken",
    "historical",
    "learner data",
    "social media",
    "poetry",
    "code switching",
    "child language",
    "niche genre",
    "nonstandard",
    #"regional variety",
    "non-Latin script",
    "duplicate",
    "too small",
    "low quality",
    "incompatible annotation",
  #  "pending review",  # no reason on record yet
)

# Treebanks excluded per language (UD language folder name -> {treebank id:
# reason}, reasons drawn from EXCLUSION_REASONS). Every other treebank of a
# language is used; a language whose treebanks are all excluded drops out
# entirely.
# Comments keep the provenance of each entry: "<version excluded since> [-f]:
# <original note>". A "?" on the version means unsure, "-f" means the
# treebank is also caught by flag_treebanks.
excluded_treebanks = {
    # "Arabic": {
    #     "NYUAD": "pending review",  # v2?: ?
    # },
    "Chinese": {
        "CFL": "learner data",  # v2? -f
    },
    "Classical_Chinese": {
        "TueCL": "too small",  # v1: comparatively small
    },
    "Czech": {
       # "PDTC": "pending review",  # filter out PDT sentences via sent_id
        "Poetry": "poetry",  # v1: poetry/genre
    },
    "English": {
        "Atis": "spoken",  # v2?: ?
       # "GENTLE": "pending review",  # v2?: ?
       # "LittlePrince": "pending review",  # v2?: ?
        "CHILDES": "child language",  # v2: child-adult spoken language
        "GUMReddit": "duplicate",  # v2?: duplication of GUM
        "ESLSpok": "spoken",  # v1
        "GUM": "spoken",  # v1: spoken (among others)
        "CTeTex": "niche genre",  # v1: technical genre
        "Pronouns": "niche genre",  # v1: niche
    },
    "Faroese": {
        "FarPaHC": "historical",  # v1
    },
    "French": {
        "ALTS": "historical",  # v2
        "PoitevinDIVITAL": "nonstandard",  # v2: poitevin-saintongeais
        "ParisStories": "spoken",  # v1
        "Rhapsodie": "spoken",  # v1
    },
    "Frisian_Dutch": {
        "Fame": "code switching",  # -f
    },
    "Galician": {
        "CTG": "low quality",  # v1: low star rating
    },
    "German": {
        "LIT": "historical",  # v1
    },
    "Gheg": {
        "GPS": "code switching",  # v2? -f: but is only gheg treebank
    },
    "Greek": {
        "GLCII": "learner data",  # v2? -f
    },
    "Guarani": {
        "OldTuDeT": "historical",  # v2? -f: old language variant, but is only guarani treebank
    },
    "Hebrew": {
        "IAHLTwiki": "incompatible annotation",  # v1?: not included, "IAHLT" as "incompatible"
        "IAHLTknesset": "spoken",  # v1
        "PostRab": "historical",  # v2?: post-Rabbinic
    },
    "Icelandic": {
        "IcePaHC": "historical",  # v1
    },
    "Irish": {
        "TwittIrish": "social media",  # v2? -f: tweets (also code switching)
    },
    "Italian": {
        "Old": "historical",  # v1
        "Valico": "learner data",  # v1: learner corpus
        "KIParlaForest": "spoken",  # v2?
        "PoSTWITA": "social media",  # v2? -f: tweets
        "TWITTIRO": "social media",  # v2? -f: tweets
    },
    "Japanese": {
        "BCCWJLUW": "duplicate",  # -: no version/reason given
        "GSDLUW": "duplicate",  # -: no version/reason given
        "PUDLUW": "duplicate",  # -: no version/reason given
    },
    "Korean": {
        "KSL": "learner data",  # v2? -f
    },
    "Latvian": {
        "Cairo": "too small",  # v1: only 20 sentences
    },
    "Maghrebi_Arabic_French": {
        "Arabizi": "code switching",  # -f
    },
    "Portuguese": {
        "DANTEStocks": "social media",  # v2? -f: tweets
    },
    "Romanian": {
        "MolDoRo": "non-Latin script",  # v2?: in cyrillic
       # "ArT": "regional variety",  # v2: included as Aromanian via add_langs (previously excluded as "dialect")
        "Nonstandard": "nonstandard",  # v1
        "TueCL": "social media",  # v2? -f: tweets
    },
    "Russian": {
        "Poetry": "poetry",  # v1: poetry/genre
    },
    "Sanskrit": {
        "UFAL": "too small",  # v1: comparatively small
    },
    "Slovenian": {
        "SST": "spoken",  # v1
    },
    "Spanish": {
        "COSER": "spoken",  # v1
    },
    "Swedish": {
        "Old": "historical",  # v2
        "SweLL": "learner data",  # v2? -f
    },
    "Telugu_English": {
        "TECT": "code switching",  # -f
    },
    "Turkish_English": {
        "BUTR": "code switching",  # -f
    },
    "Turkish_German": {
        "SAGT": "code switching",  # -f
    },
    "Vietnamese": {
        "TueCL": "spoken",  # v1
    },
}

assert all(r in EXCLUSION_REASONS for d in excluded_treebanks.values() for r in d.values())

def is_treebank_excluded(lang: str, treebank_id: str) -> bool:
    """treebank_id is the part after the last "-" of a UD folder name."""
    return treebank_id in excluded_treebanks.get(lang, ())


def treebank_exclusion_reason(lang: str, treebank_id: str) -> str | None:
    """Normalised reason (see EXCLUSION_REASONS) or None if not excluded."""
    return excluded_treebanks.get(lang, {}).get(treebank_id)


def _split_treebank_row(tb: str) -> tuple[str, str]:
    """"UD_Turkish_German-SAGT" -> ("Turkish_German", "SAGT") -- same shape
    as multiblimp.treebank._split_treebank_folder, duplicated here (not
    imported) since treebank.py already imports from this module and the
    reverse import would cycle.

    This is the language actually embedded in a "treebank" column value /
    tree.metadata["treebank"] string, which is NOT always the language
    whose pickle/parquet it was read from: a handful of bilingual/code-
    switched treebanks (e.g. UD_Turkish_German-SAGT, UD_Turkish_English-
    BUTR, UD_Telugu_English-TECT) are bundled into a DIFFERENT language's
    own pickle (Turkish's, Telugu's) at build time, but are registered in
    excluded_treebanks under their OWN embedded name ("Turkish_German",
    not "Turkish"). Checking is_treebank_excluded(current_lang, treebank_id)
    against the pickle's own language -- as every caller used to -- silently
    never matches these, so SAGT/BUTR/TECT rows leak into the pooled fit
    and the per-treebank page shows them as ordinary, unflagged tiles
    instead of excluded ones (see conversation)."""
    lang_part, _, treebank_id = tb.removeprefix("UD_").rpartition("-")
    return lang_part, treebank_id


def is_treebank_row_excluded(tb: str) -> bool:
    """is_treebank_excluded, but keyed off the treebank's OWN embedded
    language (see _split_treebank_row) rather than the caller's current
    language -- the correct check for a "treebank" column value / raw
    tree.metadata["treebank"] string, which may embed a different
    language than the one it was bundled/pickled under."""
    return is_treebank_excluded(*_split_treebank_row(tb))


def treebank_row_exclusion_reason(tb: str) -> str | None:
    """treebank_exclusion_reason, but keyed off the treebank's OWN embedded
    language -- see is_treebank_row_excluded."""
    return treebank_exclusion_reason(*_split_treebank_row(tb))


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


if __name__=="__main__":
    print(get_ud_langs("../../resources"))