"""Treebanks kept out of the pipeline, and the sign languages dropped from the
start. scripts/agreement_trees/supplementary/flag_treebanks_report.py compares UD's flags against both."""

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
    "words removed",
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
    "Arabic": {
        "NYUAD": "words removed",  # -f
    },
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
        "ESL": "words removed",  # -f
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
        "FTB": "words removed",  # -f
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
    "Hindi_English": {
        "HIENCS": "words removed",  # -f
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
        "BCCWJ": "words removed",  # -f
        "KTC": "words removed",  # -f
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
    "Mbya_Guarani": {
        "Dooley": "words removed",  # -f
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


def split_treebank_folder(tb: str) -> tuple[str, str]:
    """"UD_Turkish_German-SAGT" -> ("Turkish_German", "SAGT").

    A treebank row's embedded language can differ from the language whose
    pickle it sits in (bilingual treebanks such as Turkish_German-SAGT are
    bundled into Turkish's pickle), so row checks key on the embedded one."""
    lang_part, _, treebank_id = tb.removeprefix("UD_").rpartition("-")
    return lang_part, treebank_id


def is_treebank_row_excluded(tb: str) -> bool:
    """is_treebank_excluded for a "treebank" column / metadata value."""
    return is_treebank_excluded(*split_treebank_folder(tb))


def treebank_row_exclusion_reason(tb: str) -> str | None:
    """treebank_exclusion_reason for a "treebank" column / metadata value."""
    return treebank_exclusion_reason(*split_treebank_folder(tb))


# UD folder language names of sign languages; dropped from the start.
SIGN_LANGUAGES = {
    "French_Sign_Language",
    "Spanish_Sign_Language",
    "Swedish_Sign_Language",
}
