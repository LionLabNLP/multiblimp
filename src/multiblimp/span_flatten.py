"""Collapse a language's multiword tokens into one token per written word.

Some treebanks write one word as a multiword token and split it into pieces (UD_Amharic-ATT: አያስወጣም is the pieces
ኣል (negation) + ይ (subject marker) + ኣስወጣ (the verb) + ም). The lexicons list whole written words, and the controller of a
verb's agreement can be one of the pieces, so reinflecting a single piece yields spelling nobody writes. Languages in
languages.FLATTEN_SPAN_LANGS are therefore loaded with every multiword token as ONE token, the shape of treebanks such as
UD_Yupik-SLI, where the verb is a whole word carrying layered features (Person[subj]=3).

The new token is built from the span's main piece: the piece that is attached to something outside the word (the verb
stem ኣስወጣ in the example; the first content word if several are attached outside, a pronoun only if
there is no other). It keeps the main piece's lemma,
UPOS, XPOS, features, relation and head, and takes the written word as its form. Each other piece is handled by what it
is:

  - a pronoun piece becomes layered agreement features (Person[subj], Number[obj], ...). The layer is the piece's XPOS
    where the treebank marks it (SUBJC subject marker, OBJC object marker, POSM possessor), with iobj for an OBJC piece
    whose relation is iobj; a pronoun piece without such an XPOS goes by its relation (nsubj/expl with a Person,
    obj, iobj, or a possessor with Poss=Yes) and only if it hangs on the main piece. Two
    pieces for one layer with the same values give that layer once; with different values the word gets no such layer
    at all (two subject markers that disagree: the data does not say which belongs to the word, and the word then has
    no subject features to swap, as with a verb that has two conflicting nsubj). A pronoun piece that fits none of
    this is not carried over. A word with a piece that has no features (the auxiliary ኣል inside አስጨርሰዋለሁ, a
    subordinator ብ, a case marker ን) gets no layers either: its markers may belong to the stem or to the extra piece,
    and the extra piece is not in its feature bundle, so other words with the same bundle could not be told from it.
    Such a word is not reinflected, and its token is marked no_fill so that the lexicon fill of the extraction leaves
    its features as they are. The particle ም of a negated word (ኣል ... ም) is exempt: the negation piece already puts Polarity=Neg into the bundle;
  - an article piece gives a noun Definite=Def;
  - any other piece (negation, case marker, subordinator, auxiliary, ...) hands its own features over as plain
    features (the negation piece's Polarity=Neg, a case marker's Case=Loc), unless the main piece already has that
    feature. A piece without features adds nothing;
  - whatever pointed at a dropped piece points at the new token.

Token ids are renumbered; the main piece's original id stays in "orig_id" for the line numbers of the source file.
"""
import copy

CONTENT_UPOS = ("VERB", "NOUN", "PROPN", "ADJ", "NUM", "AUX", "PRON", "ADV")
AGREEMENT_KEYS = ("Person", "Number", "Gender", "Animacy", "Polite", "Clusivity")
NOT_HANDED_OVER = {"PronType", "Poss", "Typo", "Foreign", "Abbr", "Style", "Reflex"}   # about the piece, not the word


XPOS_SLOT = {"SUBJC": "subj", "OBJC": "obj", "POSM": "psor"}


def _slot(piece, feats, attached: bool):
    """The layer a pronoun piece's agreement features go to on the main piece, or None."""
    if piece["upos"] != "PRON":
        return None
    slot = XPOS_SLOT.get(piece["xpos"])
    if slot == "obj" and piece["deprel"] == "iobj":
        return "iobj"
    if slot or not attached:
        return slot
    relation = piece["deprel"].split(":")[0]
    if relation in ("nsubj", "expl") and "Person" in feats:
        return "subj"
    if relation in ("obj", "iobj"):
        return relation
    if feats.get("Poss") == "Yes":
        return "psor"
    return None


def has_featureless_piece(pieces, main) -> bool:
    """Whether the word holds a piece other than its main piece that has no features (not a pronoun or article); the
    particle ም of a negated word is exempt."""
    negated = any((piece["feats"] or {}).get("Polarity") == "Neg" for piece in pieces)
    return any(piece is not main and piece["upos"] not in ("PRON", "DET") and not piece["feats"]
               and not (negated and piece["upos"] == "PART" and piece["deprel"] == "discourse") for piece in pieces)


def flatten_spans(tree):
    """`tree` with every multiword token replaced by one token (see the module docstring); empty nodes are dropped, as
    for any other language. A tree without spans comes back as a list of its plain tokens."""
    words = [tok for tok in tree if isinstance(tok["id"], int)]   # the input is left as it is
    ranges = [tok for tok in tree if isinstance(tok["id"], tuple) and tok["id"][1] == "-"]
    if not ranges:
        return tree.__class__(words, tree.metadata) if len(words) != len(tree) else tree

    main_of = {}      # piece id -> id of the main piece of its span
    emit_at = {}      # id of the span's first piece -> the new token
    for span in ranges:
        first, _, last = span["id"]
        pieces = [tok for tok in words if first <= tok["id"] <= last]
        if not pieces:
            continue
        outside = [tok for tok in pieces if not (first <= tok["head"] <= last)]
        pool = outside or pieces
        main = next((tok for tok in pool if tok["upos"] in CONTENT_UPOS and tok["upos"] != "PRON"),
                    next((tok for tok in pool if tok["upos"] in CONTENT_UPOS), pool[0]))
        merged = copy.copy(main)
        merged["form"] = span["form"]
        merged["misc"] = span["misc"]
        merged["orig_id"] = main["id"]
        feats = dict(main["feats"] or {})
        layers = {}
        for piece in pieces:
            if piece is main:
                continue
            piece_feats = piece["feats"] or {}
            slot = _slot(piece, piece_feats, piece["head"] == main["id"])
            if slot is not None:
                layers.setdefault(slot, []).append({key: piece_feats[key] for key in AGREEMENT_KEYS if key in piece_feats})
            elif piece["upos"] == "DET" and piece_feats.get("PronType") == "Art" and main["upos"] in ("NOUN", "PROPN"):
                feats.setdefault("Definite", "Def")
            elif piece["upos"] != "PRON":
                for key, value in piece_feats.items():
                    if key not in NOT_HANDED_OVER:
                        feats.setdefault(key, value)
        not_swappable = has_featureless_piece(pieces, main)
        for slot, bundles in layers.items():
            if not_swappable:
                break
            if all(bundle == bundles[0] for bundle in bundles):
                for key, value in bundles[0].items():
                    feats[f"{key}[{slot}]"] = value
        merged["feats"] = feats or None
        if not_swappable:
            merged["no_fill"] = True
        for piece in pieces:
            main_of[piece["id"]] = main["id"]
        emit_at[first] = merged

    kept = []
    for tok in words:
        if tok["id"] in emit_at:
            kept.append(emit_at[tok["id"]])
        elif tok["id"] not in main_of:
            kept.append(copy.copy(tok))
    new_id = {}
    for position, tok in enumerate(kept, 1):
        new_id[tok.get("orig_id", tok["id"])] = position
    for old, main in main_of.items():
        new_id[old] = new_id[main]
    for position, tok in enumerate(kept, 1):
        head = tok["head"]
        tok["head"] = new_id.get(head, 0) if head else 0
        if tok["head"] == position:
            tok["head"] = 0
        tok["id"] = position
    return tree.__class__(kept, tree.metadata)
