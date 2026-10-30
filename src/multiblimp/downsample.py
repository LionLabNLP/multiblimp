"""Choose a fixed number of pairs per language from its correct_swaps tables, balanced across conditions.

Every condition (verb families, noun-phrase role pairs) contributes its share of the sample, and within a condition pairs are drawn greedily so that the balance features stay flat.
Reading and writing the tables is agreement_trees.eval.benchmark's job; this module only chooses rows."""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_ITEMS = 1000
LEMMA_CAP = 0.01
MIN_PAIRS = 10   # a language with fewer pairs in total is left out of the benchmark
DISTANCE_BINS = (1, 2, 4, 8)   # upper bounds of the bins 1, 2, 3-4, 5-8; larger distances share the last bin
LENGTH_BINS = (40, 80, 140, 240)   # upper bounds, in characters of the written sentence, of the bins up to 40, 41-80, 81-140, 141-240; longer ones share the last bin

ATTRACTOR_FLAGS = ("has_attractors", "only_has_fully_congruent", "only_has_weakly_congruent", "only_has_incongruent")
CATEGORICAL = ("feature_vals", "word_order", "swap_kind", "attractors", "pos_a", "pos_b", "distance", "length", "leaf_id", "treebank")
SINGLE_VALUED = tuple(name for name in CATEGORICAL if name not in ("feature_vals", "swap_kind"))   # weighted; the others are handled separately
PASS_THROUGH = ("pair_key", "surface_status")
NO_LEMMA = {"", "_"}


@dataclass
class Condition:
    name: str
    roles: tuple
    features: pd.DataFrame = None
    codes: dict = field(default_factory=dict)
    values: np.ndarray = None   # (pairs, alternatives) codes of the swap values a pair counts for, padded with 0
    shares: np.ndarray = None   # their weights per pair: 1 split over the alternatives (0 for padding)
    ambiguous: np.ndarray = None

    def __len__(self):
        return len(self.features)


def condition_roles(cond: str, deprel: str) -> tuple:
    """The two tokens of a pair: head and relation for the verb families, the role pair for noun phrases."""
    return ("head", deprel) if deprel else tuple(cond.split("_")[0].split("-"))


def value_alternatives(feature_vals: str) -> list:
    """The (original -> swapped) values a pair counts for: "Masc|Neut -> Fem" counts for Masc -> Fem and
    Neut -> Fem, one half each; the shares of a pair always sum to 1."""
    old, arrow, new = feature_vals.partition(" -> ")
    if not arrow:
        return [feature_vals]
    return [f"{o} -> {n}" for o in old.split("|") for n in new.split("|")]


def balance_columns(roles: tuple) -> list:
    """The source columns balance_features reads (those a table lacks are skipped)."""
    wanted = ["feature_vals", "leaf_id", "treebank", "sen_str", "sent_id", "tree_idx", "second_chance", "swap_kind",
              *ATTRACTOR_FLAGS, *PASS_THROUGH]
    for role in roles:
        wanted += [f"{role}_idx", f"{role}_lemma", f"{role}_pos"]
    return wanted


def balance_features(df: pd.DataFrame, roles: tuple, name: str) -> pd.DataFrame:
    """The balance features of one condition's pairs (a frame with balance_columns), one row per pair."""
    role_a, role_b = roles
    n = len(df)

    def column(col, default=None):
        return df[col] if col in df else pd.Series(default, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["feature_vals"] = column("feature_vals", "?").astype(str)
    idx_a, idx_b = (pd.to_numeric(column(f"{r}_idx"), errors="coerce") for r in roles)
    out["word_order"] = np.where(idx_a < idx_b, "a_first", "b_first")
    out["swap_kind"] = column("swap_kind", "?").astype(str)
    gap = (idx_a - idx_b).abs()
    out["distance"] = np.digitize(gap.fillna(DISTANCE_BINS[-1] + 1), DISTANCE_BINS, right=True)
    chars = df["sen_str"].astype(str).str.len() if "sen_str" in df else pd.Series(0, index=df.index)
    out["length"] = np.digitize(chars, LENGTH_BINS, right=True)
    out["pos_a"] = column(f"{role_a}_pos", role_a).astype(str)
    out["pos_b"] = column(f"{role_b}_pos", role_b).astype(str)
    out["leaf_id"] = column("leaf_id", -1).fillna(-1).astype(int).astype(str)
    out["treebank"] = column("treebank", "?").astype(str)
    flag = {flag_name: column(flag_name, False).fillna(False).astype(bool) for flag_name in ATTRACTOR_FLAGS}
    if "has_attractors" in df:
        out["attractors"] = np.select(
            [~flag["has_attractors"], flag["only_has_fully_congruent"], flag["only_has_weakly_congruent"],
             flag["only_has_incongruent"]],
            ["none", "congruent", "weakly_congruent", "incongruent"], default="mixed")
    else:
        out["attractors"] = "unknown"
    for key, role in (("lemma_a", role_a), ("lemma_b", role_b)):
        lemma = column(f"{role}_lemma", "").fillna("").astype(str).str.lower()
        out[key] = lemma.where(~lemma.isin(NO_LEMMA), "")
    if "sen_str" in df:
        out["sentence"] = df["sen_str"].astype(str)
    elif "sent_id" in df:
        out["sentence"] = column("treebank", "").astype(str) + "/" + df["sent_id"].astype(str)
    else:
        out["sentence"] = [f"{name}/{i}" for i in range(n)]
    out["second_chance"] = column("second_chance", False).fillna(False).astype(bool)
    for col in PASS_THROUGH:
        if col in df:
            out[col] = df[col]
    return out


def _codes(values: pd.Series, vocabulary: dict = None):
    """Integer codes for `values`; with a vocabulary (shared across conditions) unseen values are added to it."""
    if vocabulary is None:
        codes, _ = pd.factorize(values)
        return codes.astype(np.int64)
    return np.fromiter((vocabulary.setdefault(v, len(vocabulary)) for v in values), dtype=np.int64, count=len(values))


def _encode_values(cond: Condition):
    vocabulary = {}
    alternatives = [value_alternatives(text) for text in cond.features["feature_vals"]]
    width = max(map(len, alternatives), default=1)
    cond.values = np.zeros((len(alternatives), width), dtype=np.int64)
    cond.shares = np.zeros((len(alternatives), width))
    for i, alts in enumerate(alternatives):
        for j, alt in enumerate(alts):
            cond.values[i, j] = vocabulary.setdefault(alt, len(vocabulary))
        cond.shares[i, :len(alts)] = 1 / len(alts)
    cond.ambiguous = cond.features["feature_vals"].str.contains("|", regex=False).to_numpy()
    return len(vocabulary)


def encode(conditions: list):
    """Integer-code the balance features; lemmas and sentences share one code space across the conditions."""
    lemmas, sentences = {"": 0}, {}
    for cond in conditions:
        f = cond.features
        cond.codes = {name: _codes(f[name]) for name in (*SINGLE_VALUED, "swap_kind")}
        cond.codes["lemma_a"] = _codes(f["lemma_a"], lemmas)
        cond.codes["lemma_b"] = _codes(f["lemma_b"], lemmas)
        cond.codes["sentence"] = _codes(f["sentence"], sentences)
    return len(lemmas), len(sentences)


def select_pairs(conditions: list, n_items: int, lemma_cap: float, rng: np.random.Generator):
    """{condition name: sorted row positions}, plus how far the limits had to be relaxed.

    Conditions are served round-robin (the one with the fewest pairs so far goes next, so the shares stay equal
    and a small condition simply runs out). Within a condition, a pair is drawn with probability proportional to the
    product over the balance features of 1 / (1 + times that feature value is already in the sample). Pairs with
    an ambiguous swap value (feature_vals like "Masc|Neut -> Fem") are drawn only once the condition has no other
    eligible pair; they count fractionally for each of their values. Of the pairs left, only those whose reinflected
    token (swap_kind) has been picked least so far are drawn from, so the two sides of a noun-phrase pair of two
    dependents stay even as far as the data allows.
    A lemma may appear in at most `lemma_cap` of the sample and a sentence once; if the quota cannot be
    filled, the lemma cap rises by one step at a time, and only then sentences may repeat."""
    n_lemmas, n_sentences = encode(conditions)
    n_values = [_encode_values(c) for c in conditions]
    most_per_sentence = int(sum(np.bincount(c.codes["sentence"], minlength=n_sentences) for c in conditions).max())
    step = max(1, int(np.ceil(lemma_cap * n_items)))
    cap, sentence_limit = step, 1
    lemma_count = np.zeros(n_lemmas, dtype=np.int64)
    sentence_count = np.zeros(n_sentences, dtype=np.int64)
    taken = [np.zeros(len(c), dtype=bool) for c in conditions]
    kind_count = [np.zeros(c.codes["swap_kind"].max() + 1, dtype=np.int64) for c in conditions]
    feature_count = [{name: np.zeros(c.codes[name].max() + 1, dtype=np.int64) for name in SINGLE_VALUED}
                     for c in conditions]
    value_count = [np.zeros(n) for n in n_values]
    selected = np.zeros(len(conditions), dtype=np.int64)
    exhausted = np.zeros(len(conditions), dtype=bool)
    total = 0

    while total < n_items:
        open_conditions = np.flatnonzero(~exhausted)
        if not len(open_conditions):
            if cap < n_items:
                cap += step
            else:
                sentence_limit += 1
                if sentence_limit > most_per_sentence:
                    break
            exhausted[:] = False
            continue
        fewest = selected[open_conditions].min()
        c = int(rng.choice(open_conditions[selected[open_conditions] == fewest]))
        cond, codes = conditions[c], conditions[c].codes
        la, lb, sent = codes["lemma_a"], codes["lemma_b"], codes["sentence"]
        eligible = ~taken[c] & (sentence_count[sent] < sentence_limit)
        if cap < n_items:
            eligible &= ((lemma_count[la] < cap) | (la == 0)) & ((lemma_count[lb] < cap) | (lb == 0))
        rows = np.flatnonzero(eligible)
        if not len(rows):
            exhausted[c] = True
            continue
        if not cond.ambiguous[rows].all():
            rows = rows[~cond.ambiguous[rows]]
        kinds = codes["swap_kind"][rows]
        rows = rows[kind_count[c][kinds] == kind_count[c][kinds].min()]   # the reinflected side picked least so far
        log_weight = np.zeros(len(rows))
        log_weight -= np.log1p((cond.shares[rows] * value_count[c][cond.values[rows]]).sum(axis=1))
        for name in SINGLE_VALUED:
            log_weight -= np.log1p(feature_count[c][name][codes[name][rows]])
        for lemma_codes in (la, lb):
            log_weight -= np.log1p(np.where(lemma_codes[rows] == 0, 0, lemma_count[lemma_codes[rows]]))
        weight = np.exp(log_weight - log_weight.max())
        row = int(rng.choice(rows, p=weight / weight.sum()))

        taken[c][row] = True
        kind_count[c][codes["swap_kind"][row]] += 1
        for name in SINGLE_VALUED:
            feature_count[c][name][codes[name][row]] += 1
        np.add.at(value_count[c], cond.values[row], cond.shares[row])
        for lemma in {la[row], lb[row]} - {0}:
            lemma_count[lemma] += 1
        sentence_count[sent[row]] += 1
        selected[c] += 1
        total += 1

    chosen = {cond.name: np.flatnonzero(taken[i]) for i, cond in enumerate(conditions)}
    return chosen, {"lemma_cap_count": int(cap), "sentence_limit": int(sentence_limit)}


def language_seed(lang: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{lang}".encode()).digest()[:8], "big")
