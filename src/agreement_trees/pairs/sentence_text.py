import pandas as pd


def detokenize(tokens, no_space_after=None) -> str:
    """Plain-text sentence; follows the treebank's SpaceAfter=No marks when they line up with the tokens."""
    if no_space_after is None or len(no_space_after) != len(tokens):
        return " ".join(tokens)
    return "".join(tok + ("" if no_space else " ") for tok, no_space in zip(tokens, no_space_after)).strip()


def _swapped_kind(row: pd.Series):
    """(kind, 0-based token index, swapped form) of the role this row re-inflected, or None."""
    for col in row.index:
        if not col.startswith("swap_") or col == "swap_ufeat":
            continue
        kind = col[len("swap_"):]
        form, idx = row[col], row.get(f"{kind}_idx")
        if isinstance(form, str) and form and pd.notna(idx):
            return kind, int(idx) - 1, form
    return None


def add_sentence_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sen_str, wrong_sen_str (the sentence with the swapped form), swap_kind, swap_idx and pair_key.

    pair_key identifies a pair by treebank, sentence id, swapped token, swapped feature and swapped
    form, so it survives changes to the rest of the UD/UM data; eval hashes it with lang and condition.
    """
    sens, wrongs, kinds, idxs, keys = [], [], [], [], []
    swap_cols = [c for c in frame.columns if c.startswith("swap_") and c != "swap_ufeat"]
    needed = ["sen", "no_space_after", "treebank", "sent_id", "swap_ufeat", *swap_cols]
    needed += [f"{c[len('swap_'):]}_idx" for c in swap_cols if f"{c[len('swap_'):]}_idx" in frame.columns]
    # row-wise over just these columns: the frame has ~1,350
    for _, row in frame[list(dict.fromkeys(needed))].iterrows():
        tokens = list(row["sen"])
        nsa = row.get("no_space_after")
        nsa = None if nsa is None else list(nsa)
        swapped = _swapped_kind(row)
        sens.append(detokenize(tokens, nsa))
        if swapped is None:
            wrongs.append(None)
            kinds.append(None)
            idxs.append(None)
            keys.append(None)
            continue
        kind, idx, form = swapped
        wrong_tokens = list(tokens)
        wrong_tokens[idx] = form
        wrongs.append(detokenize(wrong_tokens, nsa))
        kinds.append(kind)
        idxs.append(idx + 1)
        keys.append(f"{row['treebank']}|{row['sent_id']}|{kind}{idx + 1}|{row.get('swap_ufeat')}|{form}")

    out = frame.copy()
    out["sen_str"] = sens
    out["wrong_sen_str"] = wrongs
    out["swap_kind"] = kinds
    out["swap_idx"] = pd.array(idxs, dtype="Int64")
    out["pair_key"] = keys
    return out
