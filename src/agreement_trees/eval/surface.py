"""Surface strings for pairs in sentences with multiword tokens (UD ID ranges such as 3-4 "du" = "de le").

Pair sentences are joined from syntactic words ("de le"); the benchmark needs the written form ("du"). For every pair
whose sentence contains a multiword token, or whose swapped word may now form one:
  a) the span is not the swapped token: render the span as its surface form in both sentences;
  a') the swapped token is a plain token: if it now forms a known decontracted sequence with its neighbours
     (de la -> de le), the wrong sentence gets the contracted form (du) when that is more frequent than the
     uncontracted sequence, as in i);
  b) the span contains the swapped token: render it in the sentence, and for the wrong sentence
       0) if the components sit in the written form separated only by hyphens (pode-se, fer-hi), put the new word in
          place of its component and keep the rest of the spelling (devem-se);
       otherwise look up the swapped components' contracted form in the language's spans:
       i)  a contracted form exists: use it if it is more frequent than the uncontracted component sequence
           in the language's treebanks (not counting span components), else keep the uncontracted words;
       ii) none exists: keep the uncontracted words (whether or not that sequence occurs in the treebanks).
A contraction only counts for a swapped word if the contracted span has that word's UPOS at that position.
The statistics are those of the language's non-excluded treebanks (agreement_trees.eval.mwt_cache).
"""
import collections
import os

import pandas as pd
import pyarrow.parquet as pq

from agreement_trees.eval.benchmark import SURFACE_COLUMNS, SURFACE_FILE, pair_files
from agreement_trees.eval.conllu_spans import Unit, language_of, norm, recase, render
from agreement_trees.eval.mwt_cache import load_stats
from agreement_trees.pairs.sentence_text import add_sentence_strings

# a plain word that now fuses with its neighbour (a') is merged only where the contraction is the norm: seen at least
# MERGE_MIN_CONTRACTED times, with the plain sequence at most MERGE_MAX_PLAIN_SHARE of that (de le -> du, but not the
# optional enclitic hyphen of pode se -> pode-se, which the sentence's own spelling does not use)
MERGE_MIN_CONTRACTED = 10
MERGE_MAX_PLAIN_SHARE = 0.1


def _plain_units(tokens, no_space_after) -> list:
    try:
        marks = [bool(m) for m in no_space_after]
    except TypeError:
        marks = []
    if len(marks) != len(tokens):
        marks = [False] * len(tokens)
    return [Unit(False, tok, i, i, [], not no_space, None, []) for i, (tok, no_space) in enumerate(zip(tokens, marks), 1)]


def hyphen_substitute(surface: str, comps, k: int, new_form: str):
    """`surface` with component k replaced by `new_form`, when the components are found in it in order, separated only
    by hyphens, with at least one hyphen (pode-se -> devem-se); None otherwise. A fused span (du = de le) and a span
    whose spelling changes with its parts (Turkish yapmış + dı) are left to the contraction lookup."""
    low, pos, found = surface.lower(), 0, []
    for comp in comps:
        comp = norm(comp)
        i = low.find(comp, pos)
        if not comp or i < 0 or any(ch != "-" for ch in low[pos:i]):
            return None
        found.append((i, i + len(comp)))
        pos = i + len(comp)
    if pos != len(low) or "-" not in surface:
        return None
    start, end = found[k]
    start += len(low[start:end]) - len(low[start:end].lstrip("-"))
    end -= len(low[start:end]) - len(low[start:end].rstrip("-"))
    return surface[:start] + recase(surface[start:end], new_form) + surface[end:]


def validate_language(stats, rows, pos_guard: bool = True, strict_merge: bool = True, enclitic: bool = True):
    """Surface strings for the rows of one language. rows: dicts with i (row id), key ((treebank file, sentence line)),
    tokens, no_space_after, swap_idx (1-based), swap_form, swap_pos. Returns ({i: (sen_str, wrong_sen_str, status)},
    Counter of statuses); rows whose sentence is untouched (no multiword token, no merge) are left out."""
    contracted, span_positions, patterns = stats.contracted, stats.positions, stats.patterns
    final, pending, tally = {}, {}, collections.Counter()

    for r in rows:
        i, swap_idx, swap_form = r["i"], int(r["swap_idx"]), r["swap_form"]
        cached = stats.sentences.get(r["key"])
        if cached is not None:
            units = [Unit(m, form, lo, hi, list(comps), space, upos, list(comp_upos))
                     for m, form, lo, hi, comps, space, upos, comp_upos in cached[1]]
        else:
            units = _plain_units(r["tokens"], r["no_space_after"])
        has_mwt = any(u.mwt for u in units)
        sen = render(units).strip()
        k = next((j for j, u in enumerate(units) if u.lo <= swap_idx <= u.hi), None)
        if k is None:
            tally["no_unit"] += 1
            continue
        u = units[k]
        if u.mwt:
            comps = list(u.comps)
            kept = hyphen_substitute(u.form, comps, swap_idx - u.lo, swap_form) if enclitic else None
            if kept is not None:
                final[i] = (sen, (render(units[:k]) + kept + (" " if u.space else "") + render(units[k + 1:])).strip(),
                            "enclitic_kept")
                continue
            comps[swap_idx - u.lo] = swap_form
            pending[i] = ("span", sen, render(units[:k]), render(units[k + 1:]), u, comps)
            continue
        upos = u.upos or r["swap_pos"]
        plain_wrong = (render(units[:k]) + swap_form + (" " if u.space else "") + render(units[k + 1:])).strip()
        # the reinflected token may now form a contractible sequence with its neighbours: de la -> de le
        # only where the language has spans and this UPOS occurs in them, at the positions and span lengths seen
        best = None
        for j, n in sorted(span_positions.get(upos, ())) if contracted else ():
            s = k - j
            if s < 0 or s + n > len(units) or any(w.mwt for w in units[s:s + n]):
                continue
            key = tuple(norm(w.form) if s + c != k else norm(swap_form) for c, w in enumerate(units[s:s + n]))
            if key not in contracted:
                continue
            if pos_guard and not any(len(p) == n and p[j] == upos for p in patterns.get(key, ())):
                continue
            total = sum(contracted[key].values())
            if best is None or (total, n) > (best[0], best[1]):
                best = (total, n, s, key)
        if best is not None:
            _, n, s, key = best
            pending[i] = ("merge", sen, plain_wrong, render(units[:s]), render(units[s + n:]), units[s], units[s + n - 1], key)
        elif has_mwt:
            final[i] = (sen, plain_wrong, "mwt_elsewhere")

    queries = {item[-1] if item[0] == "merge" else tuple(norm(c) for c in item[5]) for item in pending.values()}
    counts = stats.counts(queries) if queries else {}

    for i, item in pending.items():
        if item[0] == "merge":
            _, sen, plain_wrong, before, after, first, last, key = item
            total = sum(contracted[key].values())
            norm_contraction = total >= MERGE_MIN_CONTRACTED and counts[key] <= MERGE_MAX_PLAIN_SHARE * total
            if total > counts[key] and (norm_contraction or not strict_merge):
                surface = recase(first.form, contracted[key].most_common(1)[0][0])
                final[i] = (sen, (before + surface + (" " if last.space else "") + after).strip(), "merged")
            else:
                final[i] = (sen, plain_wrong, "kept_uncontracted")
            continue
        _, sen, before, after, u, comps = item
        key = tuple(norm(c) for c in comps)
        candidates, uncontracted_count = contracted.get(key), counts[key]
        plain = " ".join(c[:-1] if len(c) > 1 and c.endswith("_") else c for c in comps)
        if candidates and sum(candidates.values()) > uncontracted_count:
            chosen, status = recase(u.form, candidates.most_common(1)[0][0]), "contracted"
        elif candidates:
            chosen, status = plain, "uncontracted_more_frequent"
        elif uncontracted_count > 0:
            chosen, status = plain, "uncontracted_attested"
        else:
            chosen, status = plain, "uncontracted_unattested"
        final[i] = (sen, (before + chosen + (" " if u.space else "") + after).strip(), status)

    for _, _, status in final.values():
        tally[status] += 1
    return final, tally


def validate_frame(frame: pd.DataFrame, pos_guard: bool = True, stats_for=load_stats, strict_merge: bool = True,
                   enclitic: bool = True):
    """(DataFrame indexed like `frame` with sen_final, wrong_final and status -- NaN for rows whose sentence is
    untouched --, Counter of statuses) for a pairs frame with the add_sentence_strings columns (swap_kind, swap_idx)
    plus conllu_file, conllu_sent_line, sen, no_space_after, swap_<kind> and <kind>_pos. Rows may span languages
    (a Turkish cache holds Turkish_German sentences): each is validated against its own language's statistics."""
    kinds = frame["swap_kind"]
    forms = pd.Series(None, index=frame.index, dtype=object)
    poses = pd.Series(None, index=frame.index, dtype=object)
    for kind in kinds.dropna().unique():
        mask = kinds == kind
        forms[mask] = frame.loc[mask, f"swap_{kind}"]
        if f"{kind}_pos" in frame.columns:
            poses[mask] = frame.loc[mask, f"{kind}_pos"]

    files = frame["conllu_file"].astype(str).tolist()
    lines = frame["conllu_sent_line"].tolist()
    by_lang = collections.defaultdict(list)
    for i, (idx, form, swap_idx) in enumerate(zip(frame.index, forms.tolist(), frame["swap_idx"].tolist())):
        if not isinstance(form, str) or pd.isna(swap_idx) or pd.isna(lines[i]):
            continue
        by_lang[language_of(files[i])].append({
            "i": i, "key": (files[i], int(lines[i])),
            "tokens": list(frame["sen"].iloc[i]), "no_space_after": frame["no_space_after"].iloc[i],
            "swap_idx": swap_idx, "swap_form": form, "swap_pos": poses.iloc[i]})

    out = pd.DataFrame({"sen_final": None, "wrong_final": None, "status": None}, index=frame.index, dtype=object)
    tally = collections.Counter()
    for lang, rows in by_lang.items():
        result, counts = validate_language(stats_for(lang), rows, pos_guard, strict_merge, enclitic)
        tally.update(counts)
        for i, (sen, wrong, status) in result.items():
            out.iloc[i, 0], out.iloc[i, 1], out.iloc[i, 2] = sen, wrong, status
    return out, tally


def with_surface_strings(frame: pd.DataFrame, pos_guard: bool = True):
    """(kept, invalid): `frame` (a correct_swaps frame after add_sentence_strings, changed in place) with the written forms
    in sen_str and wrong_sen_str and a surface_status column; `invalid` is empty, as every pair has a written form. A frame
    without the sentence-location columns (an old cache) is returned as it is."""
    if not len(frame) or not {"conllu_file", "conllu_sent_line", "swap_kind", "sen"} <= set(frame.columns):
        return frame, frame.iloc[0:0]
    result, _ = validate_frame(frame, pos_guard)
    status = result["status"]
    changed = status.notna()
    frame.loc[changed, "sen_str"] = result.loc[changed, "sen_final"]
    frame.loc[changed, "wrong_sen_str"] = result.loc[changed, "wrong_final"]
    frame["surface_status"] = status.fillna("plain")
    return frame, frame.iloc[0:0]


def read_pair_frame(path) -> pd.DataFrame:
    """The columns of a correct_swaps file validate_frame reads; files from before the pair_key columns existed get them
    derived."""
    names = pq.ParquetFile(path).schema_arrow.names
    swap_cols = [c for c in names if c.startswith("swap_") and c not in ("swap_ufeat", "swap_kind", "swap_idx")]
    kinds = [c[len("swap_"):] for c in swap_cols]
    cols = ["sen", "no_space_after", "treebank", "sent_id", "swap_ufeat", "conllu_file", "conllu_sent_line", *swap_cols,
            *[c for kind in kinds for c in (f"{kind}_idx", f"{kind}_pos") if c in names]]
    if "pair_key" in names:
        cols += ["pair_key", "swap_kind", "swap_idx"]
        return pd.read_parquet(path, columns=cols)
    return add_sentence_strings(pd.read_parquet(path, columns=cols))


def run_validation(pairs_dir, ud_dir=None, langs=None, conds=None, report=None):
    """validate_frame over every correct_swaps under pairs_dir (the backfill for pair directories built before pair
    creation wrote the written forms): writes <language dir>/surface_strings.parquet (pair_key, sen_str, wrong_sen_str,
    surface_status for the rows it changed, dropped rows marked) and, if given, a per-directory report CSV."""
    rows_report = []
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        if not pq.ParquetFile(path).metadata.num_rows:
            continue
        frame = read_pair_frame(path)
        result, tally = validate_frame(frame)
        listed = result[result["status"].notna()]
        surface = pd.DataFrame({"pair_key": frame.loc[listed.index, "pair_key"].to_numpy(),
                                "sen_str": listed["sen_final"].to_numpy(), "wrong_sen_str": listed["wrong_final"].to_numpy(),
                                "surface_status": listed["status"].to_numpy()}, columns=SURFACE_COLUMNS)
        surface.to_parquet(os.path.join(path.parent, SURFACE_FILE), index=False)
        rows_report.append({"lang": lang, "target": os.path.relpath(path.parent, pairs_dir), "rows": len(frame),
                            **{k: int(v) for k, v in sorted(tally.items())}})
        print(lang, cond, len(frame), "rows;", len(surface), "listed in the surface file")

    if report:
        os.makedirs(os.path.dirname(report), exist_ok=True)
        pd.DataFrame(rows_report).fillna(0).to_csv(report, index=False)
