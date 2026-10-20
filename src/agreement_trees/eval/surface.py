"""Surface strings for pairs in sentences with multiword tokens (UD ID ranges such as 3-4 "du" = "de le").

correct_swaps sentences are joined from syntactic words ("de le"); the benchmark needs the written form ("du").
For every pair whose sentence contains a multiword token:
  a) the span is not the swapped token: render the span as its surface form in both sentences;
  a') the swapped token is a plain token: if it now forms a known decontracted sequence with its neighbours
     (de la -> de le), the wrong sentence gets the contracted form (du) when that is more frequent than the
     uncontracted sequence, as in i);
  b) the span contains the swapped token: render it in the sentence, and for the wrong sentence look up the
     swapped components' contracted form in the language's spans:
       i)  a contracted form exists: use it if it is more frequent than the uncontracted component sequence
           in the language's treebanks (not counting span components), else keep the uncontracted words;
       ii) none exists: keep the uncontracted words if that sequence occurs in the treebanks, else drop the pair.
"""
import collections
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from agreement_trees.eval.benchmark import SURFACE_COLUMNS, SURFACE_FILE, pair_files
from agreement_trees.eval.conllu_spans import contraction_map, iter_sentences, language_of, norm, recase, render
from agreement_trees.pairs.sentence_text import add_sentence_strings


def read_swap_rows(path) -> pd.DataFrame:
    """pair_key, conllu_file, conllu_sent_line, swap_idx and swap_form of a correct_swaps file; files from before
    the pair_key columns existed get them derived."""
    names = pq.ParquetFile(path).schema_arrow.names
    swap_cols = [c for c in names if c.startswith("swap_") and c not in ("swap_ufeat", "swap_kind", "swap_idx")]
    idx_cols = [f"{c[len('swap_'):]}_idx" for c in swap_cols if f"{c[len('swap_'):]}_idx" in names]
    if "pair_key" in names:
        cols = ["pair_key", "swap_kind", "swap_idx", "conllu_file", "conllu_sent_line", *swap_cols]
        df = pd.read_parquet(path, columns=cols)
    else:
        cols = ["sen", "no_space_after", "treebank", "sent_id", "swap_ufeat", "conllu_file", "conllu_sent_line",
                *swap_cols, *idx_cols]
        df = add_sentence_strings(pd.read_parquet(path, columns=cols))
    form = pd.Series([None] * len(df), index=df.index, dtype=object)
    for kind in df["swap_kind"].dropna().unique():
        mask = df["swap_kind"] == kind
        form[mask] = df.loc[mask, f"swap_{kind}"]
    df["swap_form"] = form
    df["conllu_file"] = df["conllu_file"].astype(str)
    return df[["pair_key", "conllu_file", "conllu_sent_line", "swap_idx", "swap_form"]].dropna().reset_index(drop=True)


def validate_language(ud_dir, lang: str, rows: pd.DataFrame):
    """Surface strings for the rows of one language (columns: target, pair_key, conllu_file, conllu_sent_line,
    swap_idx, swap_form). Returns ({target: [(pair_key, sen_str, wrong_sen_str, status)]}, {target: Counter});
    rows whose sentence is untouched (no multiword token, no merge) are left out."""
    files = sorted(Path(ud_dir).glob(f"UD_{lang}-*/*.conllu"))
    by_sentence = collections.defaultdict(list)
    for i, (conllu_file, line) in enumerate(zip(rows["conllu_file"], rows["conllu_sent_line"])):
        by_sentence[(conllu_file, int(line))].append(i)

    targets, keys = rows["target"].tolist(), rows["pair_key"].tolist()
    swap_idx, swap_form = rows["swap_idx"].tolist(), rows["swap_form"].tolist()
    stats = collections.defaultdict(collections.Counter)
    contracted, span_positions = contraction_map(files)
    final, pending = {}, {}

    for path in files:
        rel = f"{path.parent.name}/{path.name}"
        for start, text, units in iter_sentences(path):
            ids = by_sentence.get((rel, start))
            if not ids:
                continue
            has_mwt = any(u.mwt for u in units)
            sen = render(units).strip()
            for i in ids:
                stats[targets[i]]["text_mismatch"] += has_mwt and sen != text
                stats[targets[i]]["rows_with_mwt"] += has_mwt
                k = next((j for j, u in enumerate(units) if u.lo <= int(swap_idx[i]) <= u.hi), None)
                if k is None:
                    stats[targets[i]]["no_unit"] += 1
                    continue
                u = units[k]
                if u.mwt:
                    comps = list(u.comps)
                    comps[int(swap_idx[i]) - u.lo] = swap_form[i]
                    pending[i] = ("span", sen, render(units[:k]), render(units[k + 1:]), u, comps)
                    continue
                plain_wrong = (render(units[:k]) + swap_form[i] + (" " if u.space else "") + render(units[k + 1:])).strip()
                # the reinflected token may now form a contractible sequence with its neighbours: de la -> de le
                # only where the language has spans and this UPOS occurs in them, at the positions and span lengths seen
                best = None
                for j, n in sorted(span_positions.get(u.upos, ())) if contracted else ():
                    s = k - j
                    if s < 0 or s + n > len(units) or any(w.mwt for w in units[s:s + n]):
                        continue
                    key = tuple(norm(w.form) if s + c != k else norm(swap_form[i]) for c, w in enumerate(units[s:s + n]))
                    if key in contracted:
                        total = sum(contracted[key].values())
                        if best is None or (total, n) > (best[0], best[1]):
                            best = (total, n, s, key)
                if best is not None:
                    _, n, s, key = best
                    pending[i] = ("merge", sen, plain_wrong, render(units[:s]), render(units[s + n:]), units[s], units[s + n - 1], key)
                elif has_mwt:
                    final[i] = (sen, plain_wrong, "mwt_elsewhere")

    queries = {item[-1] if item[0] == "merge" else tuple(norm(c) for c in item[5]) for item in pending.values()}
    counts = collections.Counter()
    if queries:
        for path in files:
            for _, _, units in iter_sentences(path):
                forms = [None if u.mwt else norm(u.form) for u in units]
                for n in {len(q) for q in queries}:
                    for j in range(len(units) - n + 1):
                        window = tuple(forms[j:j + n])
                        if window in queries:
                            counts[window] += 1

    for i, item in pending.items():
        if item[0] == "merge":
            _, sen, plain_wrong, before, after, first, last, key = item
            if sum(contracted[key].values()) > counts[key]:
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
            final[i] = (None, None, "dropped:no_valid_form")
            continue
        final[i] = (sen, (before + chosen + (" " if u.space else "") + after).strip(), status)

    out = collections.defaultdict(list)
    for i, (sen, wrong, status) in final.items():
        out[targets[i]].append((keys[i], sen, wrong, status))
        stats[targets[i]][status] += 1
    return out, stats


def run_validation(pairs_dir, ud_dir, langs=None, conds=None, report=None):
    """validate_language over every language of the correct_swaps under pairs_dir, one language at a time (the verdicts
    need its whole treebank data). Writes <target dir>/surface_strings.parquet per target and, if given, a
    per-target report CSV."""
    files_by_lang = collections.defaultdict(list)
    for path, lang, cond, deprel in pair_files(pairs_dir, langs, conds):
        files_by_lang[lang].append(path)

    rows_report = []
    for lang, paths in sorted(files_by_lang.items()):
        frames = [read_swap_rows(path).assign(target=str(path.parent)) for path in paths if pq.ParquetFile(path).metadata.num_rows]
        if not frames:
            continue
        rows = pd.concat(frames, ignore_index=True)
        out, stats = validate_language(ud_dir, language_of(rows["conllu_file"].iloc[0]), rows)
        for target, n_rows in rows.groupby("target").size().items():
            pd.DataFrame(out.get(target, []), columns=SURFACE_COLUMNS).to_parquet(os.path.join(target, SURFACE_FILE), index=False)
            rows_report.append({"lang": lang, "target": os.path.relpath(target, pairs_dir), "rows": n_rows,
                                **{k: int(v) for k, v in sorted(stats[target].items())}})
        print(lang, rows.shape[0], "rows;", sum(len(v) for v in out.values()), "listed in the surface file")

    if report:
        os.makedirs(os.path.dirname(report), exist_ok=True)
        pd.DataFrame(rows_report).fillna(0).to_csv(report, index=False)
