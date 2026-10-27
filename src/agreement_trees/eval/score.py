import hashlib
import os

import pandas as pd
from lm_eval.api.instance import Instance


def text_hash(sen: str, wrong_sen: str) -> str:
    return hashlib.sha1(f"{sen}\n{wrong_sen}".encode("utf-8")).hexdigest()[:12]


def score_sentences(lm, sentences: list[str]) -> dict[str, float]:
    """Log-probability of each unique sentence given only BOS, via the batched loglikelihood of an
    lm-eval HFLM (the batch size is picked by its autobatching)."""
    unique = list(dict.fromkeys(sentences))
    too_long = sum(len(lm.tok_encode(sen)) >= lm.max_length for sen in unique) if hasattr(lm, "tok_encode") else 0
    if too_long:
        print(f"WARNING: {too_long} sentences reach the model's context length ({lm.max_length} tokens) and are truncated")
    requests = [Instance(request_type="loglikelihood", doc={}, arguments=("", sen), idx=0) for sen in unique]
    return dict(zip(unique, (logprob for logprob, _ in lm.loglikelihood(requests))))


def score_pairs(lm, pairs: pd.DataFrame) -> pd.DataFrame:
    """pair_id, text_hash (of the two sentences scored), both log-probabilities and their delta (sen - wrong_sen)."""
    logprobs = score_sentences(lm, pairs["sen"].tolist() + pairs["wrong_sen"].tolist())
    scores = pairs[["pair_id"]].copy()
    scores["text_hash"] = [text_hash(s, w) for s, w in zip(pairs["sen"], pairs["wrong_sen"])]
    scores["sen_logprob"] = pairs["sen"].map(logprobs)
    scores["wrong_logprob"] = pairs["wrong_sen"].map(logprobs)
    scores["delta"] = scores["sen_logprob"] - scores["wrong_logprob"]
    return scores


def accuracy(scored: pd.DataFrame, by=("lang", "phenomenon")) -> pd.DataFrame:
    """Share of pairs with delta > 0 (ties count as wrong) per group, with the pair count."""
    return (
        scored.assign(correct=scored["delta"] > 0)
        .groupby(list(by))
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )


def evaluate(model: str, pairs: pd.DataFrame, results_dir, dtype="float32", device="cuda", batch_size="auto:8",
             max_batch_size=1024, lm=None, rescore=False):
    """Score the pairs not yet in <results_dir>/<model>__<dtype>/scores.parquet with an lm-eval HFLM and write the
    scores and accuracy.csv (lang x phenomenon) there. A stored score counts only for the same pair_id and the same
    two sentences (text_hash); the scores are saved after each language. lm: an already loaded model (tests).
    rescore: score every given pair again. The new scores go to scores.rescore.parquet (saved after each language), so
    an interrupted rescore leaves scores.parquet untouched and the next --rescore picks up where it stopped; when it
    completes they replace the old scores and the staging file is removed.
    Returns (scored pairs, accuracy table, output dir)."""
    out_dir = os.path.join(results_dir, f"{model.replace('/', '__')}__{dtype}")
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.parquet")
    staging_path = os.path.join(out_dir, "scores.rescore.parquet")
    empty = pd.DataFrame(columns=["pair_id", "text_hash"])
    old = pd.read_parquet(scores_path) if os.path.exists(scores_path) else empty
    staged = pd.read_parquet(staging_path) if rescore and os.path.exists(staging_path) else empty
    if not rescore and os.path.exists(staging_path):
        print(f"NOTE: an unfinished rescore is stored in {staging_path}; run with --rescore to finish it")
    current = pairs.assign(text_hash=[text_hash(s, w) for s, w in zip(pairs["sen"], pairs["wrong_sen"])])
    done = set(zip(staged["pair_id"], staged["text_hash"])) if rescore else set(zip(old["pair_id"], old["text_hash"]))
    todo = current[[key not in done for key in zip(current["pair_id"], current["text_hash"])]]
    print(f"{len(todo)} of {len(pairs)} pairs not scored yet")

    if len(todo):
        if lm is None:
            from lm_eval.models.huggingface import HFLM

            lm = HFLM(pretrained=model, batch_size=batch_size, max_batch_size=max_batch_size, dtype=dtype, device=device)
        target = staged[~staged["pair_id"].isin(todo["pair_id"])] if rescore else old[~old["pair_id"].isin(todo["pair_id"])]
        for lang, chunk in todo.groupby("lang", sort=True):
            target = pd.concat([target, score_pairs(lm, chunk)], ignore_index=True)
            target.to_parquet(staging_path if rescore else scores_path, index=False)
            print(f"{lang}: {len(chunk)} pairs scored", flush=True)
        if rescore:
            staged = target
        else:
            old = target
    if rescore:
        old = pd.concat([old[~old["pair_id"].isin(staged["pair_id"])], staged], ignore_index=True)
        old.to_parquet(scores_path, index=False)
        if os.path.exists(staging_path):
            os.remove(staging_path)

    scored = pairs.drop(columns=["sen", "wrong_sen"]).merge(old.drop(columns="text_hash"), on="pair_id")
    acc = accuracy(scored)
    acc.to_csv(os.path.join(out_dir, "accuracy.csv"), index=False)
    return scored, acc, out_dir
