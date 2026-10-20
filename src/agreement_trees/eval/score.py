import os

import pandas as pd
from lm_eval.api.instance import Instance


def score_sentences(lm, sentences: list[str]) -> dict[str, float]:
    """Log-probability of each unique sentence given only BOS, via the batched loglikelihood of an
    lm-eval HFLM (the batch size is picked by its autobatching)."""
    unique = list(dict.fromkeys(sentences))
    requests = [Instance(request_type="loglikelihood", doc={}, arguments=("", sen), idx=0) for sen in unique]
    return dict(zip(unique, (logprob for logprob, _ in lm.loglikelihood(requests))))


def score_pairs(lm, pairs: pd.DataFrame) -> pd.DataFrame:
    """pair_id, both log-probabilities and their delta (sen - wrong_sen) for each pair."""
    logprobs = score_sentences(lm, pairs["sen_str"].tolist() + pairs["wrong_sen_str"].tolist())
    scores = pairs[["pair_id"]].copy()
    scores["sen_logprob"] = pairs["sen_str"].map(logprobs)
    scores["wrong_logprob"] = pairs["wrong_sen_str"].map(logprobs)
    scores["delta"] = scores["sen_logprob"] - scores["wrong_logprob"]
    return scores


def accuracy(scored: pd.DataFrame, by=("lang", "cond")) -> pd.DataFrame:
    """Share of pairs with delta > 0 (ties count as wrong) per group, with the pair count."""
    return (
        scored.assign(correct=scored["delta"] > 0)
        .groupby(list(by))
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )


def evaluate(model: str, pairs: pd.DataFrame, results_dir, dtype="float32", device="cuda", batch_size="auto:8",
             max_batch_size=1024):
    """Score the pairs not yet in <results_dir>/<model>__<dtype>/scores.parquet with an lm-eval HFLM and write the
    scores and accuracy.csv (lang x cond) there. Returns (scored pairs, accuracy table, output dir)."""
    out_dir = os.path.join(results_dir, f"{model.replace('/', '__')}__{dtype}")
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.parquet")
    old = pd.read_parquet(scores_path) if os.path.exists(scores_path) else pd.DataFrame(columns=["pair_id"])
    todo = pairs[~pairs["pair_id"].isin(old["pair_id"])]
    print(f"{len(todo)} of {len(pairs)} pairs not scored yet")

    if len(todo):
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=model, batch_size=batch_size, max_batch_size=max_batch_size, dtype=dtype, device=device)
        old = pd.concat([old, score_pairs(lm, todo)], ignore_index=True)
        old.to_parquet(scores_path, index=False)

    scored = pairs.drop(columns=["sen_str", "wrong_sen_str"]).merge(old, on="pair_id")
    acc = accuracy(scored)
    acc.to_csv(os.path.join(out_dir, "accuracy.csv"), index=False)
    return scored, acc, out_dir
