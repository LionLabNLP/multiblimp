import hashlib
import os

import pandas as pd
from lm_eval.api.instance import Instance


def text_hash(sen: str, wrong_sen: str) -> str:
    return hashlib.sha1(f"{sen}\n{wrong_sen}".encode("utf-8")).hexdigest()[:12]


def load_tokenizer(model: str):
    """AutoTokenizer, falling back to the plain tokenizer.json when the model's tokenizer class fails to build
    (transformers 5 AlbertTokenizer on Goldfish models, whose [CLS]/[SEP] are added tokens)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model)
    except TypeError:
        return PreTrainedTokenizerFast.from_pretrained(model)


def load_pretrained(model: str, dtype: str, device: str):
    """The model name for HFLM, or a loaded model when its config has no AutoModelForCausalLM class (multimodal
    checkpoints such as Apertus 1.5), which is loaded as a multimodal LM instead."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if type(AutoConfig.from_pretrained(model)) in AutoModelForCausalLM._model_mapping:
        return model
    import torch
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map={"": device}).eval()


def score_sentences(lm, sentences: list[str]) -> dict[str, float]:
    """Log-probability of each unique sentence given only BOS, via the batched loglikelihood of an
    lm-eval HFLM (the batch size is picked by its autobatching)."""
    unique = list(dict.fromkeys(sentences))
    too_long = sum(len(lm.tok_encode(sen)) >= lm.max_length for sen in unique) if hasattr(lm, "tok_encode") else 0
    if too_long:
        print(f"WARNING: {too_long} sentences reach the model's context length ({lm.max_length} tokens) and are truncated")
    requests = [Instance(request_type="loglikelihood", doc={}, arguments=("", sen), idx=0) for sen in unique]
    return dict(zip(unique, (logprob for logprob, _ in lm.loglikelihood(requests, disable_tqdm=True))))


def score_pairs(lm, pairs: pd.DataFrame) -> pd.DataFrame:
    """pair_id, lang, phenomenon, text_hash (of the two sentences scored), both log-probabilities and their delta (sen - wrong_sen)."""
    logprobs = score_sentences(lm, pairs["sen"].tolist() + pairs["wrong_sen"].tolist())
    scores = pairs[["pair_id", "lang", "phenomenon"]].copy()
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


def add_macro(acc: pd.DataFrame) -> pd.DataFrame:
    """The lang x phenomenon table plus a (lang, ALL) row per language (unweighted mean over its phenomena) and one
    (ALL, ALL) row (unweighted mean over the languages' rows); n is the pair count summed over the rows' cells."""
    per_lang = (acc.groupby("lang").agg(accuracy=("accuracy", "mean"), n=("n", "sum")).reset_index()
                .assign(phenomenon="ALL"))
    overall = pd.DataFrame({"lang": ["ALL"], "phenomenon": ["ALL"], "accuracy": [per_lang["accuracy"].mean()],
                            "n": [per_lang["n"].sum()]})
    return pd.concat([acc, per_lang[acc.columns], overall[acc.columns]], ignore_index=True)


STORED_COLUMNS = ["pair_id", "lang", "phenomenon", "text_hash", "sen_logprob", "wrong_logprob", "delta"]


def _read_scores(path, empty: pd.DataFrame) -> pd.DataFrame:
    """The stored scores at `path`, or `empty` when there are none or the file lacks a current column (an older
    format): those pairs are then scored again and the file is replaced."""
    if not os.path.exists(path):
        return empty
    stored = pd.read_parquet(path)
    if not set(STORED_COLUMNS) <= set(stored.columns):
        print(f"NOTE: {path} is in an older format (missing {sorted(set(STORED_COLUMNS) - set(stored.columns))}); "
              f"its {len(stored)} scores are discarded and the pairs scored again")
        return empty
    return stored


def evaluate(model: str, pairs: pd.DataFrame, results_dir, dtype="float32", device="cuda", batch_size="auto:8",
             max_batch_size=1024, lm=None, rescore=False, softmax_dtype="float32", benchmark_ids=None):
    """Score the pairs not yet in <results_dir>/<model>__<dtype>[__softmax_<dtype>]/scores.parquet with an lm-eval HFLM
    and write the scores and accuracy.csv there. accuracy.csv covers every stored score of a benchmark pair, not only
    the pairs given here (lang x phenomenon cells, a (lang, ALL) macro row per language, an (ALL, ALL) macro row); it
    is overwritten on every run. A stored score counts only for the same pair_id and the same
    two sentences (text_hash); the scores are saved after each language. lm: an already loaded model (tests).
    rescore: score every given pair again. The new scores go to scores.rescore.parquet (saved after each language), so
    an interrupted rescore leaves scores.parquet untouched and the next --rescore picks up where it stopped; when it
    completes they replace the old scores and the staging file is removed.
    benchmark_ids: the pair_ids of the whole current benchmark; stored scores of other pairs are dropped from
    scores.parquet (None keeps them).
    Returns (stored scores of the benchmark pairs, accuracy table, output dir)."""
    suffix = "" if softmax_dtype == "float32" else f"__softmax_{softmax_dtype}"
    out_dir = os.path.join(results_dir, f"{model.replace('/', '__')}__{dtype}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.parquet")
    staging_path = os.path.join(out_dir, "scores.rescore.parquet")
    empty = pd.DataFrame(columns=STORED_COLUMNS)
    old = _read_scores(scores_path, empty)
    staged = _read_scores(staging_path, empty) if rescore else empty
    if benchmark_ids is not None:
        keep = old["pair_id"].isin(set(benchmark_ids))
        if not keep.all():
            print(f"dropping {int((~keep).sum())} stored scores of pairs no longer in the benchmark")
            old = old[keep].reset_index(drop=True)
            old.to_parquet(scores_path, index=False)
        staged = staged[staged["pair_id"].isin(set(benchmark_ids))].reset_index(drop=True)
    if not rescore and os.path.exists(staging_path):
        print(f"NOTE: an unfinished rescore is stored in {staging_path}; run with --rescore to finish it")
    current = pairs.assign(text_hash=[text_hash(s, w) for s, w in zip(pairs["sen"], pairs["wrong_sen"])])
    done = set(zip(staged["pair_id"], staged["text_hash"])) if rescore else set(zip(old["pair_id"], old["text_hash"]))
    todo = current[[key not in done for key in zip(current["pair_id"], current["text_hash"])]]
    print(f"{len(todo)} of {len(pairs)} pairs not scored yet")

    if len(todo):
        if lm is None:
            from lm_eval.models.huggingface import HFLM

            lm = HFLM(pretrained=load_pretrained(model, dtype, device), tokenizer=load_tokenizer(model), batch_size=batch_size,
                      max_batch_size=max_batch_size, dtype=dtype, device=device, softmax_dtype=softmax_dtype)
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

    scored = old if benchmark_ids is None else old[old["pair_id"].isin(set(benchmark_ids))]
    ties = float((scored["delta"] == 0).mean())
    if ties > 0.005:
        print(f"WARNING: {ties:.1%} of the pairs have exactly equal scores; the dtype ({dtype}) is probably too coarse")
    acc = add_macro(accuracy(scored))
    acc.to_csv(os.path.join(out_dir, "accuracy.csv"), index=False)
    return scored, acc, out_dir
