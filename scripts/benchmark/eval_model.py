"""Score the benchmark pairs with a HuggingFace LM (lm-eval HFLM, autobatching).

Stage 4 of the benchmark: run after build_benchmark.py. Writes <results_dir>/<model>__<dtype>/scores.parquet
(pair_id, text_hash, sen_logprob, wrong_logprob, delta; pairs already in it are not rescored, --rescore scores them
again and can be resumed after an interruption) and accuracy.csv (share of delta > 0 per lang x phenomenon).

Usage (any directory):
  python3 eval_model.py --model Qwen/Qwen3-0.6B --device mps --langs French
"""
import argparse
import logging
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.benchmark import cap_per_cond, load_benchmark
from agreement_trees.eval.score import evaluate
from multiblimp.config import OUTPUT_BENCHMARK_DIR, OUTPUT_MODEL_RESULTS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench_dir", default=OUTPUT_BENCHMARK_DIR, help="build_benchmark.py output")
    parser.add_argument("--results_dir", default=OUTPUT_MODEL_RESULTS_DIR)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--phenomena", nargs="+", default=None, help="e.g. svNa spGa")
    parser.add_argument("--max_per_cond", type=int, default=0, help="extra cap per lang x phenomenon for quick tests; 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rescore", action="store_true", help="score the pairs again even if they have scores; resumable")
    parser.add_argument("--batch_size", default="auto:8")
    parser.add_argument("--max_batch_size", type=int, default=1024)
    parser.add_argument("--dtype", default="float32", help="bfloat16 rounds the summed log-probs into ties")
    parser.add_argument("--softmax_dtype", default="float32", help="dtype of the log-softmax over the vocabulary")
    parser.add_argument("--device", default="cuda", help="cuda, mps or cpu")
    parser.add_argument("--hf_token", default="hf_token.txt", help="File containing an HF token")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if os.path.exists(args.hf_token):
        with open(args.hf_token) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    pairs = load_benchmark(args.bench_dir, args.langs, args.phenomena)
    if args.max_per_cond:
        pairs = cap_per_cond(pairs, args.max_per_cond, args.seed)

    scored, acc, out_dir = evaluate(args.model, pairs, args.results_dir, args.dtype, args.device,
                                    args.batch_size, args.max_batch_size, rescore=args.rescore, softmax_dtype=args.softmax_dtype)
    print(acc.to_string(index=False))
    print(f"Overall accuracy: {(scored['delta'] > 0).mean():.3f}  ({out_dir})")
