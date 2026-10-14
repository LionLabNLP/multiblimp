# Compute budget: data pipeline + LLM scoring

Estimate for two separate jobs: (1) generating the SVA/NPA minimal-pair datasets from UD
treebanks (CPU-only), and (2) scoring a roster of LLMs on those datasets with forced-choice
log-likelihood (GPU). Target hardware for the GPU side: nodes of **8×A100 40GB**.

Numbers below are grounded in real measurements taken from this repo (dated 2026-09-28),
not pure formula guesses — see "How to estimate this yourself" at the bottom for the method,
so you can redo/refresh any of this as the model roster or dataset settings change.

## Scale, as measured

- UD source: 361 treebank folders, 3.7GB raw, 2,251,910 sentences total.
- Kept languages after exclusion filters (`get_ud_langs`): **190**.
- 54 of the 361 treebank folders exceed 10,000 sentences; capping (`--max_treebank_len 10000`,
  mandatory for NPA — see the `npa-always-capped-10k` note in project memory) reduces
  effective total corpus volume from 2.25M to **1.02M sentences** (~2.2x).
- Minimal pairs generated: SVA full run ≈ 121,305 pairs (101 languages, 6 conditions,
  `notebooks/min_pair_table.csv`). NPA in-progress run ≈ 382,868 pairs from 61/190 languages;
  projecting to all languages ≈ 700K–900K. The 2026-09-23 overview (`html-finalentropysave`)
  reports **3.29M pairs** across 824 non-empty (language, condition) cells (464 SVA-style, 360 NPA
  subgroups).
- **Scoring set (assumed): up to 500 pairs per (language, condition) cell** → `Σ min(n, 500)` over
  those 824 cells = **≈256K pairs**, used for all GPU numbers below. (Data generation on the CPU
  side is unaffected — it still processes the full corpus.)
- Mean sentence length (measured from actual generated pairs): 32 word-tokens (median 27,
  p90 59) → ≈40–45 subword tokens after BPE inflation.

## 1. CPU — data pipeline (SVA + NPA, all conditions, all languages)

### Method

- The pipeline is **embarrassingly parallel by language** (`Pipeline.__call__` loops
  `for lang in lang_candidates`, no shared state) — `wall-clock ≈ core-hours / concurrent cores`
  holds cleanly up to ~190-way parallelism.
- SVA has 27 conditions (9 relation "families" × 3 feature suffixes Na/Ga/Pa — see
  [`src/agreement_trees/vp/conditions.py`](../src/agreement_trees/vp/conditions.py)). Conditions sharing a
  family share the deprel-extraction cache (`word_order_subdir`), so siblings are cheaper
  than a cold run.
- NPA has a different structure: one expensive shared extraction (`np_types.py`) feeds
  **all 99 target_cols** defined in [`resources/npa_config.json`](../resources/npa_config.json)
  (not 5 — that was a misreading of what's been generated *so far* in `minimal_pairs/npa/`
  as if it were the full intended scope). The per-condition steps on top
  (`agreement_candidates.py`, `fit_candidate.py`) read the already-extracted parquet and are
  cheap per call, but there are 99 of them, not 5.
- **Always benchmark capped, not uncapped.** An uncapped run of the single largest treebank
  (German, HDT+GSD pooled, 205,517 raw sentences) took 607.4s for one condition; the same
  condition capped at 10,000 sentences (actual practice) took 61.2s — a 10x difference driven
  entirely by whether the cap was applied. Sizing off an uncapped run overestimates the whole
  budget by roughly 2x once you correct for the corpus-wide capping ratio.
- **Verify a remembered number against real evidence before using it.** Project memory
  claimed the `np_types.py` extraction sweep took "~3 hours, serial" — repeated here in an
  earlier draft of this doc without checking. The memory was 38 days old and self-flagged as
  needing verification. Checking the actual `treebank_features/npa/np_instances/*.parquet`
  file mtimes: 186 of 187 files were written within a single continuous 114.9-minute window
  (2026-08-20 12:45–14:40) — the real sweep took **≈1.9 hours**, not 3. (Whether that's 1.9
  core-hours depends on whether it ran with `n_jobs=1`; the memory said serial, which this
  measurement is consistent with but doesn't independently confirm.)

### Real timings (single core, isolated via `MULTIBLIMP_RUN_DIR` for the SVA rows;
the NPA rows read the existing production `np_instances` cache directly)

| Language(s) | Treebank size | Corpus | Cap | Wall-clock |
|---|---|---|---|---|
| Luxembourgish (smallest kept) | 56KB | 20 sentences (nsubj) | none | 4.8s |
| German, `svNa` (largest, HDT+GSD pooled) | 293MB | 182,276 sentences (nsubj) | none | 607.4s |
| German, `svNa` | 293MB | capped | 10,000 | 61.2s |
| German, `svGa` (sibling of svNa, cache reused) | 293MB | capped | 10,000 | 7.2s — but German verbs don't inflect for gender, so most of this is a trivial-candidate-count effect, not purely caching |
| German, `svPa` (sibling of svNa, cache reused, real agreement phenomenon) | 293MB | capped | 10,000 | 29.2s |
| German, NPA `HEAD-ADP_Case` (cached `np_instances`, no re-extraction) | — | 11,920 instances | — | 80.2s |
| **All 190 languages**, NPA `HEAD-ADP_Case` (cached `np_instances`) | — | — | 10,000 (extraction-time) | 795.9s (715.7s for 189 languages + German's 80.2s measured separately) |

### Spot-checks against the estimates below

- **SVA sibling-caching claim, directly verified.** Family total for German capped
  (`svNa` 61.2s + `svGa` 7.2s + `svPa` 29.2s) = **97.7s**, matching the doc's earlier modeled
  estimate of ~99s almost exactly. The family-cache-sharing assumption behind the SVA
  core-hour estimate holds up.
- **NPA per-condition-scaled-to-all-languages claim, directly verified**, not just
  extrapolated from German's volume share as before: running `HEAD-ADP_Case` across all 190
  languages (not just German) took 795.9s total — within 11% of the 889s the German-volume-share
  extrapolation predicted. Scaled to all 99 target_cols: 795.9s × 99 ≈ **21.9 core-hours**,
  landing right at the low end of the range below (tightened accordingly).
- **Caveat that's still open:** `HEAD-ADP_Case` happened to yield 0 correct_swaps for German
  (a low/no-yield condition) — a condition with real candidate yield generates more pairs,
  more diagnostics rows, and more HTML rendering, so this single spot-check may understate
  higher-yield conditions. Treat 21.9 core-hours as more of a floor than a settled number;
  I did not test a high-yield NPA condition across all languages.

### Estimate

| Component | Core-hours | Basis |
|---|---|---|
| SVA (27 conditions, 9 shared-cache families, capped) | ~15–21 | Uncapped extrapolation (32–47 core-h) corrected by the measured ~2.2x corpus-wide capping ratio; family-sharing assumption spot-checked directly (see above) |
| NPA shared extraction (`np_types.py`, all 99 target_cols' raw material, cap=10,000) | ~1.9–4 | File-mtime-verified wall-clock (1.9h); `n_jobs` default is 1 (matches memory's "serial" claim, though no log independently confirms this specific historical run used the default) |
| NPA per-condition stage (99 target_cols × `agreement_candidates.py`+`fit_candidate.py`) | ~22 (possible floor — see caveat above) | Directly measured: one condition run across all 190 languages = 795.9s; × 99 conditions |
| **Total** | **~39–47 core-hours** | |

### Resource request line

| Core-h | Typical cores/job | Typical mem/core |
|---|---|---|
| ~39–52 (plan ~45) | 1 (fan out one process per language, 32–64 concurrent on a node) | 2–4GB for most languages; reserve 8–16GB headroom for the largest treebanks (German, Czech PDT/CAC, Russian) — the pipeline's own auto-sizing capped a German worker at 14.4GB on an 18GB machine |

**Reconciling with an observed local run of ~1.5h:** that observation predates running NPA's
full 99-condition scope — only 5 of 99 target_cols have actually been generated so far
(`minimal_pairs/npa/` currently has 5 subdirectories). SVA (~15–21 core-h) + NPA extraction
(~1.9–4 core-h) + just the 5 NPA conditions actually run (~22–27 × 5/99 ≈ 1–1.5 core-h) ≈
18–26 core-hours — that range reconciles with ~1.5h wall-clock at a plausible 13–17-way local
parallelism. The ~39–52 core-hour total above is for the **full 99-condition NPA scope**,
which hasn't been run yet and so has no observed wall-clock to compare against — treat it as
the number that matters for sizing the eventual full run, not as something the ~1.5h
observation should already match.

At ~45 core-hours and 13–17-way real-world parallelism, full data generation finishes in a
few hours on a shared HPC node's CPU
allocation — small next to the GPU budget below.

## 2. GPU — LLM scoring (dense models only; MoE roster TBD, see caveat)

### Method

- Scoring is **forced-choice**: one teacher-forced forward pass per sentence (original +
  swapped), no generation — see [`src/lm_eval/eval_word_order.py`](../src/lm_eval/eval_word_order.py)
  (`minicons.scorer.IncrementalLMScorer`). Cost is a single forward pass, not a multi-token
  generation loop.
- Two independent axes, from two different numbers per model:
  - **Compute (GPU-hours)** scales with **active** parameters (dense: active = total; MoE:
    only routed experts fire per token).
  - **Memory (GPU count / VRAM)** scales with **total** parameters — a MoE model needs every
    expert resident even though few fire per token.
- FLOPs-per-forward-pass ≈ `2 × active_params × seq_len`. Total FLOPs = `2 (orig+swap) ×
  N_pairs × 2 × active_params × seq_len`. GPU-hours = FLOPs / (achieved FLOPs/s × 3600).
  "Achieved" is realistically 5–8% of peak for naive small-batch HF inference, ~20–25% for
  optimized batched serving (vLLM) — **this multiplier is the single biggest source of error
  in a FLOPs-based estimate and should be calibrated empirically**, not assumed.
- On A100 40GB (bf16 peak ≈312 TFLOPS), this works out to **≈0.7–2.6 GPU-hours per billion
  active parameters** for a ~1.05M-pair, ~45-token dataset; scaled to the 256K-pair scoring set that
  is **≈0.17–0.63 GPU-hours per billion active parameters** (low = vLLM,
  high = naive HF/minicons — the harness in this repo currently uses the naive path).

### Model roster (dense only — MoE excluded for now)

56 of the 66 candidate models are dense (the other 10 — Llama 4 Scout/Maverick, Gemma 4
26B-A4B, Qwen 3.6 35B-A3B, Mistral Large 3, Mistral Small 4 — are MoE and excluded from this
pass; revisit once their active-param counts are confirmed from model cards). Total dense
active-parameters summed across all 56: **≈874B**.

VRAM tiering at 40GB/card (bf16 weights ≈2.2GB/B params incl. headroom, power-of-2 tensor-parallel
sharding):

| Tier (total params) | Models | Active-B summed | GPUs/job | GB/job | GPU-hours (0.7–2.6/active-B, +comm overhead) | Programming scheme |
|---|---|---|---|---|---|---|
| ≤14B | 40 | 231 | 1 | 1–31 | 39–146 | Single-GPU, plain PyTorch/HF `.to("cuda")` |
| 22–32B | 10 | 283 | 2 | 48–70 | 56–207 (+15%) | Tensor-parallel ×2, vLLM `tensor_parallel_size=2` |
| 40B | 2 | 80 | 4 | ~88 | 17–63 (+25%) | Tensor-parallel ×4, vLLM |
| 70B | 4 | 280 | 8 (full node) | ~154 | 65–239 (+35%) | Tensor-parallel ×8, vLLM, full node per job |
| **Total** | **56** | **874** | — | — | **~177–656, plan ~365 (~475 with 30% contingency)** | |

### Practical notes

- 40 of 56 models fit on a single GPU — run up to 8 of them in parallel per node (one per
  GPU) rather than serially; this is the difference between ~365 GPU-hours taking about two days
  vs. weeks of wall-clock.
- Switching the scoring harness from `minicons`/HF to **vLLM** (`prompt_logprobs` gives
  exactly the forced-choice scores needed) is worth the engineering time given 66 models —
  it's both the low-end of the GPU-hour range above and the only practical way to run the
  22B+ tiers at all.
- The 70B tier (Apertus, Llama 3) is the only dense tier that claims a full node per job.

### Open items before finalizing

1. Look up active-parameter counts for Mistral Large 3 (675B MoE) and Mistral Small 4 (119B
   MoE) from their model cards, then fold them back in (their total-param VRAM footprint —
   ~1.35TB and ~238GB respectively in bf16 — will need multi-node tensor-parallel or
   quantization, or scoring via a hosted API that exposes raw token logprobs).
2. Decide whether Llama 4 Scout/Maverick are in scope at all given their memory footprint
   relative to likely allocation size.
3. Calibrate the 0.7–2.6 GPU-h/active-B range with one real run per tier (see below) before
   submitting a final ask.
4. Add ~30% contingency for reruns/debugging — with 12 different model families, harness
   bugs and per-language tokenizer edge cases are where wall-clock is actually lost.

## How to estimate these yourself

The recipe, so this doc can be refreshed as the dataset or model roster changes:

### CPU (or any embarrassingly-parallel data job)

1. **Confirm the workload is actually independent per unit** (here: per language — no
   shared state across `Pipeline.__call__`'s loop). If so, `core-hours` is a single number
   that doesn't care about your hardware; `wall-clock = core-hours / concurrent-workers` is
   the only thing hardware changes.
2. **Time the smallest and largest unit**, not an "average" one. The smallest tells you your
   fixed per-invocation overhead (interpreter start, lexicon loads — here ~5s, or ~33s once
   real work like tree-fitting happens even on small data). The largest tells you your worst
   case and, combined with the small one, lets you fit `cost ≈ fixed + rate × size`.
3. **Match your benchmark to how the job actually runs in production**, not its theoretical
   maximum. This is where the first pass on this estimate went wrong: benchmarking uncapped
   when actual practice always caps produced a ~2x-too-high total. If there's a standing
   operational constraint (a length cap, a memory limit, a `--force`/skip flag), the
   benchmark needs it too.
4. **Prefer a real recorded number over a derived one whenever you have one — but verify it,
   don't just repeat it.** A previously-logged run (`np_types.py`, cap=10,000, ~186 languages)
   was recorded in project memory as "~3 hours, serial." That memory was 38 days old and
   self-flagged as needing verification before being asserted as fact; an earlier draft of
   this doc repeated it anyway. Checking the actual artifact the run produced (the built
   parquet files' mtimes, clustered into a single 114.9-minute continuous window) showed the
   real number was ~1.9 hours. **A logged claim about "how long X took" is itself an
   estimate someone else made — check it against something the system produced (file
   timestamps, log files, a cache's own build date) before it becomes an input to your own
   estimate.** This applies doubly to anything phrased in a memory/note as approximate or
   unverified.
5. **Get the true unit count from the code path that actually runs, not from what's already
   been generated.** This estimate first said NPA has 5 conditions — true only of what
   happened to already exist in `minimal_pairs/npa/` from a partial run, not of
   `resources/npa_config.json`'s 99 target_cols that `fit_all.py` actually iterates over
   by default. Directory listings tell you what's been *done*; the driver script and its
   config tell you what's *in scope*. Check both, and don't conflate one for the other.
6. **Sum size across all units** (here: total sentences, post-cap, or total condition
   invocations) and multiply by your fitted rate; add per-unit fixed overhead × number of
   units. Watch for the fixed-overhead term becoming dominant once you've capped/optimized
   the variable part — at ~190 languages × (27 SVA + 99 NPA) conditions, that's tens of
   thousands of invocations before accounting for any cache-sharing, so even a small
   per-invocation constant adds up.

### GPU (or any model-inference job)

1. **Split compute from memory.** Compute (GPU-hours) is driven by what actually executes
   per token — active parameters for MoE, all of them for dense. Memory (GPU count) is
   driven by what has to be resident — total parameters, always. Conflating these is the
   single most common estimation error for a mixed dense/MoE roster.
2. **Use the standard forward-pass FLOPs approximation** (`2 × active_params × tokens`) to
   get a compute *floor*, but never present that floor as the answer — multiply by a
   realistic achieved-utilization fraction (5–8% naive small-batch, 20–30% optimized batched
   serving) and treat that multiplier as the number you're most uncertain about.
3. **Calibrate the multiplier empirically before finalizing**, the same way as the CPU side:
   run ~1,000–5,000 real pairs through one small model, one mid-size model, and (if feasible)
   one large model, on your actual target GPU and serving stack, and read off real
   sentences/sec. Interpolate the rest of the roster from active-param count within each
   architecture family rather than trusting the FLOPs floor directly.
4. **Tier the roster by total-param VRAM footprint** to get GPUs-per-job (VRAM ÷ usable
   per-GPU memory, rounded to a power of 2 for clean tensor-parallel sharding) — this is a
   separate table from the compute one, and it's usually the real constraint for an
   allocation request, not total GPU-hours (a handful of giant MoE models can dominate the
   memory ask while contributing little to the compute total, since their active-param count
   is small relative to their total size).
5. **Decide self-host vs. API per model**, not for the whole roster. A few outlier models
   (>~150B total params here) are often cheaper to score via a hosted endpoint (if it exposes
   raw token logprobs — many chat-completion APIs don't) than to fight for a multi-node
   allocation just for them.
6. **Present the ask in tiers, not one number.** A reviewer of a compute allocation trusts
   "63 models need X GPU-hours on single GPUs, 3 need Y GPU-hours on 4-GPU jobs, 3 giants
   need Z or an API budget" far more than one aggregate figure, and it's what you'll actually
   need operationally to schedule the jobs anyway.
