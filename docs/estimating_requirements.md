# How to estimate project requirements (files, storage, CPU, GPU)

A recipe for projecting resource needs for a data-pipeline + model-scoring project, written
from the lessons of sva-dt (see [compute_budget.md](compute_budget.md) for a worked example).
The goal is a defensible range with stated assumptions, not a single number.

## 0. Ground rules

1. **Separate "in scope" from "done so far".** Directory listings show what has been
   generated; the driver script and config show what *will* be. (sva-dt: 5 NPA conditions
   existed on disk, 99 were in `npa_config.json`.)
2. **Verify every input number against an artifact the system produced** (file mtimes, row
   counts, logs). A remembered "it took 3 hours" was really 1.9 h by mtimes. A table named
   `min_pair_table.csv` turned out to count the upstream public dataset, not our pipeline.
3. **Benchmark the way production runs**: same caps, filters, flags, and cache state. An
   uncapped benchmark overestimated by ~2x; a debug/uncapped output overestimated pair counts.
4. **Report ranges and name the dominant unknown.** Usually one multiplier (yield per unit,
   GPU utilization) swings the total by 3-10x. Say which one, and how to measure it.
5. **Re-measure when the code changes** (new cap, new conditions, new filters). Old output
   is evidence about old code.

## 1. Define the units and the multiplication

Write the project as `units x cost-per-unit`:

| Level | sva-dt example |
|---|---|
| Unit of independent work | language (190) |
| Work items per unit | conditions (27 SVA + 99 NPA) |
| Output records per item | minimal pairs |
| Consumers of records | models (~66) x 2 sentences per pair |

Find out (a) is the work independent per unit (then wall-clock = total / workers), (b) which
level has the heavy tail (one language had 10x+ the pairs of a typical one), and (c) whether
items share expensive intermediates (sibling conditions sharing a cache are much cheaper than
cold runs).

## 2. Number of records (the root quantity)

Everything downstream scales with record count, so get this right first.

1. Count records per item from **real output**, cheaply: parquet footers
   (`pyarrow.parquet.ParquetFile(f).metadata.num_rows`), `wc -l`, or JSON counts. No rebuild needed.
2. Check how the output was produced: was the cap applied? Which version of the code? Is it
   a full or partial language/condition set? Note the file date.
3. Extrapolate to the missing items using the **distribution**, not the mean of the items you
   happen to have. Items that exist are often the high-yield ones (2 NPA columns gave 155K
   pairs each; most of the other 97 will yield far less). Give low/likely/high.
4. Deduplicate: records that share a sentence/input across conditions can be scored once.
5. Decide whether a **subsampling cap** (per unit and item) is planned. It is the strongest
   lever on every downstream cost and should be a design decision made before the ask.

Result: `N_records` with a range, and `N_unique_inputs` if dedupe is possible.

## 3. Number of files

1. Count files per (unit, item) cell from an existing output tree:
   `find <dir> -type f | wc -l` divided by the number of cells present.
2. Multiply by the projected number of non-empty cells (not units x items; sparse cells
   often write nothing or only a stub).
3. Add stage-specific trees (decision trees, caches, per-model results, HTML).
4. Look for a file-count explosion on the *results* side: one file per
   (model x unit x item) is `models x cells`, e.g. ~1M files. Plan consolidated outputs
   (one parquet per model and unit) before it happens; shared filesystems have inode quotas
   and small-file metadata cost.

`files = cells_nonempty x files_per_cell (+ caches, + per-model outputs)`

## 4. Storage

1. **Bytes per record, measured per format.** Take real files and divide size by rows
   (columnar parquet with ~1,000 sparse feature columns was ~200 B/row; TSV with per-token
   float lists was ~1-2 KB/row). The wide diagnostic/feature tables usually dominate the
   clean deliverable by 10x or more.
2. Separate **deliverable** (what downstream needs), **working set** (caches, diagnostics,
   intermediate trees), and **inputs** (raw corpora, lexicons). Size and retain them
   differently; the deliverable is often <1 GB while the working set is tens of GB.
3. **Models:** `weights_GB ~= params(B) x bytes_per_param` (bf16 = 2, int8 = 1, fp32 = 4).
   Sum across the roster for the cache size, or plan download-score-delete and size scratch
   for the single largest model plus headroom.
4. **Results:** `rows x tokens x bytes` if you keep per-token scores, `rows x bytes` if you
   keep only sentence totals. Decide which you need; it is a 10-50x difference.
5. Add 25-50% headroom for reruns, versions, backups.

## 5. CPU

1. Time the **smallest and largest** unit (single core, production flags) and fit
   `cost ~= fixed + rate x size`. Fixed overhead (interpreter, lexicon loads, model fits)
   dominates when you have tens of thousands of tiny invocations.
2. Sum over all units after applying caps; account for shared caches (time a cold item and a
   sibling item separately).
3. `core_hours = sum(unit costs)`; `wall_clock = core_hours / effective_workers`, with
   effective workers limited by memory (largest unit's peak RSS x workers <= node RAM) and
   by the heavy-tail unit, which sets the minimum wall-clock.
4. Record **peak memory** of the heaviest unit, not just time (`/usr/bin/time -l`).
5. Report: core-hours, cores per job, GB per core, wall-clock at N workers.
6. Analysis stage: usually cheap (groupby over `records x models` rows is minutes), but
   bootstrap confidence intervals and mixed-effects fits can reach tens of core-hours; size
   RAM for the full results table (`rows x columns x 8 B`).

## 6. GPU

1. **Split compute from memory.** Compute scales with *active* parameters; GPU count scales
   with *total* parameters resident in memory. (Matters for mixture-of-experts.)
2. **Tokens:** `tokens = N_inputs x sentences_per_input x avg_tokens`. Measure avg tokens
   with each model family's tokenizer on a sample; non-Latin and low-resource scripts can be
   2-4x worse than English.
3. **FLOPs floor:** `2 x active_params x tokens` for a forward-only pass (scoring, no
   generation). GPU-hours = FLOPs / (peak FLOPs/s x utilization x 3600).
4. **Utilization is the dominant uncertainty:** ~5-8% for naive one-pair-at-a-time HF loops,
   ~20-30% for batched / vLLM serving. Small models are launch-overhead-bound, so also
   budget a per-model floor (hours) regardless of size.
5. **Calibrate, don't assume:** run ~1-5K real records through one small, one mid, one large
   model on the target GPU and stack; read off records/sec; interpolate by active params
   within a family.
6. **Memory tiering:** `GPUs_per_job = ceil_pow2(params x bytes_per_param x ~1.1 / usable_GB_per_GPU)`.
   Large tensor-parallel jobs add 15-35% communication overhead. Decide per model whether
   to self-host, quantize, or use a hosted endpoint that exposes token logprobs.
7. Present as **tiers** (models per GPU-count class, GPU-hours each), then add ~30%
   contingency for reruns and harness bugs.

## 7. Package the estimate

- A table per resource: low / planned / high, plus the single assumption that moves it most.
- Which numbers are **measured** (with date and source file), **derived**, or **guessed**.
- A short list of measurements that would tighten it (e.g. one high-yield sample of the
  unmeasured conditions, one calibration run per model tier).
- A re-run recipe so it can be refreshed when the cap, roster, or code changes.

### Quick formulas

```
records        = sum over cells (measured or distribution-extrapolated)
files          = nonempty_cells x files_per_cell + cache/result files
storage        = records x bytes_per_record + models(GB) + results
core_hours     = sum(fixed + rate x size) ; wall = core_hours / min(cores, RAM/peak_RSS)
gpu_hours      = 2 x active_params x tokens / (peak x util x 3600)   [per model]
gpus_per_job   = ceil_pow2(total_params x bytes x 1.1 / GB_per_GPU)
```
