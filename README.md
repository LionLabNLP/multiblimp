# MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs
[![arXiv](https://img.shields.io/badge/arXiv-2504.02768-<COLOR>.svg)](https://arxiv.org/abs/2504.02768)

MultiBLiMP is a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages, 6 linguistic phenomena and containing more than 125,000 minimal pairs.
This repository contains the code for creating the corpus and the scripts for LLM evaluation.

## This branch: agreement decision trees

This branch builds minimal pairs from UD treebanks and explains them with decision trees: which treebank annotations
predict whether a subject, object, indirect object or noun-phrase dependent agrees with its head. Pipeline, from raw data to pages:

1. **Resources** (`setup.sh`): UD treebanks, `resources/um2ud_annotation` (UniMorph-to-UD value mapper), treebank pickles
   (`scripts/data_caching/pickle_treebanks.py`), UD-derived tables (`scripts/data_caching/ud_tables.py` -> `resources/ud_tables/`), UniMorph and its tables in
   UD notation (`scripts/data_caching/um2ud.py` -> `resources/um_tables/`). All three scripts take `--langs`.
2. **Extraction and scan** (`cache_data.sh`): `scripts/agreement_trees/cache_treebank_features.py` writes one feature cache per language and
   family under `treebank_features/` (verb families sv, sp, sa, ov, op, oa, iov, iop, ioa; noun phrases under `npa/`);
   `find_agreement_candidates.py` scans the caches and records which (target, language) pairs have enough agreement evidence
   (`resources/agreement_langs_{vp,np}.json`).
3. **Fit** (`fit_all.py`, or one target with `fit_candidate.py`, e.g. `sv:Number`, `HEAD-DET_Number`): decision tree, minimal pairs
   (`output/minimal_pairs/`, `output/decision_trees/`) and the tree pages (`html/decision_trees/`).
4. **Overview** (`scripts/overview/run_all.sh`): per-condition indexes (`generate_html_indexes.py`), `build_stats.py` and the top-level `html/index.html`.
5. **Benchmark** (`scripts/benchmark/`): `validate_surface.py` rebuilds the written form of contractions (`de le` -> `du`) in the pairs' sentences and
   drops pairs with no valid form; `build_benchmark.py` downsamples the pairs (default 1000 per language x condition) into `output/benchmark/`;
   `eval_model.py` scores them with a HuggingFace model through lm-eval and writes the accuracy per language x condition to `output/model_results/`.

Settings are in `resources/agreement_config.json` (modes, languages, workers, scan thresholds, tree and second-chance settings) and
`resources/data_prep_config.json`. `MULTIBLIMP_RUN_DIR` redirects every generated artifact (`output/`, `html/`, `treebank_features/`) into another
directory for an isolated run. The long sweeps are memory-guarded (RAM-capped workers);
a language that fails or is killed is logged in `output/logs/cache_error_log.txt`.

Layout:

| Path | Contents |
|---|---|
| `src/multiblimp/` | UD and UniMorph layer: treebank loading, `ud_lexicon`, `um2ud`, `ud_inflector`, `downsample` |
| `src/agreement_trees/` | `extract/` (treebank -> instances, cache sweep), `pairs/` (minimal pairs), `fit/` (trees, scan), `report/` (HTML pages, overview stats and indexes), `eval/` (benchmark build, surface forms, scoring), `vp/` and `npa/` (verb and noun-phrase pipelines), `aux/` |
| `scripts/` | thin commands over `src`: `data_caching/` (resource prep), `agreement_trees/` (extraction, scan, fit; `supplementary/` holds one-off reports), `overview/`, `benchmark/` |
| `docs/` | `lookup_tables_walkthrough.md` (tables and reinflection), `review_log.md` (pipeline review), `legacy_deletion_plan.md` |

## Citation
```
@misc{jumelet2025multiblimp10massivelymultilingual,
      title={MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs}, 
      author={Jaap Jumelet and Leonie Weissweiler and Arianna Bisazza},
      year={2025},
      eprint={2504.02768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02768}, 
}
```