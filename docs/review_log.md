# Pipeline review log

Goal (day 1): go through the whole pipeline file by file, sanity check it, omit old code and
computation, reduce comments and docstrings. Order: resource prep, `scripts/agreement_trees`, then the
`src` packages in pipeline order. Reconstructed 2026-10-14 from the conversation; keep updated.

## Status

| Area | Status |
|---|---|
| Resource prep: `treebank.py`, `pickle_treebanks`, exclusions, `ud_lexicon`/`ud_tables`, `um2ud`, `unimorph` (legacy-marked), loggers | reviewed |
| `scripts/agreement_trees/`, `agreement_config` | reviewed (merged/restructured) |
| Annotation filling + reinflection: `process_treebank` fill path, `ud_inflector`, `ud_table` | reviewed |
| `sva_trees/`: `pipeline`, `cli`, `conditions`, `second_chance`, `diagnostics`, `create_pairs` | reviewed, comments trimmed (2026-10-14) |
| `npa/` (`agreement`, `np_types`, `np_cache`, `targets`, `cli`) | reviewed, comments trimmed (2026-10-14); `targets.py` has no importers (LEGACY); `pipeline.py` legacy |
| `word_order/` core: `prediction_target`, `process_treebank`, `decision_tree`, `entropy`, `fit_cli`, `agreement_scan`, `per_treebank`, `utils` | reviewed, comments trimmed (2026-10-14) |
| `word_order/` reporting: `viz_tree`, `viz_deprel`, `viz_overview`, `html/*` | reviewed for Python logic and comments; the HTML/JS template bodies were not read (see below) |
| `subj_aux/` | reviewed, trimmed, `_build_df` duplicate removed (2026-10-14) |
| `scripts/overview/` | reviewed, comments trimmed, two duplicates removed; `render_html.render` (a 2,600-line template function) not read |
| supplementary scripts | not reviewed |
| `multiblimp/` helpers: `agreement_pipeline_utils`, `condition_taxonomy`, `config`, `languages`, `swap_features` | partly (see trimming) |

## Comment / docstring trimming candidates

Share of non-blank lines that are comments or docstrings (live modules, 2026-10-14):
`subj_aux/redirect.py` 67%, `multiblimp/swap_features.py` 56% (reviewed, not
trimmed: long bracket-convention comments), `word_order/entropy.py` 53%, `word_order/tree_utils.py` 48%,
`word_order/viz_overview.py` 42%, `multiblimp/config.py` 41%,
`multiblimp/agreement_pipeline_utils.py` 36% (reviewed, not trimmed),
`word_order/utils.py` 34%.

## Unreferenced definitions (marked LEGACY in their docstrings; delete during cleanup)

- Orphaned helpers: `count_np_types` (`npa/np_types.py`), `default_leaf_threshold` (`word_order/entropy.py`,
  entropy-era threshold, superseded by `DEFAULT_LEAF_MIN_ACCURACY`), `remove_censored` (`word_order/viz_tree.py`).
- Legacy chain: `word_order/tree_utils.py` (only imported by the dead `validate_swap_directly.py`) and
  `split_node_dfs` (`process_treebank.py`, from the validation procedure).

## Decisions pending (sva_trees/create_pairs)

- `max_num_of_pairs` is never passed: the `extra_pairs` bucket is always empty and the `add_item` loop in
  `process_item` is dead (the diagnostics HTML legend already says "always 0").
- The VP `create_pairs` never gets a `context_inflector`, so `ambiguous_subjects` is always empty for VP
  (NPA passes one from its own pair function).
- `swap_target` is always `["head"]` for VP; the multi-role handling there is NPA-era generality.
- `ambiguous_readings`: Welsh variants judged correctly treated; the Portuguese bucket (427, 27% of would-be
  pairs, e.g. pluperfect vs preterite `trocaram` -> `trocara`/`trocou`) has real cases but was not fully verified.
- Frequency pruning of tied fill rows: parked.
- Rebuild of all tables/caches waits until the review is done (caches are version-stamped).

## Findings from the npa review

- Deduplicated 2026-10-14: `coverage_stats` and `leaf_keep` now live in `sva_trees/create_pairs.py` (used by VP
  and NPA); `FEATURE_ABBREV` is derived from `condition_taxonomy.NPA_FEATURE_NAMES`; `np_types.ROLE_PRIORITY`
  is `condition_taxonomy.NPA_ROLE_ORDER`. Counts unchanged (Portuguese svNa 1202 / 378; German HEAD-DET_Number 9886).
- `np.streaming` in `agreement_config.json` is always true; the non-streaming `build_np_data` and the
  `consolidate=False` option of `build_np_data_streaming` are unused fallbacks.
- NPA now runs every swap form through `sva_trees.create_pairs.process_item` (2026-10-14), so it has VP's
  checks and a `conflicting_features` bucket; its og_feats include bracketed features like VP. German 3000 trees:
  HEAD-DET_Case 18204 -> 18025 correct + 179 conflicting (e.g. `dem` -> `der` also changes Gender);
  HEAD-DET_Number 9886 -> 9867 (19 `seinen` -> `seinem`/`seiner` rows now ambiguous_readings because the
  possessor features widen the candidates); HEAD-ADJ_Gender unchanged. `process_item` takes `context_features`
  (the fixed role's possible swap-feature values, computed once per row) instead of a context inflector; VP
  passes the same empty set as before.
- `UNDEFINED` tests in `create_npa_pairs` and VP `process_item` are dead under `UDInflector` (marked LEGACY).
- `remove_censored` is legacy because forms are already scanned for placeholders at an earlier stage.

## word_order scope (only what VP/NP reach; static reachability, 2026-10-14)

Not reached by live code: `create_pairs`, `create_tree_pairs`, `plot_decision_tree`, `validate_swap_directly`,
`tree_utils` (legacy), `huggingface` (dataset upload, outside the pipeline). Unreached defs inside live modules:
`default_leaf_threshold`, `split_node_dfs`, `capitalize_first`, `remove_censored`, `write_html`.
To review, in this order: `prediction_target` (173), `process_treebank` (1273), `decision_tree` (709), `entropy` (208),
`fit_cli` (137), `agreement_scan` (413), `per_treebank` (442), `utils` (139); then reporting: `viz_tree` (1453),
`viz_deprel` (970), `viz_overview` (399), `html/html_tree` (2126), `html/html_deprel` (1836), `html/html_overview` (522).

## Portuguese `ambiguous_readings` (svNa, 378 rows / 142 tokens on the 3000-tree cache, current code)

- ~50% of rows: preterite vs pluperfect. The UD-derived table has `Tense=Pqp` for the pluperfect (por 90 rows;
  also glg, kat, kmr, lat, ron), the UniMorph-derived one folds PST+PRF into `Tense=Past`+`Aspect=Perf`
  (UM-code PRF), so a `Tense=Past` token matches both `trocou` and `trocara`. Rewriting the por UM rows
  PST+PRF to `Tense=Pqp` took ambiguous 378 -> 194 and correct_swaps 1202 -> 1294. Mapping gap, not sparsity;
  per-language decision (Latin's PRF is not the pluperfect).
- ~48%: the verb carries several readings (`Person=1,3`, `Mood=Imp,Ind`: `sabia`, `continua`); the subject
  (all 3rd person here) would resolve them. Code option, not sparsity.
- ~10%: spelling variants (`veem`/`vêem`) and a few wrong UD-derived rows; accepted.

## Pqp bridge (implemented 2026-10-14, takes effect on the next `um2ud.py` run)

`um2ud._pluperfect_is_prf` rewrites a language's UniMorph past + PRF rows to `Tense=Pqp` only if the UD-derived
table confirms it, by (lemma, form): at least 30 UD Pqp rows found among the UniMorph past rows, >= 80% of them
with a PRF reading, and <= 5% of UD past rows with only PRF readings. UniMorph has no pluperfect tag, so without the
rewrite a `Tense=Past` token matches both the preterite (PFV) and the pluperfect (PRF) rows. Result on the current
data: por, glg, ron and lat rewrite (por 3000-tree svNa: ambiguous 378 -> 194, correct 1202 -> 1294); georgian
(1 matched Pqp row), kurmanji (3), turkish and the languages without UD Pqp are left alone. The inflector's query guard no longer special-cases Pqp (see below).
Languages that fail the check keep Past + Perf, and Pqp tokens there still drop the Tense constraint (more ambiguity,
never a wrong pair).

## Inflector query guard (2026-10-14)

`UDInflector._query` used to drop every query value without a `UD2UM` entry, on every table, and thereby also
swallowed the `_missing` placeholder that dt_df carries for unstated features. Now only a UniMorph-derived table
(`expand_values=True`) drops a value, and only if it has no UniMorph tag and the table does not hold it (so a
rewritten `Tense=Pqp` passes); a UD-derived table uses every value it has, an unattested one matches nothing.
`create_pairs` and the NPA loop strip `decision_tree.MISSING` from the features they pass in. Checked on the
3000-tree caches, old vs new: Basque svNa 949 / ovNa 305 / iovNa 54 and Portuguese svNa 1202 identical; Georgian
svNa 933 -> 934 correct (the one pair: a `Tense=Imp` token whose UD fallback query used to lose its tense).

## word_order core review (2026-10-14)

Checked: `process_treebank` extraction on Basque (400 trees, nsubj and obj targets, with annotation filling)
is identical before and after the edits; every other trimmed file is AST-identical apart from docstrings and
comments (`prediction_target` only gained a `Callable` annotation fix for `head_feats`).

Legacy / orphans marked (delete during cleanup):
- `prediction_target`: eleven generic word-order targets (`amod_target` ... `advcl_noun_verb_target`) have no users;
  the live ones are `nsubj_target`, `obj_agr_target`, `iobj_agr_target`.
- `process_treebank`: the `encode_positional_features` branch (+ `child_head_distance`), the `UNDEFINED` test in
  `candidate_fills`, `require_all_children=False` (only the legacy `npa/pipeline` passes it), `split_node_dfs`.
  Removed: a stray `from pyexpat import features`, `sys.path.append("src")`, the unused `UNDEFINED` constant.
- `decision_tree`: the word-order block of `set_dt_features_in_df` (`{class}_entropy`, `num_swaps`,
  `swap_order_candidates`, `get_all_orders`) runs for every VP fit and per-treebank fit because they pass a
  target, but is meaningless for yes/no labels (NPA passes `target=None` and works); only `viz_deprel.calculate_metrics`
  reads `num_swaps` for its n_flexible columns. Also legacy: `leaf_full_entropy`, `leaf_top1_entropy`, the
  `leaf_threshold=0.1` / `predictor_var="deprel_order"` defaults of `fit_dt`, the LightGBM branch.
- `utils`: `MAX_TREEBANK_LEN`, `ALL_CORE_ARGS`, `get_all_orders`, `capitalize_first`.

Measured: the word-order block costs about 7 microseconds per row (Georgian svNa dt, 3,126 rows: 0.227 s with a
target vs 0.207 s without), so removing it is a cleanup and a smaller saved frame (the `{class}_entropy` columns,
`num_swaps` and the list column `swap_order_candidates`, which is why `create_pairs._detect_swap_col` has to
test for scalar strings), not a speedup.

Open observations: `deprel_order` and `in_question` stay in the tree fit as predictors, which is fine.
`core_args` (the main clause's subject/verb/object order, from `extract_sen_features`) describes the sentence's
root clause, not the instance's own pair, and is a leftover of the word-order project (it was a predictor_var
there). Decision 2026-10-14: keep it (and `deprel_order`, `in_question`) as predictors for now. `agreement_scan` and `fit_cli` needed no changes.

## subj_aux, reporting layer, overview (2026-10-14)

- `subj_aux`: `AuxPipeline._build_df` was a copy of `Pipeline._build_df`; it is now a `_extraction_setup` hook
  (redirect the treebank, extract through `aux_target`, keep the aux-count column) plus an `extra_always_keep`
  argument on `create_word_order_df`. Basque saNa / oaNa and German saNa frames are identical before and after.
  `redirect_nsubj_to_aux` is unused (LEGACY).
- `viz_tree._finalize_tree_html` builds a full Plotly figure (annotations, edge traces, hit-target scatter)
  that nothing uses: the v2 page renders from `node_data`. Measured ~0.08 s of 1.1 s for a 9-node tree page; the
  Python `plotly` import is only used there. LEGACY with it: `viz_tree.write_html` and `html_tree.create_html`
  (the 1,100-line v1 page template). `get_all_orders` is still called in `tree2html` but yields nothing for an
  agreement predictor.
- `viz_deprel` / `html_deprel`: every live predictor is an agreement predictor, so the classic page
  (`_create_classic_html`, 427 lines), `calculate_metrics` (word-order entropy) and the non-agreement branches
  are LEGACY; `generate_html_deprel_index` still builds `rows_six` / `rows_binary` / plot data for them on every
  run, which the rich page ignores.
- Deduplicated: `diagnostics.BUCKET_JSON_KEYS` (public) now feeds `build_stats.PUBLISHED_BUCKET_KEYS`;
  `generate_html_indexes` uses `condition_taxonomy.NPA_FEATURE_NAMES` instead of inverting `FEATURE_ABBREV`.
- Not changed: `generate_html_indexes` has its own process pool next to `agreement_pipeline_utils.run_in_pool`.
- Not read: the HTML/JS template strings (`html_tree.create_html_v2` template, `html_deprel._create_diagnostics_html`
  1,330 lines, `html_overview.create_html`, `render_html.render`); all trimming there is Python-side only.

## Supplementary scripts (`scripts/agreement_trees/supplementary/`)

Reviewed 2026-10-14. All imports resolve; none is referenced by the pipeline (standalone reports).

- `annotation_coverage.py`, `second_chance_report.py`, `flag_treebanks_report.py`: fine, repo-root paths.
- `np_morph_pct.py`: hard-coded `../../../report/np_types` replaced by `NP_REPORT_DIR` / `NP_INSTANCES_DIR` (config); repo-root `sys.path`; comments trimmed.
- `gender_overview.py`: cwd-relative `sys.path` and default output (`npa_exploration/`, does not exist) replaced by repo-root path and `output/overview/gender_overview.csv`; docs trimmed.
- `find_no_coverage_langs.py`: stale "UniMorph-schema fallback" wording and wrong script path in the output header fixed; docs trimmed.
- Nothing run (no data written); `py_compile` and import checks only.

## swap_features.py

Reviewed 2026-10-14. Live: the nine `*_any` maps, imported only by `sva_trees/conditions.py` (`INFLECTION_MAPS`). LEGACY (UniMorph-code value maps, imported via `import *` by the old `scripts/create_minimal_pairs` / `measure_agreement` only): `swap_number_subj`, `swap_number`, `swap_person`, `swap_case_any`, `swap_case`, `swap_gender_f2m`, `swap_gender_fm2n`. Rewritten with a short module docstring (bug-history comments removed, stale `UnimorphInflector` / `ud2um` references gone); all values verified identical to before. `multiblimp/inflection_maps.py` is imported only by `multiblimp/pipeline.py` (old Pipeline, legacy) and has its own copies of the old maps.

## HTML template bodies (`tree_v2_template.html`, skim)

Reviewed 2026-10-14 (skim of the JS up to the feature-table code; later drawer/pairs code only grepped).
- Fixed: `html_tree._json` now escapes `</` and `<!--` (the JSON is embedded in a `<script>`; a form/sentence containing `</script>` would have cut the page). Round trip checked.
- Open, low risk: `T.META[...]`, `tree.title` and `displayLabel` go into `innerHTML` unescaped (names come from our own config/treebank folders).
- Dead / stale in the template (cleanup candidates): `fitScale`/`contentScale` are constants 1; multi-tree selector (`TREES` has one key, `buildTreeSelector`, `selectTree`); entropy-era rows (`baseEntropy`, `reducedEntropy`, `measure` branches, `_v2_build_meta` lines 1658-9); comments about "demo trees", "Hindi/Greek trees", 900x500 canvas; about 500 comment lines, many multi-paragraph narrative; `esc` defined four times.
- Not read: CSS (lines 14-1271) and JS after ~1810 beyond greps.

Template trim (2026-10-14, follow-up): 3073 to ~2760 lines. Removed the unused multi-tree selector (`buildTreeSelector`, `selectTree`, `treeSelectRow`, its CSS), the constant `fitScale`/`contentScale`, and ~45 narrative comment blocks. One shared `esc` (also escapes quotes) now covers META values and class labels in `innerHTML`. Entropy meta rows kept (only emitted when `measure != accuracy`, i.e. never for agreement trees). Checked on four real pages (French spGa pooled, spGa/PUD, svNa, svPa): rendered DOM identical to the old template apart from the removed hidden selector div, and a scripted run (wheel zoom, drag pan, reset, click every node, pairs/features toggles, sample rows) gave identical results with no JS errors.

## Package restructure (2026-10-14)

`npa`, `subj_aux`, `sva_trees` and the shared parts of `word_order` are merged into `src/agreement_trees/` (mirrors `scripts/agreement_trees/`):
`extract/` (process_treebank, prediction_target, per_treebank), `pairs/` (create_pairs, diagnostics, second_chance), `fit/` (decision_tree, entropy, scan), `report/` (viz_*, html/), `vp/` (conditions, pipeline), `npa/` (agreement, np_types, np_cache), `aux/` (redirect, pipeline), plus top-level `config` (was multiblimp.agreement_config), `taxonomy` (condition_taxonomy), `pipeline_utils` (agreement_pipeline_utils + cache_version), `utils` (word_order.utils) and `cli` (fit_cli + sva_trees.cli + npa.cli; run_vp / run_np import their pipelines lazily).
Folded into existing modules: `lemmas` -> `multiblimp/ud_table.py`, `conllu_lines` -> `multiblimp/treebank.py`.
Checked: all 43 modules import, every script resolves its imports and the argparse scripts start (`cache_treebank_features --dry_run`), and each moved file is AST-identical to the old one apart from imports, docstrings and comments (three report files differ only in module names inside comments or one tooltip string).
Left in place (dead, now with broken imports; delete in the legacy pass): `src/word_order/{create_pairs,create_tree_pairs,tree_utils,validate_swap_directly,plot_decision_tree,huggingface}.py`, `src/npa/{pipeline,targets}.py`, the old `multiblimp/{pipeline,measure_agreement,downsample,argparse,inflection_maps}.py` chain, `scripts/{create_minimal_pairs,measure_agreement,downsample,treebank2sen}`. Older entries in this log use the old module names.

Dead files from the restructure moved to `src/legacy/` (`word_order/`, `npa/`, `multiblimp/` subfolders; see its README.txt) until the deletion pass; `src/word_order` and `src/npa` no longer exist.
