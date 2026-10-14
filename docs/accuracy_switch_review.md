# Entropy → smoothed-accuracy switch: review

APPLIED to the working tree on 2026-10-02 (uncommitted; diff saved as [accuracy_switch_applied.patch](accuracy_switch_applied.patch)). Original draft patch: accuracy_switch_draft.patch (removed, superseded) (16 files, `git apply --check` clean against the working tree; nothing applied).
Rendered drafts (svNa, from cached models; open via the `accuracy-draft-preview` launch config, port 8845):
[index.html](accuracy_draft/index.html) (deprel overview), [Nepali.html](accuracy_draft/Nepali.html) and [Basque.html](accuracy_draft/Basque.html) (tree pages).

## Decisions taken
- Leaf measure = smoothed accuracy `(n_right + 0.5) / (n + 1)`; this is what the keep rule compares, what every node/leaf label shows, and what the overview aggregates.
- Strict keep floor = flat **0.95**, `acc > 0.95` (`DEFAULT_LEAF_MIN_ACCURACY`). Replaces `default_leaf_threshold(min_samples_leaf)`.
- Second-chance floor = `min(smoothed_accuracy(msl - wrong, msl), strict)` (same depth-decayed disagreement budget).
- Overview (agreement pages): smoothing is applied **once** at the whole-dataset level, not per leaf. Root Acc = (root majority count + 0.5)/(N+1) (= the root node's `acc=`); Train Acc = (Σ leaf majority counts + 0.5)/(N+1); Δ Acc = Tree − Base in percentage points (never negative), coloured by share of the base's remaining error removed. New **Test Acc** column = raw held-out accuracy from `test_eval_`, faded when the test split has < 10 rows. Six-class/binary toggle and the DT Acc% column removed. Scatter is Base vs Tree accuracy, axes 0.5–1, y=x reference line.
- Tree page: node label `acc=0.xxx`; non-kept leaves show `not kept (acc 0.767 ≤ 0.95)`; info panel shows base/tree accuracy (train/test accuracy stay in the header chip and test row).
- Word-order pages keep entropy (`calculate_metrics` unchanged).
- sklearn `criterion="entropy"` untouched.

## What changes in outputs (measured on cached svNa/sp*/sv* data)
- Keep set: 0.95 is slightly looser than the old 0.9545-equivalent. **+2,346 kept rows of 1,446,148 (0.16%), none lost.** This includes the pure-10-row-leaf quirk (269 rows). A leaf of 29 rows with 1 wrong lands exactly on 0.95 and is *not* kept (`>`).
- Not derived from `min_samples_leaf` any more: a `--min_samples_leaf` below 10 can never reach 0.95 for a pure leaf.
- Per-leaf smoothing stacked one +0.5 penalty per leaf, making Δ Acc negative for 14 of 110 svNa languages (worst Nepali/Tatar −1.4 pp; entropy delta was never negative). With smoothing applied once (pooled), Δ Acc is ≥ 0 for all 109 rendered languages.
- Consequence: leaf `acc=` values on the tree page (smoothed per leaf, for the keep test) no longer average to the table's Train Acc; the glossary says so. Table accuracies are training accuracy, so Test Acc is shown next to them to expose overfitting.

## Corrections to earlier messages
My first impact numbers (41/98 languages with negative gain, Zazaki example, entropy 18/98) were computed on the wrong column (`deprel_order`, the Vs/sV word order, not the Yes/No agreement label). The figures above are on `head_nsubj_Number_agreement`. Also, "entropy masks this" was wrong: entropy delta is never negative here.

## Cascades found (all handled in the patch unless noted)
- `viz_overview._extract_plot_data` regex-scrapes `plotData = {six, binary}` from each deprel page → now scrapes the new array; pages built before the switch return `None` (shown as no-data) so entropy and accuracy are never mixed in one overview.
- `scripts/overview/build_stats.py` regex-scrapes `LANGUAGES.six` from each page and reads `acc` (raw train accuracy) for the dataset overview → regex updated; `acc` is still emitted in the page JSON (not shown in the table).
- `html_overview.py` (cross-condition scatter): toggle removed, axes/labels switched.
- `meta.json` key `leaf_threshold` → `leaf_min_acc` (old files are ignored, never misread as an accuracy floor); `diagnostics.py` reads the new key.
- Classic (non-rich) deprel page path can't render accuracy metrics; agreement now always uses the rich page (`use_rich_table = is_agreement`). In practice it already did (`debug_view=not generate_pairs`).
- `second_chance`, `create_pairs` (sva), `npa/agreement.py` (x2), `viz_deprel._agreement_row_stats`, `fit_candidate.py` CLI: all moved to `leaf_top1_acc` / accuracy floor. CLI `--leaf_threshold` units changed (≈0.95, not entropy).

## Not done / open
- **Caches** (deferred): cached parquets still hold `leaf_top1_entropy`, `{cls}_entropy`; the draft reads `leaf_top1_acc`/`{cls}_acc`. Renders above add `leaf_top1_acc` in-memory from each tree's counts. A stale parquet falls into the "no tree columns → keep all" branch, so migrate before building real pages. Old `meta.json` (entropy units, per-leaf dicts) needs regenerating for second-chance languages.
- Rendered overview uses old `meta.json`/pairs, so N PAIRS / bucket counts there reflect the old keep set, not the 0.95 floor.
- Not touched: the five notebooks, `deprel_entropy.csv`, `default_leaf_threshold`/`order_entropy` (now unused by sva/npa, kept for word-order and notebooks), `leaf_full_entropy` column, per-class `{cls}_acc` swap logic untested end to end, v1 tree-page node colours (still sklearn-impurity based; the v2 page now blends by `2·(1−acc)`, so mid-purity nodes are more saturated (less white) than in the entropy version, e.g. acc 0.9 blends 14% toward white instead of 34%).
- No automated tests exist for these paths; checks done: compile, render of 109-language overview + 2 tree pages (no console errors), keep-rule counts above.

## Update (word-order untouched, end-to-end test, second_chance)

**Word order is not touched.** `set_dt_features_in_df` keeps every entropy column (`leaf_top1_entropy`, `{cls}_entropy`, `keep`, `num_swaps`, `swap_order_candidates`, default threshold 0.1) exactly as before and only ADDS `leaf_top1_acc`; `word_order/create_pairs.py`, `create_tree_pairs.py` and `calculate_metrics` are unchanged. The shared tree page picks its measure per page (`meta["measure"]`: accuracy when `is_agreement_predictor(predictor_var)`, else entropy with the old `H=` labels and base/reduced-entropy rows); checked by forcing the entropy branch on a real tree. Side effect: cached parquets only need `leaf_top1_acc` added, no column renames.

**End-to-end test (isolated `MULTIBLIMP_RUN_DIR`, patched code, svNa, cached extraction copied in):**
- Basque: 651 raw / 651 kept / 2,880 pairs / 9,682 items seen, identical to the existing run; every parquet column identical to the old one except the new `leaf_top1_acc`; `meta.json` now carries `leaf_min_acc: 0.95`; index + tree page generated.
- `--second_chance` on Marathi and Albanian (both retried before): same strict keep (50 / 0), same rescued (1,712 / 141), same final keep (1,762 / 141) and 2nd-chance keep (1,712 / 141) as the old entropy-based runs. Per-leaf floors convert, e.g. depth 0: entropy cap 0.684 -> accuracy floor 0.818.

**second_chance stage:** `sva_trees/second_chance.py` is new and still untracked in git. Wired only into `agreement_trees.vp.pipeline.Pipeline` (SVA families via `agreement_trees.cli`, off by default; NPA does not use it). 38 of 770 existing `meta.json` are second-pass (svNa 19, spNa 7, spGa 6, svPa 5, svGa 1); in every svNa one the strict pass kept 0 or very few leaves.
Effect of the switch: trigger rules (keep share < 0.5, unk share <= 0.5, enough Yes rows, something rescuable) are unchanged. The lax floor is `min(smoothed_accuracy(10 - 10*budget, 10), 0.95)`, budget = 0.15 * 0.8^depth: 0.818 at depth 0, 0.885 at depth 3, 0.926 at 7, 0.945 at 12, reaching the 0.95 strict floor at depth 15 (trees stop at depth 12). Differences: (1) the strict floor is flat 0.95 while the lax anchor is still `min_samples_leaf`-based, so they only line up at the default 10; (2) retried `meta.json` store `leaf_min_acc` as a per-leaf dict, old ones (entropy units) must be regenerated; (3) tree-page 2️⃣ tags read `(acc>0.82)`.
