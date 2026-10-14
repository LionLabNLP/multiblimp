# correct_swaps.parquet -> final-dataset schema: gap report

Compared against: `output/minimal_pairs/{svNa,svPa,spGa}/*/French/correct_swaps.parquet`
(opGa files checked for Albanian only; that one was empty, so opGa is unverified).
Target schema = the MultiBLiMP-style row (`sen, verb, verb_idx, cop, ..., agreement_certainty`).
Old builders: `src/multiblimp/filters/nsubj.py` (item fields), `filters/utils.py`
(`wrong_sen`, `prefix`), `downsample.py` (attractor/weight columns).

Source facts that matter everywhere:
- `sen` is a **token array**, target wants a **string**.
- `*_idx` columns are **1-based**; index `sen` with `idx - 1` (verified: `sen[head_idx-1] == head_form_orig` and `sen[nsubj_idx-1] == nsubj_form_orig` on 100% of rows).
- `no_space_after` is a per-token bool array, same length as `sen` (verified 100%), so faithful detokenization is possible.

## 1. Directly available (rename / copy)

| target | source in correct_swaps |
|---|---|
| `child` | `child` (also `nsubj_form_orig`) |
| `child_idx` | `nsubj_idx` (role prefix differs per condition) |
| `head` | `head_form_orig` (original casing; `head_form` is lowercased) |
| `head_idx` | `head_idx` |
| `swap_head` | `swap_head` (already recased by `match_casing`) |
| `head_upos` | `head_pos` (this is UPOS; `head_xpos` is separate) |
| `child_upos` | `nsubj_pos` |
| `grammatical_feature`, `ungrammatical_feature`, `feature_vals` | split `feature_vals` on ` -> ` (e.g. `SG -> PL`; values may be `3 -> 1\|2`) |

## 2. Derivable by simple conversion

| target | how |
|---|---|
| `sen` (string) | detokenize `sen` using `no_space_after` (same logic as `tokenlist2sen` in `filters/utils.py`) |
| `wrong_sen` | copy tokens, set `[head_idx-1] = swap_head`, detokenize the same way. The role is `head` for sv*/spGa; the HTML (`_v2_pair_from_row`) detects it per row from whichever `swap_{role}` is non-null |
| `prefix` | detokenized `sen[:head_idx-1]` (old code: `range(0, swap_idx)` with 0-based idx) |
| `distance` | `head_idx - child_idx` (old code: `main_verb_idx - child_idx`; example row: ik=10, zal=9 -> -1) |
| `wo` | `SV` if `distance > 0` else `VS`. Also in `deprel_order` (`sV`/`Vs`), but `distance` is what `downsample.py` uses |
| `lang` | not a column. From the output path (`.../<Language>/`) via `lang2langcode` in `src/multiblimp/languages.py` |
| `phenomenon` | not a column. From the condition dir (svNa/svPa/spGa/opGa) plus `swap_ufeat` (Number/Person/Gender). The `SV-P` style label needs a mapping from `condition_taxonomy.py` / `sva_trees/conditions.py` |
| `inflect_item` | `aux` vs `verb`. Old code split on `cop` being null (`downsample.py`), see section 3 for the catch |
| `head_features` | rebuild a dict from the flattened `head_{Feat}` columns, dropping `NaN`/`_missing`. Check `after_head_*` is not mixed in |
| `child_features` | same from `nsubj_{Feat}` columns |
| `num_attractors`, `has_attractors`, `*_congruent_*` flags | only derivable once `attractors` exists, see section 3 |
| `sample_weight` | recomputed by `balance_df` in `downsample.py` over `BALANCE_FEATURES`. It's a function of the final selected set, not a per-row fact |

## 3. Missing, or not derivable by simple conversion

| target | problem |
|---|---|
| `verb`, `verb_idx` | **Name collision, not a match.** `verb_idx` in correct_swaps comes from `extract_sen_features` (`word_order/process_treebank.py`): the **main-clause** verb that has an `nsubj` child under root. It is sentence-level and unrelated to the pair (31% of French svPa rows have `verb_idx != head_idx`, and the example shows `découlera` vs `donne`). The target `verb` is the lexical verb heading the clause when `head` is a copula/aux. Needs the tree. Same issue for `subject_idx`/`object_idx` (sentence-level, not used by the target) |
| `cop`, `cop_idx` | Not stored. `head_pos` is only `VERB` in the French files checked, so the aux/cop-as-head case was never observed here. The old pipeline looked up a child with deprel `cop`/`aux*` of the verb (`fetch_copula`); that needs the dependency tree. The `head_child-feat_aux_*` / `head_child-pos_AUX` columns say an aux child *exists* but not which token. **Cannot be rebuilt reliably from the parquet alone.** Should be re-checked on a language with aux/cop-headed agreement |
| `attractors` (and `congruent_attractors`, `incongruent_attractors`, `num_*`, `has_attractors`, `only_has_*`) | Old definition: tokens strictly between child and head (`fetch_attractors`) that carry the agreement feature in their `feats`. Needs the **full per-token UD feats** for the span. `sen` has forms only; the parquet has no per-token feats. Attractor lists (with `feats`) must come from the treebank. `set_attractors` also needs each attractor's `feats[ufeat]` to classify congruent vs incongruent. Note the old fetch had a slice quirk (`tree[start_idx:end_idx]`), so exact reproduction needs the same indexing |
| `metadata` | Not in the parquet. Old code copied `tree.metadata` (archive, sent_id, treebank, ...) in `filters/base.py`. Only `treebank`, `sent_id`, `tree_idx` survive, so it can be partially rebuilt (or re-read from the treebank via `tree_idx`), but not the original dict |
| `agreement_certainty` (`+`, `~`, `?`) | Computed in `downsample.create_conditions` from `lang_probs` (counts of gram_val agreement per order, binomial test, `p0=0.9`, `alpha=0.1`) over the **whole language's** agreement statistics, produced by `measure_agreement.py`. Nothing per-row in correct_swaps. The SVA-tree pipeline has `leaf_top1_acc` / `leaf_decision` instead, a different notion. Need a decision: reproduce the old binomial scheme from `sva_agreement_config.json` / a measure_agreement run, or map `leaf_*` to a new label |
| `inflect_item` (reliability) | Derivable only if `cop` is: old rule is `cop` null -> `verb`, else `aux`. Without `cop`, falls back to `head_pos` (`VERB`/`AUX`) as a proxy, but that's not identical to the old rule |

## 4. In correct_swaps but not in the target (drop or keep as extras)

`leaf_*` (decision-tree diagnostics), `keep`, `no_entropy`/`yes_entropy`/`Vs_entropy`/`sV_entropy`,
`num_swaps`, `swap_order_candidates`, `swap_ufeat`, `after_head_*` (features of the reinflected form),
`core_args`, `in_question`, `head_dir`/`nsubj_dir`, all `*_under_*`, `*_child-*`, `*_sibling-*` context columns,
`treebank`, `sent_id`, `tree_idx`, `no_space_after`, `subject_idx`, `object_idx`, `head_form`/`nsubj_form` (lowercased).
(~1400 columns total; the wide `head_*`/`nsubj_*` ones are the pair's feature bundle.)

## 5. Summary

- **Easy (rename/convert):** `sen`, `wrong_sen`, `prefix`, `child`, `child_idx`, `head`, `head_idx`, `swap_head`, `head/child_upos`, `distance`, `wo`, `grammatical/ungrammatical_feature`, `head_features`, `child_features`, `lang`, `phenomenon` (needs small lookup tables).
- **Need the treebank / tree (re-read via `treebank`+`tree_idx`+`sent_id`):** `verb`, `verb_idx`, `cop`, `cop_idx`, `attractors` and every attractor-derived column, `metadata`.
- **Need a design decision:** `agreement_certainty`, `sample_weight` (depends on downsampling), and `inflect_item` if `cop` is unavailable.
- **Do not map by name:** `verb_idx` (sentence-level main verb vs clause verb).

Caveats: only French was inspected for svNa/svPa/spGa, and opGa was not (Albanian file was empty). Aux/cop-headed pairs and ergative/bracketed-feature languages (e.g. Basque, Georgian) may add more role-dependent differences.
