# Lookup tables: how they are built and how they are used

Rewritten 2026-10-14 for the parquet / `UDInflector` pipeline. The earlier version described the
CSV + pickle chain (`ud2um.py`, `pickle_unimorph.py`, `UnimorphInflector`), which is now legacy
(see the last section).

## 1. Tables per language

| Table | Source | Built by | Stored as |
|---|---|---|---|
| UD treebank trees | UD `.conllu` | `scripts/pickle/pickle_treebanks.py` | `resources/ud/ud_pickles`, `ud_typo_pickles` (typo trees kept) |
| UD-derived table | tokens of the selected treebanks | `scripts/ud_tables.py` | `resources/ud_tables/<code>.parquet` |
| UniMorph-derived table | raw UniMorph (+ `.segmentations`) | `scripts/um2ud.py` | `resources/um_tables/<code>.parquet` |

Both tables share one layout (`multiblimp/ud_table.py`), in **UD notation**: columns `lemma`, `form`,
`upos` (UD tag), `frequency` and one string column per UD feature (`Number`, `Case`, bracketed
`Number[obj]`, ...). A comma value (`Masc,Neut`) is a syncretism, an empty cell means "row does not
state the feature". Only features the mapping module knows by name are stored.

`setup.sh` order: pickle treebanks, `ud_tables.py`, UniMorph download, `um2ud.py`. `um2ud.py` needs the
UD-derived tables (see 3).

## 2. UD-derived table (`src/multiblimp/ud_lexicon.py`, `scripts/ud_tables.py`)

`create_all_ud_tables`: per language except `skip_langs`, loads the typo pickle with the default
`use_selected_treebanks=True` (excluded treebanks are not read) and runs `collect_ud_rows` per token:

1. the token is counted for the coverage statistics (before any skip);
2. skipped if: placeholder lemma (`lemmas.clean_lemma`: `_`, `unknown`, `-`, `null` unless equal to the
   form), `_` in the lemma ("multiword lemma"), typo, no feats, or UPOS with no UniMorph tag (PUNCT, SYM, X);
3. row key = (lemma.lower(), form.lower(), upos, stored features), counted;
4. `remove_duplicate_features`: of rows sharing (lemma, features) keep forms within `dup_form_threshold`
   of the most frequent; of rows sharing (lemma, form) keep feature sets within `dup_feat_threshold`
   (or that differ a lot, are not singletons, or change the UPOS). Thresholds: `ud_tables` in
   `resources/data_prep_config.json`. During this step rows are encoded as
   `<UM upos tag>;upos=<UPOS>;Feature=Value;...` strings.

Also written: the UD side of `resources/annotation_coverage.json` and the list of UD feature values with no
UniMorph tag (`output/logs/ud_tables.log`).

## 3. UniMorph-derived table (`src/multiblimp/um2ud.py`, `scripts/um2ud.py`)

`read_unimorph` reads the raw file (and `.segmentations`): only empty cells are missing (so `nan`, `null`,
`None` are real words), optional diacritic / multiples removal, Tatar to Cyrillic, lemma and form
lowercased, then per row `UD2UM_mapper.ufeats_to_dict` (tag string to `{feature: UM code}`; handles UPOS
codes, real `ARG*/PSS*` tags via the mapper, `feature_for(tag)` for everything else). Any `+` value that is
not a known compound tag (`IN+ABL`, ...) is split into one row per component (`_expand_multiple_values`).

`convert_language` turns the codes into UD values (`UM2UD`; `SHORTENED_UM_VALS` for the one/two-letter
codes of argument-marking columns) and:
- a code with no UD value stays as `UM:<code>` (the row still states something, so it matches no UD value);
  they are listed per language in `output/logs/um2ud.log`;
- `not NOM`-style values become the comma list of the language's other values (from the UniMorph and the
  UD-derived table);
- when UD merges two UniMorph values (PFV/PRF are both `Aspect=Perf`) the original codes are kept in a
  `UM-code:<Feature>` shadow column, so a lookup does not return the other value's forms;
- verb rows without VerbForm get `Fin`; Mood and Voice stay unstated;
- UniMorph has no pluperfect tag (past + PRF); where the UD-derived table confirms that PRF is the pluperfect
  (`_pluperfect_is_prf`: por, glg, ron, lat) those rows become `Tense=Pqp` and lose Aspect/shadow;
- rows lacking upos, lemma or form are dropped; duplicates removed; `frequency` = 1.

Vocabulary and pure functions live in `resources/um2ud_annotation/` (`UM2UD_mapper`, `UD2UM_mapper`):
`UM2UD`, `UD2UM`, `SHORTENED_UM_VALS`, `LAYERED_FEAT_SEP`, `to_um_value`, `ufeats_to_dict`, `stored_upos`,
`um_tag_upos`, `expand_ud_feats`.

## 4. In memory (`UDTable`, `UDSlice`)

One `UDSlice` per UPOS, rows sorted by form; each feature cell is a uint64 bit mask over the feature's
values (0 = not stated, at most 63 values per feature). A row agrees with the token if
`cell == 0 or cell & token_bits` (this also covers comma values); a form is a contiguous slice.

Query values: a UniMorph-derived table drops a value it cannot hold (no UniMorph tag and not in the table); a UD-derived
table uses every value it has. Callers strip the `_missing` placeholder (`decision_tree.MISSING`) first.

`expand_values` marks tables whose rows use UniMorph's spelling of a UD value (`Tense=Imp` is
`Past` + `Aspect=Imp`, `UD_VALUE_EXPANSIONS`): True for the UniMorph-derived table, False for the raw
UD-derived one. Queries are expanded only against such tables.

`UDSlice.unify` (annotation filling): a query feature whose value the table has never seen is ignored;
the tight match keeps the rows with the fewest differences (a stated feature the token lacks, a token
feature the row does not state); `Number`, `Gender`, `Person`, `Degree` are not scored unless the token
has them; competing values are merged. Where readings stay ambiguous (`ambiguous_readings`, e.g. German
gender of determiners) the fill is a comma value.

## 5. Two consumers

**Annotation filling** (`src/word_order/process_treebank.py`): `extract_node_features` calls `expand_anno`
with `um_lookup_table(inflector)` (UniMorph-derived table), then again with `ud_lookup_table(inflector)`
(UD-derived table) for what is still missing. `candidate_fills` queries with the token's features
(lowercased form, `clean_lemma` lemma) and `_unify_rows` merges the matching rows. Both lookup functions
are in `ud_table.py`.

**Reinflection** (`src/multiblimp/ud_inflector.py`): `load_ud_inflector(lang, langcode, unimorph_args,
inflection_map, resource_dir)` returns `(inflector, num_lemmas, num_forms)`. The outer `UDInflector` is
built on the UniMorph-derived table, with the UD-derived one as `fallback`; a UD-only language has an empty
outer table. The swap column comes from the condition's inflection map
(`sva_trees/conditions.INFLECTION_MAPS[(deprel, feature_suffix)]`, passed in by `Pipeline`): bracketed
candidates (`Number[abs]`) first, plain only if none qualifies. `inflect` finds the form's rows, takes the
rows of the same lemma with the swapped value, and requires the swap feature to be stated for any-value
swaps. Verbal rows with a Mood are taken as finite (`_assume_finite`). `get_form_features` returns the
feature readings of a form. Per-row layered overrides and slot aliases come from `create_pairs`
(`resolve_layered_head_key`, `slot_suffixes`).

## 6. Rules to keep in mind

- Everything is lowercased (form, lemma) in both tables and in the queries.
- Placeholder lemmas are dropped everywhere (`lemmas.clean_lemma`).
- `+` in raw UniMorph is an ambiguity unless it is a known compound tag.
- UniMorph gaps (UD value with no UM code, `Tense=Imp` etc.) are bridged only for UniMorph-derived rows.
- Missing pairs caused by labelling inconsistencies between UniMorph and the treebank are accepted.

## 7. Legacy (kept marked, to delete later)

`UnimorphInflector` (inference, `load_inflector`, pickle methods), `PosLookup`, the lexicon pickles
(`resources/unimorph/um_pickles`, `resources/ud_unimorph/ud_pickles`), the `ud_unimorph/<code>` TSVs
(no longer written), `scripts/pickle/pickle_unimorph.py`, the UM-coded maps in `swap_features.py`, the
`UNDEFINED` check in `sva_trees/create_pairs.py`, and the old minimal-pair chain. The treebank pickles
are **not** legacy. Full list: project memory `project_legacy_code_unused`.
