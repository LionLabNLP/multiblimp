# MultiBLiMP 2.0 – {lang} agreement pages (data-debugging build{treebanks_title})

Decision-tree pages and per-condition TSVs for {lang} (UD 2.18, `--debug` mode{treebanks_sentence}),
for finding annotation errors and gaps.

## Run

    python3 serve.py [port]     # http://localhost:{port}/decision_trees/

Needs Python 3 and an internet connection.

## Contents

- `site/decision_trees/` – the overview, and one folder per condition ({n_conditions} in total): scatter plot and table,
  the default tree (`{lang}.html`, all rows, unknown ones split by cause) and the unknown-excluded view
  (`{lang}__asis.html`, via the page's toggle). Noun phrases are under `npa/`.
- `export/<condition>.tsv` – every instance of the condition. `_summary.tsv` has the rows per label. Look
  here to find the full list of sentences with labels and their relevant annotation and conllu file references.

## Debug mode

- A condition needs at least one Agree or Disagree row. A tree needs two classes.
- Unknown rows are kept and split by cause ("Verb's Gender unknown"); greyed out in the unknown-excluded view.
- The tested feature is blocked on both tokens, and lexical features (PronType, Abbr, ...) are not conditions.

## TSV columns

`label` (disagree, agree, unk_head, unk_dep, unk_both; disagree first), `treebank`, `sent_id`,
`conllu_file`, `sent_line`, `link` (Grew query), `sentence` (`[form|role]` marks the two tokens); then per token
(`head` and `nsubj`/`obj`/`iobj`, or a noun-phrase role such as `det`): `_id`, `_line`, `_form`, `_lemma`, `_upos`,
`_deprel`, `_xpos`, the tested feature, `_feats`, `_raw_feats` (the file's FEATS field verbatim) and `_<Feature>_keys`
(the key(s) the file uses for the tested feature, e.g. `Number` or `Number[subj]`; `[subj]` and `[nsubj]` are folded
into the plain feature before comparing, so `_feats` shows the folded value and these two columns show the original).

## Caveats: what is not included

- **Verb forms:** no VerbForm is assumed. Main verbs are finite (`sv`, `ov`, `iov`), participles (`sp`, `op`, `iop`)
  or, for subjects only, any other form including no annotation (`sx`: infinitives, supines, converbs, ...).
  Objects of other forms are not covered. Auxiliaries and copulas are taken in any form.
- **Verb heads:** subjects and objects of `VERB` heads, plus subjects of auxiliaries and copulas (the
  subject-auxiliary conditions). Subjects of a non-verbal predicate with no copula are not covered.
- **Arguments:** only nominal ones (NOUN, PROPN, PRON). Clausal subjects are left out. If a head has two
  arguments with the same relation, only the last one is kept.
- **Noun phrases:** NOUN, PROPN or PRON heads with a `det`, `nummod`, `amod`, `case` or possessive-pronoun
  dependent. Other dependents (relative clauses, nominal modifiers) are ignored, and of several dependents of the
  same part of speech only the rightmost is kept. Bare nouns are left out, and an NP is in a condition's file only if
  one of the two tokens carries the feature.
- **Parts of speech never looked at:** ADV, CCONJ, SCONJ, PART, INTJ, PUNCT, SYM, X. Only VERB and AUX (as
  heads), NOUN, PROPN and PRON (as arguments and noun-phrase heads), and DET, NUM, ADJ and ADP (as noun-phrase
  dependents) take part.
- **Relations not covered:** a predicate adjective or noun agreeing with its subject (only the copula is
  checked), apposition, relative pronouns and their antecedents, conjuncts, clausal subjects.
- **Layered verb features:** `[subj]` and `[nsubj]` keys are folded into the plain feature before comparing. Case-named
  keys (`[erg]`, `[abs]`) are found only through the dependent's own `Case`. A relation subtype that names a verb slot
  (Abkhaz `iobj:lo` with `[lo]`) is compared with that slot; if a head has several such dependents only the plain `iobj`
  is kept. Slots no relation points to are never compared. `<Feature>_keys` in the TSVs shows which keys a token carries.
- **Conditions with no Agree or Disagree row** have no TSV. They were tried, and are listed with their label
  counts in `export/_not_exported.tsv`.

{extra}
