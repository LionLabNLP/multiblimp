"""The UD-derived lexicon: every (lemma, form, upos, features) a language's treebank attests,
written as resources/ud_tables/<langcode>.parquet (multiblimp.ud_table).

Tokens without usable lemma/features are skipped, and rows that contradict a much more frequent
one (same lemma and features with another form, or same lemma and form with other features) are
dropped as annotation errors; the thresholds are ud_tables.dup_form_threshold / dup_feat_threshold
in resources/data_prep_config.json.
"""
import sys
from collections import Counter, defaultdict
from typing import *

from tqdm import tqdm

from .annotation_coverage import (
    count_token, count_unmapped, finish_ud_stats, new_ud_stats, table_coverage, update_coverage,
)
from .config import REPO_ROOT
from .languages import lang2langcode, skip_langs
from .ud_table import clean_lemma
from .treebank import Treebank, has_typo
from .ud_table import is_stored_feature, table_path, write_ud_table

sys.path.append(REPO_ROOT)
from resources.um2ud_annotation.UD2UM_mapper import UD2UM, feat_val_to_um_tag, upos_to_um_tag


def _value_is_mapped(feat: str, raw_val: str) -> bool:
    """Whether a UD feature value has a UniMorph tag (layered features: any of its values)."""
    base_feat, bracket, _ = feat.partition("[")
    if bracket:
        return any(UD2UM.get((base_feat, val)) is not None for val in raw_val.split(","))
    return feat_val_to_um_tag(feat, raw_val) is not None


def create_all_ud_tables(
    ud_langs: List[str],
    verbose=False,
    resource_dir=".",
    dup_form_threshold=100.0,
    dup_feat_threshold: float = 100.0,
    load_from_pickle=False,
    log=None,
):
    for lang in tqdm(sorted(ud_langs)):
        if lang in skip_langs:
            continue

        isolang = lang2langcode(lang)
        treebank = Treebank(
            lang.replace(" ", "_"),
            remove_diacritics=False,
            load_from_pickle=load_from_pickle,
            resource_dir=resource_dir,
            remove_typo=False,
            pickle_path="ud/ud_typo_pickles",
        )

        coverage = new_ud_stats()
        rows = collect_ud_rows(
            treebank, verbose=verbose, dup_form_threshold=dup_form_threshold,
            dup_feat_threshold=dup_feat_threshold, coverage=coverage,
        )
        if rows:
            write_ud_table(rows, table_path(isolang))
        update_coverage(lang, "ud", finish_ud_stats(coverage))
        update_coverage(lang, "ud_table", table_coverage(table_path(isolang)))
        counts = (lang, len({r[0] for r in rows}), len({r[1] for r in rows}), len(rows))
        if log is not None:
            log.info("\t".join(map(str, counts)))
        else:
            print(*counts, sep="\t")


def collect_ud_rows(
    treebank: Treebank,
    verbose: bool = False,
    dup_form_threshold: float = 100.0,
    dup_feat_threshold: float = 100.0,
    coverage: Optional[dict] = None,
) -> list:
    """The treebank's lexicon rows, (lemma, form, upos, {feature: value}, frequency), for
    multiblimp.ud_table.write_ud_table. coverage: a multiblimp.annotation_coverage.new_ud_stats()
    dict to fill with the token counts (every token is counted before any skip rule applies)."""
    ud_keys = []

    iterator = tqdm(treebank) if verbose else treebank

    def skip(reason: str) -> None:
        if coverage is not None:
            coverage["skipped"][reason] += 1

    for tree in iterator:
        treebank_name = tree.metadata.get("treebank", "").split("/")[0]
        for token in tree:
            form = token["form"]
            upos = token["upos"]
            lemma = token["lemma"]
            token_feats = token["feats"] or {}
            if coverage is not None:
                count_token(coverage, treebank_name, upos, token_feats)
                count_unmapped(coverage, token_feats, _value_is_mapped)

            # "_" is UD's missing value; some treebanks put a placeholder lemma on unlemmatized tokens
            if clean_lemma(lemma, form) is None:
                skip("no lemma")
                continue

            if "_" in lemma:
                skip("multiword lemma")
                continue

            if has_typo(token):
                skip("typo")
                continue

            if len(token_feats) == 0:
                skip("no feats")
                continue

            if upos_to_um_tag(upos) is None:
                skip("upos has no UniMorph tag")
                continue

            stored = {f: ",".join(sorted(set(v.split(",")))) for f, v in token_feats.items() if is_stored_feature(f)}
            if stored:
                ud_keys.append((lemma.lower(), form.lower(), upos, tuple(sorted(stored.items()))))

    # duplicate removal on (lemma, form, ("upos=<UPOS>", "Feat=Val", ...), frequency)
    ud_freqs = [
        (lemma, form, (f"upos={upos}",) + tuple(f"{f}={v}" for f, v in feats), freq)
        for (lemma, form, upos, feats), freq in Counter(ud_keys).items()
    ]
    rows = []
    for lemma, form, encoded, freq in remove_duplicate_features(ud_freqs, dup_form_threshold, dup_feat_threshold):
        feats = dict(part.split("=", 1) for part in encoded[1:])
        rows.append((lemma, form, encoded[0][len("upos="):], feats, freq))
    if coverage is not None:
        coverage["rows_raw"], coverage["rows"] = len(ud_freqs), len(rows)
    return rows


def _pos_class(features: tuple) -> str:
    """The UniMorph POS class of a row's UPOS (NOUN and PROPN are both N): rows of one class are compared as
    the same word class."""
    return upos_to_um_tag(features[0][len("upos="):]).split(".")[0]


def remove_duplicate_features(
    row_freqs: List[Tuple[str, str, tuple, int]],
    dup_form_threshold: float,
    dup_feat_threshold: float,
) -> List[Tuple[str, str, tuple, int]]:
    """
    Form mismatch might indicate annotation error, since we can expect
    a lemma+feat combination to usually map to a single form
    (lemma1, form1, feat1)  <> (lemma1, form2, feat1)

    Feature mismatch might indicate feature annotation error, but here
    we only do this if the less frequent triple occurs once. Different
    lemma+form combinations can be instantiations of different feature
    sets, e.g. I saw_PRS / I saw_PST / you saw_PST
    (lemma1, form1, feat1)  <> (lemma1, form1, feat2)
    """
    lemma_feat2form = defaultdict(list)
    lemma_form2feat = defaultdict(list)

    for lemma, form, features, freq in row_freqs:
        lemma_feat2form[lemma, features].append((form, freq))
        lemma_form2feat[lemma, form].append((features, freq))

    dedup_forms = []
    for (lemma, features), forms in lemma_feat2form.items():
        if len(forms) == 1:
            form, freq = forms[0]
            dedup_forms.append((lemma, form, features, freq))
        else:
            sorted_forms: List[Tuple[str, int]] = sorted(
                forms, key=lambda x: x[1], reverse=True
            )
            most_common_form = sorted_forms[0]
            # If the most frequent form is _n_ times more frequent we remove that item
            dedup_forms.append(
                (lemma, most_common_form[0], features, most_common_form[1])
            )
            for next_common_form in sorted_forms[1:]:
                if (most_common_form[1] / next_common_form[1]) <= dup_form_threshold:
                    dedup_forms.append(
                        (lemma, next_common_form[0], features, next_common_form[1])
                    )

    dedup_feats = []
    for (lemma, form), all_features in lemma_form2feat.items():
        if len(all_features) == 1:
            features, freq = all_features[0]
            dedup_feats.append((lemma, form, features, freq))
        else:
            sorted_feats: List[Tuple[tuple, int]] = sorted(
                all_features, key=lambda x: x[1], reverse=True
            )
            most_common_feat = sorted_feats[0]
            # If the most frequent feature is _n_ times more frequent we remove that item
            dedup_feats.append((lemma, form, most_common_feat[0], most_common_feat[1]))
            for next_common_feat in sorted_feats[1:]:
                mismatching_feats = set(next_common_feat[0]).symmetric_difference(most_common_feat[0])

                if (
                    ((most_common_feat[1] / next_common_feat[1]) <= dup_feat_threshold)
                    or (len(mismatching_feats) > 2)
                    or (next_common_feat[1] > 1)
                    or (_pos_class(most_common_feat[0]) != _pos_class(next_common_feat[0]))
                ):
                    dedup_feats.append(
                        (lemma, form, next_common_feat[0], next_common_feat[1])
                    )

    # the table's row order is part of its content (ties in lookups), so the sort key keeps the order the rows
    # had as ";"-joined "<UniMorph POS tag>;upos=<UPOS>;Feat=Val" strings
    def sort_key(row):
        lemma, form, features, freq = row
        return repr((lemma, form, ";".join((upos_to_um_tag(features[0][len("upos="):]),) + features), freq))

    dedup_rows = sorted(set(dedup_forms) & set(dedup_feats), key=sort_key)

    return dedup_rows
