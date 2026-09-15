import os
import re
import sys
import random

sys.path.append("../")

from word_order.process_treebank import create_word_order_df, read_df
from word_order.prediction_target import PredictionTarget
from multiblimp.languages import gblang2udlang, get_ud_langs
from multiblimp.config import TREEBANK_FEATURES_DIR

random.seed(42)


def extract_npa_df(lang: str, target: PredictionTarget, resource_dir: str,
                    word_order_dir: str, max_treebank_len: int | None = None,
                    lexicalize: bool = True, never_skip: bool = False,
                    require_all_children: bool = True):
    """Extract the raw NPA instance dataframe for one language/target.

    Deliberately does the extraction step only: no unimorph/UD-inflection
    lookup (fetch_all=False), no agreement labels (agreement_feats=None), no
    decision tree, no minimal pairs. Agreement computation is benched for now
    -- this is for eyeballing what instances UD's det/amod/case annotation
    actually gives us before deciding how to compute agreement over them.

    require_all_children: if False, a head yields an instance as soon as any
        one of target.child_deprels is present (rather than requiring all of
        them at once) -- see word_order.process_treebank.extract_instances.

    Caches to "{word_order_dir}/{lang}.parquet", same convention as
    sva_trees.pipeline, so repeat runs across det/amod/case/langs don't
    re-parse treebanks. word_order_dir must be distinct per (target,
    require_all_children) combination the caller cares about -- see
    run_extraction_sweep's default, which keys it off target.child_deprels.
    """
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    cache_path = os.path.join(word_order_dir, f"{cached_lang}.parquet")

    if not never_skip and os.path.exists(cache_path):
        return read_df(lang, word_order_dir=word_order_dir)

    return create_word_order_df(
        lang=lang,
        target=target,
        resource_dir=resource_dir,
        save_to=word_order_dir,
        max_treebank_len=max_treebank_len,
        drop_singleton_columns=False,  # exploratory pass: keep everything visible
        lexicalize=lexicalize,
        fetch_all=False,
        require_all_children=require_all_children,
    )


def default_eyeball_columns(df, child_deprels: list[str]) -> list[str]:
    """Columns worth looking at first for a manual sample: sentence
    metadata, head/child form+lemma+pos for every deprel in child_deprels,
    and any single-feature columns (head_Number, det_Case, ...) -- skips the
    sibling-/child-deprel/-pos presence columns extract_node_features also
    computes, which are noisy for a first eyeball pass.
    """
    meta_cols = [c for c in ("sen", "treebank", "sent_id") if c in df.columns]
    prefixes = ["head", *child_deprels]
    core_cols = [
        c for prefix in prefixes
        for c in (f"{prefix}_form", f"{prefix}_lemma", f"{prefix}_pos")
        if c in df.columns
    ]
    feat_pattern = re.compile(
        rf"^({'|'.join(re.escape(p) for p in prefixes)})_[A-Z][a-z]+$"
    )
    feat_cols = [c for c in df.columns if feat_pattern.match(c)]

    seen = set()
    ordered = []
    for c in meta_cols + core_cols + feat_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def sample_rows(df, n: int = 20, seed: int = 42, columns: list[str] | None = None):
    """A small random sample of rows for manual inspection."""
    if len(df) == 0:
        return df
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    return sample[columns] if columns else sample


def npa_cache_key(target: PredictionTarget, require_all_children: bool = True) -> str:
    """Directory-name component identifying a (target, require_all_children)
    combination, so distinct sweeps never collide on the same cache path --
    e.g. det_target alone caches under "det", while a det+amod+case target
    caches under "det_amod_case" (or "det_amod_case_any" when
    require_all_children=False), never under plain "det".
    """
    key = "_".join(target.child_deprels)
    if len(target.child_deprels) > 1 and not require_all_children:
        key += "_any"
    return key


def run_extraction_sweep(target: PredictionTarget,
                          resource_dir: str = "../../resources",
                          word_order_dir: str | None = None,
                          langs: list[str] | None = None,
                          never_skip: bool = False,
                          max_treebank_len: int | None = None,
                          n_samples: int = 20,
                          require_all_children: bool = True):
    """Extract `target` across `langs` (default: every available UD language)
    and print a sample per language, for a first look at what instances the
    extraction step produces. No agreement/decision-tree/pairs step -- see
    module docstring on extract_npa_df.
    """
    cache_key = npa_cache_key(target, require_all_children)
    word_order_dir = word_order_dir or os.path.join(TREEBANK_FEATURES_DIR, "npa", cache_key)
    langs = sorted(langs) if langs else get_ud_langs(resource_dir)

    for lang in langs:
        df = extract_npa_df(lang, target, resource_dir, word_order_dir,
                             max_treebank_len=max_treebank_len,
                             never_skip=never_skip,
                             require_all_children=require_all_children)
        if len(df) == 0:
            print(f"{lang}: no {cache_key} instances found")
            continue
        cols = default_eyeball_columns(df, target.child_deprels)
        sample = sample_rows(df, n=n_samples, columns=cols)
        print(f"\n=== {lang}: {len(df)} instances ===")
        print(sample.to_string(index=False))
