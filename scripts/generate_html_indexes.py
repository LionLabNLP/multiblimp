"""Regenerate every already-built NPA condition's diagnostics CSV + deprel
index.html, then the cross-pipeline decision_trees/index.html overview --
purely from on-disk output/decision_trees + output/minimal_pairs artifacts,
no refitting.

Why this works without refitting: npa.agreement.run_agreement_pipeline
already skips its own fit/render/pairs steps per language whenever their
cached .joblib/.html/pairs-dir already exist and never_skip=False -- so
calling it again on languages that are already fully built degrades to
exactly "regenerate diagnostics + index", which is otherwise unconditional
at the end of every call. This script's only real job is discovering which
(target_col, languages) are already built, so it doesn't have to be told by
hand and doesn't accidentally build anything new.

Scope: NPA only for now. sva_trees.pipeline.Pipeline/subj_aux.pipeline.
SubjAuxPipeline have the same free-reindex property, but re-driving them
generically needs each target's own PredictionTarget/predictor_var/label
metadata, which isn't centralized anywhere yet -- left as a follow-up.

Run from scripts/: python generate_html_indexes.py
"""
import glob
import os
import sys
from pathlib import Path

sys.path.append("../src")

from npa.agreement import run_agreement_pipeline, FEATURE_ABBREV

INV_FEATURE_ABBREV = {v: k for k, v in FEATURE_ABBREV.items()}
from word_order.viz_overview import generate_html_overview_index
from multiblimp.config import (
    TREEBANK_FEATURES_DIR, OUTPUT_DECISION_TREES_DIR, HTML_DECISION_TREES_DIR,
)

RESOURCE_DIR = "../resources"
INSTANCES_DIR = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")


def _npa_identifier_to_target_col(npa_identifier: str) -> str:
    """Inverse of npa.agreement.npa_id -- "HEAD-DET_N" -> "HEAD-DET_Number".
    The on-disk directory name is the abbreviated identifier, not the real
    target_col (run_agreement_pipeline's own argument), so this has to be
    reversed rather than assumed. Relies on FEATURE_ABBREV being injective
    (true today: N/G/P/C/Def/Deg/PT/NT/Pos are all distinct)."""
    roles, abbr = npa_identifier.rsplit("_", 1)
    return f"{roles}_{INV_FEATURE_ABBREV.get(abbr, abbr)}"


def built_npa_conditions() -> dict[str, list[str]]:
    """target_col -> [languages already fit for it], read off whatever's
    already on disk under output/decision_trees/npa/*/*.joblib -- not
    npa_config.json, which lists CANDIDATE languages, most not yet run.

    Uses the .joblib filename directly as the language name (not reversed
    through gblang2udlang), so the rare language whose on-disk cached_lang
    differs from its true UD name is silently skipped rather than mismatched
    -- run_agreement_pipeline already skips it too if the parquet lookup
    that follows from it misses, same as any other unknown language.
    """
    conditions = {}
    for cond_dir in sorted(glob.glob(os.path.join(OUTPUT_DECISION_TREES_DIR, "npa", "*"))):
        if not os.path.isdir(cond_dir):
            continue
        target_col = _npa_identifier_to_target_col(os.path.basename(cond_dir))
        langs = sorted(Path(p).stem for p in glob.glob(os.path.join(cond_dir, "*.joblib")))
        if langs:
            conditions[target_col] = langs
    return conditions


if __name__ == "__main__":
    conditions = built_npa_conditions()
    print(f"{len(conditions)} npa condition(s) already built: {list(conditions)}")

    for target_col, langs in conditions.items():
        print(f"\n=== {target_col} ({len(langs)} languages) ===")
        run_agreement_pipeline(
            target_col=target_col,
            langs=langs,
            instances_dir=INSTANCES_DIR,
            resource_dir=RESOURCE_DIR,
            never_skip=False,
            build_pairs=True,
        )

    print("\nGenerating cross-pipeline overview index")
    generate_html_overview_index(html_directory=HTML_DECISION_TREES_DIR)
