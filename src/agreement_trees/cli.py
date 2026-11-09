"""Command line of the verb-phrase and noun-phrase fits (scripts/agreement_trees/fit_candidate.py): arguments,
language selection and the run_vp / run_np entry points. Everything that is a setting rather than a per-run
choice lives in resources/agreement_config.json."""
import argparse
import dataclasses
import os
import subprocess
import sys

from agreement_trees.config import (
    AGREEMENT_CONFIG_PATH, VP_FAMILIES, load_agreement_config, parse_vp_target, vp_scan_key, vp_target_id,
)
from agreement_trees.fit.scan import np_pair_re
from multiblimp.config import RESOURCE_DIR
from multiblimp.languages import get_ud_langs
from agreement_trees.pairs.second_chance import SecondChanceConfig

CACHE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "scripts", "agreement_trees", "cache_treebank_features.py")


def check_target(target: str) -> None:
    """Exit with a hint unless `target` is a verb-phrase id ("sv:Number", "svNa") or a noun-phrase id
    ("HEAD-DET_Number"). Anything else would otherwise run as a noun-phrase target and find no languages."""
    if parse_vp_target(target) is not None or np_pair_re().match(target):
        return
    hint = ""
    for family in sorted(VP_FAMILIES, key=len, reverse=True):
        if target.startswith(family) and len(target) > len(family):
            hint = (f" Did you mean {family}:{target[len(family):]}? Only Number, Gender and Person have a short "
                    f"form ({family}Na, {family}Ga, {family}Pa).")
            break
    raise SystemExit(f"unrecognised target {target!r}: use family:feature (e.g. sv:Number, sa:Case; families "
                     f"{', '.join(VP_FAMILIES)}) or ROLE1-ROLE2_Feature (e.g. HEAD-DET_Number).{hint}")


def parse_fit_args(argv=None):
    """(args, config). Worker defaults come from the config; flags override."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=AGREEMENT_CONFIG_PATH)
    config = load_agreement_config(pre.parse_known_args(argv)[0].config)
    workers = config["workers"]

    parser = argparse.ArgumentParser(
        description="Fit the decision tree, minimal pairs and pages for one agreement target: "
                    "a verb-phrase target ('sv:Number', or its fixed id 'svNa') or a noun-phrase "
                    "target ('HEAD-DET_Number').")
    parser.add_argument("target", help="'family:feature' (sv, sp, sa, ov, op, oa, iov, iop, ioa) "
                                       "or 'ROLE1-ROLE2_Feature'")
    parser.add_argument("--config", default=AGREEMENT_CONFIG_PATH, help="Agreement config JSON.")
    parser.add_argument("--langs", "-l", nargs="*", default=[],
                        help="Only consider these languages. Either way, only those the agreement scan found worth "
                             "attempting for this target are fit (a live scan if it is not in the results).")
    parser.add_argument("--recache_data", "-rc", action="store_true",
                        help="Re-extract each language's feature cache, then redo everything downstream.")
    parser.add_argument("--refit_trees", "-rf", action="store_true",
                        help="Redo everything downstream of the cached extraction (included/excluded "
                             "split, tree fit, minimal pairs, pages) without re-extracting.")
    parser.add_argument("--redo_pairs", action="store_true",
                        help="Rebuild the minimal pairs and the pooled tree page from the existing fits (no refit, per-treebank "
                             "trees reused). Pairs written by an older version are rebuilt without it.")
    parser.add_argument("--keep_unk", action="store_true",
                        help="Keep unk-labeled rows in the tree fit instead of dropping them.")
    parser.add_argument("--incl_unk_trees", action="store_true",
                        help="Also fit an incl.-unk variant ({lang}__unk) of each tree, pooled and "
                             "per treebank, for inspection; minimal pairs are unaffected.")
    parser.add_argument("--treebanks", nargs="*", default=[],
                        help="Only these treebanks of each language ('UD_Swedish-Talbanken' or 'Talbanken'); "
                             "their rows alone form the pooled tree. Default: all selected treebanks.")
    parser.add_argument("--skip_per_treebank", action="store_true",
                        help="Skip the analysis-only tree per treebank.")
    parser.add_argument("--include_excluded", action="store_true",
                        help="Also fit/render trees for the language's excluded treebanks (never used "
                             "for the pooled tree or minimal pairs). Needs per-treebank trees.")
    parser.add_argument("--no_index", action="store_true",
                        help="Skip this run's diagnostics table and deprel index.html (fit_all.py "
                             "builds them once afterwards).")
    parser.add_argument("--n_jobs", "-j", type=int, default=workers["n_jobs"],
                        help="Languages processed in parallel.")
    parser.add_argument("--max_worker_mem_gb", "-m", type=float, default=workers["max_worker_mem_gb"],
                        help="RAM cap per worker in GB. Default: available RAM x mem_headroom / n_jobs.")
    parser.add_argument("--max_tasks_per_child", type=int, default=workers["max_tasks_per_child"],
                        help="Languages per worker before it is recycled.")
    parser.add_argument("--force", action="store_true", help="Skip the check for other running instances.")
    parser.add_argument("--single_aux_only", action="store_true",
                        help="Auxiliary targets: only single-aux instances, not stacked aux.")
    args = parser.parse_args(argv)
    check_target(args.target)
    return args, config


def second_chance_config(fit_cfg: dict):
    """SecondChanceConfig from the config, None when disabled."""
    sc = fit_cfg["second_chance"]
    if not sc["enabled"]:
        return None
    return SecondChanceConfig(lax_threshold=sc["lax_threshold"])


class TargetSkipped(Exception):
    """No language is eligible for the target; fit_candidate exits SKIPPED_EXIT."""


SKIPPED_EXIT = 3


def extract_missing(mode: str, langs: list[str], args) -> None:
    """Run the caching script for languages with no current cache (a language that
    came back empty has a sentinel, so it is not repeated). It caches every
    family/part for them, so the result serves all targets."""
    cmd = [sys.executable, CACHE_SCRIPT, f"--{mode}", "--config", args.config,
           "--n_jobs", str(args.n_jobs), "--langs", *langs]
    if args.force:
        cmd.append("--force")
    print(f"Extracting {len(langs)} language(s) without a current {mode} cache: {', '.join(langs)}", flush=True)
    if subprocess.run(cmd).returncode:
        raise SystemExit(f"{CACHE_SCRIPT} failed; see output/logs/cache_error_log.txt")


def target_langs(mode: str, target: str, args, config: dict) -> list[str]:
    """The languages that qualify for this target, among --langs, else the config's "langs", else all UD
    languages: out-of-date caches (and, for named languages, missing ones) are extracted first, the rest
    judged from the saved scan where valid (a scan now, saved, for the others). Raises TargetSkipped when
    none qualifies."""
    from agreement_trees.fit.scan import cache_lang, cache_state, resolve_target_langs
    named = [cache_lang(lang) for lang in args.langs or config["langs"]]
    universe = named or get_ud_langs(RESOURCE_DIR)
    key = vp_scan_key(*parse_vp_target(target)) if mode == "vp" else target
    family = parse_vp_target(target)[0] if mode == "vp" else None
    # named languages are extracted when uncached; with no list, only out-of-date caches
    # are refreshed (an uncached language is simply not considered)
    to_extract = ("missing", "outdated") if named else ("outdated",)
    stale = [lang for lang in universe if cache_state(mode, family, lang) in to_extract]
    if stale:
        extract_missing(mode, stale, args)
    langs, info = resolve_target_langs(mode, target, universe, config["scan"])
    if not info["n_cached"]:
        raise TargetSkipped(f"no {mode} feature caches found to check {key} against")
    if not langs:
        raise TargetSkipped(f"none of the {info['n_cached']} cached languages qualifies for {key} "
                            f"({info['n_saved']} judged from the saved scan, {info['n_scanned_now']} scanned now)")
    print(f"{len(langs)}/{info['n_cached']} cached languages worth attempting for {key} "
          f"({info['n_saved']} from the saved scan, {info['n_scanned_now']} scanned now)")
    not_eligible = [lang for lang in named if lang not in langs] if args.langs else []
    if not_eligible:
        print(f"Skipping --langs not eligible for {key}: {', '.join(not_eligible)}")
    return langs


def run_vp(target: str, args, config: dict) -> None:
    """Fit one verb-phrase target ("sv:Number" / "svNa" / "sv:Case").

    A fixed condition (Number/Gender/Person) uses its hand-written swap map;
    any other feature swaps to whatever other value the lemma's paradigm
    attests. Everything else comes from the family's Na condition: Na/Ga/Pa and
    candidates share the cache, deprel, head label and unimorph settings.
    """
    from agreement_trees.vp.conditions import CONDITIONS
    from agreement_trees.vp.pipeline import Pipeline
    family, feature = parse_vp_target(target)
    base = CONDITIONS[f"{family}Na"]
    target_id = vp_target_id(family, feature)
    fixed = CONDITIONS.get(target_id)
    if fixed is not None:
        pipeline_target, predictor_var, inflection_map = fixed.make_target(), fixed.predictor_var, fixed.inflection_map
    else:
        pipeline_target = dataclasses.replace(base.make_target(), swap_feat=feature)
        predictor_var = f"head_{base.deprel}_{feature}_agreement"
        inflection_map = (feature, None)

    kwargs = {}
    if base.is_aux:
        from agreement_trees.aux.pipeline import AuxPipeline
        pipeline_cls = AuxPipeline
        kwargs["aux_target"] = base.aux_target
        kwargs["include_multi_aux"] = not args.single_aux_only
        if args.single_aux_only:
            target_id += "_single"
    else:
        pipeline_cls = Pipeline

    fit = config["fit"]
    pipeline = pipeline_cls(
        target=pipeline_target,
        predictor_var=predictor_var,
        langs=target_langs("vp", target, args, config),
        inflection_map=inflection_map,
        unimorph_args=base.unimorph_args,
        deprel_dir=base.deprel,
        resource_dir=RESOURCE_DIR,
        word_order_dir=base.word_order_dir(),
        max_treebank_len=config["vp"]["max_treebank_len"],
        never_skip=args.recache_data,
        never_skip_fit=args.refit_trees,
        redo_pairs=args.redo_pairs,
        rm_columns=list(base.rm_columns),
        head_label=base.head_label,
        target_id=target_id,
        n_jobs=args.n_jobs,
        max_worker_mem_gb=args.max_worker_mem_gb,
        mem_headroom=config["workers"]["mem_headroom"],
        force=args.force,
        drop_unk=not args.keep_unk,
        max_tasks_per_child=args.max_tasks_per_child,
        per_treebank=not args.skip_per_treebank,
        include_excluded=args.include_excluded,
        fetch_all=True,
        incl_unk=args.incl_unk_trees,
        second_chance=second_chance_config(fit),
        build_index=not args.no_index,
        only_treebanks=args.treebanks,
        threshold=fit["leaf_threshold"],
        min_samples_leaf=fit["min_samples_leaf"],
        max_depth=fit["max_depth"],
        test_size=fit["test_size"],
        impurity=fit["impurity"],
        **kwargs,
    )
    pipeline()


def run_np(target: str, args, config: dict) -> None:
    """Fit one noun-phrase target ("HEAD-DET_Number")."""
    from agreement_trees.npa.agreement import canonical_target_col, run_agreement_pipeline
    if args.single_aux_only:
        raise SystemExit("--single_aux_only only applies to auxiliary verb-phrase targets")
    canonical = canonical_target_col(target)
    if canonical != target:
        print(f"{target} doesn't exist as written (pairs are stored in role priority order) -- using {canonical}")
    fit, np_cfg = config["fit"], config["np"]
    run_agreement_pipeline(
        target_col=canonical,
        langs=target_langs("np", canonical, args, config),
        resource_dir=RESOURCE_DIR,
        never_skip=args.recache_data or args.refit_trees,
        redo_pairs=args.redo_pairs,
        recache_data=args.recache_data,
        extract=dict(max_treebank_len=np_cfg["max_treebank_len"], streaming=np_cfg["streaming"],
                     chunk_size=np_cfg["chunk_size"], fetch_all=True),
        drop_unk=not args.keep_unk,
        max_depth=fit["max_depth"],
        min_samples_leaf=fit["min_samples_leaf"],
        test_size=fit["test_size"],
        leaf_threshold=fit["leaf_threshold"],
        impurity=fit["impurity"],
        per_treebank=not args.skip_per_treebank,
        include_excluded=args.include_excluded,
        second_chance=second_chance_config(fit),
        build_index=not args.no_index,
        incl_unk=args.incl_unk_trees,
        n_jobs=args.n_jobs,
        max_worker_mem_gb=args.max_worker_mem_gb,
        mem_headroom=config["workers"]["mem_headroom"],
        max_tasks_per_child=args.max_tasks_per_child,
        force=args.force,
        only_treebanks=args.treebanks,
    )
