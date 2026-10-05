"""Small infrastructure helpers shared by sva_trees.pipeline.Pipeline,
subj_aux.pipeline.AuxPipeline, npa.agreement, and multiblimp.pipeline
(memory capping, the concurrent-instance guard, reading back fit_dt's
unk-drop counts, and matching a reinflected token's casing to its swap
candidate's) -- previously two private, near-identical copies (one per
module), which is exactly the kind of thing that quietly drifts out of sync
over time (it already had: subj_aux's _limit_process_memory and sva_trees's
_limit_worker_memory were the same function under two names). Pooled here,
in multiblimp/ rather than word_order/, matching where both callers already
get their other shared cross-cutting helpers from (multiblimp.languages,
multiblimp.unimorph) -- neither sva_trees nor subj_aux/word_order/npa "owns"
this, so neither should host it as a private helper the others reach into.
"""

import ctypes
import math
import os
import subprocess
import json
import threading
import time


def total_system_memory_bytes():
    """Best-effort total physical RAM, POSIX only. None if undetectable
    (e.g. Windows). Only used as a fallback when /proc/meminfo isn't
    available -- prefer available_system_memory_bytes for anything
    memory-cap-related.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def available_system_memory_bytes():
    """Best-effort currently-available memory."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return total_system_memory_bytes()


def find_other_running_instances(script_path):
    """PIDs of other processes that appear to be running this same script:
    another live invocation (command line mentions the script's filename,
    via pgrep -f) plus any orphaned ProcessPoolExecutor worker/
    resource_tracker left behind by a run of this script whose top-level
    process already died. The latter matters because a killed run doesn't
    take its workers down with it -- max_tasks_per_child recycling means
    they're mid-language at any given moment, so they keep running to
    completion (still writing output) after their parent is gone, then exit
    on their own once done. Until they exit they're a second writer racing
    the next run, but pgrep -f can't see them: a spawned worker's command
    line is just "-c from multiprocessing.spawn import spawn_main; ..." with
    no mention of script_path anywhere. They're identified instead by ppid 1
    (reparented to launchd/init, i.e. orphaned) plus a cwd matching this
    script's own directory (every Pipeline/np_types run's workers inherit
    their cwd from the driver process, which always runs from its own
    script's directory).

    script_path: sys.argv[0] as invoked (relative or absolute).

    Returns [] (skips the check) if pgrep/ps/lsof aren't available rather
    than blocking the run.
    """
    script_name = os.path.basename(script_path)
    pids = set()

    try:
        result = subprocess.run(
            ["pgrep", "-f", script_name], capture_output=True, text=True
        )
        pids.update(int(p) for p in result.stdout.split() if p.strip().isdigit())
    except (FileNotFoundError, OSError):
        pass

    try:
        script_dir = os.path.dirname(os.path.abspath(script_path))
        ps_out = subprocess.run(
            ["ps", "-eo", "pid,ppid,command"], capture_output=True, text=True
        ).stdout.splitlines()[1:]
        orphan_candidates = []
        for line in ps_out:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, ppid, command = parts
            if ppid == "1" and (
                "multiprocessing.spawn" in command
                or "multiprocessing.resource_tracker" in command
            ):
                orphan_candidates.append(int(pid))
        for pid in orphan_candidates:
            lsof_out = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True, text=True,
            ).stdout.splitlines()
            cwd = next((l[1:] for l in lsof_out if l.startswith("n")), None)
            if cwd == script_dir:
                pids.add(pid)
    except (FileNotFoundError, OSError, IndexError, ValueError):
        pass

    return [p for p in pids if p != os.getpid()]


def limit_process_memory(max_bytes):
    """Caps this process's virtual address space (RLIMIT_AS) so running out
    of memory raises a catchable MemoryError / crashes just this process,
    instead of letting the OS OOM-killer kill an arbitrary process
    system-wide (which is what takes down unrelated services, not just this
    one). Used as a ProcessPoolExecutor initializer (each worker calls it
    once, on itself, before running any language) -- see each pipeline's
    own _worker_mem_bytes for how the per-worker cap is sized down as
    n_jobs grows, so total memory across all workers stays bounded
    regardless of n_jobs.
    """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (ImportError, ValueError, OSError):
        pass


def guard_process_memory(max_bytes, poll_s=0.5, grace_s=10.0):
    """limit_process_memory plus an RSS watchdog, for platforms where
    RLIMIT_AS isn't enforced (macOS). A daemon thread polls this process's
    resident set; past max_bytes it raises MemoryError in the main thread
    (caught per language by the caller, so the run continues), and if usage
    is still over the cap grace_s later it hard-exits the process instead.
    Needs psutil; without it only the RLIMIT_AS cap applies.
    """
    limit_process_memory(max_bytes)
    try:
        import psutil
    except ImportError:
        return
    proc = psutil.Process()
    main_id = threading.main_thread().ident

    def watch():
        over_since = None
        while True:
            time.sleep(poll_s)
            if proc.memory_info().rss < max_bytes:
                over_since = None
                continue
            if over_since is None:
                over_since = time.monotonic()
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(main_id), ctypes.py_object(MemoryError))
            elif time.monotonic() - over_since > grace_s:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def match_casing(original: str, reinflected: str) -> str:
    """Match a reinflected/swapped token's casing to the swap candidate it
    came from. multiblimp.unimorph.UnimorphInflector.lemma2form (what every
    inflector.inflect() call ultimately bottoms out in) reads word forms
    verbatim off a UniMorph/UD-derived paradigm table -- usually all
    lowercase, regardless of how the original token was actually cased in
    its sentence (sentence-initial capital, an all-caps acronym, ...). Left
    alone, that silently lowercases the reinflected replacement even though
    nothing about the swap itself should change its casing convention.

    All-caps is checked first: a single-letter original (e.g. "A") makes
    both isupper() and istitle() true, where .upper() and .capitalize()
    happen to agree anyway, but checking isupper() first keeps a real
    all-caps token (e.g. "NASA") from being title-cased instead.
    """
    if not isinstance(original, str) or not isinstance(reinflected, str):
        return reinflected
    if original.isupper():
        return reinflected.upper()
    if original.istitle():
        return reinflected.capitalize()
    return reinflected


def read_unk_counts(decision_trees_dir: str, lang: str) -> dict:
    """word_order.decision_tree.fit_dt's <lang>_unk_counts.json ({"head_unk":
    n, "nsubj_unk": n, "both_unk": n}), the same shape fit_dt returns
    directly on a fresh fit. {} for languages with no file yet (never fit a
    tree, e.g. trivial single-class languages, or too few post-drop rows)
    -- create_pairs treats a falsy unk_counts the same as None.
    """
    fn = os.path.join(decision_trees_dir, f"{lang}_unk_counts.json")
    if not os.path.exists(fn):
        return {}
    with open(fn) as f:
        return json.load(f)


def write_label_distribution(decision_trees_dir: str, lang: str, label_distribution: dict) -> None:
    """Raw label counts (Yes/No/unk for SVA/subj_aux, yes/no/unk for NPA)
    over the language's full pre-fit data, written unconditionally by
    sva_trees.pipeline.Pipeline/subj_aux.pipeline.AuxPipeline/npa.agreement
    regardless of whether a tree was fit or minimal pairs were ever
    generated. In a normal pairs-enabled run this same information also
    reaches word_order.viz_deprel.generate_html_deprel_index via the
    pairs-derived "diag" blob's meta.json (sva_trees.diagnostics.
    _read_label_distribution) -- this file is the fallback for when that
    doesn't exist (the data-debugging --debug mode, which never runs
    create_pairs), read back by read_label_distribution below.
    """
    os.makedirs(decision_trees_dir, exist_ok=True)
    fn = os.path.join(decision_trees_dir, f"{lang}_label_distribution.json")
    with open(fn, "w") as f:
        json.dump(label_distribution, f)


def read_label_distribution(decision_trees_dir: str, lang: str) -> dict:
    """See write_label_distribution. {} if never written (e.g. a language
    skipped before reaching that point)."""
    fn = os.path.join(decision_trees_dir, f"{lang}_label_distribution.json")
    if not os.path.exists(fn):
        return {}
    with open(fn) as f:
        return json.load(f)


def wilson_lower_bound(both: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score confidence interval (default z=1.96,
    i.e. ~95%) for the true rate of `both`/`total` -- used by
    build_lang_config in place of a flat min-count or min-percentage cutoff
    to decide whether a (target_col, lang)'s joint attestation is real
    signal or plausible annotation noise (see conversation: a raw `both`
    count alone can't tell "8/50, a real minority pattern" apart from
    "8/50000, probably a few stray tags", and a raw percentage alone is
    unstable at small `total` -- 2/20 and 2000/20000 are both "10%" but
    very different strength of evidence). The Wilson lower bound adapts
    to both at once: a small `total` needs a much higher observed rate to
    clear a given floor than a large `total` does, roughly matching how
    much a human would trust each case.

    0.0 for total==0 (no co-occurrences at all -- shouldn't reach here in
    practice, since build_lang_config only calls this for target_cols that
    already have a nonzero `total`, but kept safe for direct callers).
    """
    if total == 0:
        return 0.0
    p = both / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, center - margin)


def passes_agreement_bar(yes: int, no: int, total: int, wilson_floor: float = 0.01,
                         min_both: int = 10, min_yes: int = 10,
                         yes_rate_floor: float = 0.15) -> bool:
    """Whether a (target_col, lang) is worth attempting: enough jointly-tagged
    rows (both=yes+no, Wilson-floored against the co-occurrence total), AND
    enough actual "yes" rows, whose Wilson lower bound against both must clear
    yes_rate_floor. The yes conditions matter because pairs are only ever
    built from "yes" items -- a lexical feature like Derivation or PronType
    is tagged on both sides constantly but almost always with different
    values ("no"), so it clears the both-based bar without being agreement."""
    both = yes + no
    return (both >= min_both and wilson_lower_bound(both, total) >= wilson_floor
            and yes >= min_yes and wilson_lower_bound(yes, both) >= yes_rate_floor)
