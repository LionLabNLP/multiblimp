"""Small infrastructure helpers shared by sva_trees.pipeline.Pipeline and
subj_aux.pipeline.SubjAuxPipeline (memory capping, the concurrent-instance
guard, and reading back fit_dt's unk-drop counts) -- previously two private,
near-identical copies (one per module), which is exactly the kind of thing
that quietly drifts out of sync over time (it already had: subj_aux's
_limit_process_memory and sva_trees's _limit_worker_memory were the same
function under two names). Pooled here, in multiblimp/ rather than word_order/,
matching where both callers already get their other shared cross-cutting
helpers from (multiblimp.languages, multiblimp.unimorph) -- neither
sva_trees nor subj_aux/word_order "owns" this, so neither should host it as
a private helper the other reaches into.
"""

import os
import subprocess
import json


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


def find_other_running_instances(script_name):
    """PIDs of other processes whose command line mentions script_name,
    excluding this process itself. Uses pgrep -f; returns [] (skips the
    check) if pgrep isn't available rather than blocking the run.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", script_name], capture_output=True, text=True
        )
    except (FileNotFoundError, OSError):
        return []
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
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
