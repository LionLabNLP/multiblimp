import gc
import os
import re
import shutil
import sys
from collections import Counter
from itertools import combinations

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append("../")

from word_order.process_treebank import (
    load_treebank, get_all_feats, records_to_df, extract_node_features, _build_tree_maps,
    tree2sen, tree_no_space_after,
)

# Direct dependents that count toward an NP's UPOS-type projection, and the
# UPOS each must carry to qualify -- mirrors npa.targets' det/nummod/amod/
# case/nmod:poss targets (head_pos=NOUN/PROPN/PRON), but scans ALL matching
# dependents of a head (e.g. two amod children), not the single
# child-per-deprel that word_order.process_treebank.extract_instances keeps
# (see its deprel2child dict: last child of a given deprel wins, others are
# dropped). That's fine for extract_instances' per-instance-agreement
# purpose, but it means a noun with two adjectives only ever contributes one
# amod_* column there -- wrong for reconstructing a phrase's full UPOS
# sequence. This reimplements the direct-dependents scan straight over the
# treebank instead.
#
# nmod is scoped down to possessive pronouns specifically ("a friend of
# mine"), not nominal modifiers generally ("the president of the company") --
# an earlier version qualified any NOUN/PROPN/PRON nmod child, but that's
# both too broad (any nmod, not just possessive ones) and, per an empirical
# sweep comparing candidate selection criteria (see conversation), the most
# reliable signal for "this is a possessive" turned out to be neither a
# upos+Poss=Yes filter nor a upos=PRON+Case=Gen filter -- both under- and
# over-select depending on the language -- but simply UD's own "nmod:poss"
# deprel subtype, which is the most complete across the sample. So: key on
# the literal deprel string "nmod:poss" (not "nmod"), restricted to PRON
# (the noun/proper-noun genitive-possessor case, e.g. Turkish izafet or
# English "John's", is deliberately excluded -- those aren't pronouns).
QUALIFYING_DEPS = {
    "det": {"DET"},
    "nummod": {"NUM"},
    "amod": {"ADJ"},
    "case": {"ADP"},
    "nmod:poss": {"PRON"},
}


def np_upos_type(tree, noun_token, qualifying_deps=QUALIFYING_DEPS) -> str:
    """Linear UPOS sequence of noun_token's own direct qualifying dependents
    (det/nummod/amod/case/nmod:poss matching their expected POS) plus the
    noun itself, in surface (token id) order -- e.g. "DET ADJ ADJ NOUN" or
    "NOUN ADP". A bare noun with no qualifying dependents returns just its own UPOS
    ("NOUN"/"PROPN"/"PRON"). Non-qualifying dependents (e.g. an advmod modifying an
    adjective rather than the noun itself) are not part of the projection --
    same "direct dependents only" definition npa.targets' NOUN-headed
    targets already use.
    """
    span = [(noun_token["id"], noun_token["upos"])]
    for token in tree:
        if token["head"] != noun_token["id"]:
            continue
        allowed_pos = qualifying_deps.get(token["deprel"])
        if allowed_pos is not None and token["upos"] in allowed_pos:
            span.append((token["id"], token["upos"]))
    span.sort(key=lambda x: x[0])
    return " ".join(upos for _, upos in span)


def count_np_types(lang: str, resource_dir: str, max_treebank_len: int | None = None,
                    head_pos=("NOUN", "PROPN", "PRON"), qualifying_deps=QUALIFYING_DEPS) -> Counter:
    """Counter of NP UPOS-type string -> occurrence count for one language."""
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
    counts = Counter()
    for tree in treebank:
        for token in tree:
            if token["upos"] in head_pos:
                counts[np_upos_type(tree, token, qualifying_deps)] += 1
    return counts


# Canonical role order for an NP's tokens: the head always sorts first
# (fixed role label "HEAD", see np_roles), then dependents by deprel
# priority (det, nummod, amod, case; nmod:poss's role is "PRON" -- since
# unlike the head it isn't relabeled, so it gets its own tier, trailing
# case). Matches npa.targets' child_deprels order. Pair-column names
# ("HEAD-DET", not "DET-HEAD") follow this order regardless of the
# language's actual word order, so they read consistently across languages.
ROLE_PRIORITY = {"HEAD": 0, "DET": 1, "NUM": 2, "ADJ": 3, "ADP": 4, "PRON": 5}


def np_roles(tree, noun_token, qualifying_deps=QUALIFYING_DEPS) -> list[tuple[str, dict]]:
    """Ordered (role, token) pairs for one NP: the head (always labeled the
    fixed role "HEAD", regardless of whether it's NOUN, PROPN, or PRON --
    this pools NOUN-, PROPN-, and PRON-headed NPs' feature/agreement
    columns together as "HEAD_Gender"/"HEAD-DET_Gender" rather than
    splitting them into parallel "NOUN_Gender"/"PROPN_Gender"/"PRON_Gender"
    sets; np_type still preserves the NOUN-vs-PROPN-vs-PRON distinction
    separately), then its qualifying dependents. Dependent role labels are
    UPOS-based ("DET", "ADJ", "ADP", "PRON" for nmod:poss) with no
    disambiguation for repeats: if a noun has two amod children, only the
    rightmost (surface-order-last) one ends up under role "ADJ" -- the
    earlier one is simply overwritten and dropped from this per-instance
    role/feature view. np_type (see np_upos_type) still records the full
    repeated-UPOS sequence ("DET ADJ ADJ NOUN"); only the role-keyed
    feature/agreement columns collapse repeats down to one slot per UPOS, to
    keep the role vocabulary (and so the pairwise-column space) fixed and
    small rather than growing per language with however many modifiers its
    longest attested NP happens to stack. Returned in ROLE_PRIORITY order
    (head, det, nummod, amod, case, nmod:poss), not surface order.
    """
    dependents = []
    for token in tree:
        if token["head"] != noun_token["id"]:
            continue
        allowed_pos = qualifying_deps.get(token["deprel"])
        if allowed_pos is not None and token["upos"] in allowed_pos:
            dependents.append(token)
    dependents.sort(key=lambda t: t["id"])  # surface order -> last (rightmost) wins below

    roles = {"HEAD": noun_token}
    for token in dependents:
        roles[token["upos"]] = token  # overwrite: only the rightmost of a given upos survives

    return sorted(roles.items(), key=lambda role_token: ROLE_PRIORITY.get(role_token[0], 99))


def pairwise_agreement(val1, val2) -> str:
    """"yes"/"no"/"unk" for two feature values -- unk if either side is
    undefined, yes/no otherwise. Collapses the finer Yes/No/+-/-- scheme
    word_order.process_treebank.extract_instances uses for SVA down to the
    3-way label requested here.
    """
    if val1 is None or val2 is None:
        return "unk"
    return "yes" if val1 == val2 else "no"


def np_instance(tree, noun_token, all_feats: set, all_deprel: set, all_pos: set,
                 tree_maps, meta: dict, qualifying_deps=QUALIFYING_DEPS,
                 roles: list[tuple[str, dict]] | None = None) -> dict:
    """One row: np_type, per-role node features, and pairwise agreement
    columns.

    Per-role features are the full spread word_order.process_treebank.
    extract_node_features computes for SVA's head/nsubj items -- deprel,
    pos, idx, form/lemma, raw morphological feats, this token's own
    head_pos/head_deprel/dir/grandhead_deprel, path-to-root ("under_X" for
    every deprel in all_deprel), and sibling-/child-of-this-token presence
    and lemma/feat indicators -- computed identically for every role (HEAD
    and every dependent), not just the head. `all_feats`/`all_deprel`/
    `all_pos` should be the language's frequency-filtered sets
    (word_order.process_treebank.get_all_feats); `tree_maps` should be
    word_order.process_treebank._build_tree_maps(tree), built once per tree
    by the caller (build_np_data) rather than once per NP instance.
    target=None is passed to extract_node_features: the only place it reads
    `target` is a `prefix=="head"` branch, and our roles are always "HEAD"
    (uppercase) or a dependent's UPOS, so that branch never fires.

    Pairwise agreement, by contrast, stays scoped to just the raw
    morphological feats (not the fuller node-feature spread above) for
    every (role1, role2) pair actually present in this NP x every feature
    present on at least one side of that pair -- "yes"/"no"/"unk" is a
    narrow, well-defined comparison; deprel/pos/idx/sibling/child columns
    aren't agreement targets, just extra predictors.

    `roles`: pass np_roles(...)'s already-computed result if the caller
    needs it anyway (e.g. build_np_data checks len(roles) to skip bare
    heads before calling this at all) to avoid recomputing it here.
    """
    if roles is None:
        roles = np_roles(tree, noun_token, qualifying_deps)

    instance = dict(meta)
    instance["np_type"] = np_upos_type(tree, noun_token, qualifying_deps)

    role_feats = {}
    for role, token in roles:
        node_features = extract_node_features(
            token, tree, tree_maps, role, all_feats, all_deprel, all_pos,
            None, lexicalize=True,
            excluded_deprels=set(qualifying_deps) if role == "HEAD" else None,
        )
        instance.update(node_features)
        role_feats[role] = {
            feat: node_features.get(f"{role}_{feat}")
            for feat in all_feats
            if node_features.get(f"{role}_{feat}") is not None
        }

    # Default assumption: a NOUN/PROPN role without an explicit Person
    # annotation is assumed 3rd person, but only when Person is attested
    # (explicitly annotated) on at least one other role in this same NP --
    # i.e. Person is a relevant category here, just not marked on this token.
    roles_with_person = {role for role, _ in roles if "Person" in role_feats[role]}
    for role, token in roles:
        if (
            token["upos"] in ("NOUN", "PROPN")
            and role not in roles_with_person
            and roles_with_person
        ):
            role_feats[role]["Person"] = "3"
            instance[f"{role}_Person"] = "3"

    for (role1, _), (role2, _) in combinations(roles, 2):
        feats1, feats2 = role_feats[role1], role_feats[role2]
        for feat in feats1.keys() | feats2.keys():
            instance[f"{role1}-{role2}_{feat}"] = pairwise_agreement(
                feats1.get(feat), feats2.get(feat)
            )

    return instance


def _np_records_for_tree(tree, tree_idx: int, all_feats: set, all_deprel: set, all_pos: set,
                          head_pos, qualifying_deps) -> tuple[list[dict], Counter]:
    """One tree's worth of build_np_data's inner loop body -- every
    NOUN/PROPN/PRON head's np_instance record (skipping bare heads, which
    only contribute to `counts`, see build_np_data's own docstring) plus the
    np_type/bare-UPOS counts for this tree alone. Factored out of
    build_np_data so build_np_data_streaming can reuse the exact same
    per-tree logic while flushing records to disk every `chunk_size` of them
    instead of accumulating the whole treebank's records in memory at once
    -- keeping both entry points behaviorally identical is the point; a
    change to one must not silently diverge from the other.
    """
    tree_maps = _build_tree_maps(tree)
    meta = {
        "treebank": tree.metadata["treebank"].split("/")[0],
        "sent_id": tree.metadata["sent_id"],
        "tree_idx": tree_idx,
        "sen": tree2sen(tree),
        "no_space_after": tree_no_space_after(tree),
    }
    records = []
    counts = Counter()
    for token in tree:
        if token["upos"] not in head_pos:
            continue
        roles = np_roles(tree, token, qualifying_deps)
        if len(roles) <= 1:  # bare head: no qualifying dependents
            counts[token["upos"]] += 1
            continue
        instance = np_instance(tree, token, all_feats, all_deprel, all_pos,
                                tree_maps, meta, qualifying_deps, roles=roles)
        counts[instance["np_type"]] += 1
        records.append(instance)
    return records, counts


def build_np_data(lang: str, resource_dir: str, max_treebank_len: int | None = None,
                   head_pos=("NOUN", "PROPN", "PRON"), qualifying_deps=QUALIFYING_DEPS):
    """One treebank pass producing both the (fast, small) NP UPOS-type
    Counter -- covering every NOUN/PROPN/PRON head, including bare ones with
    no qualifying dependents -- and the (heavier) per-instance DataFrame
    with the full per-role node-feature spread and pairwise agreement
    columns (see np_instance), which only covers heads with at least one
    qualifying dependent.

    A bare head (np_roles returns only "HEAD") is dropped before any of the
    expensive per-role work: no extract_node_features call, no tree-context
    lookups, nothing beyond the np_roles scan itself -- it can never
    contribute a pairwise agreement column (combinations of a single role is
    empty), so there's nothing for np_instance to add for it. Still counted
    under its own bare-UPOS np_type ("NOUN"/"PROPN"/"PRON") in `counts`.

    Holds every record for the whole (possibly capped) treebank in memory at
    once -- np_instance's per-role node-feature spread is ~13k columns wide
    (see extract_node_features' deprel/feat/pos cross-product columns), so
    peak memory scales with treebank size and can reach the tens of GB for
    a large, morphologically rich language even at a few thousand sentences.
    Use build_np_data_streaming instead for a language too large to hold
    fully in memory this way.

    Returns (counts: Counter, df: pd.DataFrame).
    """
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
    all_feats, _, all_deprel, all_pos = get_all_feats(treebank)

    counts = Counter()
    records = []
    for tree_idx, tree in enumerate(treebank):
        tree_records, tree_counts = _np_records_for_tree(
            tree, tree_idx, all_feats, all_deprel, all_pos, head_pos, qualifying_deps
        )
        records.extend(tree_records)
        counts.update(tree_counts)

    df = records_to_df(records)
    return counts, df


def _decode_dictionary_columns(table: "pa.Table") -> "pa.Table":
    """Cast every dictionary-encoded (pandas "category") column in `table`
    to its plain value type (e.g. dictionary<string> -> string).

    Each chunk's dictionary is local -- built independently from just that
    chunk's own distinct values (see records_to_df/_categorize) -- and
    pandas/pyarrow default to the narrowest index width that fits it
    (int8, up to 127 distinct values). Concatenating many chunks' local
    dictionaries for a high-cardinality column (e.g. "HEAD_form"/
    "HEAD_lemma", lexicalized -- easily >127 distinct forms across a few
    hundred chunks) without reconciling them raises "Integer value ... not
    in range" (verified empirically): pa.concat_tables doesn't unify
    dictionaries or widen the index type on its own. Decoding to plain
    string before concatenation sidesteps this entirely -- parquet still
    dictionary-encodes repeated string values at the file-format level
    regardless, so this doesn't meaningfully cost storage; it just isn't
    pandas "category" dtype on read-back (an "object"/string column
    instead), which every npa/ reader treats the same way regardless
    (value_counts/notna/equality all work identically on either dtype).

    Builds the new column list first and constructs the table in one
    pa.Table.from_arrays call rather than calling set_column once per
    dictionary column found -- Table.set_column/append_column rebuild the
    table's internal column list on every call, making a per-column loop
    over np_instance's ~13k-wide schema O(n^2) (measured: ~34s for one
    20k-column table) instead of the ~0.05s a single one-shot rebuild
    takes. This exact bug is why an early version of
    build_np_data_streaming was ~30x SLOWER than build_np_data despite
    using a fraction of the memory -- fixed here and in _consolidate_parts.
    """
    columns = [
        column.cast(field.type.value_type) if pa.types.is_dictionary(field.type) else column
        for field, column in zip(table.schema, table.columns)
    ]
    return pa.Table.from_arrays(columns, names=table.column_names)


def _resolve_field_types(schemas: list["pa.Schema"]) -> dict[str, "pa.DataType"]:
    """{field_name: resolved_type} across every part schema, preferring a
    "real" type over pyarrow's null/double placeholder types wherever a
    field disagrees between chunks.

    A chunk-local column that's entirely empty gets one of two placeholder
    types depending on exactly HOW it ended up all-empty (see
    _consolidate_parts): pa.null() when every record explicitly carried the
    key as Python None, or double (pandas' default fill for a column some
    records never set the key for at all) when a chunk mixes explicit-None
    records with key-absent ones for the same column -- np_instance's
    per-role feature dict production does exactly this (e.g. a
    "{role}_sibling-feat_{deprel}_{feat}" key is only ever present at all
    when that role's token has a head, see extract_node_features; explicit
    None vs the key never existing both occur for it, depending on the
    row). Neither placeholder reflects the column's real type -- once ANY
    chunk has real data in that column (e.g. plain string, after
    _decode_dictionary_columns has already run), that's the type every
    chunk's version should end up as.

    Only null/floating placeholder types ever get overridden this way; if
    EVERY chunk agrees on a type (including double, e.g. a column that is
    genuinely empty across the entire language, or one that's genuinely
    numeric everywhere), that type is kept as-is -- there's nothing more
    specific to prefer.
    """
    resolved: dict[str, "pa.DataType"] = {}
    for schema in schemas:
        for field in schema:
            current = resolved.get(field.name)
            if current is None or (
                current != field.type
                and (pa.types.is_null(current) or pa.types.is_floating(current))
            ):
                resolved[field.name] = field.type
    return resolved


def _consolidate_parts(part_paths: list[str], out_path: str) -> None:
    """Merge build_np_data_streaming's per-chunk parquet files into one
    `out_path` file, preserving the single-file-per-language convention
    every other npa/ module expects (agreement_candidates.py,
    npa.agreement.run_agreement_pipeline, np_morph_pct.py all read
    treebank_features/npa/np_instances/{lang}.parquet as one file).

    Three things a naive pyarrow.dataset.dataset(paths) read gets wrong
    here, all verified empirically against this exact data:

    1. Different chunks can have different column sets (e.g. a chunk with
       no ADJ-headed NP has no "ADJ_*" columns at all -- see np_instance) --
       a bare ds.dataset(paths) silently uses the FIRST file's schema and
       drops any column that file didn't happen to have, unlike pandas'
       own pd.concat (which unions columns automatically). Passing an
       explicit target schema fixes this.
    2. Each chunk's dictionary-encoded (category) columns use a LOCAL
       dictionary/index width -- concatenating many chunks' worth of a
       high-cardinality lexicalized column (e.g. "HEAD_form") without
       reconciling them overflows pyarrow's int8 indices ("Integer value
       128 not in range"). _decode_dictionary_columns sidesteps this by
       decoding every such column to plain string before concatenation.
    3. A field that's entirely empty in one chunk but has real values in
       another can disagree on TYPE, not just presence -- e.g. "double"
       (pandas' fill for a column some records never set the key for, see
       _resolve_field_types) vs "string". pa.unify_schemas refuses to
       merge these (ArrowTypeError), and even given an explicit target
       schema, ds.dataset can't implicitly cast double -> string either
       (ArrowNotImplementedError) -- so any column whose type disagrees
       with the resolved target is manually replaced with a null array of
       the target type (valid exactly because a type mismatch here only
       ever arises from an all-empty placeholder column in the first
       place -- see _resolve_field_types).

    Builds each fixed table's full column list first and constructs it via
    one pa.Table.from_arrays call, same reasoning as
    _decode_dictionary_columns -- looping set_column/append_column once
    per field over np_instance's ~13k-wide schema is O(n^2) and was the
    single biggest cost in an early version of this function (~30x
    slower overall than build_np_data despite using far less memory).

    This step's own peak memory is back to ~the size of the full
    consolidated table (pyarrow.Table, not pandas -- somewhat more compact,
    but not a hard bound) -- see build_np_data_streaming's docstring for
    when to skip it (consolidate=False) instead.
    """
    tables = [_decode_dictionary_columns(pq.read_table(p)) for p in part_paths]
    resolved_types = _resolve_field_types([t.schema for t in tables])
    all_names = list(resolved_types)

    fixed_tables = []
    for table in tables:
        existing = dict(zip(table.column_names, table.columns))
        columns = [
            existing[name] if name in existing and existing[name].type == resolved_types[name]
            else pa.nulls(len(table), type=resolved_types[name])
            for name in all_names
        ]
        fixed_tables.append(pa.Table.from_arrays(columns, names=all_names))

    combined = pa.concat_tables(fixed_tables)
    pq.write_table(combined, out_path)


def build_np_data_streaming(lang: str, resource_dir: str, out_path: str,
                             max_treebank_len: int | None = None, chunk_size: int = 500,
                             head_pos=("NOUN", "PROPN", "PRON"), qualifying_deps=QUALIFYING_DEPS,
                             consolidate: bool = True) -> Counter:
    """Memory-bounded sibling of build_np_data: same per-tree logic (see
    _np_records_for_tree, shared by both), but flushes accumulated records
    to a parquet file on disk every `chunk_size` NP-instance records instead
    of holding the whole (possibly multi-hundred-thousand-sentence)
    treebank's records in memory at once. Peak memory during the actual
    tree-processing loop is ~O(chunk_size x width) -- roughly constant,
    independent of total treebank size -- rather than build_np_data's
    O(treebank_size x width), where width is np_instance's ~13k columns.
    Chunked by RECORD count, not tree count (contrast word_order.
    process_treebank.extract_features's tree-count batch_size): trees vary
    widely in how many NP instances they contribute, and it's the record
    count x width product that actually drives memory, not the tree count.

    Does not return a DataFrame (the whole point is to avoid ever holding
    one for the full language) -- writes directly to `out_path` and returns
    just `counts` (the Counter, same as build_np_data's — for save_counts'
    np_type CSV). A caller wanting the data back should read `out_path`
    afterward, same as every other npa/ consumer already does.

    consolidate=True (default) merges the per-chunk parts into one
    `out_path` file at the end (see _consolidate_parts) -- keeps the
    on-disk convention every other npa/ module expects, at the cost of that
    merge step's own peak memory being back to ~full-table size (just
    working from compact on-disk parquet/Arrow data rather than the
    original wide Python-dict records, so noticeably lighter than
    build_np_data's peak in practice, but not a hard bound the way the
    chunked accumulation phase is). For a language too large even for that
    final merge (i.e. still OOMs with consolidate=True), pass
    consolidate=False: the per-chunk parts are left on disk at
    f"{out_path}.parts/part-*.parquet" and never merged -- a caller can
    read them back with pyarrow.dataset + pa.unify_schemas (the same
    technique _consolidate_parts uses) without ever materializing the
    whole thing in one process, but every other npa/ module would need
    updating to read that layout instead of a single file; out of scope
    here since nothing currently needs it.

    Returns `counts` (Counter) -- {} (empty) if the language has no
    NOUN/PROPN/PRON heads at all (mirrors build_np_data: nothing gets
    written to `out_path` in that case, matching np_morph_pct.py's existing
    "no np_type column" handling for a degenerate/empty parquet).
    """
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
    all_feats, _, all_deprel, all_pos = get_all_feats(treebank)

    parts_dir = f"{out_path}.parts"
    os.makedirs(parts_dir, exist_ok=True)
    part_paths = []
    chunk_records = []
    counts = Counter()

    def flush():
        if not chunk_records:
            return
        chunk_df = records_to_df(chunk_records)
        part_path = os.path.join(parts_dir, f"part-{len(part_paths):05d}.parquet")
        chunk_df.to_parquet(part_path, index=False)
        part_paths.append(part_path)
        chunk_records.clear()

    for tree_idx, tree in enumerate(treebank):
        tree_records, tree_counts = _np_records_for_tree(
            tree, tree_idx, all_feats, all_deprel, all_pos, head_pos, qualifying_deps
        )
        chunk_records.extend(tree_records)
        counts.update(tree_counts)
        if len(chunk_records) >= chunk_size:
            flush()
    flush()

    del treebank
    gc.collect()

    if not part_paths:
        os.rmdir(parts_dir)
        return counts

    if consolidate:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _consolidate_parts(part_paths, out_path)
        shutil.rmtree(parts_dir)

    return counts


# Matches a genuine UD morphological feature name (Gender, Case, PronType,
# ...) -- same convention sva_trees/create_pairs.py's kind_feat_cols regex
# uses (rf"^{kind}_([A-Z][a-z]+)"). np_instance() now also stores, per role,
# the same non-morphological node features SVA collects (deprel/pos/idx/
# head_pos/head_deprel/dir/grandhead_deprel/under_*/sibling-*/child-*) --
# all lowercase- or hyphen-led -- so this regex is what keeps those out of
# _np_roles_and_feats/morph_value_counts, which are specifically about
# morphological-feature agreement, not the fuller predictor set.
_FEAT_NAME_RE = re.compile(r"^[A-Z][a-z]")


def _np_roles_and_feats(df) -> dict[str, list[str]]:
    """role -> morphological feature names present as "{role}_{feature}"
    columns in `df`, restricted to roles that actually occur at least once
    in `df` (so a np_type group that never includes e.g. ADJ doesn't
    spuriously report ADJ features just because the column exists elsewhere
    in a wider dataframe).

    Role names come from a small fixed vocabulary (HEAD, DET, NUM, ADJ, ADP,
    NOUN/PROPN/PRON) with no repeat-numbering suffixes (see np_roles), so no
    role name is ever a string-prefix of another -- prefix-matching
    "{role}_" against df.columns is unambiguous. Two further filters narrow
    the match to genuine morphological features (see _FEAT_NAME_RE): "-" is
    excluded (np_instance()'s pairwise "{role1}-{role2}_{feature}" columns,
    and any of the hyphenated "sibling-*"/"child-*" node-feature columns),
    and the remainder after the role prefix must look like a UD feature name
    (excludes "{role}_deprel"/"_pos"/"_idx"/"_dir"/"_head_pos"/"_form"/
    "_lemma"/etc. and the "under_{deprel}" family).
    """
    all_roles = {c[: -len("_form")] for c in df.columns if c.endswith("_form")}
    present_roles = [r for r in all_roles if df[f"{r}_form"].notna().any()]

    role_feats = {}
    for role in present_roles:
        prefix = f"{role}_"
        role_feats[role] = [
            col[len(prefix):] for col in df.columns
            if col.startswith(prefix) and "-" not in col
            and _FEAT_NAME_RE.match(col[len(prefix):])
        ]

    ordered = sorted(present_roles, key=lambda r: ROLE_PRIORITY.get(r, 99))
    return {r: role_feats[r] for r in ordered}


def morph_value_counts(df) -> dict[str, int]:
    """Raw occurrence counts (not yet normalized -- meant to be summed
    across groups before dividing by a separately-tracked total, so pooling
    many languages doesn't require holding their instance frames
    concatenated) for:

    - single-role feature values: "{role}_{feat}_{value}" -> how many rows
      have that role's feature set to that value.
    - pairwise joint-value matches: "{role1}-{role2}_{feat}_{value}" -> how
      many rows have BOTH role1 and role2 set to that same value. This is a
      finer-grained companion to np_instance()'s coarse yes/no/unk
      "{role1}-{role2}_{feat}" agreement column: it shows which value an
      agreement centers on (e.g. HEAD-DET_Case_Nom), not just that one
      exists.
    - pairwise mismatches: "{role1}-{role2}_{feat}_mismatch" -> how many
      rows have both sides defined but set to DIFFERENT values (e.g.
      HEAD_Gender=Fem, DET_Gender=Masc). Without this, a disagreeing row
      contributes to none of the per-value match counts above and is
      simply invisible to this stat. One aggregate bucket per (role pair,
      feature) rather than a full (value1, value2) confusion matrix --
      matches the coarse agreement column's own yes/no/unk granularity.
    """
    role_feats = _np_roles_and_feats(df)
    counts = Counter()

    for role, feats in role_feats.items():
        for feat in feats:
            col = f"{role}_{feat}"
            for value, n in df[col].value_counts(dropna=True).items():
                counts[f"{role}_{feat}_{value}"] += int(n)

    roles = list(role_feats)
    for role1, role2 in combinations(roles, 2):
        shared_feats = set(role_feats[role1]) & set(role_feats[role2])
        for feat in shared_feats:
            col1, col2 = f"{role1}_{feat}", f"{role2}_{feat}"
            both = df[[col1, col2]].dropna()
            # .astype(object): the two columns are independently-built
            # categoricals (records_to_df), so their category sets can
            # differ even when the underlying values overlap -- pandas
            # refuses to compare categoricals with mismatched categories.
            v1 = both[col1].astype(object)
            v2 = both[col2].astype(object)
            is_match = v1 == v2
            for value, n in both[is_match][col1].value_counts().items():
                counts[f"{role1}-{role2}_{feat}_{value}"] += int(n)
            # Rows where both sides are defined but disagree (e.g.
            # HEAD_Gender=Fem, DET_Gender=Masc) would otherwise vanish
            # entirely -- neither v1's nor v2's value gets a match-count
            # above, so a mismatch was previously invisible to this stat.
            # One aggregate bucket rather than a full (v1, v2) confusion
            # matrix, matching the "yes"/"no"/"unk" agreement column's own
            # granularity.
            mismatch_n = int((~is_match).sum())
            if mismatch_n:
                counts[f"{role1}-{role2}_{feat}_mismatch"] += mismatch_n

    return dict(counts)


def np_type_morph_counts(df, group_col: str = "np_type") -> dict[str, tuple[int, dict]]:
    """{np_type: (group_size, morph_value_counts dict)} for every np_type
    group in df. group_size is the % denominator a caller should use.
    """
    return {
        np_type: (len(group), morph_value_counts(group))
        for np_type, group in df.groupby(group_col, observed=True)
    }
