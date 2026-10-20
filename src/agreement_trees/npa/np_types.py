import gc
import os
import re
import shutil
from collections import Counter
from itertools import combinations

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from agreement_trees.extract.process_treebank import (
    load_treebank, get_all_feats, records_to_df, extract_node_features, _build_tree_maps,
    tree2sen, tree_no_space_after, values_overlap, filled_feats, conllu_sentence_meta, attractor_cols,
)
from agreement_trees.config import lexical_features
from agreement_trees.pipeline_utils import CACHE_VERSION, _KEY as _CACHE_KEY
from agreement_trees.taxonomy import NPA_ROLE_ORDER
from multiblimp.excluded_treebanks import is_treebank_row_excluded
from multiblimp.languages import lang2langcode
from multiblimp.ud_inflector import load_ud_inflector
from multiblimp.ud_table import ud_lookup_table, um_lookup_table

# Lookup tables for annotation filling: UniMorph plus its UD-derived fallback.
NP_LOOKUP_UNIMORPH_ARGS = {"combine_um_ud": True, "remove_multiword_forms": True}


def np_lookup_tables(lang: str, resource_dir: str):
    """(um_data, ud_data) for fetch_all. Only the lookup tables are used, so the
    inflection map is a placeholder."""
    inflector, *_ = load_ud_inflector(
        lang=lang, langcode=lang2langcode(lang), unimorph_args=NP_LOOKUP_UNIMORPH_ARGS,
        inflection_map=("Number", None), resource_dir=resource_dir,
    )
    return um_lookup_table(inflector), ud_lookup_table(inflector)

# Direct dependents that count toward an NP (deprel -> the UPOS each must carry). All matching
# dependents of a head are scanned, unlike extract_instances, which keeps one child per deprel.
# nmod is limited to UD's "nmod:poss" subtype on PRON children ("a friend of mine"): it was the most
# complete possessive signal across languages; genitive nouns (Turkish izafet, English "John's") are out.
QUALIFYING_DEPS = {
    "det": {"DET"},
    "nummod": {"NUM"},
    "amod": {"ADJ"},
    "case": {"ADP"},
    "nmod:poss": {"PRON"},
}


def np_upos_type(tree, noun_token, qualifying_deps=QUALIFYING_DEPS) -> str:
    """UPOS sequence of the noun and its direct qualifying dependents in surface order ("DET ADJ ADJ NOUN");
    a bare noun gives just its UPOS. Other dependents (e.g. an advmod of an adjective) are not part of it."""
    span = [(noun_token["id"], noun_token["upos"])]
    for token in tree:
        if token["head"] != noun_token["id"]:
            continue
        allowed_pos = qualifying_deps.get(token["deprel"])
        if allowed_pos is not None and token["upos"] in allowed_pos:
            span.append((token["id"], token["upos"]))
    span.sort(key=lambda x: x[0])
    return " ".join(upos for _, upos in span)


# Canonical role order of an NP's tokens: head first, then det, nummod, amod, case, nmod:poss (role "PRON").
# Pair columns ("HEAD-DET", never "DET-HEAD") follow it whatever the language's word order.
ROLE_PRIORITY = NPA_ROLE_ORDER


def np_roles(tree, noun_token, qualifying_deps=QUALIFYING_DEPS) -> list[tuple[str, dict]]:
    """Ordered (role, token) pairs for one NP, in ROLE_PRIORITY order: the head (always role "HEAD", whether
    NOUN, PROPN or PRON, so their columns are pooled; np_type keeps the distinction) and its qualifying
    dependents under UPOS-based roles. Repeats collapse to one slot per UPOS (the rightmost wins) to keep
    the column space small; np_type still records the full sequence."""
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
    """"yes"/"no"/"unk" for two feature values: unk if either side is undefined."""
    if val1 is None or val2 is None:
        return "unk"
    return "yes" if (val1 == val2 or values_overlap(val1, val2)) else "no"


def np_instance(tree, noun_token, all_feats: set, all_deprel: set, all_pos: set,
                 tree_maps, meta: dict, qualifying_deps=QUALIFYING_DEPS,
                 roles: list[tuple[str, dict]] | None = None,
                 um_data=None, ud_data=None, fetch_all: bool = False) -> dict:
    """One row: np_type, per-role node features and pairwise agreement columns.

    Per-role features are the full extract_node_features spread, computed identically for the head and
    every dependent. `all_feats`/`all_deprel`/`all_pos` are the language's frequency-filtered sets
    (get_all_feats); `tree_maps` is _build_tree_maps(tree), built once per tree by the caller.
    Pairwise agreement covers only the raw morphological feats, for every role pair present x every
    feature stated on at least one side. `roles`: np_roles(...)'s result, if the caller already has it.
    """
    if roles is None:
        roles = np_roles(tree, noun_token, qualifying_deps)

    instance = dict(meta)
    instance["np_type"] = np_upos_type(tree, noun_token, qualifying_deps)

    role_feats = {}
    assumed_person = []
    for role, token in roles:
        node_features = extract_node_features(
            token, tree, tree_maps, role, all_feats, all_deprel, all_pos,
            None, lexicalize=True,
            excluded_deprels=set(qualifying_deps) if role == "HEAD" else None,
            # xpos is kept out of the tree fit (some tagsets spell out the feature, see
            # agreement_trees.npa.agreement.npa_fit_omit_feats) but extracted for display
            include_xpos=True,
            um_data=um_data, ud_data=ud_data, fetch_all=fetch_all,
        )
        instance.update(node_features)
        role_feats[role] = {
            feat: node_features.get(f"{role}_{feat}")
            for feat in all_feats
            if node_features.get(f"{role}_{feat}") is not None
        }

    # a NOUN/PROPN role without Person is assumed 3rd person when another role of this NP states Person
    roles_with_person = {role for role, _ in roles if "Person" in role_feats[role]}
    for role, token in roles:
        if (
            token["upos"] in ("NOUN", "PROPN")
            and role not in roles_with_person
            and roles_with_person
        ):
            role_feats[role]["Person"] = "3"
            instance[f"{role}_Person"] = "3"
            assumed_person.append(f"{role}:Person")
    instance["filled_feats"] = filled_feats(dict(roles), assumed_person)

    for (role1, tok1), (role2, tok2) in combinations(roles, 2):
        feats1, feats2 = role_feats[role1], role_feats[role2]
        for feat in (feats1.keys() | feats2.keys()) - lexical_features():
            instance[f"{role1}-{role2}_{feat}"] = pairwise_agreement(
                feats1.get(feat), feats2.get(feat)
            )
        refs = {feat: feats1.get(feat) if feats1.get(feat) is not None else feats2.get(feat)
                for feat in (feats1.keys() | feats2.keys()) - lexical_features()}
        instance.update(attractor_cols(tree, tok1["id"], tok2["id"], refs, f"{role1}-{role2}",
                                         all_feats - lexical_features(), um_data, ud_data, fetch_all))

    return instance


def _np_records_for_tree(tree, tree_idx: int, all_feats: set, all_deprel: set, all_pos: set,
                          head_pos, qualifying_deps, um_data=None, ud_data=None,
                          fetch_all: bool = False) -> tuple[list[dict], Counter]:
    """One tree's NP records (every head with a qualifying dependent) and np_type counts; bare heads are
    only counted. Shared by build_np_data and build_np_data_streaming, which must stay identical."""
    tree_maps = _build_tree_maps(tree)
    meta = {
        "treebank": tree.metadata["treebank"].split("/")[0],
        "sent_id": tree.metadata["sent_id"],
        "tree_idx": tree_idx,
        **conllu_sentence_meta(tree),
        "sen": tree2sen(tree),
        "no_space_after": tree_no_space_after(tree),
    }
    records = []
    counts = Counter()
    # excluded treebanks' records are kept (split off at read time, see
    # agreement_trees.npa.agreement.split_excluded) but never counted in the np_type distribution
    counted = not is_treebank_row_excluded(meta["treebank"])
    for token in tree:
        if token["upos"] not in head_pos:
            continue
        roles = np_roles(tree, token, qualifying_deps)
        if len(roles) <= 1:  # bare head: no qualifying dependents
            if counted:
                counts[token["upos"]] += 1
            continue
        instance = np_instance(tree, token, all_feats, all_deprel, all_pos,
                                tree_maps, meta, qualifying_deps, roles=roles,
                                um_data=um_data, ud_data=ud_data, fetch_all=fetch_all)
        if counted:
            counts[instance["np_type"]] += 1
        records.append(instance)
    return records, counts


def build_np_data(lang: str, resource_dir: str, max_treebank_len: int | None = None,
                   head_pos=("NOUN", "PROPN", "PRON"), qualifying_deps=QUALIFYING_DEPS,
                   fetch_all: bool = False):
    """One treebank pass giving the np_type Counter (every head, bare ones included) and the per-instance
    DataFrame (heads with at least one qualifying dependent; bare heads are dropped before any per-role work).

    Holds all records in memory (the frame is ~13k columns wide), so use build_np_data_streaming for large
    languages. All treebanks are read, selected and excluded alike (records carry a "treebank" column,
    split again at read time); `counts` covers the selected ones only.

    Returns (counts, df).
    """
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len,
                             use_selected_treebanks=False)
    all_feats, _, all_deprel, all_pos = get_all_feats(treebank)
    um_data, ud_data = np_lookup_tables(lang, resource_dir) if fetch_all else (None, None)

    counts = Counter()
    records = []
    for tree_idx, tree in enumerate(treebank):
        tree_records, tree_counts = _np_records_for_tree(
            tree, tree_idx, all_feats, all_deprel, all_pos, head_pos, qualifying_deps,
            um_data, ud_data, fetch_all
        )
        records.extend(tree_records)
        counts.update(tree_counts)

    df = records_to_df(records)
    return counts, df


def _decode_dictionary_columns(table: "pa.Table") -> "pa.Table":
    """Cast dictionary-encoded columns to their plain value type. Chunk-local dictionaries have the
    narrowest index width (int8), and concatenating many of them for a high-cardinality column overflows
    ("Integer value ... not in range"). Builds the table in one from_arrays call: per-column set_column
    over ~13k columns is quadratic."""
    columns = [
        column.cast(field.type.value_type) if pa.types.is_dictionary(field.type) else column
        for field, column in zip(table.schema, table.columns)
    ]
    return pa.Table.from_arrays(columns, names=table.column_names)


def _resolve_field_types(schemas: list["pa.Schema"]) -> dict[str, "pa.DataType"]:
    """{field_name: type} across the part schemas, preferring a real type over the null/double placeholders
    that all-empty chunk columns get (pa.null() or pandas' double fill); types all chunks agree on are kept."""
    resolved: dict[str, "pa.DataType"] = {}
    for schema in schemas:
        for field in schema:
            current = resolved.get(field.name)
            if current is not None and current != field.type and all(
                    pa.types.is_integer(t) or pa.types.is_floating(t) for t in (current, field.type)):
                resolved[field.name] = pa.float64()
            elif current is None or (
                current != field.type
                and (pa.types.is_null(current) or pa.types.is_floating(current))
            ):
                resolved[field.name] = field.type
    return resolved


BATCH_ROWS = 50_000


def _as_type(column: "pa.ChunkedArray", target: "pa.DataType") -> "pa.ChunkedArray":
    """The column as `target`; a column of another type that cannot be cast has to be all-empty (a placeholder)
    and becomes nulls."""
    if column.type == target:
        return column
    try:
        return column.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        if column.null_count < len(column):
            raise
        return pa.nulls(len(column), type=target)


def _leaf_to_column(path: str) -> str:
    return path.removesuffix(".list.element")


def _nonempty_columns(part_paths: list[str]) -> set[str]:
    """Names of the columns with at least one value in some part, read off the parts' row-group statistics
    (no data is loaded). A column without statistics counts as non-empty, except null-typed ones, which never
    have any. The NP schema is rows x (role x child relation x feature): Czech has ~14k columns of which ~9k
    are empty in every part, and keeping those through the merge made it need tens of GB."""
    keep: set[str] = set()
    for path in part_paths:
        parquet_file = pq.ParquetFile(path)
        meta = parquet_file.metadata
        null_typed = {f.name for f in parquet_file.schema_arrow if pa.types.is_null(f.type)}
        for i in range(meta.num_columns):
            name = _leaf_to_column(meta.schema.column(i).path)
            if name in keep or name in null_typed:
                continue
            for g in range(meta.num_row_groups):
                group = meta.row_group(g)
                stats = group.column(i).statistics
                if stats is None or not stats.has_null_count or stats.null_count < group.num_rows:
                    keep.add(name)
                    break
    return keep


def _decoded_schema(schema: "pa.Schema", keep: set[str]) -> "pa.Schema":
    """The part's schema as _decode_dictionary_columns leaves it, restricted to the non-empty columns."""
    return pa.schema(
        [pa.field(f.name, f.type.value_type if pa.types.is_dictionary(f.type) else f.type)
         for f in schema if f.name in keep],
        metadata=schema.metadata)


def _true_only_columns(part_paths: list[str], resolved_types: dict) -> set[str]:
    """Boolean columns that hold True and no False in any part. Their gaps are filled with False, as
    process_treebank._categorize does on a whole frame; a part alone cannot decide that."""
    candidates = [n for n, t in resolved_types.items() if pa.types.is_boolean(t)]
    has_false: set[str] = set()
    for path in part_paths:
        present = set(pq.read_schema(path).names)
        names = [n for n in candidates if n in present and n not in has_false]
        if not names:
            continue
        table = _decode_dictionary_columns(pq.read_table(path, columns=names))
        has_false.update(n for n in names if pc.any(pc.invert(table[n])).as_py())
    return set(candidates) - has_false


def _consolidate_parts(part_paths: list[str], out_path: str) -> None:
    """Merge the per-chunk parquet files into the single `out_path` file the other npa modules read.
    A plain dataset read gets three things wrong: chunks can have different column sets (the first file's
    schema would drop columns), dictionary columns need decoding first, and an all-empty column can differ
    in type between chunks (replaced by a null array of the resolved type). Columns empty in every part are
    dropped first (see _nonempty_columns) and the output is written part by part in row groups, so the
    peak is one batch of parts, not the whole consolidated table (Czech needed tens of GB for that)."""
    keep = _nonempty_columns(part_paths)
    schemas = [_decoded_schema(pq.read_schema(p), keep) for p in part_paths]
    resolved_types = _resolve_field_types(schemas)
    all_names = list(resolved_types)
    true_only = _true_only_columns(part_paths, resolved_types)
    target = pa.schema([(name, resolved_types[name]) for name in all_names],
                       metadata={**(schemas[0].metadata or {}), _CACHE_KEY: str(CACHE_VERSION).encode()})

    tmp_path = out_path + ".tmp"
    pending, pending_rows = [], 0
    with pq.ParquetWriter(tmp_path, target) as writer:
        for path in part_paths:
            present = set(pq.read_schema(path).names)
            table = _decode_dictionary_columns(pq.read_table(path, columns=[n for n in all_names if n in present]))
            existing = dict(zip(table.column_names, table.columns))
            columns = [
                _as_type(existing[name], resolved_types[name]) if name in existing
                else pa.nulls(len(table), type=resolved_types[name])
                for name in all_names
            ]
            columns = [pc.fill_null(column, False) if name in true_only else column
                       for name, column in zip(all_names, columns)]
            pending.append(pa.Table.from_arrays(columns, schema=target))
            pending_rows += len(table)
            if pending_rows >= BATCH_ROWS:
                writer.write_table(pa.concat_tables(pending))
                pending, pending_rows = [], 0
        if pending:
            writer.write_table(pa.concat_tables(pending))
    os.replace(tmp_path, out_path)


def build_np_data_streaming(lang: str, resource_dir: str, out_path: str,
                             max_treebank_len: int | None = None, chunk_size: int = 500,
                             head_pos=("NOUN", "PROPN", "PRON"), qualifying_deps=QUALIFYING_DEPS,
                             consolidate: bool = True, fetch_all: bool = False) -> Counter:
    """Memory-bounded sibling of build_np_data: same per-tree logic, but records are flushed to a parquet
    part every `chunk_size` records (counted by records, not trees: records x width drives memory), so peak
    memory during extraction is roughly constant. Writes `out_path` and returns only `counts`; nothing is
    written when the language has no NOUN/PROPN/PRON heads.

    consolidate=True merges the parts into one `out_path` file (the layout the other npa modules read); False
    leaves them in f"{out_path}.parts/" for a language too large for that merge (no reader supports that layout).
    Reads every treebank, counts selected ones only, as in build_np_data.
    """
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len,
                             use_selected_treebanks=False)
    all_feats, _, all_deprel, all_pos = get_all_feats(treebank)
    um_data, ud_data = np_lookup_tables(lang, resource_dir) if fetch_all else (None, None)

    parts_dir = f"{out_path}.parts"
    os.makedirs(parts_dir, exist_ok=True)
    part_paths = []
    chunk_records = []
    counts = Counter()

    def flush():
        if not chunk_records:
            return
        chunk_df = records_to_df(chunk_records, fill_true_only=False)
        part_path = os.path.join(parts_dir, f"part-{len(part_paths):05d}.parquet")
        chunk_df.to_parquet(part_path, index=False)
        part_paths.append(part_path)
        chunk_records.clear()

    for tree_idx, tree in enumerate(treebank):
        tree_records, tree_counts = _np_records_for_tree(
            tree, tree_idx, all_feats, all_deprel, all_pos, head_pos, qualifying_deps,
            um_data, ud_data, fetch_all
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


# A genuine UD feature name (Gender, Case, ...); keeps the lowercase- or hyphen-led node-feature columns
# (deprel, pos, sibling-*, ...) out of the morphological stats below.
_FEAT_NAME_RE = re.compile(r"^[A-Z][a-z]")


def _np_roles_and_feats(df) -> dict[str, list[str]]:
    """role -> feature names present as "{role}_{feature}" columns, for roles that occur in `df`. Role names
    never prefix one another; "-" columns (pairwise, sibling-*, child-*) and non-feature names are excluded."""
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
    """Raw occurrence counts, to be summed across groups before normalising:
    - "{role}_{feat}_{value}": rows where that role's feature has that value;
    - "{role1}-{role2}_{feat}_{value}": rows where both roles have that same value (which value an
      agreement centres on);
    - "{role1}-{role2}_{feat}_mismatch": rows where both are defined but differ (one bucket, not a confusion matrix).
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
            # object dtype: independently built categoricals can have different category sets
            v1 = both[col1].astype(object)
            v2 = both[col2].astype(object)
            is_match = v1 == v2
            for value, n in both[is_match][col1].value_counts().items():
                counts[f"{role1}-{role2}_{feat}_{value}"] += int(n)
            mismatch_n = int((~is_match).sum())
            if mismatch_n:
                counts[f"{role1}-{role2}_{feat}_mismatch"] += mismatch_n

    return dict(counts)


def np_type_morph_counts(df, group_col: str = "np_type") -> dict[str, tuple[int, dict]]:
    """{np_type: (group_size, morph_value_counts)} per np_type group; group_size is the % denominator."""
    return {
        np_type: (len(group), morph_value_counts(group))
        for np_type, group in df.groupby(group_col, observed=True)
    }
