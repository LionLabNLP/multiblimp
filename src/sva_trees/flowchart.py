"""
HTML flowchart rendering for create_pairs(): visualizes how each row ends up in a
process_item() diagnostics bucket, with example rows attached to each outcome.
"""

import html as html_lib
import re

import pandas as pd

from word_order.utils import build_grew_link


# ── Sentence reconstruction ──────────────────────────────────────────────────
# Mirrors word_order/create_pairs.py's get_sen_str/get_swapped_sen_str: correct
# spacing needs the live tree's SpaceAfter=No misc annotations, which aren't
# carried in the saved "sen" column (just bare form strings). When a treebank
# is available we look the tree up by tree_idx and use its real SpaceAfter
# data; otherwise we fall back to a punctuation heuristic.

def _no_space_afters(tree) -> list:
    return [(tok["misc"] or {}).get("SpaceAfter") == "No" for tok in tree]


def _render_token(tok, highlighted: bool) -> str:
    """Escape a token and wrap it in <strong> if it's an agreement-relevant token
    (the head/child whose feature is being checked or was swapped)."""
    text = html_lib.escape(str(tok))
    return f"<strong>{text}</strong>" if highlighted else text


def _join_with_spacing(tokens, no_space_afters, highlight_idx=frozenset()) -> str:
    parts = [
        _render_token(tok, i in highlight_idx) + ("" if no_space else " ")
        for i, (tok, no_space) in enumerate(zip(tokens, no_space_afters))
    ]
    return "".join(parts).strip()


_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'", "’", "%"}


def _fallback_join(tokens, highlight_idx=frozenset()) -> str:
    """Heuristic spacing used when no treebank is available: no space before
    closing punctuation. Less accurate than real SpaceAfter data."""
    parts = []
    for i, tok in enumerate(tokens):
        if i > 0 and str(tok) not in _NO_SPACE_BEFORE:
            parts.append(" ")
        parts.append(_render_token(tok, i in highlight_idx))
    return "".join(parts)


def _detect_swap_col(row):
    """Find the swap_<kind> column process_item set on this row, if any.

    Only matches scalar string values, since the source df already has unrelated
    columns sharing the "swap_" prefix (e.g. swap_order_candidates, a list)."""
    for col in row.index:
        if col.startswith("swap_") and isinstance(row.get(col), str) and row[col]:
            return col, col[len("swap_"):]
    return None, None


def _resolve_kind(row, default_kind="head"):
    _, kind = _detect_swap_col(row)
    return kind or default_kind


# ── Metadata: feats + treebank link ──────────────────────────────────────────

def _feats_summary(row, prefix, highlight_feat=None) -> str:
    """Collect this node's set morphological features (e.g. head_Number, head_Case)
    into a collapsed <details> list, one Feat=Val per line — same pattern as the
    "_features" columns in word_order/html/html_tree.py's example table, so a long
    feature set never overflows into neighboring cells. Column naming matches
    extract_node_features's f"{prefix}_{feat}" convention; excludes the many other
    {prefix}_-prefixed columns (form, idx, deprel, sibling-*, child-*, ...) since
    those never look like a bare CamelCase feature name.

    highlight_feat: the feature actually being swapped/compared (e.g. "Number"),
    bolded in the list so it's easy to spot among the node's other features."""
    pat = re.compile(rf"^{re.escape(prefix)}_([A-Z][a-zA-Z]*)$")
    pairs = []
    highlight_val = None
    for col in row.index:
        match = pat.match(col)
        if not match:
            continue
        val = row[col]
        if pd.isna(val) or val in (None, "None", "", "_missing"):
            continue
        feat = match.group(1)
        val_str = html_lib.escape(str(val))
        line = f"{feat}={val_str}"
        if highlight_feat and feat == highlight_feat:
            line = f'<strong class="swap-feat">{line}</strong>'
            highlight_val = val_str
        pairs.append((feat, line))

    if not pairs:
        return "&mdash;"

    pairs.sort(key=lambda p: p[0])
    items = "<br>".join(line for _, line in pairs)
    badge = (
        f' &middot; <span class="swap-feat-badge">{html_lib.escape(highlight_feat)}={highlight_val}</span>'
        if highlight_val is not None else ""
    )
    return f'<details><summary>{len(pairs)} feats{badge}</summary>{items}</details>'


def _treebank_link(row) -> str:
    """grew.fr query-link for a single example row — shared with word_order/
    viz_tree.py's build_treebank_links via word_order.utils.build_grew_link."""
    form_cols = [c for c in row.index if c.endswith("_form") and pd.notna(row.get(c))]
    form_values = [row[c] for c in form_cols]
    link = build_grew_link(row.get("treebank"), row.get("sent_id"), form_values)
    return link if link is not None else "&mdash;"


def _highlight_indices(row, kind, child_deprel=None) -> set:
    """0-based positions in `sen` of the agreement-relevant tokens: the swapped
    node (e.g. head) and, when known, its agreement partner (e.g. nsubj)."""
    idx = set()
    idx_col = f"{kind}_idx"
    if idx_col in row.index and pd.notna(row.get(idx_col)):
        idx.add(int(row[idx_col]) - 1)
    if child_deprel:
        child_idx_col = f"{child_deprel}_idx"
        if child_idx_col in row.index and pd.notna(row.get(child_idx_col)):
            idx.add(int(row[child_idx_col]) - 1)
    return idx


def _sentence_pair(row, treebank=None, child_deprel=None, default_kind="head"):
    """Return (before, after) HTML sentence strings for one example row, with
    agreement-relevant tokens wrapped in <strong>. `after` is None when the
    bucket has no swap_<kind> value (e.g. no_candidates).

    Spacing precedence: the row's own "no_space_after" column (saved alongside
    "sen" at extraction time, process_treebank.py's tree_no_space_after) if
    present, else a live treebank lookup by tree_idx (for data saved before
    "no_space_after" existed), else the punctuation heuristic."""
    sen = row.get("sen")
    if sen is None:
        return None, None
    sen = list(sen)

    swap_col, kind = _detect_swap_col(row)
    kind = kind or default_kind
    idx_col = f"{kind}_idx"
    highlight_idx = _highlight_indices(row, kind, child_deprel)

    swapped_tokens = None
    if swap_col is not None and idx_col in row.index and pd.notna(row.get(idx_col)):
        swapped_tokens = list(sen)
        swapped_tokens[int(row[idx_col]) - 1] = row[swap_col]

    nsa = row.get("no_space_after")
    if nsa is not None and len(nsa) == len(sen):
        before = _join_with_spacing(sen, nsa, highlight_idx)
        after = _join_with_spacing(swapped_tokens, nsa, highlight_idx) if swapped_tokens else None
        return before, after

    if treebank is not None and "tree_idx" in row.index and pd.notna(row.get("tree_idx")):
        try:
            tree = treebank[int(row["tree_idx"])]
            nsa = _no_space_afters(tree)
            if len(nsa) == len(sen):
                before = _join_with_spacing(sen, nsa, highlight_idx)
                after = _join_with_spacing(swapped_tokens, nsa, highlight_idx) if swapped_tokens else None
                return before, after
        except (IndexError, KeyError, TypeError):
            pass  # fall through to the heuristic below

    before = _fallback_join(sen, highlight_idx)
    after = _fallback_join(swapped_tokens, highlight_idx) if swapped_tokens else None
    return before, after


# ── Table rendering ───────────────────────────────────────────────────────────

def _fmt_cell(val, max_len=40):
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple, set)):
        val = ", ".join(map(str, val))
    text = str(val)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return html_lib.escape(text)


def _fmt_sentence(html_text):
    """`html_text` comes pre-escaped (with <strong> highlights) from _sentence_pair;
    do not re-escape or truncate by character, which would corrupt the tags."""
    return html_text if html_text is not None else "&mdash;"


def _diverse_sample(item_df: pd.DataFrame, max_examples: int, by: str = "feature_vals") -> pd.DataFrame:
    """Sample up to max_examples rows, covering as many distinct `by` values as
    possible first (e.g. both "SG -> PL" and "PL -> SG"), rather than just the
    first N rows which could all be a single, over-represented category."""
    if by not in item_df.columns or item_df[by].nunique(dropna=True) <= 1:
        return item_df.head(max_examples)

    groups = [g for _, g in item_df.groupby(by, sort=False)]
    picks = []
    round_idx = 0
    while len(picks) < max_examples and any(len(g) > round_idx for g in groups):
        for g in groups:
            if len(picks) >= max_examples:
                break
            if len(g) > round_idx:
                picks.append(g.iloc[round_idx])
        round_idx += 1

    return pd.DataFrame(picks) if picks else item_df.head(max_examples)


def _examples_table_html(item_df: pd.DataFrame, max_examples: int, treebank=None,
                          child_deprel=None, default_kind="head", swap_feature=None) -> str:
    """Small HTML table with up to max_examples sample rows from one diagnostics bucket,
    covering as many distinct feature_vals categories (e.g. SG -> PL vs PL -> SG) as
    the example budget allows."""
    if item_df is None or len(item_df) == 0:
        return '<p class="empty">no examples</p>'

    preferred = [c for c in item_df.columns if c.endswith("_form") or c in ("feature_vals", "alternatives")]
    forms = sorted(c for c in preferred if c.endswith("_form"))
    rest = [c for c in preferred if c not in forms]
    cols = (forms + rest)[:6] or list(item_df.columns[:6])

    has_sentence = "sen" in item_df.columns
    has_treebank = "treebank" in item_df.columns and "sent_id" in item_df.columns
    sample = _diverse_sample(item_df, max_examples)

    meta_cols = []
    if default_kind:
        meta_cols.append(f"{default_kind}_feats")
        meta_cols.append(f"{default_kind}_feats_after")
    if child_deprel:
        meta_cols.append(f"{child_deprel}_feats")
    if has_treebank:
        meta_cols.append("treebank")

    header_cols = (["before", "after"] if has_sentence else []) + cols + meta_cols
    header = "".join(f"<th>{html_lib.escape(c)}</th>" for c in header_cols)
    body_rows = ""
    for _, row in sample.iterrows():
        cells = ""
        if has_sentence:
            before, after = _sentence_pair(row, treebank=treebank, child_deprel=child_deprel,
                                            default_kind=default_kind)
            cells += f'<td class="sentence">{_fmt_sentence(before)}</td>'
            cells += f'<td class="sentence">{_fmt_sentence(after)}</td>'
        cells += "".join(f"<td>{_fmt_cell(row[c])}</td>" for c in cols)
        if default_kind:
            kind = _resolve_kind(row, default_kind)
            cells += f'<td class="feats">{_feats_summary(row, kind, highlight_feat=swap_feature)}</td>'
            cells += f'<td class="feats">{_feats_summary(row, f"after_{kind}", highlight_feat=swap_feature)}</td>'
        if child_deprel:
            cells += f'<td class="feats">{_feats_summary(row, child_deprel, highlight_feat=swap_feature)}</td>'
        if has_treebank:
            cells += f'<td class="treebank">{_treebank_link(row)}</td>'
        body_rows += f"<tr>{cells}</tr>"

    table = f'<table class="examples"><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>'
    return f'<div class="examples-scroll">{table}</div>'


# ── Flowchart page ────────────────────────────────────────────────────────────

def render_flowchart_html(diagnostic_dfs: dict, out_file: str, max_examples: int = 5, treebank=None,
                           child_deprel=None, default_kind="head", swap_feature=None) -> None:
    """Render an HTML flowchart of the process_item branch logic, with up to
    max_examples sample rows (before/after sentence + key columns) per outcome bucket.

    treebank: optional fallback, only used when a row's own "no_space_after" column
        (saved alongside "sen" since process_treebank.py's tree_no_space_after) is
        missing — i.e. for dt_df built before that column existed. A loaded treebank
        (list of conllu TokenLists) indexable by tree_idx, e.g. from
        word_order.process_treebank.load_treebank(lang, resource_dir,
        max_treebank_len=...) using the SAME max_treebank_len as when the dt_df was
        built. If neither is available, falls back further to a punctuation heuristic.
    child_deprel: the agreement partner's deprel (e.g. "nsubj"), used together with
        default_kind (e.g. "head") to bold both agreement-relevant tokens in every
        before/after sentence example, not just the ones that had a swap.
    swap_feature: the morphological feature actually being swapped/compared (e.g.
        "Number", parsed from swap_feat), highlighted within each node's feats list.
    """

    def outcome(item_type, label, color):
        item_df = diagnostic_dfs.get(item_type, pd.DataFrame())
        n = len(item_df)
        examples = _examples_table_html(item_df, max_examples, treebank=treebank,
                                         child_deprel=child_deprel, default_kind=default_kind,
                                         swap_feature=swap_feature)
        cls = "outcome" + (" outcome-empty" if n == 0 else "")
        return f'''<div class="{cls}" style="--dot:{color}">
      <div class="outcome-title"><span class="outcome-dot"></span>{html_lib.escape(label)}<span class="count">n={n}</span></div>
      <details><summary>examples (up to {max_examples})</summary>{examples}</details>
    </div>'''

    def conn(label_html, explain):
        return f'<div class="connector"><span>{label_html}</span><div class="explain">{explain}</div></div>'

    def node(text_html, explain=None, cls="decision"):
        explain_html = f'<div class="node-explain">{explain}</div>' if explain else ""
        return f'<div class="node {cls}">{text_html}{explain_html}</div>'

    body = f'''
  <div class="flow">
    {node("inflector.inflect(form, features)",
          "Try to re-inflect the word into a different grammatical value (e.g. singular &rarr; plural).",
          cls="start")}

    {conn("swap_forms is None", "No matching entry was found for this word at all &mdash; inflection wasn&rsquo;t even attempted.")}
    {outcome("no_candidates", "no_candidates", "#94a3b8")}

    {conn("swap_forms == [] (found, but no inflected candidates)", "The word was found, but no valid re-inflected form could be generated for it.")}
    {outcome("no_inflections", "no_inflections", "#94a3b8")}

    {conn("for each swap_form in swap_forms", "One or more candidate re-inflected forms came back; each one is checked in turn below.")}
    {node("swap_form == form ?", "Does the re-inflected word look exactly the same as the original, with no visible change?")}

    <div class="branch">
      <div class="branch-col">
        {conn("yes", "The spelling didn&rsquo;t change at all, so there&rsquo;s no usable swap here.")}
        {outcome("same_forms", "same_forms", "#64748b")}
      </div>
      <div class="branch-col">
        {conn("no &rarr; get_form_features(swap_form)", "The spelling changed &mdash; now check what grammatical value that new form actually carries.")}
        {node("feature_vals &cap; swap_feature_vals = &empty;<br>and UNDEFINED not in swap_feature_vals ?",
              "Does the new form&rsquo;s value genuinely differ from the original, and is it clearly defined (not ambiguous)?")}

        <div class="branch">
          <div class="branch-col">
            {conn("no, overlap &gt; 0", "The new form could still carry the original value too, so the swap didn&rsquo;t cleanly change anything.")}
            {outcome("same_features", "same_features", "#ef4444")}
          </div>
          <div class="branch-col">
            {conn("no, UNDEFINED present", "The new form&rsquo;s grammatical value couldn&rsquo;t be confidently determined.")}
            {outcome("undefined_features", "undefined_features", "#f59e0b")}
          </div>
          <div class="branch-col">
            {conn("yes", "The new form clearly and only carries the target value &mdash; a real change.")}
            {node("swap_feature_vals non-empty<br>and within max_num_of_pairs quota ?",
                  "Did we get a usable value for the new form, and is there still room left in the per-feature-pair balancing quota?")}

            <div class="branch">
              <div class="branch-col">
                {conn("no (empty result)", "No usable feature value came back for the new form.")}
                {outcome("undefined_features", "undefined_features (empty)", "#f59e0b")}
              </div>
              <div class="branch-col">
                {conn("no (quota exceeded)", "This feature combination already has enough examples elsewhere, so it&rsquo;s skipped to keep the dataset balanced.")}
                {outcome("extra_pairs", "extra_pairs", "#a78bfa")}
              </div>
              <div class="branch-col">
                {conn("yes &rarr; check subject overlap", "Looks like a valid swap &mdash; now make sure the subject itself doesn&rsquo;t already carry the new value.")}
                {node("child_features &cap; swap_feature_vals &ne; &empty; ?",
                      "Does the subject&rsquo;s own features already match what the verb was just swapped to?")}

                <div class="branch">
                  <div class="branch-col">
                    {conn("yes", "The subject could still agree with the new form by coincidence, so this isn&rsquo;t a clean disagreement case.")}
                    {outcome("ambiguous_subjects", "ambiguous_subjects", "#eab308")}
                  </div>
                  <div class="branch-col">
                    {conn("no", "No overlap &mdash; swapping the verb creates a genuine agreement mismatch with the subject.")}
                    {outcome("correct_swaps", "correct_swaps ✓", "#22c55e")}
                    {conn("+ if multiple swap_forms existed", "When more than one valid re-inflected spelling was found for this value, it&rsquo;s also logged here for visibility.")}
                    {outcome("multi_now_valid", "multi_now_valid", "#38bdf8")}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
'''

    page = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>create_pairs bucket flowchart</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}

  :root {{
    --bg: #f5f5f4;
    --surface: #ffffff;
    --surface-raised: #fafaf9;
    --border: #e7e5e4;
    --border-subtle: #f0efee;
    --text-primary: #1c1917;
    --text-secondary: #78716c;
    --text-tertiary: #a8a29e;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --font-ui: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius: 10px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  }}

  body {{
    font-family: var(--font-ui);
    background: var(--bg);
    color: var(--text-primary);
    padding: clamp(14px, 3vw, 32px);
  }}

  h1 {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
  p.subtitle {{
    color: var(--text-secondary); font-size: 12.5px; margin-bottom: 20px; max-width: 720px;
  }}

  .flow-wrap {{ overflow-x: auto; padding-bottom: 16px; width: 100%; }}
  .flow {{
    display: flex; flex-direction: column; align-items: center;
    width: 100%; max-width: 1400px; margin: 0 auto;
  }}

  .node {{
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow-sm); padding: 10px 16px; text-align: center;
    max-width: min(360px, 82vw); font-size: 13px; font-family: var(--font-mono);
  }}
  .node.decision {{ border-style: dashed; border-color: var(--text-tertiary); font-weight: 500; }}
  .node.start {{ font-weight: 600; }}
  .node-explain {{
    font-family: var(--font-ui); font-size: 10.5px; font-weight: 400; font-style: italic;
    color: var(--text-secondary); margin-top: 7px; line-height: 1.4;
  }}

  .connector {{
    display: flex; flex-direction: column; align-items: center; padding: 8px 0;
    text-align: center; max-width: min(280px, 82vw); gap: 4px;
  }}
  .connector::before {{ content: ""; width: 1px; height: 10px; background: var(--border); }}
  .connector span {{
    display: inline-flex; align-items: center; background: var(--surface-raised);
    border: 1px solid var(--border); border-radius: 20px; padding: 3px 10px;
    font-size: 10.5px; font-weight: 500; color: var(--text-secondary);
  }}
  .connector .explain {{
    font-family: var(--font-ui); font-size: 9.5px; font-style: italic;
    color: var(--text-tertiary); line-height: 1.35; max-width: 230px;
  }}

  .branch {{
    display: flex; flex-wrap: wrap; justify-content: center; gap: 24px;
    align-items: flex-start; border-top: 1px solid var(--border);
    padding-top: 16px; margin-top: 8px; width: 100%;
  }}
  .branch-col {{ display: flex; flex-direction: column; align-items: center; }}

  .outcome {{
    border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-sm);
    padding: 10px 14px; min-width: 190px; max-width: min(760px, 94vw);
    text-align: center; background: var(--surface); transition: box-shadow .15s;
  }}
  .outcome:hover {{ box-shadow: var(--shadow-md); }}
  .outcome:not(.outcome-empty) {{ background: #ecfdf5; }}
  .outcome-title {{
    font-weight: 600; font-size: 12.5px; display: flex; align-items: center;
    justify-content: center; gap: 6px;
  }}
  .outcome-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--dot, var(--accent)); flex-shrink: 0; }}
  .count {{
    color: var(--text-tertiary); font-family: var(--font-mono); font-size: 10.5px;
    font-weight: 400; margin-left: 2px;
  }}

  details {{ margin-top: 8px; text-align: left; }}
  summary {{
    cursor: pointer; color: var(--accent); font-size: 10.5px; font-weight: 500;
    list-style: none;
  }}
  summary::-webkit-details-marker {{ display: none; }}
  summary::before {{ content: "▸ "; }}
  details[open] summary::before {{ content: "▾ "; }}
  p.empty {{ color: var(--text-tertiary); font-size: 10.5px; margin-top: 6px; }}

  .examples-scroll {{ overflow-x: auto; max-width: 100%; }}
  table.examples {{ border-collapse: collapse; margin-top: 8px; font-size: 11.5px; }}
  table.examples th, table.examples td {{
    border-bottom: 1px solid var(--border-subtle); padding: 5px 10px; text-align: left;
    white-space: nowrap; font-family: var(--font-mono); vertical-align: top;
  }}
  table.examples th {{
    color: var(--text-tertiary); font-weight: 600; font-family: var(--font-ui);
    text-transform: uppercase; letter-spacing: .03em; font-size: 9.5px; white-space: nowrap;
  }}
  table.examples tbody tr:hover td {{ background: var(--accent-light); }}
  table.examples td.sentence {{ white-space: normal; max-width: 320px; line-height: 1.45; }}
  table.examples td.sentence strong {{ color: var(--accent); font-weight: 700; }}
  table.examples td.feats {{ white-space: normal; max-width: 190px; color: var(--text-secondary); }}
  table.examples td.feats strong.swap-feat {{ color: var(--accent); font-weight: 700; }}
  table.examples td.feats .swap-feat-badge {{ color: var(--accent); font-weight: 600; }}
  table.examples td.treebank a {{ color: var(--accent); text-decoration: none; }}
  table.examples td.treebank a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
  <h1>create_pairs bucket flowchart</h1>
  <p class="subtitle">Traces the branch logic in process_item() for this run. Each outcome box shows how many rows landed there and up to {max_examples} example rows (covering distinct feature_vals categories where possible), with before/after sentences{" (real SpaceAfter spacing)" if treebank is not None else " (heuristic spacing — no treebank passed in)"}.</p>
  <div class="flow-wrap">
{body}
  </div>
</body>
</html>'''

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(page)
