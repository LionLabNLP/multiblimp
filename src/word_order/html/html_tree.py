import json
import os
import re
import numpy as np
import pandas as pd

from ..utils import build_grew_link


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json(obj):
    return json.dumps(obj, cls=_NumpyEncoder)


def create_html(meta, node_samples, node_data, hex_colors, classes, div_id):
    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
      *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

      :root {{
        color-scheme: light;
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
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.04);
      }}
      /* Same palette as viz_deprel.py's html_deprel.py, for a consistent
         look across the deprel index and every tree page it links to. */
      :root[data-theme="dark"] {{
        color-scheme: dark;
        --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
        --border: #38332a; --border-subtle: #2a261e;
        --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
        --accent: #6ea8ff; --accent-light: #1c2a42;
      }}
      @media (prefers-color-scheme: dark) {{
        :root:not([data-theme="light"]) {{
          color-scheme: dark;
          --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
          --border: #38332a; --border-subtle: #2a261e;
          --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
          --accent: #6ea8ff; --accent-light: #1c2a42;
        }}
      }}

      body {{
        font-family: var(--font-ui);
        background: var(--bg);
        color: var(--text-primary);
        overflow: hidden;
        height: 100vh;
      }}

      .page {{
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
      }}

      .tree-panel {{
        flex: 0 0 auto;
        height: 58%;
        position: relative;
        overflow: hidden;
        background: var(--bg);
      }}

      .table-panel {{
        flex: 1 1 auto;
        overflow-y: auto;
        background: var(--surface);
        border-top: 1px solid var(--border);
        font-family: var(--font-ui);
        font-size: 14px;
        display: flex;
        flex-direction: column;
      }}

      .table-panel-header {{
        padding: 16px 20px 12px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
        position: sticky;
        top: 0;
        z-index: 10;
        text-align: center;
      }}

      .table-panel-header h2 {{
        font-size: 14px;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}

      .table-panel-body {{
        padding: 0 20px 20px;
        flex: 1;
      }}

      .splitter {{
        height: 5px;
        width: 100%;
        cursor: row-resize;
        background: var(--border);
        transition: background 0.15s;
        z-index: 20;
        flex-shrink: 0;
      }}

      .splitter:hover {{ background: #a8a29e; }}

      /* ── Info panel ── */
      .info-panel {{
        position: absolute;
        top: 16px;
        left: 16px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow-md);
        z-index: 100;
        width: 240px;
        overflow: hidden;
        font-family: var(--font-ui);
        transition: width 0.3s ease;
      }}

      .info-panel.minimized {{
        width: auto;
      }}

      .info-panel-header {{
        padding: 10px 14px;
        /* Deliberately NOT var(--text-primary): this is a fixed inverted
           dark bar (white text) regardless of theme, not a themed surface --
           --text-primary flips to a light colour in dark mode, which would
           turn this into a light bar with barely-visible white text. */
        background: #1c1917;
        color: white;
        position: relative;
      }}

      .minimize-button {{
        position: absolute;
        top: 10px;
        right: 10px;
        width: 20px;
        height: 20px;
        border: none;
        background: rgba(255,255,255,0.2);
        color: white;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        line-height: 1;
        transition: background 0.2s;
        padding: 0;
      }}

      .minimize-button:hover {{
        background: rgba(255,255,255,0.3);
      }}

      .minimize-button:active {{
        background: rgba(255,255,255,0.4);
      }}

      .info-panel-header .language-name {{
        font-size: 15px;
        font-weight: 600;
        line-height: 1.2;
      }}

      .info-panel-header .accuracy-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 5px;
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 500;
        color: rgba(255,255,255,0.9);
      }}

      .accuracy-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #4ade80;
      }}

      .info-panel-meta {{
        padding: 8px 14px;
        border-bottom: 1px solid var(--border-subtle);
        transition: opacity 0.3s ease, max-height 0.3s ease;
        max-height: 500px;
        overflow: hidden;
      }}

      .info-panel.minimized .info-panel-meta,
      .info-panel.minimized .info-panel-legend,
      .info-panel.minimized .toggle-row,
      .info-panel.minimized .branch-legend {{
        opacity: 0;
        max-height: 0;
        padding-top: 0;
        padding-bottom: 0;
        border: none;
      }}

      .meta-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
      }}

      .meta-key {{
        font-size: 11px;
        color: var(--text-tertiary);
        font-weight: 500;
      }}

      .meta-val {{
        font-size: 11px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-weight: 500;
      }}

      .info-panel-legend {{
        padding: 8px 14px 10px;
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 500px;
        overflow: hidden;
      }}

      .toggle-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-top: 1px solid var(--border-subtle);
        cursor: pointer;
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 100px;
        overflow: hidden;
      }}

      .legend-title {{
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-tertiary);
        margin-bottom: 6px;
      }}

      .legend-item {{
        display: grid;
        grid-template-columns: 9px 28px 28px 1fr;
        align-items: center;
        gap: 5px;
        padding: 3px 0;
      }}

      .legend-swatch {{
        width: 9px; height: 9px;
        border-radius: 2px;
        flex-shrink: 0;
      }}

      .legend-label {{
        font-size: 11px;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-weight: 500;
      }}

      .legend-count {{
        font-size: 10px;
        color: var(--text-tertiary);
        font-family: var(--font-mono);
        text-align: right;
      }}

      .legend-bar-track {{
        height: 6px;
        background: var(--border-subtle);
        border-radius: 3px;
        overflow: hidden;
      }}

      .legend-bar-fill {{
        height: 100%;
        border-radius: 3px;
      }}

      /* ── Branch legend ── */
      .branch-legend {{
        display: flex;
        gap: 10px;
        padding: 10px 14px;
        border-top: 1px solid var(--border-subtle);
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 100px;
        overflow: hidden;
      }}

      .branch-pill {{
        display: flex;
        align-items: center;
        gap: 5px;
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 500;
        color: var(--text-secondary);
      }}

      .branch-line {{
        width: 16px; height: 2.5px;
        border-radius: 2px;
      }}

      /* ── Per-node bar card (permanently visible, JS-positioned) ── */
      #node-card-layer {{
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
      }}

      .node-dist-card {{
        position: absolute;
        background: var(--surface);
        border: 1px solid rgba(0,0,0,0.13);
        border-radius: 4px;
        padding: 4px 5px 3px;
        display: flex;
        flex-direction: row;
        align-items: flex-end;
        gap: 3px;
        pointer-events: all;
        transform: translateX(-50%);
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        cursor: default;
        z-index: 50;
      }}

      .node-dist-card.expanded {{
        transform: translateX(-50%) scale(2);
        transform-origin: top center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
        z-index: 200;
      }}

      .node-dist-col {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1px;
        width: 19px;
      }}

      .node-dist-bar-wrap {{
        width: 14px;
        height: 30px;
        display: flex;
        align-items: flex-end;
      }}

      .node-dist-bar {{
        width: 100%;
        border-radius: 2px 2px 0 0;
        min-height: 1px;
      }}

      .node-dist-count {{
        font-size: 7px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        text-align: center;
        line-height: 1.2;
        white-space: nowrap;
      }}

      .node-dist-label {{
        font-size: 8.5px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        text-align: center;
        line-height: 1.2;
      }}

      /* ── Node summary row: decision path + distribution side by side ── */
      .node-summary-row {{
        display: flex;
        align-items: stretch;
        border-bottom: 1px solid var(--border);
      }}

      #node-path-panel {{
        flex: 1 1 auto;
        min-width: 0;
      }}

      #node-dist-panel {{
        flex: 1 1 auto;
        min-width: 0;
      }}

      /* ── Node distribution panel in table header ── */
      .node-dist-panel {{
        display: flex;
        flex-direction: row;
        align-items: flex-end;
        justify-content: center;
        gap: 6px;
        padding: 12px 16px;
        height: 100%;
        background: var(--surface-raised);
      }}

      .node-dist-panel-col {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
        width: 28px;
      }}

      .node-dist-panel-bar-wrap {{
        width: 28px;
        height: 60px;
        display: flex;
        align-items: flex-end;
      }}

      .node-dist-panel-bar {{
        width: 100%;
        border-radius: 3px 3px 0 0;
        min-height: 2px;
      }}

      .node-dist-panel-count {{
        font-size: 12px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        text-align: center;
        line-height: 1.3;
        white-space: nowrap;
      }}

      .node-dist-panel-label {{
        font-size: 9px;
        color: var(--text-secondary);
        font-family: var(--font-mono);
        text-align: center;
        line-height: 1.3;
      }}

      /* ── Decision path visualization ── */
      .decision-path-panel {{
        height: 100%;
        padding: 12px 20px;
        border-right: 1px solid var(--border-subtle);
        background: var(--surface);
        overflow-y: auto;
      }}

      .decision-path-title {{
        font-size: 10px;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
      }}

      .decision-path {{
        display: flex;
        flex-direction: column;
        gap: 4px;
      }}

      .path-step {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        line-height: 1.4;
      }}

      .path-arrow {{
        color: var(--text-tertiary);
        font-size: 10px;
        flex-shrink: 0;
      }}

      .path-condition {{
        font-family: var(--font-mono);
        color: var(--text-primary);
        font-weight: 500;
      }}

      .path-branch {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 600;
        margin-left: 6px;
      }}

      .path-branch.true {{
        background: rgba(22, 163, 74, 0.1);
        color: #16a34a;
      }}

      .path-branch.false {{
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
      }}

      /* ── Right panel table ── */
      .group-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 14px 0 6px;
      }}

      .group-swatch {{
        width: 10px; height: 10px;
        border-radius: 2px;
        flex-shrink: 0;
      }}

      .group-title {{
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary);
      }}

      .group-count {{
        margin-left: auto;
        font-size: 13px;
        color: var(--text-tertiary);
        font-family: var(--font-mono);
      }}

      .node-table {{
        border-collapse: collapse;
        width: 100%;
        table-layout: fixed;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
      }}

      .node-table thead th {{
        background: var(--surface-raised);
        color: var(--text-secondary);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 6px 10px;
        text-align: left;
        border-bottom: 1px solid var(--border);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      .node-table td {{
        padding: 5px 10px;
        border-bottom: 1px solid var(--border-subtle);
        vertical-align: top;
        color: var(--text-primary);
        font-size: 13px;
        line-height: 1.35;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
      }}

      .node-table td:has(details[open]) {{
        white-space: normal;
        overflow: visible;
        max-width: none;
      }}

      .node-table td details[open] {{
        word-break: break-all;
        white-space: normal;
      }}

      .node-table th.col-sen_str, .node-table td.col-sen_str {{
        width: 45%;
        white-space: normal;
        word-break: break-word;
      }}

      .node-table th.col-other, .node-table td.col-other {{
        width: auto;
        white-space: nowrap;
      }}

      .node-table tbody tr:last-child td {{ border-bottom: none; }}
      .node-table tbody tr:hover td {{ background: var(--accent-light); }}
      .node-table td strong {{ color: var(--accent); font-weight: 700; }}

      .node-table a {{
        color: var(--accent);
        text-decoration: none;
        font-weight: 500;
      }}

      .node-table a:hover {{ text-decoration: underline; }}

      .empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 100%;
        min-height: 300px;
        color: var(--text-tertiary);
        gap: 8px;
      }}

      .empty-state-icon {{ font-size: 32px; opacity: 0.4; }}
      .empty-state-text {{ font-size: 13px; font-weight: 500; }}
      .empty-state-sub {{ font-size: 12px; color: var(--text-tertiary); }}

      /* ── Toggle ── */
      .toggle-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        border-top: 1px solid var(--border-subtle);
        cursor: pointer;
        user-select: none;
      }}

      .toggle-row:hover {{ background: var(--surface-raised); }}

      .toggle-label {{
        font-size: 11px;
        font-weight: 500;
        color: var(--text-secondary);
      }}

      .toggle-switch {{
        width: 28px;
        height: 16px;
        border-radius: 8px;
        background: var(--accent);
        position: relative;
        transition: background 0.2s;
        flex-shrink: 0;
      }}

      .toggle-switch.off {{ background: var(--border); }}

      .toggle-switch::after {{
        content: "";
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: white;
        top: 2px;
        left: 14px;
        transition: left 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
      }}

      .toggle-switch.off::after {{ left: 2px; }}
    </style>

    <div class="page">

      <div class="info-panel" id="info-panel">
        <div class="info-panel-header">
          <button class="minimize-button" id="minimize-btn" onclick="toggleMinimize()" title="Minimize panel">−</button>
          <div class="language-name">{meta.get('Language', 'Decision Tree').replace('_', ' ')}</div>
          <div class="accuracy-badge">
            <span class="accuracy-dot"></span>
            {meta['accuracy']} accuracy
          </div>
        </div>
        {"<div class='info-panel-meta'>" + meta['rows'] + "</div>" if meta.get('rows') else ""}
        <div class="info-panel-legend">
          <div class="legend-title">Classes · dataset distribution</div>
          {meta['legend_items']}
        </div>
        <div class="toggle-row" id="dist-toggle-row" onclick="toggleDistributions()">
          <span class="toggle-label">Show node distributions</span>
          <span class="toggle-switch" id="dist-toggle-switch"></span>
        </div>
        <div class="branch-legend">
          <div class="branch-pill">
            <span class="branch-line" style="background:#dc2626;"></span>False
          </div>
          <div class="branch-pill">
            <span class="branch-line" style="background:#16a34a;"></span>True
          </div>
        </div>
      </div>

      <div class="tree-panel">
        <div id="node-card-layer"></div>
      </div>

      <div class="splitter" id="dragbar"></div>

      <div class="table-panel">
        <div class="table-panel-header">
          <h2 id="table-panel-title">Examples</h2>
        </div>
        <div class="node-summary-row">
          <div id="node-path-panel" style="display:none;"></div>
          <div id="node-dist-panel" style="display:none;"></div>
        </div>
        <div class="table-panel-body">
          <div class="empty-state" id="node-table">
            <div class="empty-state-icon">⬡</div>
            <div class="empty-state-text">No node selected</div>
            <div class="empty-state-sub">Click any node in the tree to explore examples</div>
          </div>
        </div>
      </div>
    </div>

    <script>
    const nodeSamples = {_json(node_samples)};
    const nodeData = {_json({str(k): v for k, v in node_data.items()})};
    const classColors = {_json(dict(zip([str(c) for c in classes], hex_colors)))};

    // ── Info panel minimize/expand ──
    function toggleMinimize() {{
      const panel = document.getElementById("info-panel");
      const btn = document.getElementById("minimize-btn");
      const isMinimized = panel.classList.toggle("minimized");
      btn.textContent = isMinimized ? "+" : "−";
      btn.title = isMinimized ? "Expand panel" : "Minimize panel";
    }}

    // ── Per-node distribution bar cards ──
    const cardLayer = document.getElementById("node-card-layer");

    function buildCard(nd) {{
      const dist = nd.dist;
      const maxCnt = Math.max(...dist.map(d => d.cnt), 1);
      const BAR_MAX_H = 30;

      const card = document.createElement("div");
      card.className = "node-dist-card";

      dist.forEach(d => {{
        const h = Math.max(1, Math.round((d.cnt / maxCnt) * BAR_MAX_H));
        const dim = d.cnt === 0 ? "opacity:0.18;" : "";
        const col = document.createElement("div");
        col.className = "node-dist-col";
        col.style.cssText = dim;
        // No count in the compact view — only bar + label
        col.innerHTML = `
          <div class="node-dist-bar-wrap">
            <div class="node-dist-bar" style="height:${{h}}px;background:${{d.color}};"></div>
          </div>
          <div class="node-dist-label">${{d.cls}}</div>`;
        card.appendChild(col);
      }});

      // Hover: expand to 2× and show counts
      card.addEventListener("mouseenter", function() {{
        card.classList.add("expanded");
        // Inject count elements
        card.querySelectorAll(".node-dist-col").forEach((col, idx) => {{
          const existing = col.querySelector(".node-dist-count");
          if (!existing) {{
            const countEl = document.createElement("div");
            countEl.className = "node-dist-count";
            countEl.textContent = dist[idx].cnt;
            // Insert between bar and label
            const label = col.querySelector(".node-dist-label");
            col.insertBefore(countEl, label);
          }}
        }});
      }});

      card.addEventListener("mouseleave", function() {{
        card.classList.remove("expanded");
        card.querySelectorAll(".node-dist-count").forEach(el => el.remove());
      }});

      return card;
    }}

    function toggleDistributions() {{
      showDistributions = !showDistributions;
      document.getElementById("dist-toggle-switch").classList.toggle("off", !showDistributions);
      renderNodeCards();
    }}

    // ── Example table ──
    function renderNodeOverview(nodeId) {{
      const container = document.getElementById("node-table");
      const titleEl = document.getElementById("table-panel-title");
      const distPanel = document.getElementById("node-dist-panel");
      const pathPanel = document.getElementById("node-path-panel");

      const groups = [];
      for (const [predictorValue, nodeMap] of Object.entries(nodeSamples)) {{
        const entry = nodeMap[nodeId];
        if (entry && entry.count > 0) {{
          groups.push({{
            predictor: predictorValue,
            rows: entry.rows,
            count: entry.count,
            total_count: entry.total_count
          }});
        }}
      }}

      const totalSamples = groups.reduce((s,g) => s + g.total_count, 0);
      titleEl.textContent = totalSamples > 0
        ? `Node ${{nodeId}} · ${{totalSamples}} samples`
        : `Node ${{nodeId}}`;

      // ── Build decision path from root to this node ──
      const path = [];
      let currentNode = nodeId;
      const nd = nodeData[String(nodeId)];

      console.log("Building path for node:", nodeId);
      console.log("Node data:", nd);

      while (nd && nodeData[String(currentNode)]) {{
        const current = nodeData[String(currentNode)];
        console.log("Current node:", currentNode, "Rule:", current.rule, "Parent:", current.parent);
        if (current.rule) {{
          path.unshift({{
            rule: current.rule[0],
            isTrueBranch: current.rule[1]
          }});
        }}
        if (current.parent === null || current.parent === undefined) break;
        currentNode = current.parent;
      }}

      console.log("Built path:", path);

      // Render decision path
      if (path.length > 0) {{
        let pathHtml = '<div class="decision-path-panel">';
        pathHtml += '<div class="decision-path-title">Decision Path</div>';
        pathHtml += '<div class="decision-path">';

        path.forEach((step, idx) => {{
          const branchClass = step.isTrueBranch ? 'true' : 'false';
          const branchLabel = step.isTrueBranch ? 'TRUE' : 'FALSE';
          const arrow = idx === 0 ? '▶' : '↳';

          pathHtml += `
            <div class="path-step">
              <span class="path-arrow">${{arrow}}</span>
              <span class="path-condition">${{step.rule}}</span>
              <span class="path-branch ${{branchClass}}">${{branchLabel}}</span>
            </div>`;
        }});

        pathHtml += '</div></div>';
        pathPanel.innerHTML = pathHtml;
        pathPanel.style.display = "block";
        console.log("Path panel HTML set");
      }} else {{
        pathPanel.style.display = "none";
        console.log("No path to display");
      }}

      // ── Distribution bar panel ──
      if (nd) {{
        const dist = nd.dist;
        const maxCnt = Math.max(...dist.map(d => d.cnt), 1);
        const BAR_MAX_H = 60;
        let panelHtml = '<div class="node-dist-panel">';
        dist.forEach(d => {{
          const h = Math.max(2, Math.round((d.cnt / maxCnt) * BAR_MAX_H));
          const dim = d.cnt === 0 ? "opacity:0.18;" : "";
          panelHtml += `
            <div class="node-dist-panel-col" style="${{dim}}">
              <div class="node-dist-panel-bar-wrap">
                <div class="node-dist-panel-bar" style="height:${{h}}px;background:${{d.color}};"></div>
              </div>
              <div class="node-dist-panel-count">${{d.cnt}}</div>
              <div class="node-dist-panel-label">${{d.cls}}</div>
            </div>`;
        }});
        panelHtml += "</div>";
        distPanel.innerHTML = panelHtml;
        distPanel.style.display = "block";
      }} else {{
        distPanel.style.display = "none";
      }}

      if (groups.length === 0) {{
        container.innerHTML = `<div class="empty-state">
          <div class="empty-state-icon">∅</div>
          <div class="empty-state-text">No examples for this node</div></div>`;
        return;
      }}

      groups.sort((a, b) => b.total_count - a.total_count);
      let html = "";

      for (const group of groups) {{
        const color = classColors[group.predictor] || "#888";
        const pct = totalSamples > 0 ? (group.total_count / totalSamples * 100).toFixed(1) : "0.0";
        const shownLabel = group.count < group.total_count
          ? `showing ${{group.count}} of ${{group.total_count}}`
          : `${{group.total_count}}`;
        html += `
          <div class="group-header">
            <span class="group-swatch" style="background:${{color}};"></span>
            <span class="group-title">${{group.predictor}}</span>
            <span class="group-count">${{shownLabel}} (${{pct}}%)</span>
          </div>`;

        if (!group.rows || group.rows.length === 0) {{
          html += `<p style="color:var(--text-tertiary);font-size:12px;padding:4px 0 12px;">No sample rows available.</p>`;
          continue;
        }}

        const cols = Object.keys(group.rows[0]);
        html += "<table class='node-table'><thead><tr>";
        cols.forEach(c => {{
          const cls = c === "sen_str" ? "col-sen_str" : "col-other";
          html += `<th class="${{cls}}">${{c}}</th>`;
        }});
        html += "</tr></thead><tbody>";
        group.rows.forEach(r => {{
          html += "<tr>";
          cols.forEach(c => {{
            const cls = c === "sen_str" ? "col-sen_str" : "col-other";
            const raw = r[c] == null ? "" : String(r[c]);
            let content;
            if (c.endsWith("_features")) {{
              content = "<details class='feat-details'><summary>features...</summary>" + raw.split(",").join("<br>") + "</details>";
            }} else {{
              content = raw;
            }}
            html += "<td class='" + cls + "'>" + content + "</td>";
          }});
          html += "</tr>";
        }});
        html += "</tbody></table>";
      }}

      container.innerHTML = html;
      linkFeatureDetails(container);
    }}

    // Feature lists for different roles in the same example row (e.g. subject
    // vs. verb) are separate <details> elements. Link them per-row so that
    // opening/closing one toggles the others in that row to match, letting
    // you compare both sides' features at a glance instead of expanding
    // each one by hand.
    function linkFeatureDetails(root) {{
      root.querySelectorAll("tr").forEach(tr => {{
        const detailsEls = tr.querySelectorAll("details.feat-details");
        if (detailsEls.length < 2) return;
        detailsEls.forEach(d => {{
          d.addEventListener("toggle", () => {{
            detailsEls.forEach(other => {{
              if (other !== d) other.open = d.open;
            }});
          }});
        }});
      }});
    }}

    // ── Plotly setup ──
    const plot = document.getElementById("{div_id}");
    document.querySelector(".tree-panel").prepend(plot);
    plot.style.width = "100%";
    plot.style.height = "100%";
    Plotly.Plots.resize(plot);

    function scaleTreeFont() {{
      const w = window.innerWidth;
      const newSize = w < 1400 ? 10 : w < 1920 ? 11 : 13;
      const annotations = plot.layout.annotations;
      for (let i = 0; i < annotations.length; i++) {{
        annotations[i].font.size = newSize;
      }}
      Plotly.relayout(plot, {{ annotations }});
    }}

    let showDistributions = true;

    function renderNodeCards() {{
      cardLayer.innerHTML = "";
      if (!showDistributions) return;

      // Plotly renders each annotation as <g class="annotation" data-index="N">
      // We read the annotation's stored name (= sklearn node_id) via plot.layout.annotations[N].name
      const panelRect = document.querySelector(".tree-panel").getBoundingClientRect();
      const annotGroups = plot.querySelectorAll("g.annotation");

      annotGroups.forEach((g, annotIdx) => {{
        // Get the sklearn node_id from the annotation's name field
        const ann = plot.layout.annotations[annotIdx];
        if (!ann) return;
        const nodeId = ann.name;
        if (nodeId === undefined) return;
        const nd = nodeData[nodeId];
        if (!nd) return;

        const bgRect = g.querySelector("rect.bg");
        if (!bgRect) return;

        const bbox = bgRect.getBoundingClientRect();
        const cx = bbox.left + bbox.width / 2 - panelRect.left;
        const top = bbox.bottom - panelRect.top;

        const card = buildCard(nd);
        card.style.left = cx + "px";
        card.style.top = top + "px";
        cardLayer.appendChild(card);
      }});
    }}

    scaleTreeFont();

    plot.on("plotly_afterplot", renderNodeCards);
    plot.on("plotly_relayout", renderNodeCards);
    window.addEventListener("resize", () => {{
      Plotly.Plots.resize(plot);
      scaleTreeFont();
      // cards re-render via plotly_afterplot triggered by resize
    }});

    plot.on("plotly_hover", function(e) {{
      const drag = plot.querySelector(".nsewdrag");
      if (drag) drag.style.cursor = "pointer";
    }});

    plot.on("plotly_unhover", function() {{
      const drag = plot.querySelector(".nsewdrag");
      if (drag) drag.style.cursor = "default";
    }});

    plot.on("plotly_click", function(e) {{
      const nodeId = e.points[0].customdata.node_id;
      renderNodeOverview(nodeId);
    }});

    // ── Draggable splitter ──
    const dragbar = document.getElementById("dragbar");
    const treePanel = document.querySelector(".tree-panel");
    let isDragging = false;

    dragbar.addEventListener("mousedown", (e) => {{
      isDragging = true;
      document.body.style.cursor = "row-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
    }});
    document.addEventListener("mouseup", () => {{
      isDragging = false;
      document.body.style.cursor = "default";
      document.body.style.userSelect = "";
    }});
    document.addEventListener("mousemove", (e) => {{
      if (!isDragging) return;
      const page = document.querySelector(".page");
      const rect = page.getBoundingClientRect();
      const pct = ((e.clientY - rect.top) / rect.height) * 100;
      if (pct > 20 && pct < 80) {{
        treePanel.style.height = pct + "%";
        treePanel.style.flex = "0 0 auto";
        Plotly.Plots.resize(plot);
      }}
    }});
    </script>
    """


# ── v2 tree page ──────────────────────────────────────────────────────────
# v2 integration (see the integration plan): tree + node-click samples
# browsing, ported from the "Minimal-Pair Attention Trace" mockup's already-
# verified HTML/CSS/JS, now driven by real per-language data instead of two
# hand-picked demo trees. Step 2 adds the generated-pairs section itself --
# correct_swaps_df=None (its default) emits an empty PAIRS, which the ported
# JS still renders gracefully (a real "N leaves kept, 0 shown" header, no
# pair cards) for callers/languages that don't have one.

_V2_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "tree_v2_template.html")

# The mockup's CSS only themes this fixed agreement-classifier vocabulary
# (--yes/--no/--unk custom properties); anything else falls back to its own
# real hex color from the v1 palette rather than a var(...) reference that
# doesn't exist. Covers every SVA/NPA condition this generator targets --
# multi-class word-order predictors (get_all_orders/only_show_real_orders)
# aren't in scope for v2 yet.
_V2_SEMANTIC_COLOR_VARS = {"yes": "var(--yes)", "no": "var(--no)", "unk": "var(--unk)"}

_TREEBANK_LINK_RE = re.compile(r"href='([^']*)'.*?>([^<]*)</a>")
_FORM_KEY_RE = re.compile(r"^(.+)_form$")


def _v2_class_color(cls, hex_color):
    return _V2_SEMANTIC_COLOR_VARS.get(str(cls).lower(), hex_color)


def _v2_parse_treebank_link(html_str):
    """v1's row dicts carry treebank_link as a ready-made <a href='...'>NAME</a>
    string (built by build_treebank_links) -- v2 wants the href and the link
    text as separate fields instead, so it can render its own compact arrow
    (title=name) rather than embedding v1's whole anchor tag verbatim."""
    m = _TREEBANK_LINK_RE.search(html_str or "")
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _v2_strong_to_b(s):
    # get_samples/_build_feat_columns marks the decisive feature with
    # <strong>...</strong> (matching v1's own on-page convention); the v2
    # template's featEntry() looks for <b> specifically instead.
    return s.replace("<strong>", "<b>").replace("</strong>", "</b>")


def _v2_detect_role_prefixes(node_samples, head_label):
    """The two role column prefixes (e.g. nsubj/head, or DET/HEAD) vary by
    condition and aren't passed to this function directly -- but every
    sample row get_samples produced carries exactly two "{prefix}_form"
    columns, so the first real row found tells us both prefixes without
    needing the caller to know them ahead of time. Falls back to a generic
    nsubj/head guess only if no samples exist at all for this tree."""
    for node_map in node_samples.values():
        for bucket in node_map.values():
            for row in bucket.get("rows", []):
                prefixes = [m.group(1) for m in (_FORM_KEY_RE.match(k) for k in row) if m]
                if len(prefixes) >= 2:
                    head_matches = [p for p in prefixes if p.lower() == str(head_label).lower()]
                    role_b = head_matches[0] if head_matches else prefixes[0]
                    role_a = next((p for p in prefixes if p != role_b), prefixes[-1])
                    return role_a, role_b
    return "nsubj", "head"


def _v2_node_samples(node_id, node_samples, role_a, role_b):
    """Flattens get_samples's {label: {node_id: {rows, count, total_count}}}
    into one plain list of rows for this node_id, each row carrying its own
    label inline -- the shape the v2 template's per-node sample table wants,
    since (unlike v1's client-side grouping) it renders one node's samples
    from a single already-flat array."""
    rows_out = []
    for label, node_map in node_samples.items():
        bucket = node_map.get(node_id, node_map.get(str(node_id)))
        if not bucket:
            continue
        for r in bucket.get("rows", []):
            href, name = _v2_parse_treebank_link(r.get("treebank_link", ""))
            rows_out.append({
                "sentence": r.get("sen_str", ""),
                "verb": r.get(f"{role_b}_form", ""),
                "verbFeats": [_v2_strong_to_b(v) for v in r.get(f"{role_b}_features", [])],
                "nsubj": r.get(f"{role_a}_form", ""),
                "nsubjFeats": [_v2_strong_to_b(v) for v in r.get(f"{role_a}_features", [])],
                "label": label,
                "treebankLink": href,
                "treebankName": name,
            })
    return rows_out


def _v2_layout(node_data):
    """Maps compute_tree_layout's coordinates (leaves at integer x 0..n-1,
    y = -depth -- relative DFS positions, never meant to be read as pixels)
    onto the template's pixel-space canvas (y increasing downward). Canvas
    size is derived from the actual tree's width/depth rather than the
    mockup's fixed 900x500, which only ever had to fit two hand-placed demo
    trees of five and seven nodes."""
    NODE_SPACING_X, NODE_SPACING_Y = 190, 170
    MARGIN_X, MARGIN_Y = 100, 60
    max_leaf_x = max((nd["x"] for nd in node_data.values()), default=0)
    max_depth = max((-nd["y"] for nd in node_data.values()), default=0)
    canvas_w = int(max_leaf_x * NODE_SPACING_X + 2 * MARGIN_X) or 900
    canvas_h = int(max_depth * NODE_SPACING_Y + 2 * MARGIN_Y) or 500
    screen = {
        i: (MARGIN_X + nd["x"] * NODE_SPACING_X, MARGIN_Y + (-nd["y"]) * NODE_SPACING_Y)
        for i, nd in node_data.items()
    }
    return screen, canvas_w, canvas_h


def _v2_leaf_qualifying_classes(dist):
    # Mirrors _finalize_tree_html's own leaf_label logic exactly (majority
    # class plus any other within 50% of it) -- duplicated here rather than
    # shared since v1 computes and discards this locally, never persisting
    # it into node_data.
    if not dist:
        return []
    sorted_d = sorted(dist, key=lambda d: d["cnt"], reverse=True)
    top_cnt = sorted_d[0]["cnt"]
    quals = [sorted_d[0]["cls"]]
    for d in sorted_d[1:]:
        if d["cnt"] > 0 and d["cnt"] >= 0.5 * top_cnt:
            quals.append(d["cls"])
    return quals


def _v2_raw_pair_role_prefixes(correct_swaps_df):
    """Like _v2_detect_role_prefixes, but scoped to correct_swaps.parquet's
    own raw columns -- unlike node_samples' rows, this dataframe never went
    through _relabel_head_columns, so its head-role prefix is always
    literally "head" (SVA/subj_aux) or "HEAD" (npa), matched case-
    insensitively rather than against head_label."""
    prefixes = [
        m.group(1) for m in (_FORM_KEY_RE.match(c) for c in correct_swaps_df.columns) if m
    ]
    head_matches = [p for p in prefixes if p.lower() == "head"]
    raw_role_b = head_matches[0] if head_matches else (prefixes[0] if prefixes else "head")
    raw_role_a = next((p for p in prefixes if p != raw_role_b), prefixes[-1] if len(prefixes) > 1 else "nsubj")
    return raw_role_a, raw_role_b


def _v2_pair_swap_role(row, raw_role_a, raw_role_b):
    """Which raw role this row's create_pairs reinflection actually swapped
    -- 'A' or 'B' -- detected from whichever swap_{role} column holds a real
    value on this row (mirrors create_pairs.py's own _detect_swap_col).
    Recomputed per row rather than assumed fixed for the whole tree, since a
    merged-direction bucket (e.g. npa's non-head pairs) can swap either role
    row by row."""
    for role, letter in ((raw_role_b, "B"), (raw_role_a, "A")):
        val = row.get(f"swap_{role}")
        if isinstance(val, str) and val:
            return letter
    return None


def _v2_pair_feat_list(row, prefix, decisive_feat):
    """{prefix}_{Feat}=val strings for one correct_swaps.parquet row's role,
    lemma/deprel/upos first then morphological feats sorted -- same
    inclusion/ordering convention as create_pairs.py's own _feats_summary
    and viz_tree.py's _build_feat_columns. Bolds the decisive feature with
    <b>, matching _v2_strong_to_b's node-sample markup, so the template's
    featEntry() finds it the same way regardless of source."""
    pat = re.compile(rf"^{re.escape(prefix)}_([A-Z][a-zA-Z]*(?:\[[a-z]+\])?)$")
    morph = []
    for col, val in row.items():
        m = pat.match(col)
        if not m or pd.isna(val) or val in (None, "None", "", "_missing", "nan"):
            continue
        feat = m.group(1)
        entry = f"{feat}={val}"
        if decisive_feat and feat == decisive_feat:
            entry = f"<b>{entry}</b>"
        morph.append((feat, entry))
    morph.sort(key=lambda p: p[0])

    extra = []
    for extra_feat, suffix in (("lemma", "lemma"), ("deprel", "deprel"), ("upos", "pos")):
        val = row.get(f"{prefix}_{suffix}")
        if pd.notna(val) and val not in (None, "None", "", "_missing", "_"):
            extra.append(f"{extra_feat}={val}")
    return extra + [e for _, e in morph]


def _v2_sentence_html(sen, highlight_idx):
    return " ".join(
        f"<strong>{tok}</strong>" if i in highlight_idx else str(tok)
        for i, tok in enumerate(sen)
    )


def _v2_pair_treebank_link(row):
    form_values = [v for c, v in row.items() if c.endswith("_form") and pd.notna(v)]
    link = build_grew_link(row.get("treebank"), row.get("sent_id"), form_values)
    return _v2_parse_treebank_link(link or "")


def _v2_pair_from_row(row, raw_role_a, raw_role_b, decisive_feat):
    """One T.PAIRS entry from a single correct_swaps.parquet row, or None if
    the row can't be turned into one (no sentence, no detectable swap role,
    or a missing token index)."""
    sen = row.get("sen")
    if sen is None:
        return None
    sen = list(sen)

    swap_role = _v2_pair_swap_role(row, raw_role_a, raw_role_b)
    if swap_role is None:
        return None
    swap_prefix = raw_role_b if swap_role == "B" else raw_role_a
    other_prefix = raw_role_a if swap_role == "B" else raw_role_b

    idx = row.get(f"{swap_prefix}_idx")
    if pd.isna(idx):
        return None
    other_idx = row.get(f"{other_prefix}_idx")

    # 1-indexed CoNLL-U token ids -- subtract 1 before indexing into `sen`
    # (see the memory note on this exact off-by-one, caught building the
    # mockup's own hand-picked pairs from this same parquet family).
    highlight_idx = {int(idx) - 1}
    if pd.notna(other_idx):
        highlight_idx.add(int(other_idx) - 1)

    swap_form = row.get(f"swap_{swap_prefix}")
    swapped_sen = list(sen)
    swapped_sen[int(idx) - 1] = swap_form

    href, name = _v2_pair_treebank_link(row)
    after_value = row.get(f"after_{swap_prefix}_{decisive_feat}") if decisive_feat else None

    return {
        "origSentence": _v2_sentence_html(sen, highlight_idx),
        "swapSentence": _v2_sentence_html(swapped_sen, highlight_idx),
        "nsubj": row.get(f"{raw_role_a}_form", ""),
        "verb": row.get(f"{raw_role_b}_form", ""),
        "nsubjFeats": _v2_pair_feat_list(row, raw_role_a, decisive_feat),
        "verbFeats": _v2_pair_feat_list(row, raw_role_b, decisive_feat),
        "ungrammatical": swap_form,
        "swapRole": swap_role,
        "afterValue": None if pd.isna(after_value) else str(after_value),
        "treebankLink": href,
        "treebankName": name,
    }


def _v2_build_pairs(correct_swaps_df, decisive_feat, max_per_leaf=15):
    """Builds T.PAIRS from create_pairs' own "correct_swaps" bucket, capped
    to max_per_leaf real rows per leaf (page-size budget, same idea as
    node samples' own max_rows) -- and the true per-leaf total, for
    entry["correctSwaps"] (the leaf badge's real pair count, independent of
    how many are actually materialized into PAIRS).

    Returns (pairs, total_by_leaf, swapped_role) -- swapped_role ('A'/'B',
    or None if no pairs) is the majority swap direction observed among the
    emitted pairs, used as T.swappedRole's tree-level fallback; each pair
    also carries its own swapRole, so a genuinely mixed-direction bucket
    (e.g. npa's merged non-head pairs) still renders correctly per pair.
    """
    if correct_swaps_df is None or len(correct_swaps_df) == 0 or "leaf_id" not in correct_swaps_df.columns:
        return [], {}, None

    raw_role_a, raw_role_b = _v2_raw_pair_role_prefixes(correct_swaps_df)

    valid = correct_swaps_df.dropna(subset=["leaf_id"])
    total_by_leaf = {
        str(int(k)): int(v) for k, v in valid["leaf_id"].value_counts().items()
    }
    sampled = valid.groupby("leaf_id", group_keys=False).head(max_per_leaf)

    pairs = []
    role_votes = {"A": 0, "B": 0}
    for row in sampled.to_dict("records"):
        pair = _v2_pair_from_row(row, raw_role_a, raw_role_b, decisive_feat)
        if pair is None:
            continue
        pair["leaf"] = str(int(row["leaf_id"]))
        pairs.append(pair)
        role_votes[pair["swapRole"]] += 1

    swapped_role = "A" if role_votes["A"] > role_votes["B"] else "B" if pairs else None
    return pairs, total_by_leaf, swapped_role


def _v2_build_nodes(node_data, screen, leaf_threshold, pair_counts=None):
    nodes = {}
    for i, nd in node_data.items():
        x, y = screen[i]
        is_leaf = nd["is_leaf"]
        branch = "root"
        if nd["parent"] is not None:
            _, is_true = nd["rule"]
            branch = "true" if is_true else "false"
        entry = {
            "x": round(x, 1), "y": round(y, 1),
            "branch": branch,
            "n": nd["n"], "H": round(nd["H"], 4),
            "leaf": is_leaf,
            "corr": nd.get("corr") or [],
            "dist": [
                {"cls": d["cls"], "cnt": d["cnt"], "color": _v2_class_color(d["cls"], d["color"])}
                for d in nd["dist"]
            ],
        }
        if is_leaf:
            quals = _v2_leaf_qualifying_classes(nd["dist"])
            entry["rule"] = "predict: " + (" / ".join(quals) if quals else "?")
            entry["predicted"] = " / ".join(quals)
            entry["majorityLabel"] = quals[0] if quals else None
            entry["keep"] = bool(leaf_threshold is not None and nd["H"] < leaf_threshold)
            # "kept" is the majority class's own count, not the leaf's full n.
            majority_cnt = next(
                (d["cnt"] for d in nd["dist"] if d["cls"] == entry["majorityLabel"]), 0
            )
            entry["kept"] = majority_cnt if entry["keep"] else 0
            entry["correctSwaps"] = (pair_counts or {}).get(str(i), 0)
        else:
            entry["rule"] = nd.get("own_rule") or ""
            entry["ruleFeat"] = None
            entry["ruleRole"] = None
            entry["ruleValue"] = None
        nodes[str(i)] = entry
    return nodes


def _v2_build_edges(node_data):
    edges = []
    for i, nd in node_data.items():
        if nd["parent"] is not None:
            _, is_true = nd["rule"]
            edges.append([nd["parent"], i, "true" if is_true else "false"])
    return edges


def _v2_build_paths(node_data):
    paths = {}
    for i in node_data:
        path, cur = [i], i
        while node_data[cur]["parent"] is not None:
            cur = node_data[cur]["parent"]
            path.append(cur)
        paths[str(i)] = list(reversed(path))
    return paths


def _v2_decisive_feat(predictor_display):
    # meta["Predictor"] (_display_predictor_var's output) is a string like
    # "head_nsubj_Number" or "Verb_nsubj_Number" -- the decisive agreement
    # feature is always its last underscore segment.
    if not predictor_display:
        return None
    return str(predictor_display).rsplit("_", 1)[-1]


def _v2_build_meta(meta, leaf_threshold):
    m = {
        "language": meta.get("Language") or meta.get("language") or "",
        "accuracy": meta.get("accuracy", ""),
        "baseEntropy": meta.get("base entropy", ""),
        "reducedEntropy": meta.get("reduced entropy", ""),
        "nodes": meta.get("Nodes", ""),
        "depth": meta.get("Depth", ""),
        "trainingSamples": meta.get("Training samples", ""),
        "predictor": meta.get("Predictor", ""),
    }
    if leaf_threshold is not None:
        m["keepThreshold"] = f"entropy < {leaf_threshold:g}"
        m["keepThresholdH"] = leaf_threshold
    return m


def _v2_build_classes(classes, hex_colors, root_dist_counts):
    return [
        {"cls": cls, "cnt": int(cnt), "color": _v2_class_color(cls, hexc)}
        for cls, hexc, cnt in zip(classes, hex_colors, root_dist_counts)
    ]


def create_html_v2(meta, node_samples, node_data, hex_colors, classes, root_dist_counts,
                    leaf_threshold=None, head_label="head", correct_swaps_df=None):
    screen, canvas_w, canvas_h = _v2_layout(node_data)
    decisive_feat = _v2_decisive_feat(meta.get("Predictor"))
    pairs, pair_counts, swapped_role = _v2_build_pairs(correct_swaps_df, decisive_feat)
    nodes = _v2_build_nodes(node_data, screen, leaf_threshold, pair_counts=pair_counts)
    edges = _v2_build_edges(node_data)
    paths = _v2_build_paths(node_data)
    role_a, role_b = _v2_detect_role_prefixes(node_samples, head_label)
    node_samples_v2 = {
        str(i): _v2_node_samples(i, node_samples, role_a, role_b) for i in node_data
    }
    tree_meta = _v2_build_meta(meta, leaf_threshold)
    tree_obj = {
        "title": f"{tree_meta['language']} · {meta.get('Predictor', '')}",
        "roleA": role_a.lower(),
        "roleB": role_b.lower(),
        "decisiveFeat": decisive_feat,
        # Tree-level fallback only -- each pair also carries its own
        # swapRole, since a merged-direction bucket can swap either role row
        # by row (see _v2_build_pairs). 'B' when there are no pairs at all
        # (no visible effect, matches SVA/subj_aux's always-head convention).
        "swappedRole": swapped_role or "B",
        "defaultNode": 0,
        "CANVAS_W": canvas_w,
        "CANVAS_H": canvas_h,
        "META": tree_meta,
        "CLASSES": _v2_build_classes(classes, hex_colors, root_dist_counts),
        "NODES": nodes,
        "EDGES": edges,
        "PATHS": paths,
        "NODE_SAMPLES": node_samples_v2,
        "PAIRS": pairs,
    }
    tree_js = (
        "const TREE_DATA = " + _json(tree_obj) + ";\n"
        "  const TREES = { main: TREE_DATA };\n"
        "  let currentTreeKey = 'main';\n"
    )
    with open(_V2_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    body = template.replace("__V2_TREE_DATA_JS__", tree_js)
    return "<!doctype html>\n<html lang=\"en\">\n" + body + "\n</html>\n"


def write_html_v2(node_samples, node_data, out_file, classes, hex_colors, root_dist_counts,
                   meta, leaf_threshold=None, head_label="head", correct_swaps_df=None):
    html = create_html_v2(
        meta, node_samples, node_data, hex_colors, classes, root_dist_counts,
        leaf_threshold=leaf_threshold, head_label=head_label, correct_swaps_df=correct_swaps_df,
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)


def write_placeholder_html(out_file, predictor_var, label, meta=None, sample_rows=None):
    # `label`: {value: count} across ALL training samples (build_placeholder_args
    # always passes the full value_counts, not a subsample) -- a single key means
    # every row genuinely shares one value; multiple keys means the tree wasn't
    # fit for some other reason (e.g. too few samples) despite a mixed label set.
    if len(label) == 1:
        heading = "Single label"
        description = (
            f"All training samples for <b>{predictor_var}</b> share one "
            "agreement label — no decision tree was needed."
        )
        (sole_label,) = label.keys()
        label_html = f'<div class="label">{sole_label}</div>'
    else:
        heading = "Too few samples to fit a tree"
        description = (
            f"Training samples for <b>{predictor_var}</b> don&rsquo;t share one "
            "agreement label, but there weren&rsquo;t enough of them to fit a "
            "decision tree — showing the raw label distribution instead."
        )
        label_html = "".join(
            f'<div class="label">{cls}: {n}</div>'
            for cls, n in sorted(label.items(), key=lambda kv: -kv[1])
        )

    # Machine-readable distribution for downstream consumers (e.g.
    # viz_deprel.py's overview table) to read without regex-scraping label_html.
    distribution_json = json.dumps(
        {str(k): int(n) for k, n in label.items()}, ensure_ascii=False
    )

    meta_rows = ""
    if meta:
        for key, val in meta.items():
            meta_rows += f'<div class="meta-row"><span class="meta-key">{key}</span><span class="meta-val">{val}</span></div>'

    table_html = ""
    if sample_rows:
        cols = list(sample_rows[0].keys())
        table_html += "<table class='ex-table'><thead><tr>"
        for c in cols:
            table_html += f"<th>{c}</th>"
        table_html += "</tr></thead><tbody>"
        for row in sample_rows:
            table_html += "<tr>"
            for c in cols:
                val = row[c]
                if val is None:
                    content = ""
                elif c.endswith("_features"):
                    # unpack list the same way as tree pages
                    if isinstance(val, list):
                        items = "<br>".join(
                            v for v in val if not str(v).endswith("=nan")
                        )
                    else:
                        # stored as string repr of list
                        import ast

                        try:
                            items = "<br>".join(
                                v
                                for v in ast.literal_eval(str(val))
                                if not str(v).endswith("=nan")
                            )
                        except Exception:
                            items = str(val)
                    content = (
                        f"<details class='feat-details'><summary>features...</summary>{items}</details>"
                    )
                else:
                    content = str(val)
                table_html += f"<td>{content}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>No decision tree to display</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f5f4; --surface: #ffffff; --surface-raised: #fafaf9;
      --border: #e7e5e4; --border-subtle: #f0efee;
      --text-primary: #1c1917; --text-secondary: #78716c; --text-tertiary: #a8a29e;
      --accent: #2563eb; --accent-light: #eff6ff;
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
      --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
      --border: #38332a; --border-subtle: #2a261e;
      --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
      --accent: #6ea8ff; --accent-light: #1c2a42;
    }}
    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        color-scheme: dark;
        --bg: #16140f; --surface: #221f19; --surface-raised: #1c1a15;
        --border: #38332a; --border-subtle: #2a261e;
        --text-primary: #f0ede6; --text-secondary: #a39a8a; --text-tertiary: #766d5e;
        --accent: #6ea8ff; --accent-light: #1c2a42;
      }}
    }}
    body {{ font-family: "DM Sans", sans-serif; margin: 0; background: var(--bg); color: var(--text-primary); }}
    /* Stacked (top: title/caption/label/distribution card, bottom: example
       items), mirroring the real tree pages' layout -- there, a compact
       floating info-panel card sits above the (much larger) tree diagram
       area, with the examples section below that. This placeholder has no
       diagram to show, so the card just sits in normal flow at the top
       instead of floating, but keeps the same "compact bordered/shadowed
       card, not a full-width band" treatment -- that's what makes the
       meta-rows' space-between layout read as compact key-value pairs
       rather than stretching key and value apart across the full page. */
    .layout {{ display: flex; flex-direction: column; min-height: 100vh; }}
    .sidebar {{
        max-width: 420px; margin: 1.5rem 1.5rem 0; background: var(--surface);
        border: 1px solid var(--border); border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem;
    }}
    h2 {{ font-size: 1.1rem; margin: 0; }}
    p  {{ color: var(--text-secondary); font-size: 0.875rem; margin: 0; line-height: 1.5; }}
    .label {{ display: inline-block; padding: 0.3rem 0.9rem;
              background: var(--bg); border-radius: 999px; font-weight: 600;
              font-size: 1rem; border: 1px solid var(--border);
              margin: 0 0.4rem 0.4rem 0; }}
    .labels {{ display: flex; flex-wrap: wrap; }}
    .meta {{ font-size: 0.8rem; }}
    .meta-row {{ display: flex; justify-content: space-between; padding: 0.2rem 0;
                 border-bottom: 1px solid var(--border); }}
    .meta-key {{ color: var(--text-tertiary); }}
    .meta-val {{ font-family: "JetBrains Mono", monospace; color: var(--text-secondary); }}
    .main {{ padding: 1.5rem; }}
    .main h3 {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.05em; color: var(--text-secondary); margin: 0 0 0.75rem; }}
    .ex-table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem;
                 border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    .ex-table th {{ background: var(--surface-raised); color: var(--text-secondary); font-size: 0.7rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.04em; padding: 6px 10px;
                    text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    .ex-table td {{ padding: 5px 10px; border-bottom: 1px solid var(--border-subtle); vertical-align: top;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }}
    .ex-table td:first-child {{ white-space: normal; word-break: break-word; max-width: none; }}
    .ex-table tbody tr:hover td {{ background: var(--accent-light); }}
    .ex-table a {{ color: var(--accent); text-decoration: none; font-weight: 500; }}
    .ex-table a:hover {{ text-decoration: underline; }}
    .ex-table tbody tr:last-child td {{ border-bottom: none; }}
    .ex-table td strong {{ color: var(--accent); font-weight: 700; }}
  </style>
</head>
<body>
  <script type="application/json" id="label-distribution">{distribution_json}</script>
  <div class="layout">
    <div class="sidebar">
      <div>
        <h2>{heading}</h2>
        <p>{description}</p>
      </div>
      <div class="labels">{label_html}</div>
      <div class="meta">{meta_rows}</div>
    </div>
    <div class="main">
      <h3>Example items (sample of {len(sample_rows) if sample_rows else 0})</h3>
      {table_html}
    </div>
  </div>
  <script>
    // Feature lists for different roles in the same example row (e.g. subject
    // vs. verb) are separate <details> elements. Link them per-row so that
    // opening/closing one toggles the others in that row to match.
    document.querySelectorAll(".ex-table tr").forEach(tr => {{
      const detailsEls = tr.querySelectorAll("details.feat-details");
      if (detailsEls.length < 2) return;
      detailsEls.forEach(d => {{
        d.addEventListener("toggle", () => {{
          detailsEls.forEach(other => {{
            if (other !== d) other.open = d.open;
          }});
        }});
      }});
    }});
  </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
