def create_html(
    rows_six, rows_binary, plot_data_six_json, plot_data_binary_json, trivial_note="",
    header_cells="", diagnostics_enabled=False,
    languages_six_json="[]", languages_binary_json="[]",
    leaf_threshold=None, agreement_label="Subject-Verb", head_role_label="head",
    subject_label="subject", nsubj_label="nsubj",
):
    """Dispatches to one of two full, independent page templates.

    diagnostics_enabled=False (the default, and the only path any pre-existing
    caller exercises) renders the exact original table-based page, unchanged.

    diagnostics_enabled=True renders the newer page with an expandable
    create_pairs diagnostics panel per language -- only meaningful for the
    SVA/agreement target, and only when the caller actually has per-language
    diagnostics JSON to give it (languages_six_json/languages_binary_json;
    rows_six/rows_binary/header_cells are unused on this path).

    leaf_threshold: the entropy cutoff N KEEP/the bucket breakdown were computed
    with (see sva_trees.create_pairs.create_pairs) -- shown as a header badge
    and interpolated into the N KEEP legend text on the diagnostics-enabled
    page only. None omits the badge (e.g. a caller that doesn't know it).

    agreement_label: prose label for the page subtitle on the diagnostics-enabled
    page (e.g. "Subject-Verb" / "Subject-Auxiliary"), so pages for different
    agreement targets don't all say "Subject-Verb".

    head_role_label: prose label standing in for the generic "head" role in the
    diagnostics-enabled page's "Dropped before fitting" stats/tooltips and legend
    text (e.g. "Verb" for SVA, "Aux" for subj_aux). "head" (default) is a no-op.

    subject_label: prose label standing in for the generic "subject" role in the
    "ambiguous subject" bucket's tooltips/legend text (e.g. "HEAD" for an NPA
    target_col whose fixed/comparison role is the NP head). "subject" (default)
    is a no-op.

    nsubj_label: prose label standing in for the generic "nsubj" role in the
    "{nsubj_label} unk" stat's label/tooltip (the fixed/comparison role's own
    missing-feature count -- same underlying role as subject_label, just a
    separate historical wording). "nsubj" (default) is a no-op.
    """
    if diagnostics_enabled:
        return _create_diagnostics_html(
            plot_data_six_json, plot_data_binary_json, trivial_note,
            languages_six_json, languages_binary_json, leaf_threshold,
            agreement_label, head_role_label, subject_label, nsubj_label,
        )
    return _create_classic_html(
        rows_six, rows_binary, plot_data_six_json, plot_data_binary_json,
        trivial_note, header_cells,
    )


def _create_classic_html(
    rows_six, rows_binary, plot_data_six_json, plot_data_binary_json, trivial_note="",
    header_cells="",
):
    return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MultiBLiMP v2 - Language Overview</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            :root {{
                --bg: #f5f5f4;
                --card: #ffffff;
                --text: #1c1917;
                --accent: #2563eb;
                --border: #e7e5e4;
                --hover: #f5f5f5;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                font-family: 'DM Sans', system-ui, sans-serif;
                background: var(--bg);
                color: var(--text);
                padding: 2rem;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: var(--card);
                padding: 2.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                border: 1px solid var(--border);
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 2rem;
            }}
            .title-section {{
                flex: 1;
            }}
            h1 {{
                margin: 0 0 0.5rem 0;
                font-size: 1.75rem;
                font-weight: 600;
                color: var(--text);
            }}
            .subtitle {{
                color: #78716c;
                font-size: 0.95rem;
            }}
            .controls {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}
            .back-btn {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.45rem 0.875rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: white;
                color: #57534e;
                font-family: 'DM Sans', system-ui, sans-serif;
                font-size: 0.875rem;
                font-weight: 500;
                text-decoration: none;
                transition: border-color 0.15s, color 0.15s, background 0.15s;
            }}
            .back-btn:hover {{
                border-color: var(--accent);
                color: var(--accent);
                background: #eff6ff;
                text-decoration: none;
            }}
            .back-btn svg {{
                flex-shrink: 0;
            }}
            .controls label {{
                font-size: 0.875rem;
                font-weight: 500;
                color: #57534e;
            }}
            select {{
                padding: 0.5rem 2rem 0.5rem 0.875rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: white;
                color: var(--text);
                font-family: 'DM Sans', system-ui, sans-serif;
                font-size: 0.875rem;
                font-weight: 500;
                cursor: pointer;
                appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2357534e' d='M6 8L2 4h8z'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 0.625rem center;
            }}
            select:hover {{
                border-color: var(--accent);
            }}
            select:focus {{
                outline: none;
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1.5rem;
            }}
            thead {{
                position: sticky;
                top: 0;
                z-index: 10;
                background: var(--hover);
                border-bottom: 2px solid var(--border);
            }}
            th {{
                padding: 0.875rem 1rem;
                text-align: left;
                font-weight: 600;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.025em;
                color: #57534e;
                cursor: pointer;
                user-select: none;
                position: relative;
                background: var(--hover);
            }}
            th:hover {{
                background: #ececeb;
            }}
            th.sortable::after {{
                content: '⇅';
                position: absolute;
                right: 0.5rem;
                opacity: 0.3;
                font-size: 0.75rem;
            }}
            th.sort-asc::after {{
                content: '↑';
                opacity: 1;
            }}
            th.sort-desc::after {{
                content: '↓';
                opacity: 1;
            }}
            td {{
                padding: 0.875rem 1rem;
                border-bottom: 1px solid var(--border);
            }}
            tbody tr:hover {{
                background: var(--hover);
            }}
            tbody tr:last-child td {{
                border-bottom: none;
            }}
            td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(7), td:nth-child(8) {{
                text-align: right;
                font-variant-numeric: tabular-nums;
            }}
            a {{
                color: var(--accent);
                text-decoration: none;
                font-weight: 500;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            #scatterPlot {{
                width: 100%;
                height: 500px;
                margin-bottom: 2rem;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: white;
            }}
            .trivial-note {{
                font-size: 0.85rem;
                color: #78716c;
                margin-bottom: 1rem;
                padding: 0.6rem 0.875rem;
                background: #fafaf9;
                border: 1px solid var(--border);
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title-section">
                    <h1>MultiBLiMP v2 - Language Overview</h1>
                    <div class="subtitle">Subject-Verb agreement prediction through decision trees</div>
                </div>
                <div class="controls">
                    <a href="../" class="back-btn">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10 12L6 8L10 4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Overview
                    </a>
                    <label for="entropyType">Entropy Type:</label>
                    <select id="entropyType">
                        <option value="six" selected>Six-class</option>
                        <option value="binary">Binary (majority vs. rest)</option>
                    </select>
                </div>
            </div>

            <div id="scatterPlot"></div>

            {trivial_note}

            <table id="dataTable">
                <thead>
                    <tr>
                        {header_cells}
                    </tr>
                </thead>
                <tbody id="tableBody">
                    {rows_six}
                </tbody>
            </table>
        </div>

        <script>
            // Store both data sets for table and plot
            const tableData = {{
                six: `{rows_six}`,
                binary: `{rows_binary}`
            }};

            const plotData = {{
                six: {plot_data_six_json},
                binary: {plot_data_binary_json}
            }};

            const table = document.getElementById('dataTable');
            const tbody = document.getElementById('tableBody');
            const headers = table.querySelectorAll('th.sortable');
            const entropySelect = document.getElementById('entropyType');

            let currentSort = {{ column: 0, ascending: true }};
            let currentEntropyType = 'six';

            // Initialize plot
            renderPlot(currentEntropyType);

            // Sort by language (alphabetically) on load
            sortTable(0, true, true);

            // Handle entropy type change
            entropySelect.addEventListener('change', (e) => {{
                currentEntropyType = e.target.value;
                tbody.innerHTML = tableData[currentEntropyType];
                renderPlot(currentEntropyType);
                // Re-apply current sort
                sortTable(currentSort.column, currentSort.ascending, true);
            }});

            // Render scatter plot
            function renderPlot(entropyType) {{
                const data = plotData[entropyType];

                const trace = {{
                    x: data.map(d => d.base),
                    y: data.map(d => d.reduced),
                    mode: 'markers',
                    type: 'scatter',
                    text: data.map(d => d.name),
                    customdata: data.map(d => [d.url, d.n_items]),
                    hovertemplate: '<b>%{{text}}</b><br>' +
                                'Base Entropy: %{{x:.3f}}<br>' +
                                'Reduced Entropy: %{{y:.3f}}<br>' +
                                'N Items: %{{customdata[1]:,}}<br>' +
                                '<extra></extra>',
                    marker: {{
                        size: data.map(d => d.n_items),
                        sizemode: 'area',
                        sizeref: 2 * Math.max(...data.map(d => d.n_items)) / (40 ** 2),
                        sizemin: 4,
                        color: data.map(d => d.color),
                        opacity: 0.6,
                        line: {{
                            color: '#1e40af',
                            width: 1
                        }}
                    }}
                }};

                const layout = {{
                    title: {{
                        text: 'Base Entropy vs. Reduced Entropy',
                        font: {{
                            family: 'DM Sans, system-ui, sans-serif',
                            size: 16,
                            color: '#1c1917'
                        }}
                    }},
                    xaxis: {{
                        title: 'Base Entropy',
                        gridcolor: '#e7e5e4',
                        zeroline: false
                    }},
                    yaxis: {{
                        title: 'Reduced Entropy',
                        gridcolor: '#e7e5e4',
                        zeroline: false
                    }},
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#ffffff',
                    font: {{
                        family: 'DM Sans, system-ui, sans-serif',
                        color: '#1c1917'
                    }},
                    hovermode: 'closest',
                    margin: {{ t: 50, r: 30, b: 50, l: 60 }}
                }};

                const config = {{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false
                }};

                Plotly.newPlot('scatterPlot', [trace], layout, config);

                // Add click event to navigate to language page
                document.getElementById('scatterPlot').on('plotly_click', function(data) {{
                    const url = data.points[0].customdata[0];
                    if (url) {{
                        window.location.href = url;
                    }}
                }});
            }}

            // Handle column header clicks
            headers.forEach(header => {{
                header.addEventListener('click', () => {{
                    const column = parseInt(header.dataset.column);
                    const ascending = currentSort.column === column ? !currentSort.ascending : true;
                    sortTable(column, ascending);
                }});
            }});

            function sortTable(column, ascending, skipToggle = false) {{
                const rows = Array.from(tbody.querySelectorAll('tr'));

                rows.sort((a, b) => {{
                    const aVal = parseFloat(a.children[column].dataset.sort) || a.children[column].dataset.sort;
                    const bVal = parseFloat(b.children[column].dataset.sort) || b.children[column].dataset.sort;

                    if (typeof aVal === 'string' && typeof bVal === 'string') {{
                        return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                    }}
                    return ascending ? aVal - bVal : bVal - aVal;
                }});

                rows.forEach(row => tbody.appendChild(row));

                if (!skipToggle) {{
                    // Update header indicators
                    headers.forEach(h => {{
                        h.classList.remove('sort-asc', 'sort-desc');
                    }});
                    headers[column].classList.add(ascending ? 'sort-asc' : 'sort-desc');

                    currentSort = {{ column, ascending }};
                }} else {{
                    headers[0].classList.add('sort-asc');
                }}
            }}
        </script>
    </body>
    </html>"""


def _create_diagnostics_html(
    plot_data_six_json, plot_data_binary_json, trivial_note,
    languages_six_json, languages_binary_json, leaf_threshold=None,
    agreement_label="Subject-Verb", head_role_label="head",
    subject_label="subject", nsubj_label="nsubj",
):
    """Language-overview page for the SVA/agreement pipeline: same scatter
    plot, six/binary toggle and sortable columns as the classic page, plus an
    expandable create_pairs diagnostics panel per language. Row markup is
    built client-side from languages_six_json/languages_binary_json (one
    object per language, matching sva_trees.diagnostics.diagnostics_row_to_json's
    "diag" shape plus the metrics fields already computed for the classic
    page) rather than server-rendered, so colours -- which read CSS custom
    properties for light/dark theming -- and sorting can both work without a
    server round-trip.
    """
    threshold_badge = (
        f'<span class="threshold-badge" title="The entropy cutoff N KEEP and the '
        f'bucket breakdown were computed with -- a row is only “kept” if its '
        f'leaf entropy is below this.">Keep threshold: entropy &lt; {leaf_threshold:g}</span>'
        if leaf_threshold is not None else ""
    )
    threshold_text = f"entropy &lt; {leaf_threshold:g}" if leaf_threshold is not None else "the run's leaf-confidence filter"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MultiBLiMP v2 - Language Overview</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f5f5f4;
            --card: #ffffff;
            --text: #1c1917;
            --text-muted: #78716c;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --border: #e7e5e4;
            --hover: #f5f5f5;
            --detail-bg: #fafaf9;
            --detail-border: #e7e5e4;
            --warn: #d97706;
            --font-body: 'DM Sans', system-ui, sans-serif;
            /* percentage colour scale: vermilion -> violet -> teal. A single
               steady hue rotation from 358deg down to 176deg (the "short
               way", through magenta/violet/blue), never wrapping the other
               way through orange/yellow/green. Red only ever appears at the
               very 0% endpoint, never as a colour the gradient sweeps
               through on its way elsewhere; green never appears at all. */
            --pct-lo-h: 358; --pct-lo-s: 75%; --pct-lo-l: 48%;
            --pct-hi-h: 176; --pct-hi-s: 55%; --pct-hi-l: 36%;
            /* Same rotation's 25/50/75% waypoints, only needed so the small
               static legend bar (a plain CSS linear-gradient, which
               interpolates colour stops through RGB, not around the hue
               wheel) can approximate the true polar path pctColor()
               computes per-cell below. */
            --pct-q1-h: 312;  --pct-q1-s: 70%; --pct-q1-l: 45%;
            --pct-mid-h: 267; --pct-mid-s: 65%; --pct-mid-l: 42%;
            --pct-q3-h: 222;  --pct-q3-s: 60%; --pct-q3-l: 39%;
        }}
        :root[data-theme="dark"] {{
            color-scheme: dark;
            --bg: #16140f; --card: #221f19; --text: #f0ede6; --text-muted: #a39a8a;
            --accent: #6ea8ff; --accent-soft: #1c2a42; --border: #38332a; --hover: #2a261e;
            --detail-bg: #1c1a15; --detail-border: #38332a; --warn: #fbbf24;
            --pct-lo-l: 58%; --pct-hi-l: 56%;
            --pct-q1-l: 57%; --pct-mid-l: 57%; --pct-q3-l: 57%;
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                color-scheme: dark;
                --bg: #16140f; --card: #221f19; --text: #f0ede6; --text-muted: #a39a8a;
                --accent: #6ea8ff; --accent-soft: #1c2a42; --border: #38332a; --hover: #2a261e;
                --detail-bg: #1c1a15; --detail-border: #38332a; --warn: #fbbf24;
                --pct-lo-l: 58%; --pct-hi-l: 56%;
                --pct-q1-l: 57%; --pct-mid-l: 57%; --pct-q3-l: 57%;
            }}
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            min-height: 100vh;
            font-family: var(--font-body);
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card);
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.75rem;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .title-section {{ flex: 1; min-width: 260px; }}
        h1 {{ margin: 0 0 0.5rem 0; font-size: 1.75rem; font-weight: 600; text-wrap: balance; }}
        .subtitle {{ color: var(--text-muted); font-size: 0.92rem; line-height: 1.5; max-width: 62ch; }}
        .top-controls {{ display: flex; align-items: center; gap: 0.75rem; flex-shrink: 0; padding-top: 0.25rem; }}
        .back-btn {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.45rem 0.875rem; border: 1px solid var(--border); border-radius: 6px;
            background: var(--card); color: var(--text-muted);
            font-family: inherit; font-size: 0.875rem; font-weight: 500;
            text-decoration: none; transition: border-color 0.15s, color 0.15s, background 0.15s;
        }}
        .back-btn:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); text-decoration: none; }}
        .top-controls label {{ font-size: 0.875rem; font-weight: 500; color: var(--text-muted); }}
        select {{
            padding: 0.5rem 2rem 0.5rem 0.875rem; border: 1px solid var(--border); border-radius: 6px;
            background: var(--card); color: var(--text); font-family: inherit;
            font-size: 0.875rem; font-weight: 500; cursor: pointer; appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2357534e' d='M6 8L2 4h8z'/%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 0.625rem center;
        }}
        select:hover {{ border-color: var(--accent); }}
        select:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }}

        #scatterPlot {{
            width: 100%; height: 500px; margin-bottom: 1.5rem;
            border: 1px solid var(--border); border-radius: 8px; background: var(--card);
        }}
        .trivial-note {{
            font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.25rem;
            padding: 0.6rem 0.875rem; background: var(--detail-bg); border: 1px solid var(--border); border-radius: 6px;
        }}

        .legend-row {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }}
        .threshold-badge {{
            font-size: 0.78rem; font-weight: 500; color: var(--text-muted);
            background: var(--detail-bg); border: 1px solid var(--detail-border);
            border-radius: 999px; padding: 0.3rem 0.75rem; cursor: default;
        }}
        .controls {{ display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 1.25rem; }}
        .btn {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.5rem 0.9rem; border: 1px solid var(--border); border-radius: 6px;
            background: var(--card); color: var(--text-muted); font-family: inherit;
            font-size: 0.82rem; font-weight: 500; cursor: pointer;
            transition: border-color .15s, color .15s, background .15s;
        }}
        .btn:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }}
        .toggle-btn.active {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }}
        .toggle-btn.active::before {{
            content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
            background: var(--accent); margin-right: 0.35rem;
        }}

        /* legend modal */
        .modal-overlay {{
            display: none; position: fixed; inset: 0; background: rgba(15, 13, 9, 0.5);
            align-items: center; justify-content: center; padding: 2rem; z-index: 50;
        }}
        .modal-overlay.open {{ display: flex; }}
        .modal {{
            background: var(--card); border: 1px solid var(--border); border-radius: 12px;
            width: 100%; max-width: 660px; max-height: 82vh; overflow-y: auto;
            padding: 1.75rem 2rem 2rem; box-shadow: 0 16px 48px rgba(0,0,0,0.28);
        }}
        .modal-head {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.4rem; gap: 1rem; }}
        .modal-head h2 {{ margin: 0; font-size: 1.15rem; font-weight: 600; }}
        .modal-head p {{ margin: 0.3rem 0 0; font-size: 0.82rem; color: var(--text-muted); }}
        .modal-close {{
            background: none; border: 1px solid var(--border); border-radius: 6px;
            width: 1.9rem; height: 1.9rem; flex-shrink: 0; display: inline-flex;
            align-items: center; justify-content: center; cursor: pointer; color: var(--text-muted);
            transition: color .15s, border-color .15s, background .15s;
        }}
        .modal-close:hover {{ color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }}
        .modal.modal-wide {{ max-width: 1100px; }}

        /* examples modal: the pre-rendered HTML from sva_trees.create_pairs.
           bucket_examples_html (same markup/classes the former standalone
           flowchart page used) is dropped in as-is, so these rules mirror
           that page's example-table CSS, translated onto this page's tokens. */
        .examples-scroll {{ overflow-x: auto; max-width: 100%; }}
        .examples-scroll table.examples {{
            border-collapse: collapse; margin-top: 0.4rem; font-size: 0.78rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }}
        .examples-scroll table.examples th, .examples-scroll table.examples td {{
            border-bottom: 1px solid var(--detail-border); padding: 0.35rem 0.7rem;
            text-align: left; white-space: nowrap; vertical-align: top;
        }}
        .examples-scroll table.examples th {{
            color: var(--text-muted); font-weight: 600; font-family: var(--font-body);
            text-transform: uppercase; letter-spacing: 0.03em; font-size: 0.66rem; white-space: nowrap;
        }}
        .examples-scroll table.examples tbody tr:hover td {{ background: var(--hover); }}
        .examples-scroll table.examples td.sentence {{ white-space: normal; max-width: 320px; line-height: 1.45; }}
        .examples-scroll table.examples td.sentence strong {{ color: var(--accent); font-weight: 700; }}
        .examples-scroll table.examples td.feats {{ white-space: normal; max-width: 190px; color: var(--text-muted); }}
        .examples-scroll table.examples td.feats strong.swap-feat {{ color: var(--accent); font-weight: 700; }}
        .examples-scroll table.examples td.feats .swap-feat-badge {{ color: var(--accent); font-weight: 600; }}
        .examples-scroll table.examples td.treebank a {{ color: var(--accent); text-decoration: none; }}
        .examples-scroll table.examples td.treebank a:hover {{ text-decoration: underline; }}
        .examples-scroll table.examples details {{ margin: 0; }}
        .examples-scroll table.examples summary {{ cursor: pointer; color: var(--accent); font-size: 0.72rem; list-style: none; }}
        .examples-scroll table.examples summary::-webkit-details-marker {{ display: none; }}
        .examples-scroll table.examples summary::before {{ content: "▸ "; }}
        .examples-scroll table.examples details[open] summary::before {{ content: "▾ "; }}
        p.empty {{ color: var(--text-muted); font-size: 0.78rem; margin: 0.4rem 0 0; }}

        .legend-section {{ margin-bottom: 1.4rem; }}
        .legend-section:last-child {{ margin-bottom: 0; }}
        .legend-section h3 {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);
            margin: 0 0 0.65rem; font-weight: 600; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
        }}
        dl.legend-list {{ margin: 0; display: flex; flex-direction: column; gap: 0.55rem; }}
        dl.legend-list div {{ display: flex; gap: 1rem; align-items: baseline; }}
        dl.legend-list dt {{ flex: 0 0 152px; font-weight: 600; font-size: 0.82rem; color: var(--text); }}
        dl.legend-list dd {{ margin: 0; font-size: 0.82rem; color: var(--text-muted); line-height: 1.45; }}
        .ratio-scroll {{ overflow-x: auto; border: 1px solid var(--detail-border); border-radius: 8px; }}
        table.ratio-table {{ border-collapse: collapse; width: 100%; min-width: 780px; font-size: 0.78rem; }}
        table.ratio-table th, table.ratio-table td {{
            padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--detail-border);
            text-align: left; vertical-align: top; line-height: 1.4;
        }}
        table.ratio-table thead th {{
            font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted);
            font-weight: 600; background: var(--hover); border-bottom: 1px solid var(--border); white-space: nowrap;
        }}
        table.ratio-table tbody tr:last-child td {{ border-bottom: none; }}
        table.ratio-table code {{
            background: var(--hover); border: 1px solid var(--border); border-radius: 4px;
            padding: 0.05rem 0.3rem; font-size: 0.72rem; white-space: nowrap;
        }}
        table.ratio-table td:nth-child(4) {{ color: var(--text-muted); }}

        .legend-group {{ display: flex; align-items: center; gap: 0.9rem; }}
        .scale-legend {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.74rem; color: var(--text-muted); }}
        .scale-legend .legend-title {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 0.15rem; }}
        .scale-legend .bar {{
            width: 68px; height: 8px; border-radius: 4px; border: 1px solid var(--border);
            background: linear-gradient(to right,
              hsl(var(--pct-lo-h) var(--pct-lo-s) var(--pct-lo-l)) 0%,
              hsl(var(--pct-q1-h) var(--pct-q1-s) var(--pct-q1-l)) 25%,
              hsl(var(--pct-mid-h) var(--pct-mid-s) var(--pct-mid-l)) 50%,
              hsl(var(--pct-q3-h) var(--pct-q3-s) var(--pct-q3-l)) 75%,
              hsl(var(--pct-hi-h) var(--pct-hi-s) var(--pct-hi-l)) 100%);
        }}
        .scale-legend.inverted .bar {{
            background: linear-gradient(to right,
              hsl(var(--pct-hi-h) var(--pct-hi-s) var(--pct-hi-l)) 0%,
              hsl(var(--pct-q3-h) var(--pct-q3-s) var(--pct-q3-l)) 25%,
              hsl(var(--pct-mid-h) var(--pct-mid-s) var(--pct-mid-l)) 50%,
              hsl(var(--pct-q1-h) var(--pct-q1-s) var(--pct-q1-l)) 75%,
              hsl(var(--pct-lo-h) var(--pct-lo-s) var(--pct-lo-l)) 100%);
        }}

        .table-scroll {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }}
        .grid-table {{ min-width: 760px; }}
        /* Pinned to the viewport bottom (not the table's own bottom edge) so
           horizontal scrolling stays reachable without first scrolling all
           the way down a tall page -- e.g. once several rows have their
           Results/Diagnostics panel expanded. left/width are set by JS to
           match #tableScroll's own rect (see refreshFixedHscroll) -- NOT
           left/right:0, which would span the full viewport and usually end
           up wider than the table's actual content, leaving nothing for
           this bar itself to scroll even while the real table overflows.
           Hidden by default so it never flashes before JS decides the table
           actually overflows horizontally. */
        .fixed-hscroll {{
            display: none; position: fixed; bottom: 0;
            height: 14px; overflow-x: auto; overflow-y: hidden;
            background: var(--card); border-top: 1px solid var(--border);
            z-index: 40;
        }}
        .fixed-hscroll #fixedHscrollInner {{ height: 1px; }}
        .row {{ display: flex; align-items: center; width: max-content; min-width: 100%; }}
        .thead-row {{ background: var(--hover); border-bottom: 2px solid var(--border); }}
        .cell {{
            padding: 0.8rem 0.9rem; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--text-muted); font-weight: 600; flex: 0 0 auto; line-height: 1.25;
        }}
        .thead-row .cell {{ white-space: normal; cursor: pointer; user-select: none; position: relative; }}
        .thead-row .cell.sortable:hover {{ color: var(--accent); }}
        .thead-row .cell.sortable::after {{ content: '⇅'; margin-left: 0.3rem; opacity: 0.35; font-size: 0.7rem; }}
        .thead-row .cell.sort-asc::after {{ content: '↑'; opacity: 1; }}
        .thead-row .cell.sort-desc::after {{ content: '↓'; opacity: 1; }}
        .c-chev  {{ width: 2.25rem; padding-right: 0; }}
        .c-lang  {{ width: 9rem; overflow: hidden; text-overflow: ellipsis; }}
        .c-dist  {{ width: 6.5rem; }}
        .c-base  {{ width: 6.75rem; text-align: right; }}
        .c-reduced {{ width: 7.25rem; text-align: right; }}
        .c-delta {{ width: 5.5rem; text-align: right; }}
        .c-acc   {{ width: 5.25rem; text-align: right; }}
        .c-raw   {{ width: 5.25rem; text-align: right; }}
        .c-keep  {{ width: 5.25rem; text-align: right; }}
        .c-pairs {{ width: 5.25rem; text-align: right; }}

        .lang-row {{ border-bottom: 1px solid var(--border); cursor: pointer; transition: background .1s; }}
        .lang-row:hover {{ background: var(--hover); }}
        .lang-row.is-open {{ background: var(--accent-soft); }}
        .lang-row .cell {{
            padding: 0.75rem 0.9rem; font-size: 0.88rem; text-transform: none; letter-spacing: 0;
            color: var(--text); font-weight: 400; font-variant-numeric: tabular-nums; white-space: nowrap;
        }}
        .lang-row .c-base, .lang-row .c-reduced, .lang-row .c-delta,
        .lang-row .c-acc, .lang-row .c-raw, .lang-row .c-keep, .lang-row .c-pairs {{ text-align: right; }}
        .lang-name {{ font-weight: 600 !important; }}
        .lang-name a {{ color: var(--accent); text-decoration: none; }}
        .lang-name a:hover {{ text-decoration: underline; }}
        .dist-bar {{
            display: flex; width: 5.5rem; height: 8px; border-radius: 4px;
            overflow: hidden; background: var(--border);
        }}
        .dist-bar span {{ height: 100%; }}
        .dist-bar span:not(:last-child) {{ border-right: 1px solid rgba(0, 0, 0, 0.65); }}

        .chevron {{
            display: inline-flex; width: 1.4rem; height: 1.4rem; align-items: center; justify-content: center;
            border-radius: 5px; color: var(--text-muted); transition: transform .18s ease, background .15s, color .15s;
        }}
        /* Scoped to .c-chev specifically, not a bare ".chevron" descendant
           match -- .bucket-chevron-btn's icon is ALSO just a ".chevron"
           element nested inside the same .lang-row, so an unscoped rule here
           would rotate/recolour it too whenever Results opens, even though
           only .c-chev's own chevron should react to "is-open". */
        .lang-row:hover .c-chev .chevron {{ color: var(--accent); background: var(--card); }}
        .lang-row.is-open .c-chev .chevron {{ transform: rotate(90deg); color: var(--accent); }}

        .c-bchev {{ width: 1.75rem; padding: 0.75rem 0 0.75rem 0.4rem; display: flex; justify-content: center; }}
        .bucket-chevron-btn {{ background: none; border: none; padding: 0; cursor: pointer; display: inline-flex; }}
        .bucket-chevron-btn .chevron {{ color: var(--text-muted); }}
        .lang-row:hover .bucket-chevron-btn .chevron {{ color: var(--accent); background: var(--card); }}
        .lang-row.buckets-open .bucket-chevron-btn .chevron {{ transform: rotate(90deg); color: var(--accent); }}

        .bucket-zone {{
            display: flex; align-items: stretch; gap: 0; max-width: 0; overflow: hidden; opacity: 0;
            transition: max-width .32s ease, opacity .22s ease, margin .32s ease; flex: 0 0 auto;
            border-left: 1px dashed var(--border); border-right: 1px dashed var(--border);
        }}
        .buckets-open .bucket-zone {{ max-width: 900px; opacity: 1; margin: 0 0.5rem; }}
        .bucket-item {{
            /* justify-content: flex-start (not center) so label/value/count
               always start at the same offset in every item, regardless of
               whether that item also has an "examples" button below --
               .bucket-zone's align-items:stretch already equalizes item
               heights, so centering each item's own (variable-length)
               content is what was throwing the value/count lines out of
               alignment with their neighbours. */
            display: flex; flex-direction: column; align-items: flex-end; justify-content: flex-start;
            gap: 0.15rem; padding: 0.75rem 0.75rem; width: 6rem;
        }}
        .bucket-item.no-data {{ width: auto; color: var(--text-muted); font-size: 0.78rem; white-space: nowrap; }}
        .bucket-item .b-label {{
            /* Fixed 2-line reservation (min-height + flex-end) so short labels
               ("no match") and wrapping ones ("undefined feature") both push
               b-value to the exact same offset -- otherwise a 1-line label
               leaves b-value sitting a full line higher than a neighbour
               whose label wrapped, misaligning the percentage row itself
               even though the earlier flex-start fix already lined up each
               item's own top edge. */
            font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.045em;
            color: var(--text-muted); font-weight: 600; text-align: right;
            line-height: 1.25; min-height: 2.5em;
            display: flex; flex-direction: column; align-items: flex-end; justify-content: flex-end;
        }}
        .bucket-item .b-value {{ font-size: 0.88rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.3; }}
        .bucket-item .b-count {{ font-size: 0.78rem; font-weight: 400; color: var(--text-muted); font-variant-numeric: tabular-nums; }}
        .b-examples-btn {{
            /* margin-top: auto (not a fixed 0.15rem) pushes it to the bottom
               of the stretched .bucket-item regardless of how tall the
               label/value/count block above it is -- .bucket-zone's
               align-items:stretch already equalizes every item's height in a
               row, so without this, an item's button lands wherever its own
               content happens to end, not level with its neighbours'. */
            background: none; border: none; padding: 0; margin-top: auto; cursor: pointer;
            font-family: inherit; font-size: 0.68rem; font-weight: 500; color: var(--accent);
        }}
        .b-examples-btn:hover {{ text-decoration: underline; }}

        /* N PAIRS / "valid from multi": the whole value becomes the examples
           trigger (no separate label, unlike the bucket items above), so it
           has to inherit this cell's own colour/size/alignment rather than
           impose its own -- it's standing in for plain text, not a widget. */
        .cell-examples-btn {{
            background: none; border: none; padding: 0; margin: 0; cursor: pointer;
            font: inherit; color: inherit; font-variant-numeric: tabular-nums;
        }}
        .cell-examples-btn:hover {{ text-decoration: underline; }}

        .detail-row {{ display: none; }}
        .detail-row.is-open {{ display: flex; }}
        .detail-row {{ border-bottom: 1px solid var(--border); background: var(--detail-bg); }}
        .detail-inner {{ padding: 1.1rem 1.4rem 1.4rem 3rem; display: flex; flex-direction: column; gap: 1.1rem; width: 100%; }}
        .stat-rows {{ display: grid; grid-template-columns: repeat(4, max-content); column-gap: 1.6rem; row-gap: 0.55rem; align-items: start; }}
        .stat {{ display: flex; flex-direction: column; gap: 0.15rem; }}
        .stat .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
        .stat .value {{ font-size: 0.92rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
        .stat .value .sub {{ font-size: 0.74rem; font-weight: 500; color: var(--text-muted); }}
        .section-label {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);
            margin: 0 0 0.55rem; font-weight: 600;
        }}
        .distribution-line {{
            font-size: 0.82rem; color: var(--text); background: var(--card); border: 1px solid var(--detail-border);
            border-radius: 6px; padding: 0.6rem 0.8rem; font-variant-numeric: tabular-nums; line-height: 1.6;
        }}
        .prob-pills {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
        .prob-pill {{
            font-size: 0.78rem; font-variant-numeric: tabular-nums; background: var(--card);
            border: 1px solid var(--detail-border); border-radius: 999px; padding: 0.3rem 0.75rem;
        }}
        .prob-pill b {{ font-weight: 600; }}
        .no-diag-note {{ font-size: 0.8rem; color: var(--text-muted); margin: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-section">
                <h1>MultiBLiMP v2 - Language Overview</h1>
                <div class="subtitle">
                    {agreement_label} agreement prediction through decision trees. Click a language row
                    (or the <strong>Results</strong> button) to open its create_pairs results below.
                    The small chevron after N Keep (or the <strong>Diagnostics</strong> button) is
                    separate: it opens just the bucket breakdown, inline, right where those items
                    dropped out of the pipeline.
                </div>
            </div>
            <div class="top-controls">
                <a href="../" class="back-btn">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M10 12L6 8L10 4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Overview
                </a>
                <label for="entropyType">Entropy Type:</label>
                <select id="entropyType">
                    <option value="six" selected>Six-class</option>
                    <option value="binary">Binary (majority vs. rest)</option>
                </select>
            </div>
        </div>

        <div id="scatterPlot"></div>

        {trivial_note}

        <div class="legend-row">
            <button class="btn" id="legendBtn" type="button">Legend</button>
            {threshold_badge}
        </div>
        <div class="controls">
            <div class="legend-group">
                <div class="scale-legend" title="Vermilion at 0%, violet in the middle, teal at 100%. Used wherever a high percentage is good (UM/UM+UD coverage, DT Acc%, N KEEP/N RAW, N PAIRS/N KEEP, Δ Entropy/Base Entropy).">
                    <span class="legend-title">Coverage</span>
                    <span>0%</span><span class="bar"></span><span>100%</span>
                </div>
                <div class="scale-legend inverted" title="Teal at 0%, violet in the middle, vermilion at 100% — the same scale mirrored. Used in the bucket breakdown (a high % means more rows dropped out at that step) and for Base/Reduced Entropy.">
                    <span class="legend-title">Item loss</span>
                    <span>0%</span><span class="bar"></span><span>100%</span>
                </div>
            </div>
            <button class="btn toggle-btn" id="resultsBtn" type="button" aria-pressed="false" title="Show the per-language coverage &amp; volume panel below each row (UD candidates, forms of interest, UM/UM+UD coverage, unk-drop counts).">Results</button>
            <button class="btn toggle-btn" id="diagnosticsBtn" type="button" aria-pressed="false" title="Show the per-language bucket breakdown row (no match / no inflection / same inflection / same feature / undefined feature / ambiguous {subject_label}).">Diagnostics</button>
        </div>

        <div class="table-scroll" id="tableScroll">
            <div class="grid-table" id="gridTable">
                <div class="row thead-row" id="theadRow">
                    <div class="cell c-chev"></div>
                    <div class="cell c-lang sortable" data-column="name">Language</div>
                    <div class="cell c-dist" title="The language's full Yes / No / unk label mix, before the decision tree drops unk rows. Always ordered Yes → No → unk.">Distribution</div>
                    <div class="cell c-base sortable" data-column="base" title="Entropy of the target label distribution before any splitting — the uninformed baseline.">Base Entropy</div>
                    <div class="cell c-reduced sortable" data-column="reduced" title="Entropy remaining after fitting the decision tree; lower means the tree explains more of the outcome.">Reduced Entropy</div>
                    <div class="cell c-delta sortable" data-column="delta" title="Base Entropy minus Reduced Entropy — how much uncertainty the tree removes, as a fraction of Base Entropy.">Δ Entropy</div>
                    <div class="cell c-acc sortable" data-column="acc" title="The decision tree's accuracy scored on its own training data.">DT Acc%</div>
                    <div class="cell c-raw sortable" data-column="nRaw" title="Candidate rows before the entropy/leaf-confidence filter is applied.">N RAW</div>
                    <div class="cell c-keep sortable" data-column="nKeep" title="Of N RAW, the rows kept after the leaf-confidence filter — the ones actually attempted.">N KEEP</div>
                    <div class="cell c-bchev"></div>
                    <div class="bucket-zone">
                        <div class="bucket-item" title="no match: no candidate swap form was found for this row at all."></div><div class="bucket-item" title="no inflection: a swap form was targeted, but the inflector had nothing to offer for it."></div><div class="bucket-item" title="same inflection: the re-inflected form came back identical to the original surface form."></div>
                        <div class="bucket-item" title="same feature: the re-inflected form's feature value still overlapped the original — the swap didn't actually change it."></div><div class="bucket-item" title="undefined feature: the target feature was undefined or missing for this form."></div><div class="bucket-item" title="ambiguous {subject_label}: the row's {subject_label} couldn't be uniquely resolved."></div>
                    </div>
                    <div class="cell c-pairs sortable" data-column="nPairs" title="Rows that ended up as a correctly re-inflected minimal pair, as a fraction of N KEEP.">N PAIRS</div>
                </div>
                <div id="tableBody"></div>
            </div>
        </div>
    </div>

    <div class="fixed-hscroll" id="fixedHscroll"><div id="fixedHscrollInner"></div></div>

    <div class="modal-overlay" id="legendOverlay">
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="legendTitle">
        <div class="modal-head">
          <div>
            <h2 id="legendTitle">Legend</h2>
            <p>What every label on this page means.</p>
          </div>
          <button class="modal-close" id="legendClose" type="button" aria-label="Close legend">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/></svg>
          </button>
        </div>

        <div class="legend-section">
          <h3>Table columns</h3>
          <dl class="legend-list">
            <div><dt>Distribution</dt><dd>The language's full Yes / No / unk label mix, before the decision tree drops unk rows. Always ordered Yes &rarr; No &rarr; unk, coloured to match the tree pages. Hover a bar for exact percentages.</dd></div>
            <div><dt>Base Entropy</dt><dd>Entropy of the target label distribution before any splitting. The uninformed baseline. Coloured on the item-loss scale (teal = low = good), relative to 1 bit or the highest base entropy in this table, whichever is larger.</dd></div>
            <div><dt>Reduced Entropy</dt><dd>Entropy remaining after fitting the decision tree; lower means the tree explains more of the outcome. Coloured on the item-loss scale (teal = low = good), on the same 1-bit-or-highest-base-entropy scale as Base Entropy above, so the two columns are directly comparable -- teal here means a language landed on a meaningfully low absolute entropy, not just a low entropy relative to its own (possibly already-low) starting point.</dd></div>
            <div><dt>Δ Entropy</dt><dd>Base Entropy minus Reduced Entropy. How much uncertainty the tree removes. Coloured on the coverage scale (teal = good), by what fraction of the base entropy that delta represents, not the raw number. 0.3 off a base of 0.4 is a much bigger win than 0.3 off a base of 2.0.</dd></div>
            <div><dt>DT Acc%</dt><dd>The decision tree's accuracy scored on its own training data. Coloured on the coverage scale (teal = high = good).</dd></div>
            <div><dt>N RAW</dt><dd>Candidate rows before the entropy/leaf-confidence filter is applied. Not coloured. There's no cheap per-language reference point (like treebank size) to scale it against, and not informative enough to be worth adding one for.</dd></div>
            <div><dt>N KEEP</dt><dd>Of those, the rows kept after the leaf-confidence filter ({threshold_text}). The ones actually attempted. Coloured on the coverage scale, as a fraction of N RAW &mdash; faded (with a hover tooltip) when N RAW is below 10, since that fraction is too noisy to mean much at that size (2/2 reads as a perfect 100% off pure small-sample luck). The colour still shows, just dimmed, rather than disappearing outright.</dd></div>
            <div><dt>N PAIRS</dt><dd>Rows that ended up as a correctly re-inflected minimal pair. Coloured on the coverage scale, as a fraction of N KEEP, with the same under-10 fade as N KEEP above (checked against N KEEP this time, not N RAW). Clickable when example rows are available &mdash; opens the same examples modal as the bucket breakdown.</dd></div>
          </dl>
        </div>

        <div class="legend-section">
          <h3>Results (below-row panel)</h3>
          <dl class="legend-list">
            <div><dt># UD candidates (raw)</dt><dd>Same value as N RAW above. Candidate rows before the entropy/leaf-confidence filter.</dd></div>
            <div><dt># UD candidates (kept)</dt><dd>Same value as N KEEP above. The rows kept after that filter.</dd></div>
            <div><dt># forms of interest</dt><dd>Distinct {head_role_label}/{nsubj_label} forms eligible for this prediction target at all, whether or not they ended up labelled "Yes".</dd></div>
            <div><dt>% covered by UM</dt><dd>Share of those forms found directly in UniMorph's inflection tables.</dd></div>
            <div><dt>% covered by UM+UD</dt><dd>Share covered once the UD-derived fallback is added on top of UniMorph.</dd></div>
            <div><dt># UM lemmas / forms</dt><dd>Size of the UniMorph inflection table available for this language.</dd></div>
            <div><dt># valid from multi</dt><dd>Minimal pairs that came from a row with more than one candidate swap form, as a share of all pairs. Clickable when example rows are available.</dd></div>
            <div><dt>Feature distribution</dt><dd>Tally of attempted feature-value swaps, as "FROM &rarr; TO: count", across every attempt that reached the comparison step.</dd></div>
            <div><dt>P(value|value)</dt><dd>Of the attempts that started at that value, the fraction where re-inflecting still landed back on the same value. I.e. how often the inflector failed to actually change it.</dd></div>
          </dl>
        </div>

        <div class="legend-section">
          <h3>Diagnostics (bucket breakdown)</h3>
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0 0 0.8rem;line-height:1.5;">
            Any bucket with at least one row (here, and N PAIRS / "valid from multi" in the
            Results panel above) has an <b>examples</b> link showing real sample rows &mdash;
            reconstructed sentences, feature values, and a treebank link &mdash; for that
            language and outcome.
          </p>
          <dl class="legend-list">
            <div><dt>no match</dt><dd>No candidate swap form was found for this row at all.</dd></div>
            <div><dt>no inflection</dt><dd>A swap form was targeted, but the inflector had nothing to offer for it.</dd></div>
            <div><dt>same inflection</dt><dd>The re-inflected form came back identical to the original surface form.</dd></div>
            <div><dt>same feature</dt><dd>The re-inflected form's feature value still overlapped the original. The swap didn't actually change it.</dd></div>
            <div><dt>undefined feature</dt><dd>The target feature was undefined or missing for this form.</dd></div>
            <div><dt>ambiguous {subject_label}</dt><dd>The row's {subject_label} couldn't be uniquely resolved.</dd></div>
          </dl>
        </div>

        <div class="legend-section">
          <h3>Numerator / denominator reference</h3>
          <p style="font-size:0.8rem;color:var(--text-muted);margin:0 0 0.8rem;line-height:1.5;">
            What each percentage above is actually a fraction of. "Row" scope means once per
            candidate row; "item" scope means once per attempted swap_form. A single row can
            produce more than one item, so row- and item-scoped percentages use different
            denominators. <code>extra_pairs</code> isn't shown as a percentage in the table, but
            shares the same item-scoped pool as the four buckets above.
          </p>
          <div class="ratio-scroll">
            <table class="ratio-table">
              <thead>
                <tr><th>Column</th><th>Numerator</th><th>Denominator</th><th>Denominator comes from</th><th>Scope</th><th>Why</th></tr>
              </thead>
              <tbody>
                <tr><td>% covered by UM</td><td><code>num_covered_um</code></td><td># forms of interest</td><td><code>num_forms_of_interest</code></td><td>forms</td><td>Distinct {head_role_label}/{nsubj_label} forms found in UniMorph alone.</td></tr>
                <tr><td>% covered by UM+UD</td><td><code>num_covered_um_ud</code></td><td># forms of interest</td><td><code>num_forms_of_interest</code></td><td>forms</td><td>Distinct {head_role_label}/{nsubj_label} forms found via UniMorph or the UD-derived fallback.</td></tr>
                <tr><td>% no match</td><td># no match (<code>no_candidates</code>)</td><td># UD candidates (kept)</td><td><code>n_keep</code></td><td>row</td><td>Fires once per candidate row, in place of the swap_form loop. The two are mutually exclusive.</td></tr>
                <tr><td>% no inflection</td><td># no inflection (<code>no_inflections</code>)</td><td># UD candidates (kept)</td><td><code>n_keep</code></td><td>row</td><td>Same as "no match": fires instead of entering the swap_form loop.</td></tr>
                <tr><td>% same inflection</td><td># same inflection (<code>same_forms</code>)</td><td>swap_items</td><td><code>items_seen &minus; no_match &minus; no_inflection</code></td><td>item</td><td>Fires once per attempted swap_form; a row can yield more than one item.</td></tr>
                <tr><td>% same feature</td><td># same feature (<code>same_features</code>)</td><td>swap_items</td><td><code>items_seen &minus; no_match &minus; no_inflection</code></td><td>item</td><td>Fires once per attempted swap_form.</td></tr>
                <tr><td>% undefined feature</td><td># undefined feature (<code>undefined_features</code>)</td><td>swap_items</td><td><code>items_seen &minus; no_match &minus; no_inflection</code></td><td>item</td><td>Fires once per attempted swap_form.</td></tr>
                <tr><td>% ambiguous {subject_label}</td><td># ambiguous {subject_label} (<code>ambiguous_subjects</code>)</td><td>swap_items</td><td><code>items_seen &minus; no_match &minus; no_inflection</code></td><td>item</td><td>Fires once per attempted swap_form.</td></tr>
                <tr><td>(not shown) extra_pairs</td><td><code>extra_pairs</code></td><td>swap_items</td><td><code>items_seen &minus; no_match &minus; no_inflection</code></td><td>item</td><td>Would collect otherwise-valid swaps (same eligibility as correct_swaps) that get capped once their FROM&rarr;TO combination hits the max_num_of_pairs limit. Always 0 today since create_pairs is never called with that limit set.</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="legend-section">
          <h3>Colour scales</h3>
          <dl class="legend-list">
            <div><dt>Coverage</dt><dd>Vermilion (red) at 0%, violet around the middle, teal at 100%. Used where a high percentage is good (UM/UM+UD coverage, DT Acc%, N KEEP as a fraction of N RAW, N PAIRS as a fraction of N KEEP, and Δ Entropy as a fraction of Base Entropy).</dd></div>
            <div><dt>Item loss</dt><dd>Teal at 0%, violet around the middle, vermilion at 100%, same scale mirrored. Used in the bucket breakdown (a high percentage means more rows dropped out at that step) and for Base/Reduced Entropy, scaled against the highest base entropy shown in this table rather than a fixed 0&ndash;100 range.</dd></div>
          </dl>
        </div>
      </div>
    </div>

    <div class="modal-overlay" id="examplesOverlay">
      <div class="modal modal-wide" role="dialog" aria-modal="true" aria-labelledby="examplesTitle">
        <div class="modal-head">
          <div>
            <h2 id="examplesTitle">Examples</h2>
            <p id="examplesSubtitle"></p>
          </div>
          <button class="modal-close" id="examplesClose" type="button" aria-label="Close examples">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/></svg>
          </button>
        </div>
        <div id="examplesBody"></div>
      </div>
    </div>

    <script>
    const LANGUAGES = {{ six: {languages_six_json}, binary: {languages_binary_json} }};
    const plotData = {{ six: {plot_data_six_json}, binary: {plot_data_binary_json} }};

    const BUCKETS = [
      {{ key: "no_match", label: "no match", desc: "No candidate swap form was found for this row at all." }},
      {{ key: "no_inflection", label: "no inflection", desc: "A swap form was targeted, but the inflector had nothing to offer for it." }},
      {{ key: "same_inflection", label: "same inflection", desc: "The re-inflected form came back identical to the original surface form." }},
      {{ key: "same_feature", label: "same feature", desc: "The re-inflected form's feature value still overlapped the original — the swap didn't actually change it." }},
      {{ key: "undefined_feature", label: "undefined feature", desc: "The target feature was undefined or missing for this form." }},
      {{ key: "ambiguous_subject", label: "ambiguous {subject_label}", desc: "The row's {subject_label} couldn't be uniquely resolved." }},
    ];

    // Fixed left-to-right order for the distribution bar -- matches
    // sva_trees.pipeline's palette_map order, never sorted by size, so a bar
    // reads the same way across every language and every render. A language
    // whose predictor_var was never simplified (Pipeline.simplify=False)
    // carries "+-"/"--" instead of "unk"; both are covered here so either
    // convention renders, though a run only ever produces one or the other.
    // Lowercase "yes"/"no" alongside SVA's own "Yes"/"No": NPA's pairwise
    // target_cols (e.g. "HEAD-DET_Number") use lowercase labels throughout
    // (see word_order.utils.is_agreement_predictor's module comment) --
    // "unk" already happened to match both conventions, but "yes"/"no"
    // didn't, so a label_distribution built from an NPA target_col only
    // ever matched the "unk" entry here, silently dropping its Yes/No
    // segments from the bar instead of rendering them.
    const LABEL_ORDER = [
      ["Yes", "#31cb9f"],
      ["yes", "#31cb9f"],
      ["No", "#f16393"],
      ["no", "#f16393"],
      ["unk", "#b893de"],
      ["+-", "#e5c64d"],
      ["--", "#b893de"],
    ];

    function renderDistBar(dist) {{
      if (!dist || Object.keys(dist).length === 0) {{
        return `<span class="dist-bar" style="background:none;"></span>`;
      }}
      const total = Object.values(dist).reduce((a, b) => a + b, 0);
      if (total === 0) return `<span class="dist-bar" style="background:none;"></span>`;
      const present = LABEL_ORDER.filter(([label]) => dist[label] > 0);
      const title = present
        .map(([label]) => `${{label}} ${{(dist[label] / total * 100).toFixed(1)}}%`)
        .join("  ·  ");
      const segs = present
        .map(([label, color]) => `<span style="width:${{dist[label] / total * 100}}%;background:${{color}};"></span>`)
        .join("");
      return `<div class="dist-bar" title="${{escapeAttr(title)}}">${{segs}}</div>`;
    }}

    // Read the scale's two endpoints from CSS ONCE at load, not per cell --
    // getComputedStyle() forces a style recalc, so doing this per coloured
    // cell would be needless repeated work at real language counts.
    const PCT_SCALE = (() => {{
      const root = getComputedStyle(document.documentElement);
      const num = (name) => parseFloat(root.getPropertyValue(name));
      return {{
        h0: num("--pct-lo-h"), s0: num("--pct-lo-s"), l0: num("--pct-lo-l"),
        h1: num("--pct-hi-h"), s1: num("--pct-hi-s"), l1: num("--pct-hi-l"),
      }};
    }})();

    // Same rationale as PCT_SCALE: Plotly's layout takes plain colour
    // strings, not CSS vars, so the scatter plot needs its own theme read.
    const CHART_THEME = (() => {{
      const root = getComputedStyle(document.documentElement);
      const v = (name) => root.getPropertyValue(name).trim();
      return {{ text: v("--text"), border: v("--border") }};
    }})();

    // Vermilion -> violet -> teal, and never green. Hue rotates steadily in
    // one direction, numerically decreasing from 358 down to 176 (the
    // "short way" through magenta/violet/blue) rather than increasing up
    // through orange/yellow/green -- that arc is what keeps green off the
    // scale entirely, and it means red only ever appears at the literal 0%
    // endpoint, never as a colour the gradient sweeps through on its way
    // elsewhere. invert=true mirrors the whole thing, for metrics where a
    // high % is bad (item loss).
    function pctColor(pct, invert = false) {{
      const {{ h0, s0, l0, h1, s1, l1 }} = PCT_SCALE;
      let t = Math.max(0, Math.min(100, pct)) / 100;
      if (invert) t = 1 - t;
      const h = h0 + (h1 - h0) * t;
      const s = s0 + (s1 - s0) * t;
      const l = l0 + (l1 - l0) * t;
      return `hsl(${{h.toFixed(0)}}deg ${{s.toFixed(0)}}% ${{l.toFixed(0)}}%)`;
    }}

    // For low-N cells: the colour still renders, blended mostly back toward
    // the theme's default text colour (not left at full opacity -- plain
    // opacity blends with whatever's behind it and reads as low-contrast
    // rather than "dimmed").
    function dimColorStyle(color, dim, reason) {{
      if (!dim) return ` style="color:${{color}}"`;
      return ` style="color:color-mix(in srgb, ${{color}} 40%, var(--text) 60%)" title="${{reason}}"`;
    }}

    const MIN_N_FOR_COLOR = 10;
    function investmentColor(numerator, denominator) {{
      const pct = denominator > 0 ? (numerator / denominator) * 100 : 0;
      return pctColor(Math.max(0, Math.min(100, pct)));
    }}

    // Entropy has no fixed 0-100 scale (its ceiling depends on how many
    // labels the target has), so colour it relative to a ceiling rather
    // than assuming a theoretical max -- maxBaseEntropy (table-wide,
    // computed in renderAll/generate_rows below), shared by both Base and
    // Reduced Entropy so they're directly comparable at a glance. invert
    // stays at its default true for both: lower is better, teal at the low
    // end.
    function entropyColor(value, maxEntropy, invert = true) {{
      return pctColor(Math.min(100, (value / maxEntropy) * 100), invert);
    }}

    // Delta entropy isn't meaningful in isolation -- 0.3 off a base of 0.4
    // is a much bigger win than 0.3 off a base of 2.0 -- so colour it by
    // what fraction of the base entropy got explained away instead.
    //
    // base == 0 means there was no entropy to explain in the first place
    // (e.g. a single-sample or perfectly pure language) -- that's a
    // trivial, not a meritorious, 0.000, and must not be painted the same
    // "fully explained" colour as a language whose tree genuinely explained
    // away 100% of a real base entropy. Returns null for that case; callers
    // should fall back to a neutral/muted colour instead of colouring it.
    function deltaColor(base, delta) {{
      if (base <= 0) return null;
      const pct = Math.max(0, Math.min(100, (delta / base) * 100));
      return pctColor(pct);
    }}

    function chevronSvg() {{
      return `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    }}

    function escapeAttr(s) {{
      return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;");
    }}

    // The "examples" button is only shown when there's actually at least one
    // row in that bucket AND create_pairs wrote an "examples" entry into this
    // language's meta.json (older runs, or ones that hit the save_to
    // try/except in pipeline.py, won't have one -- diag.examples is then
    // undefined/empty).
    function renderBucketZone(diag, langName) {{
      if (!diag) {{
        return `<div class="bucket-item no-data">No diagnostics available</div>`;
      }}
      return BUCKETS.map(b => {{
        const [n, pct] = diag.buckets[b.key] || [0, 0];
        const color = pctColor(pct, true);
        const hasExamples = n > 0 && diag.examples && diag.examples[b.key];
        return `
          <div class="bucket-item" title="${{escapeAttr(b.desc)}} (${{n.toLocaleString()}} rows, ${{pct.toFixed(1)}}% of swap attempts)">
            <span class="b-label">${{b.label}}</span>
            <span class="b-value" style="color:${{color}}">${{pct.toFixed(1)}}%</span>
            <span class="b-count">${{n.toLocaleString()}}</span>
            ${{hasExamples ? `<button class="b-examples-btn" type="button" data-lang="${{escapeAttr(langName)}}" data-bucket="${{b.key}}">examples</button>` : ""}}
          </div>`;
      }}).join("");
    }}

    function multiValueHtml(lang, diag) {{
      const n = diag.nValidFromMulti.toLocaleString();
      const sub = `<span class="sub"> &nbsp;(${{(diag.ratioValidFromMulti * 100).toFixed(1)}}% of pairs)</span>`;
      const hasExamples = diag.examples && diag.examples.multi_now_valid && diag.nValidFromMulti > 0;
      const n_html = hasExamples
        ? `<button class="cell-examples-btn" type="button" data-lang="${{escapeAttr(lang.name)}}" data-bucket="multi_now_valid">${{n}}</button>`
        : n;
      return n_html + sub;
    }}

    function unkValueHtml(lang, diag, bucketKey, count) {{
      const n = count.toLocaleString();
      const hasExamples = diag.examples && diag.examples[bucketKey] && count > 0;
      return hasExamples
        ? `<button class="cell-examples-btn" type="button" data-lang="${{escapeAttr(lang.name)}}" data-bucket="${{bucketKey}}">${{n}}</button>`
        : n;
    }}

    function renderDetail(lang) {{
      const diag = lang.diag;
      const probEntries = diag ? Object.entries(diag.probs) : [];
      return `
        <div class="detail-inner">
          <div>
            <p class="section-label">Coverage &amp; volume</p>
            <div class="stat-rows">
              <div class="stat" style="grid-column:1;grid-row:1;"><span class="label" title="Candidate rows before the entropy/leaf-confidence filter is applied. Same value as N RAW in the table."># UD candidates (raw)</span><span class="value">${{lang.nRaw.toLocaleString()}}</span></div>
              <div class="stat" style="grid-column:2;grid-row:1;"><span class="label" title="Distinct {head_role_label}/{nsubj_label} forms eligible for this prediction target at all, whether or not they ended up labelled &quot;Yes&quot;."># forms of interest</span><span class="value">${{diag ? diag.nForms.toLocaleString() : "—"}}</span></div>

              <div class="stat" style="grid-column:1;grid-row:2;"><span class="label" title="Share of forms of interest covered once the UD-derived fallback is added on top of UniMorph.">% covered by UM+UD</span><span class="value"${{diag ? ` style="color:${{pctColor(diag.pctUMUD)}}"` : ""}}>${{diag ? diag.pctUMUD.toFixed(1) + "%" : "—"}}</span></div>
              <div class="stat" style="grid-column:2;grid-row:2;"><span class="label" title="Share of forms of interest found directly in UniMorph's inflection tables.">% covered by UM</span><span class="value"${{diag ? ` style="color:${{pctColor(diag.pctUM)}}"` : ""}}>${{diag ? diag.pctUM.toFixed(1) + "%" : "—"}}</span></div>
              <div class="stat" style="grid-column:3;grid-row:2;"><span class="label" title="Size of the UniMorph lemma inventory available for this language."># UM lemmas</span><span class="value">${{diag ? diag.nLemma.toLocaleString() : "—"}}</span></div>
              <div class="stat" style="grid-column:4;grid-row:2;"><span class="label" title="Size of the UniMorph inflected-form inventory available for this language."># UM forms</span><span class="value">${{diag ? diag.nForm.toLocaleString() : "—"}}</span></div>

              <div class="stat" style="grid-column:1;grid-row:3;"><span class="label" title="Of N RAW, the rows kept after the leaf-confidence filter. Same value as N KEEP in the table."># UD candidates (kept)</span><span class="value">${{lang.nKeep.toLocaleString()}}</span></div>
              <div class="stat" style="grid-column:2;grid-row:3;"><span class="label" title="Rows that ended up as a correctly re-inflected minimal pair. Same value as N PAIRS in the table."># minimal pairs</span><span class="value">${{lang.nPairs.toLocaleString()}}</span></div>
              <div class="stat" style="grid-column:3;grid-row:3;"><span class="label" title="Minimal pairs that came from a row with more than one candidate swap form, as a share of all pairs."># valid from multi</span><span class="value">${{diag ? multiValueHtml(lang, diag) : "—"}}</span></div>
            </div>
          </div>

          <div>
            <p class="section-label" title="Rows dropped from the decision tree fit because the target feature was missing on the {head_role_label}, the {nsubj_label}, or both -- fit_dt's UNK_LABELS drop (see the --keep_unk script flag).">Dropped before fitting (missing feature annotation)</p>
            <div class="stat-rows">
              <div class="stat" style="grid-column:1;grid-row:1;"><span class="label" title="Rows dropped because the {head_role_label}'s target feature was missing, but the {nsubj_label}'s was present."># {head_role_label} unk</span><span class="value">${{diag ? unkValueHtml(lang, diag, "head_unk", diag.headUnk) : "—"}}</span></div>
              <div class="stat" style="grid-column:2;grid-row:1;"><span class="label" title="Rows dropped because the {nsubj_label}'s target feature was missing, but the {head_role_label}'s was present."># {nsubj_label} unk</span><span class="value">${{diag ? unkValueHtml(lang, diag, "nsubj_unk", diag.nsubjUnk) : "—"}}</span></div>
              <div class="stat" style="grid-column:3;grid-row:1;"><span class="label" title="Rows dropped because both the {head_role_label}'s and the {nsubj_label}'s target feature were missing."># both unk</span><span class="value">${{diag ? unkValueHtml(lang, diag, "both_unk", diag.bothUnk) : "—"}}</span></div>
            </div>
          </div>

          ${{!diag ? `<p class="no-diag-note">No further diagnostics available for this language (it may not have been processed by create_pairs, or the diagnostics step failed for it).</p>` : ""}}

          ${{diag && diag.distribution ? `
          <div>
            <p class="section-label">Feature distribution (from &rarr; to)</p>
            <div class="distribution-line">${{diag.distribution}}</div>
          </div>` : ""}}

          ${{probEntries.length ? `
          <div>
            <p class="section-label">P(value | value) &mdash; re-inflection reliability</p>
            <div class="prob-pills">
              ${{probEntries.map(([v, p]) => `<span class="prob-pill">P(${{v}}|${{v}}) = <b>${{p.toFixed(3)}}</b></span>`).join("")}}
            </div>
          </div>` : ""}}
        </div>`;
    }}

    // ---- state ----
    let currentEntropyType = "six";
    let currentSort = {{ column: "name", ascending: true }};
    const openResults = new Set();   // language names with the Results panel open
    const openBuckets = new Set();   // language names with the bucket breakdown open

    const SORT_ACCESSORS = {{
      name: (l) => l.name.toLowerCase(),
      base: (l) => l.base,
      reduced: (l) => l.reduced,
      delta: (l) => l.delta,
      acc: (l) => l.acc,
      nRaw: (l) => l.nRaw,
      nKeep: (l) => l.nKeep,
      nPairs: (l) => l.nPairs,
    }};

    const gridTable = document.getElementById("gridTable");
    const tableBody = document.getElementById("tableBody");
    const theadRow = document.getElementById("theadRow");

    // Pinned-to-viewport-bottom horizontal scrollbar, mirroring #tableScroll's
    // native one -- see .fixed-hscroll's CSS comment for why. Kept correct
    // through every width-changing event via a ResizeObserver -- but NOT one
    // observing #gridTable itself: #gridTable is a plain block box (width:
    // auto), so a child row overflowing it wider never resizes #gridTable's
    // own box, and the observer would simply never fire for that. Each
    // .row IS sized with width:max-content though, so it genuinely resizes
    // as its bucket-zone's width transitions open/closed -- observing the
    // rows themselves is what makes the bar track that animation live,
    // frame by frame, not just snap to the right value sometime after.
    const tableScrollEl = document.getElementById("tableScroll");
    const fixedHscroll = document.getElementById("fixedHscroll");
    const fixedHscrollInner = document.getElementById("fixedHscrollInner");
    let syncingHscroll = false;

    function refreshFixedHscroll() {{
      // Match #tableScroll's own rect, not the viewport's: a fixed bar
      // spanning the full window would (as here) usually be WIDER than the
      // table's total content, even while the table itself -- inset by the
      // page's padding/max-width -- genuinely overflows and needs to
      // scroll. Anchoring to the same left offset and width as the real
      // scroll container is what makes this bar's own overflow match it.
      const rect = tableScrollEl.getBoundingClientRect();
      fixedHscroll.style.left = rect.left + "px";
      fixedHscroll.style.width = rect.width + "px";
      fixedHscrollInner.style.width = gridTable.scrollWidth + "px";
      fixedHscroll.style.display =
        gridTable.scrollWidth > tableScrollEl.clientWidth ? "block" : "none";
    }}
    tableScrollEl.addEventListener("scroll", () => {{
      if (syncingHscroll) return;
      syncingHscroll = true;
      fixedHscroll.scrollLeft = tableScrollEl.scrollLeft;
      syncingHscroll = false;
    }});
    fixedHscroll.addEventListener("scroll", () => {{
      if (syncingHscroll) return;
      syncingHscroll = true;
      tableScrollEl.scrollLeft = fixedHscroll.scrollLeft;
      syncingHscroll = false;
    }});
    window.addEventListener("resize", refreshFixedHscroll);
    // Re-pointed at each .row (see comment above) rather than #gridTable.
    // #theadRow carries its own bucket-zone (see syncHeaderBuckets below) and
    // is static markup that's never torn down, so it's observed once, here.
    // The per-language rows ARE torn down and rebuilt on every renderAll(),
    // so those are disconnected and re-observed there each time -- but that
    // disconnect() call clears every observation this observer holds,
    // #theadRow included, so renderAll re-observes it too, not just the rows.
    const rowResizeObserver = new ResizeObserver(refreshFixedHscroll);
    rowResizeObserver.observe(theadRow);
    // Belt-and-suspenders alongside the observer above: guarantees one
    // correct, final refresh once a bucket-zone's open/close transition
    // (max-width/opacity/margin) actually finishes, regardless of any
    // ResizeObserver timing quirk. transitionend bubbles, so one delegated
    // listener here covers every row's bucket-zone.
    gridTable.addEventListener("transitionend", refreshFixedHscroll);

    function syncHeaderBuckets() {{
      theadRow.classList.toggle("buckets-open", openBuckets.size > 0);
    }}

    function renderAll() {{
      const list = LANGUAGES[currentEntropyType];
      // Base entropy: anchored at 1 bit (the true ceiling for a binary
      // Yes/No split) rather than just "worst on screen" -- otherwise a
      // table where every language happens to sit close together stretches
      // that narrow band across the whole gradient and makes mediocre
      // values look artificially bad. Math.max(...) still wins when the
      // real data exceeds 1 (possible for six-class entropy, never binary).
      // Lower is better, teal at the low end.
      //
      // Reduced entropy: scaled on the SAME table-wide ceiling as Base
      // Entropy, deliberately not each row's own base (that was the
      // previous scheme). reduced/base and delta/base are complementary
      // fractions of the same number, so scaling Reduced per-row against
      // its own base made its colour exactly reproduce Delta Entropy's (t
      // and 1-t cancel through the two invert directions) -- zero
      // additional information over the Δ Entropy column, since it's
      // literally the same colour every time, not just visually similar.
      // Scaling against maxBaseEntropy instead makes Base/Reduced/Delta
      // three genuinely distinct reads: Base = absolute starting
      // uncertainty, Delta = fraction of that language's own uncertainty
      // explained away (relative effort), Reduced = absolute uncertainty
      // actually left over -- so at a glance you can see which languages
      // landed on a meaningfully low entropy in absolute terms, not just
      // which ones improved a lot relatively.
      const maxBaseEntropy = Math.max(1, ...list.map(l => l.base));
      const acc = SORT_ACCESSORS[currentSort.column];
      const sorted = [...list].sort((a, b) => {{
        const av = acc(a), bv = acc(b);
        if (typeof av === "string") {{
          return currentSort.ascending ? av.localeCompare(bv) : bv.localeCompare(av);
        }}
        return currentSort.ascending ? av - bv : bv - av;
      }});

      tableBody.innerHTML = "";
      rowResizeObserver.disconnect();
      rowResizeObserver.observe(theadRow);
      sorted.forEach((lang) => {{
        const langRow = document.createElement("div");
        langRow.className = "row lang-row";
        if (openResults.has(lang.name)) langRow.classList.add("is-open");
        if (openBuckets.has(lang.name)) langRow.classList.add("buckets-open");

        const nameStyle = lang.color ? ` style="color:${{lang.color}};font-weight:600;"` : "";
        const nameHtml = lang.langUrl
          ? `<a href="${{lang.langUrl}}"${{nameStyle}}>${{lang.name}}</a>`
          : `<span${{nameStyle}}>${{lang.name}}</span>`;

        const hasPairsExamples = lang.diag && lang.diag.examples && lang.diag.examples.correct_swaps && lang.nPairs > 0;
        const nPairsText = lang.nPairs.toLocaleString();
        const nPairsHtml = hasPairsExamples
          ? `<button class="cell-examples-btn" type="button" data-lang="${{escapeAttr(lang.name)}}" data-bucket="correct_swaps">${{nPairsText}}</button>`
          : nPairsText;

        langRow.innerHTML = `
          <div class="cell c-chev"><span class="chevron">${{chevronSvg()}}</span></div>
          <div class="cell c-lang lang-name">${{nameHtml}}</div>
          <div class="cell c-dist">${{renderDistBar(lang.diag && lang.diag.labelDistribution)}}</div>
          <div class="cell c-base" style="color:${{entropyColor(lang.base, maxBaseEntropy)}}">${{lang.base.toFixed(3)}}</div>
          <div class="cell c-reduced" style="color:${{entropyColor(lang.reduced, maxBaseEntropy)}}">${{lang.reduced.toFixed(3)}}</div>
          <div class="cell c-delta" style="color:${{deltaColor(lang.base, lang.delta) || "var(--text-muted)"}}">${{lang.delta.toFixed(3)}}</div>
          <div class="cell c-acc" style="color:${{pctColor(lang.acc * 100)}}">${{(lang.acc * 100).toFixed(1)}}%</div>
          <div class="cell c-raw">${{lang.nRaw.toLocaleString()}}</div>
          <div class="cell c-keep"${{dimColorStyle(investmentColor(lang.nKeep, lang.nRaw), lang.nRaw < MIN_N_FOR_COLOR, `N RAW is only ${{lang.nRaw}} — too small a sample for this ratio to be meaningful`)}}>${{lang.nKeep.toLocaleString()}}</div>
          <div class="cell c-bchev"><button class="bucket-chevron-btn" type="button" aria-label="Toggle bucket breakdown"><span class="chevron">${{chevronSvg()}}</span></button></div>
          <div class="bucket-zone">${{renderBucketZone(lang.diag, lang.name)}}</div>
          <div class="cell c-pairs"${{dimColorStyle(investmentColor(lang.nPairs, lang.nKeep), lang.nKeep < MIN_N_FOR_COLOR, `N KEEP is only ${{lang.nKeep}} — too small a sample for this ratio to be meaningful`)}}>${{nPairsHtml}}</div>
        `;

        const detailRow = document.createElement("div");
        detailRow.className = "row detail-row";
        if (openResults.has(lang.name)) detailRow.classList.add("is-open");
        detailRow.innerHTML = renderDetail(lang);

        langRow.addEventListener("click", () => {{
          const open = langRow.classList.toggle("is-open");
          detailRow.classList.toggle("is-open", open);
          if (open) openResults.add(lang.name); else openResults.delete(lang.name);
        }});

        const bucketBtn = langRow.querySelector(".bucket-chevron-btn");
        bucketBtn.addEventListener("click", (e) => {{
          e.stopPropagation();
          const open = langRow.classList.toggle("buckets-open");
          if (open) openBuckets.add(lang.name); else openBuckets.delete(lang.name);
          syncHeaderBuckets();
        }});

        tableBody.appendChild(langRow);
        tableBody.appendChild(detailRow);
        rowResizeObserver.observe(langRow);
      }});

      syncHeaderBuckets();
      theadRow.querySelectorAll(".cell.sortable").forEach(h => {{
        h.classList.remove("sort-asc", "sort-desc");
        if (h.dataset.column === currentSort.column) {{
          h.classList.add(currentSort.ascending ? "sort-asc" : "sort-desc");
        }}
      }});
      // Explicit call, not left to ResizeObserver's implicit initial fire
      // alone: that fire is async and its exact timing isn't guaranteed
      // across environments, so without this the bar can stay wrongly
      // hidden until *something* later happens to trigger a resize.
      refreshFixedHscroll();
    }}

    theadRow.querySelectorAll(".cell.sortable").forEach(header => {{
      header.addEventListener("click", () => {{
        const column = header.dataset.column;
        currentSort.ascending = currentSort.column === column ? !currentSort.ascending : true;
        currentSort.column = column;
        renderAll();
      }});
    }});

    const resultsBtn = document.getElementById("resultsBtn");
    const diagnosticsBtn = document.getElementById("diagnosticsBtn");

    resultsBtn.addEventListener("click", () => {{
      const rows = document.querySelectorAll(".lang-row");
      const shouldOpen = Array.from(rows).some(r => !r.classList.contains("is-open"));
      rows.forEach(r => r.classList.toggle("is-open", shouldOpen));
      document.querySelectorAll(".detail-row").forEach(r => r.classList.toggle("is-open", shouldOpen));
      LANGUAGES[currentEntropyType].forEach(l => {{
        if (shouldOpen) openResults.add(l.name); else openResults.delete(l.name);
      }});
      resultsBtn.classList.toggle("active", shouldOpen);
      resultsBtn.setAttribute("aria-pressed", String(shouldOpen));
    }});

    diagnosticsBtn.addEventListener("click", () => {{
      const rows = document.querySelectorAll(".lang-row");
      const shouldOpen = Array.from(rows).some(r => !r.classList.contains("buckets-open"));
      rows.forEach(r => r.classList.toggle("buckets-open", shouldOpen));
      LANGUAGES[currentEntropyType].forEach(l => {{
        if (shouldOpen) openBuckets.add(l.name); else openBuckets.delete(l.name);
      }});
      syncHeaderBuckets();
      diagnosticsBtn.classList.toggle("active", shouldOpen);
      diagnosticsBtn.setAttribute("aria-pressed", String(shouldOpen));
    }});

    const legendBtn = document.getElementById("legendBtn");
    const legendOverlay = document.getElementById("legendOverlay");
    const legendClose = document.getElementById("legendClose");
    legendBtn.addEventListener("click", () => legendOverlay.classList.add("open"));
    legendClose.addEventListener("click", () => legendOverlay.classList.remove("open"));
    legendOverlay.addEventListener("click", (e) => {{ if (e.target === legendOverlay) legendOverlay.classList.remove("open"); }});

    // ---- examples modal: per-bucket sample rows, pre-rendered server-side
    // by sva_trees.create_pairs.bucket_examples_html and dropped in as-is. One
    // shared overlay, content swapped in per click via a delegated listener
    // on tableBody (rows are re-created on every renderAll(), so per-button
    // listeners would need re-attaching). Covers all 11 buckets, not just
    // the 6 in the bucket zone -- correct_swaps (N PAIRS), multi_now_valid
    // ("valid from multi"), and head_unk/nsubj_unk/both_unk (linked from
    // the "Dropped before fitting" stats) link in from the Results panel
    // instead. ----
    const examplesOverlay = document.getElementById("examplesOverlay");
    const examplesClose = document.getElementById("examplesClose");
    const examplesTitle = document.getElementById("examplesTitle");
    const examplesSubtitle = document.getElementById("examplesSubtitle");
    const examplesBody = document.getElementById("examplesBody");

    // BUCKETS only covers the 6 "problem" buckets shown in the bucket zone.
    // correct_swaps (linked from N PAIRS), multi_now_valid (linked from
    // "valid from multi"), and head_unk/nsubj_unk/both_unk (linked from the
    // "Dropped before fitting" stats) aren't in that list, so they need
    // their own labels and their own way to look up a count (they're not
    // in diag.buckets).
    const EXTRA_BUCKET_LABELS = {{
      correct_swaps: "minimal pairs",
      multi_now_valid: "valid from multi",
      head_unk: "{head_role_label} unk",
      nsubj_unk: "{nsubj_label} unk",
      both_unk: "both unk",
    }};
    function bucketLabel(key) {{
      const b = BUCKETS.find(b => b.key === key);
      return b ? b.label : (EXTRA_BUCKET_LABELS[key] || key);
    }}
    function bucketCount(lang, key) {{
      if (key === "correct_swaps") return lang.nPairs;
      if (key === "multi_now_valid") return lang.diag.nValidFromMulti;
      if (key === "head_unk") return lang.diag.headUnk;
      if (key === "nsubj_unk") return lang.diag.nsubjUnk;
      if (key === "both_unk") return lang.diag.bothUnk;
      return (lang.diag.buckets[key] || [0, 0])[0];
    }}

    function openExamplesModal(langName, bucketKey) {{
      const lang = LANGUAGES[currentEntropyType].find(l => l.name === langName);
      if (!lang || !lang.diag || !lang.diag.examples || !lang.diag.examples[bucketKey]) return;
      examplesTitle.textContent = `${{lang.name}} — ${{bucketLabel(bucketKey)}}`;
      const n = bucketCount(lang, bucketKey);
      examplesSubtitle.textContent = `Sample rows that landed in this bucket (n=${{n.toLocaleString()}}).`;
      examplesBody.innerHTML = lang.diag.examples[bucketKey];
      linkFeatureDetails(examplesBody);
      examplesOverlay.classList.add("open");
    }}
    function closeExamplesModal() {{ examplesOverlay.classList.remove("open"); }}

    // Feature lists for different roles in the same example row (e.g.
    // {nsubj_label} vs. {head_role_label}) are separate <details> elements.
    // Link them per-row so that opening/closing one toggles the others in
    // that row to match, letting you compare both sides' features at a
    // glance instead of expanding each one by hand.
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

    tableBody.addEventListener("click", (e) => {{
      const btn = e.target.closest(".b-examples-btn, .cell-examples-btn");
      if (!btn) return;
      e.stopPropagation();
      openExamplesModal(btn.dataset.lang, btn.dataset.bucket);
    }});
    examplesClose.addEventListener("click", closeExamplesModal);
    examplesOverlay.addEventListener("click", (e) => {{ if (e.target === examplesOverlay) closeExamplesModal(); }});

    document.addEventListener("keydown", (e) => {{
      if (e.key !== "Escape") return;
      legendOverlay.classList.remove("open");
      closeExamplesModal();
    }});

    // ---- scatter plot: unchanged from the classic page ----
    function renderPlot(entropyType) {{
      const data = plotData[entropyType];
      const trace = {{
        x: data.map(d => d.base), y: data.map(d => d.reduced),
        mode: 'markers', type: 'scatter', text: data.map(d => d.name),
        customdata: data.map(d => [d.url, d.n_items]),
        hovertemplate: '<b>%{{text}}</b><br>Base Entropy: %{{x:.3f}}<br>Reduced Entropy: %{{y:.3f}}<br>N Items: %{{customdata[1]:,}}<br><extra></extra>',
        marker: {{
          size: data.map(d => d.n_items), sizemode: 'area',
          sizeref: 2 * Math.max(...data.map(d => d.n_items)) / (40 ** 2), sizemin: 4,
          color: data.map(d => d.color), opacity: 0.6, line: {{ color: '#1e40af', width: 1 }},
        }},
      }};
      const layout = {{
        title: {{ text: 'Base Entropy vs. Reduced Entropy', font: {{ family: 'DM Sans, system-ui, sans-serif', size: 16, color: CHART_THEME.text }} }},
        xaxis: {{ title: 'Base Entropy', gridcolor: CHART_THEME.border, zeroline: false }},
        yaxis: {{ title: 'Reduced Entropy', gridcolor: CHART_THEME.border, zeroline: false }},
        plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
        font: {{ family: 'DM Sans, system-ui, sans-serif', color: CHART_THEME.text }},
        hovermode: 'closest', margin: {{ t: 50, r: 30, b: 50, l: 60 }},
      }};
      const config = {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'], displaylogo: false }};
      Plotly.newPlot('scatterPlot', [trace], layout, config);
      document.getElementById('scatterPlot').on('plotly_click', function(data) {{
        const url = data.points[0].customdata[0];
        if (url) window.location.href = url;
      }});
    }}

    document.getElementById('entropyType').addEventListener('change', (e) => {{
      currentEntropyType = e.target.value;
      renderPlot(currentEntropyType);
      renderAll();
    }});

    renderPlot(currentEntropyType);
    renderAll();
    </script>
</body>
</html>"""
