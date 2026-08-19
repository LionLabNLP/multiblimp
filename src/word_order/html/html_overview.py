def create_html(sections_html, all_data_json):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MultiBLiMP v2 - Agreement Overview</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f5f5f4;
            --card: #ffffff;
            --panel-bg: #fafaf9;
            --text: #1c1917;
            --text-muted: #78716c;
            --text-subtle: #57534e;
            --accent: #2563eb;
            --border: #e7e5e4;
            --border-hover: #c7c4c0;
            --grid-line: #e7e5e4;
            --diagonal-line: #d4d0cb;
            --hover: #f5f5f5;
        }}
        /* Same palette as viz_deprel.py's html_deprel.py and viz_tree.py's
           html_tree.py, for a consistent look across every page in the site. */
        :root[data-theme="dark"] {{
            color-scheme: dark;
            --bg: #16140f; --card: #221f19; --panel-bg: #1c1a15;
            --text: #f0ede6; --text-muted: #a39a8a; --text-subtle: #a39a8a;
            --accent: #6ea8ff; --border: #38332a; --border-hover: #4d4636;
            --grid-line: #38332a; --diagonal-line: #4d4636; --hover: #2a261e;
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                color-scheme: dark;
                --bg: #16140f; --card: #221f19; --panel-bg: #1c1a15;
                --text: #f0ede6; --text-muted: #a39a8a; --text-subtle: #a39a8a;
                --accent: #6ea8ff; --border: #38332a; --border-hover: #4d4636;
                --grid-line: #38332a; --diagonal-line: #4d4636; --hover: #2a261e;
            }}
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
            max-width: 1400px;
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
            gap: 2rem;
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
        .description {{
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 700px;
            margin: 0.5rem 0 0;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-shrink: 0;
            padding-top: 0.25rem;
        }}
        .controls label {{
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-subtle);
        }}
        select {{
            padding: 0.5rem 2rem 0.5rem 0.875rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--card);
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
        /* the dropdown arrow is a static data-URI SVG (can't use a CSS var
           for its fill), so give dark mode its own lighter-arrow version */
        :root[data-theme="dark"] select {{
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23a39a8a' d='M6 8L2 4h8z'/%3E%3C/svg%3E");
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) select {{
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23a39a8a' d='M6 8L2 4h8z'/%3E%3C/svg%3E");
            }}
        }}
        select:hover {{ border-color: var(--accent); }}
        select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }}
        .group-section {{ margin-top: 2rem; }}
        .group-section:first-of-type {{ margin-top: 0; }}
        .group-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            color: var(--text);
        }}
        /* One per subgroup within a group that has them (currently just
           "Noun Phrase", subdivided by role pair -- e.g. "Head–
           Determiner") -- a group without subgroups renders its .grid
           directly under .group-title instead, no .subgroup-section
           wrapper at all. */
        .subgroup-section {{ margin-top: 1.5rem; }}
        .subgroup-section:first-of-type {{ margin-top: 0; }}
        .subgroup-title {{
            font-size: 0.875rem;
            font-weight: 600;
            margin: 0 0 0.85rem;
            color: var(--text-subtle);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}
        .panel {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--panel-bg);
            overflow: hidden;
            transition: box-shadow 0.15s ease, border-color 0.15s ease;
            cursor: pointer;
        }}
        .panel:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.10);
            border-color: var(--border-hover);
        }}
        .mini-plot {{
            width: 100%;
            height: 260px;
        }}
        .panel-label {{
            padding: 0.6rem 1rem 0.75rem;
            text-align: center;
            border-top: 1px solid var(--border);
            background: var(--card);
        }}
        .panel-label a {{
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.01em;
        }}
        .panel-label a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-section">
                <h1>MultiBLiMP v2 &mdash; Agreement Overview</h1>
                <p class="description">
                    This page gives an overview of agreement predictability across dependency relations and languages,
                    grouped by what the subject agrees with (verb, participle, auxiliary, ...). Each panel shows the
                    base entropy vs. reduced entropy (after fitting a decision tree) for a specific agreement feature
                    across all available languages. Click a panel to explore it in detail, or click a point to go
                    directly to a specific language.
                </p>
            </div>
            <div class="controls">
                <label for="entropyType">Entropy type:</label>
                <select id="entropyType">
                    <option value="six" selected>Six-class</option>
                    <option value="binary">Binary (majority vs. rest)</option>
                </select>
            </div>
        </div>

{sections_html}
    </div>

    <script>
        const allData = {all_data_json};

        let currentType = 'six';

        // Read once at load, after the theme CSS (light or dark) has applied --
        // Plotly's own layout config takes plain color strings, not CSS vars, so
        // this is the only way its mini-charts can follow the page's theme.
        const THEME = (() => {{
            const root = getComputedStyle(document.documentElement);
            const v = (name) => root.getPropertyValue(name).trim();
            return {{
                text: v('--text'),
                grid: v('--grid-line'),
                diagonal: v('--diagonal-line'),
                accent: v('--accent'),
            }};
        }})();

        function renderMiniPlot(el, points) {{
            const url = el.dataset.url;

            const baseLine = (() => {{
                const vals = points.map(d => d.base);
                const mn = Math.min(...vals), mx = Math.max(...vals);
                return {{ x: [mn, mx], y: [mn, mx] }};
            }})();

            const traceLine = {{
                x: baseLine.x,
                y: baseLine.y,
                mode: 'lines',
                type: 'scatter',
                line: {{ color: THEME.diagonal, width: 1, dash: 'dash' }},
                hoverinfo: 'skip',
                showlegend: false,
            }};

            const tracePoints = {{
                x: points.map(d => d.base),
                y: points.map(d => d.reduced),
                mode: 'markers',
                type: 'scatter',
                text: points.map(d => d.name),
                customdata: points.map(d => [d.url, d.n_items]),
                hovertemplate: '<b>%{{text}}</b><br>Base: %{{x:.3f}}<br>Reduced: %{{y:.3f}}<br>N: %{{customdata[1]:,}}<extra></extra>',
                marker: {{
                    size: points.map(d => d.n_items),
                    sizemode: 'area',
                    sizeref: 2 * Math.max(...points.map(d => d.n_items)) / (20 ** 2),
                    sizemin: 3,
                    color: points.map(d => d.color ?? THEME.accent),
                    opacity: 0.75,
                    line: {{
                        color: points.map(d => d.color ?? THEME.accent),
                        width: 0.5
                        }}
                     }},
                showlegend: false,
            }};

            const layout = {{
                margin: {{ t: 12, r: 12, b: 40, l: 44 }},
                xaxis: {{
                    title: {{ text: 'Base entropy', font: {{ size: 10 }} }},
                    gridcolor: THEME.grid,
                    zeroline: false,
                    tickfont: {{ size: 9 }},
                }},
                yaxis: {{
                    title: {{ text: 'Reduced entropy', font: {{ size: 10 }} }},
                    gridcolor: THEME.grid,
                    zeroline: false,
                    tickfont: {{ size: 9 }},
                }},
                // Transparent, not a fixed white/panel color: .panel paints its
                // own var(--panel-bg) behind the chart, light or dark.
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {{ family: 'DM Sans, system-ui, sans-serif', color: THEME.text }},
                hovermode: 'closest',
            }};

            const config = {{
                responsive: true,
                displayModeBar: false,
            }};

            Plotly.newPlot(el, [traceLine, tracePoints], layout, config);

            // Plotly fires its event synchronously before the DOM click bubbles to
            // the panel, so setting this flag here prevents the panel handler below
            // from also navigating when the user clicked a specific data point.
            el.on('plotly_click', function(evt) {{
                const pt = evt.points[0];
                if (pt && pt.customdata && pt.customdata[0]) {{
                    el.parentElement._pointClicked = true;
                    window.location.href = pt.customdata[0];
                }}
            }});
        }}

        function renderAll(type) {{
            document.querySelectorAll('.mini-plot').forEach(el => {{
                const deprel = el.dataset.deprel;
                const data = allData[deprel];
                if (data && data[type]) {{
                    renderMiniPlot(el, data[type]);
                }}
            }});
        }}

        // Set up panel-level navigation once — clicking the panel background
        // navigates to the deprel index, unless a scatter point was clicked.
        document.querySelectorAll('.panel').forEach(panel => {{
            panel.addEventListener('click', function() {{
                if (panel._pointClicked) {{
                    panel._pointClicked = false;
                    return;
                }}
                const url = panel.querySelector('.mini-plot').dataset.url;
                if (url) window.location.href = url;
            }});
        }});

        renderAll('six');

        document.getElementById('entropyType').addEventListener('change', (e) => {{
            currentType = e.target.value;
            renderAll(currentType);
        }});
    </script>
</body>
</html>"""
