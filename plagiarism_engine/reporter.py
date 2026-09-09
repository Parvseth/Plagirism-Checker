"""
reporter.py
===========
Multi-format report generation for plagiarism analysis results.

Formats
-------
* JSON  — machine-readable structured report
* CSV   — spreadsheet-compatible pair table
* HTML  — standalone, self-contained premium dark-theme dashboard
          (inline CSS + SVG heatmap, no external CDN deps)

Public API
----------
ReportWriter
  .to_json(results, names, texts)      -> str
  .to_csv(results)                     -> str
  .to_html(results, names, texts, mat) -> str
  .save(content, path)                 -> Path
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .analyzer import PairResult


# ---------------------------------------------------------------------------
# Diff snippet helper
# ---------------------------------------------------------------------------

def _diff_snippet(text_a: str, text_b: str, n: int = 5) -> str:
    """Return a unified diff of the two longest shared sub-sequences."""
    words_a = text_a.split()[:300]
    words_b = text_b.split()[:300]
    diff = list(
        difflib.unified_diff(
            words_a,
            words_b,
            lineterm="",
            n=n,
        )
    )
    return " ".join(diff[:60]) if diff else ""


# ---------------------------------------------------------------------------
# SVG Heatmap renderer
# ---------------------------------------------------------------------------

def _svg_heatmap(names: list[str], matrix: np.ndarray) -> str:
    """Generate a self-contained SVG heatmap for the similarity matrix."""
    n = len(names)
    cell = 72
    label_pad = 140
    size = n * cell + label_pad

    def rgb(score: float) -> str:
        # 0 → deep blue, 0.5 → teal, 1.0 → vivid red
        r = int(min(255, score * 2 * 255))
        g = int(min(255, (1 - abs(score - 0.5) * 2) * 200))
        b = int(min(255, (1 - score) * 2 * 255))
        return f"#{r:02x}{g:02x}{b:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'font-family="Inter, sans-serif">',
        '<rect width="100%" height="100%" fill="#0f1117"/>',
    ]

    # Column labels (rotated)
    for j, name in enumerate(names):
        x = label_pad + j * cell + cell // 2
        short = (name[:10] + "…") if len(name) > 11 else name
        lines.append(
            f'<text x="{x}" y="{label_pad - 8}" '
            f'text-anchor="start" fill="#e2e8f0" font-size="11" '
            f'transform="rotate(-40,{x},{label_pad - 8})">{short}</text>'
        )

    # Row labels
    for i, name in enumerate(names):
        y = label_pad + i * cell + cell // 2 + 4
        short = (name[:14] + "…") if len(name) > 15 else name
        lines.append(
            f'<text x="{label_pad - 8}" y="{y}" '
            f'text-anchor="end" fill="#e2e8f0" font-size="11">{short}</text>'
        )

    # Cells
    for i in range(n):
        for j in range(n):
            score = float(matrix[i, j])
            x = label_pad + j * cell
            y = label_pad + i * cell
            color = rgb(score)
            label = f"{score:.2f}"
            text_fill = "#ffffff" if score > 0.5 else "#e2e8f0"
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color}" rx="4"/>'
                f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" '
                f'text-anchor="middle" fill="{text_fill}" '
                f'font-size="12" font-weight="600">{label}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------

class ReportWriter:
    """Converts analysis results into JSON, CSV, or HTML reports."""

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(
        self,
        results: list[PairResult],
        names: list[str],
        texts: list[str],
    ) -> str:
        payload = {
            "summary": {
                "total_documents": len(names),
                "total_pairs": len(results),
                "suspicious_pairs": sum(
                    1 for r in results if r.ensemble_score >= 0.75
                ),
                "max_similarity": max(
                    (r.ensemble_score for r in results), default=0.0
                ),
            },
            "documents": [
                {"name": n, "word_count": len(t.split())}
                for n, t in zip(names, texts)
            ],
            "pairs": [r.to_dict() for r in results],
        }
        return json.dumps(payload, indent=2)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, results: list[PairResult]) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "doc_a",
                "doc_b",
                "ensemble_score",
                "semantic_score",
                "tfidf_score",
                "jaccard_score",
                "risk_level",
                "text_a_length",
                "text_b_length",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def to_html(
        self,
        results: list[PairResult],
        names: list[str],
        texts: list[str],
        matrix: np.ndarray,
    ) -> str:
        suspicious = [r for r in results if r.ensemble_score >= 0.75]
        svg = _svg_heatmap(names, matrix)

        # Pair rows
        pair_rows = ""
        for r in results:
            color_map = {
                "🔴 High": "#ff4d4d",
                "🟡 Medium": "#f5a623",
                "🟢 Low": "#4caf50",
                "⚪ None": "#888",
            }
            badge_color = color_map.get(r.risk_level, "#888")
            pair_rows += f"""
            <tr>
              <td>{r.doc_a}</td>
              <td>{r.doc_b}</td>
              <td>
                <div class="score-bar-wrap">
                  <div class="score-bar" style="width:{r.ensemble_score*100:.1f}%;background:{badge_color}"></div>
                  <span>{r.ensemble_score:.3f}</span>
                </div>
              </td>
              <td>{r.semantic_score:.3f}</td>
              <td>{r.tfidf_score:.3f}</td>
              <td>{r.jaccard_score:.3f}</td>
              <td><span class="badge" style="background:{badge_color}20;color:{badge_color};border:1px solid {badge_color}40">{r.risk_level}</span></td>
            </tr>"""

        total_docs = len(names)
        total_pairs = len(results)
        suspicious_count = len(suspicious)
        max_score = max((r.ensemble_score for r in results), default=0.0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Plagiarism Analysis Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
      --border: #2e3348; --text: #e2e8f0; --muted: #8892a4;
      --accent: #7c6aff; --accent2: #06b6d4; --danger: #ff4d4d;
      --warn: #f5a623; --ok: #4caf50;
    }}
    body {{ background: var(--bg); color: var(--text); font-family: "Inter", sans-serif; min-height: 100vh; }}

    header {{
      background: linear-gradient(135deg, #1a1d27 0%, #0d1021 100%);
      border-bottom: 1px solid var(--border);
      padding: 2rem 3rem;
      display: flex; align-items: center; gap: 1.5rem;
    }}
    header .logo {{ font-size: 2.5rem; }}
    header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }}
    header p {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }}

    .container {{ max-width: 1200px; margin: 0 auto; padding: 2.5rem 2rem; }}

    /* Stat cards */
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem; }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      text-align: center;
    }}
    .stat-card .value {{ font-size: 2.5rem; font-weight: 700; line-height: 1; }}
    .stat-card .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }}

    /* Section headers */
    .section {{ margin-bottom: 2.5rem; }}
    .section-title {{
      font-size: 1.1rem; font-weight: 600;
      color: var(--text);
      margin-bottom: 1rem;
      display: flex; align-items: center; gap: 0.5rem;
    }}
    .section-title::after {{
      content: ""; flex: 1; height: 1px; background: var(--border); margin-left: 0.5rem;
    }}

    /* Heatmap */
    .heatmap-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      overflow-x: auto;
    }}

    /* Pair table */
    .table-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead tr {{ background: var(--surface2); }}
    th {{
      padding: 0.85rem 1rem;
      text-align: left;
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    td {{
      padding: 0.85rem 1rem;
      font-size: 0.88rem;
      border-top: 1px solid var(--border);
    }}
    tr:hover td {{ background: var(--surface2); transition: background 0.15s; }}

    /* Score bar */
    .score-bar-wrap {{
      display: flex; align-items: center; gap: 0.6rem;
    }}
    .score-bar {{
      height: 6px; border-radius: 3px;
      min-width: 4px; max-width: 120px;
      flex-shrink: 0;
    }}

    /* Badge */
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.65rem;
      border-radius: 99px;
      font-size: 0.78rem;
      font-weight: 500;
      white-space: nowrap;
    }}

    /* Footer */
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
      padding: 2rem;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <span class="logo">🔍</span>
    <div>
      <h1>Plagiarism Analysis Report</h1>
      <p>AI-powered ensemble similarity detection across {total_docs} documents</p>
    </div>
  </header>

  <div class="container">

    <!-- Stats -->
    <div class="stats">
      <div class="stat-card">
        <div class="value" style="color:var(--accent)">{total_docs}</div>
        <div class="label">Documents Analyzed</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:var(--accent2)">{total_pairs}</div>
        <div class="label">Pairs Compared</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:var(--danger)">{suspicious_count}</div>
        <div class="label">Suspicious Pairs</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color:var(--warn)">{max_score:.2f}</div>
        <div class="label">Max Similarity</div>
      </div>
    </div>

    <!-- Heatmap -->
    <div class="section">
      <div class="section-title">📊 Similarity Heatmap</div>
      <div class="heatmap-wrap">
        {svg}
      </div>
    </div>

    <!-- Pair Table -->
    <div class="section">
      <div class="section-title">📋 All Pair Results</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Document A</th>
              <th>Document B</th>
              <th>Ensemble Score</th>
              <th>Semantic</th>
              <th>TF-IDF</th>
              <th>Jaccard</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {pair_rows}
          </tbody>
        </table>
      </div>
    </div>

  </div>
  <footer>Generated by Plagiarism Checker v2.0 · Ensemble AI Detection Engine</footer>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    def save(self, content: str, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p
