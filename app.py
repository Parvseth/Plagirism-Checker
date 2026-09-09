"""
app.py — Plagiarism Checker v2
==============================
Premium Streamlit web interface for multi-algorithm academic plagiarism
detection.  Powered by the plagiarism_engine package.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# ── make sure plagiarism_engine is importable regardless of cwd ──────────────
sys.path.insert(0, str(Path(__file__).parent))

from plagiarism_engine import (
    EmbeddingEngine,
    PairResult,
    ReportWriter,
    SimilarityAnalyzer,
    extract_all,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlagiarismIQ — AI Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- global reset & fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: "Inter", sans-serif; }

/* ---- dark background ---- */
.stApp { background: #0f1117; }
section[data-testid="stSidebar"] { background: #131620 !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #8892a4; }

/* ---- metric cards ---- */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2e3348;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label { color: #8892a4 !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; }

/* ---- alerts ---- */
.risk-high  { background:#ff4d4d1a; border-left:4px solid #ff4d4d; color:#ff4d4d; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
.risk-med   { background:#f5a6231a; border-left:4px solid #f5a623; color:#f5a623; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
.risk-low   { background:#4caf501a; border-left:4px solid #4caf50; color:#4caf50; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }
.risk-none  { background:#88888818; border-left:4px solid #888; color:#888; border-radius:8px; padding:.75rem 1rem; margin:.5rem 0; }

/* ---- expander ---- */
.stExpander { background:#1a1d27 !important; border:1px solid #2e3348 !important; border-radius:12px !important; }

/* ---- tabs ---- */
button[data-baseweb="tab"] { font-size:0.9rem; font-weight:500; }

/* ---- progress ---- */
.stProgress > div > div { background: linear-gradient(90deg,#7c6aff,#06b6d4); }

/* ---- buttons ---- */
.stDownloadButton button {
    background: linear-gradient(135deg,#7c6aff,#06b6d4) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- file uploader ---- */
[data-testid="stFileUploader"] {
    background: #1a1d27;
    border: 2px dashed #2e3348;
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    threshold = st.slider(
        "🎯 Plagiarism Threshold",
        min_value=0.40,
        max_value=0.99,
        value=0.75,
        step=0.01,
        help="Pairs with ensemble score ≥ this value are flagged as suspicious.",
    )

    st.markdown("### 🧮 Algorithm Weights")
    w_semantic = st.slider("Semantic (BERT)", 0.0, 1.0, 0.55, 0.05)
    w_tfidf = st.slider("TF-IDF n-gram", 0.0, 1.0, 0.30, 0.05)
    w_jaccard = st.slider("Jaccard Index", 0.0, 1.0, 0.15, 0.05)

    total_w = w_semantic + w_tfidf + w_jaccard
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f} — they will be normalised.")
        norm = total_w
        w_semantic /= norm
        w_tfidf /= norm
        w_jaccard /= norm

    st.markdown("---")
    st.markdown("### 🤖 Embedding Model")
    model_choice = st.selectbox(
        "Model",
        [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ],
        help="all-MiniLM-L6-v2 is fastest. all-mpnet-base-v2 is most accurate.",
    )

    st.markdown("---")
    st.caption("PlagiarismIQ v2.0 · Ensemble AI Detection")


# ── header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="
  background: linear-gradient(135deg,#1a1d27 0%,#0d1021 100%);
  border: 1px solid #2e3348;
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
">
  <span style="font-size:3rem">🔍</span>
  <div>
    <h1 style="margin:0;font-size:1.9rem;font-weight:700;
               background:linear-gradient(90deg,#7c6aff,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">
      PlagiarismIQ
    </h1>
    <p style="margin:0.3rem 0 0;color:#8892a4;font-size:0.95rem">
      AI-powered ensemble plagiarism detection · Semantic + TF-IDF + Jaccard
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── file upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📂 Upload documents to compare (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    help="Upload at least 2 documents to begin analysis.",
)

if not uploaded_files:
    st.info("👆 Upload two or more documents to start plagiarism analysis.")
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Please upload at least **2 documents** to compare.")
    st.stop()

# ── extract text ──────────────────────────────────────────────────────────────
with st.spinner("📄 Extracting text from documents…"):
    texts_dict: dict[str, str] = {}
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        for uf in uploaded_files:
            dst = Path(tmp) / uf.name
            dst.write_bytes(uf.getbuffer())
            try:
                extracted = extract_all([dst])
                if extracted:
                    texts_dict[uf.name] = list(extracted.values())[0]
                else:
                    errors.append(uf.name)
            except Exception as e:
                errors.append(f"{uf.name}: {e}")

if errors:
    for err in errors:
        st.error(f"❌ Failed to read: {err}")

names = list(texts_dict.keys())
texts = list(texts_dict.values())

if len(names) < 2:
    st.error("Not enough readable documents to compare.")
    st.stop()

# ── embed ─────────────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="🤖 Generating semantic embeddings…")

@st.cache_resource(show_spinner=False)
def get_engine(model_name: str) -> EmbeddingEngine:
    return EmbeddingEngine(model_name=model_name)

engine = get_engine(model_choice)

with st.spinner("🤖 Computing embeddings…"):
    embeddings = engine.encode(texts)

progress_bar.progress(50, text="📐 Running similarity analysis…")

# ── analyze ───────────────────────────────────────────────────────────────────
analyzer = SimilarityAnalyzer(
    weights=(w_semantic, w_tfidf, w_jaccard),
    threshold=threshold,
)
results: list[PairResult] = analyzer.analyze(names, texts, embeddings)
mat: np.ndarray = analyzer.similarity_matrix(names, results)
suspicious = analyzer.flag_suspicious(results)

progress_bar.progress(100, text="✅ Analysis complete!")
progress_bar.empty()

# ── metric strip ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documents", len(names))
c2.metric("🔗 Pairs Analyzed", len(results))
c3.metric("⚠️ Suspicious Pairs", len(suspicious))
max_score = max((r.ensemble_score for r in results), default=0.0)
c4.metric("🔴 Max Similarity", f"{max_score:.1%}")

st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_heat, tab_pairs, tab_detail, tab_export = st.tabs(
    ["📊 Heatmap", "📋 Pair Results", "🔬 Detail View", "⬇️ Export"]
)

# ── heatmap tab ───────────────────────────────────────────────────────────────
with tab_heat:
    import plotly.express as px
    import pandas as pd

    df_heat = pd.DataFrame(mat, index=names, columns=names)
    fig = px.imshow(
        df_heat,
        color_continuous_scale="RdYlBu_r",
        zmin=0,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Ensemble Similarity Matrix",
    )
    fig.update_layout(
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font=dict(color="#e2e8f0", family="Inter"),
        title_font_size=18,
        coloraxis_colorbar=dict(
            tickfont=dict(color="#e2e8f0"),
            title=dict(text="Score", font=dict(color="#e2e8f0")),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(tickfont=dict(size=11), side="bottom")
    fig.update_yaxes(tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

# ── pair results tab ──────────────────────────────────────────────────────────
with tab_pairs:
    import pandas as pd

    if suspicious:
        st.markdown(f"#### 🚨 {len(suspicious)} suspicious pair(s) detected")
        for r in suspicious:
            risk_css = {
                "🔴 High": "risk-high",
                "🟡 Medium": "risk-med",
            }.get(r.risk_level, "risk-low")
            st.markdown(
                f'<div class="{risk_css}">'
                f"<strong>{r.doc_a}</strong> ↔ <strong>{r.doc_b}</strong> &nbsp;|&nbsp; "
                f"Ensemble: <strong>{r.ensemble_score:.3f}</strong> &nbsp;|&nbsp; {r.risk_level}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No suspicious pairs detected above the threshold.")

    st.markdown("#### All Pairs")
    df_pairs = pd.DataFrame([r.to_dict() for r in results])
    st.dataframe(
        df_pairs.style.background_gradient(
            subset=["ensemble_score"], cmap="RdYlGn_r", vmin=0, vmax=1
        ).format(
            {
                "ensemble_score": "{:.3f}",
                "semantic_score": "{:.3f}",
                "tfidf_score": "{:.3f}",
                "jaccard_score": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=400,
    )

# ── detail view tab ───────────────────────────────────────────────────────────
with tab_detail:
    if len(results) == 0:
        st.info("No pairs to show.")
    else:
        pair_labels = [f"{r.doc_a} ↔ {r.doc_b}" for r in results]
        selected_label = st.selectbox("Select a pair to inspect", pair_labels)
        selected = results[pair_labels.index(selected_label)]

        col_a, col_b = st.columns(2)
        idx_a = names.index(selected.doc_a)
        idx_b = names.index(selected.doc_b)

        with col_a:
            st.markdown(f"**📄 {selected.doc_a}**")
            st.markdown(
                f"<div style='background:#1a1d27;border:1px solid #2e3348;"
                f"border-radius:10px;padding:1rem;max-height:320px;overflow-y:auto;"
                f"font-size:0.85rem;color:#e2e8f0;white-space:pre-wrap;'>"
                f"{texts[idx_a][:2000]}{'…' if len(texts[idx_a]) > 2000 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(f"**📄 {selected.doc_b}**")
            st.markdown(
                f"<div style='background:#1a1d27;border:1px solid #2e3348;"
                f"border-radius:10px;padding:1rem;max-height:320px;overflow-y:auto;"
                f"font-size:0.85rem;color:#e2e8f0;white-space:pre-wrap;'>"
                f"{texts[idx_b][:2000]}{'…' if len(texts[idx_b]) > 2000 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Score Breakdown")
        import plotly.graph_objects as go

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Bar(
                x=["Semantic", "TF-IDF n-gram", "Jaccard"],
                y=[
                    selected.semantic_score,
                    selected.tfidf_score,
                    selected.jaccard_score,
                ],
                marker_color=["#7c6aff", "#06b6d4", "#4caf50"],
                text=[
                    f"{selected.semantic_score:.3f}",
                    f"{selected.tfidf_score:.3f}",
                    f"{selected.jaccard_score:.3f}",
                ],
                textposition="outside",
            )
        )
        fig_radar.update_layout(
            paper_bgcolor="#1a1d27",
            plot_bgcolor="#1a1d27",
            font=dict(color="#e2e8f0", family="Inter"),
            yaxis=dict(range=[0, 1.1], gridcolor="#2e3348"),
            xaxis=dict(gridcolor="#2e3348"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Risk badge
        risk_css = {
            "🔴 High": "risk-high",
            "🟡 Medium": "risk-med",
            "🟢 Low": "risk-low",
            "⚪ None": "risk-none",
        }.get(selected.risk_level, "risk-none")
        st.markdown(
            f'<div class="{risk_css}" style="font-size:1rem">'
            f"Risk Level: <strong>{selected.risk_level}</strong> &nbsp;|&nbsp; "
            f"Ensemble Score: <strong>{selected.ensemble_score:.4f}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── export tab ────────────────────────────────────────────────────────────────
with tab_export:
    writer = ReportWriter()

    st.markdown("### 📥 Download Results")

    col_j, col_c, col_h = st.columns(3)

    with col_j:
        json_str = writer.to_json(results, names, texts)
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name="plagiarism_report.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption("Machine-readable structured report with all metrics.")

    with col_c:
        csv_str = writer.to_csv(results)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_str,
            file_name="plagiarism_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Spreadsheet-compatible pair result table.")

    with col_h:
        html_str = writer.to_html(results, names, texts, mat)
        st.download_button(
            "⬇️ Download HTML Report",
            data=html_str,
            file_name="plagiarism_report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.caption("Self-contained dashboard (no internet required).")
