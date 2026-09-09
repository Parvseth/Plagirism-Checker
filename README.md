# PlagiarismIQ 🔍

> **AI-Powered Ensemble Plagiarism Detection** — Multi-algorithm academic integrity analysis with semantic understanding, n-gram fingerprinting, and a premium interactive dashboard.

[![CI](https://github.com/your-org/plagiarism-checker/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/plagiarism-checker/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Coverage](https://img.shields.io/badge/Coverage-86%25-brightgreen)](https://github.com/your-org/plagiarism-checker)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-57%20passed-brightgreen)](https://github.com/your-org/plagiarism-checker)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)

---

## 🌟 Why PlagiarismIQ?

Most plagiarism checkers use a single algorithm — PlagiarismIQ uses **three complementary algorithms** fused into an ensemble score, catching everything from verbatim copying to deep paraphrasing:

| Algorithm | Catches | Technique |
|---|---|---|
| 🧠 **Semantic Cosine** | Paraphrasing, synonym substitution | BERT sentence embeddings |
| 📝 **TF-IDF n-gram** | Verbatim sections, shuffled sentences | Character n-gram (3–5) TF-IDF |
| 🔤 **Jaccard Index** | Word-level overlap | Set intersection / union |

---

## ✨ Features

| Category | Feature |
|---|---|
| 📄 **File Support** | TXT, PDF, DOCX with encoding auto-detection |
| 🤖 **AI Models** | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, multilingual |
| ⚡ **Smart Caching** | SHA-256 fingerprint disk cache — re-encode only changed docs |
| 📊 **Interactive Heatmap** | Plotly similarity matrix with hover tooltips |
| 🔬 **Per-pair Drill-down** | Side-by-side text view + per-algorithm score bar chart |
| ⬇️ **Export** | JSON, CSV, self-contained HTML report |
| 🖥️ **CLI** | Rich colored output, batch processing, all export formats |
| ✅ **Tests** | 57 tests, 86% coverage, multi-OS CI/CD |
| 🛡️ **CI/CD** | GitHub Actions: lint + test matrix + security audit |

---

## 🏗️ Architecture

```
plagiarism_engine/
├── __init__.py      ← Public API surface
├── extractor.py     ← Multi-format text extraction (TXT/PDF/DOCX)
├── embedder.py      ← SentenceTransformer engine + SHA-256 cache
├── analyzer.py      ← Ensemble similarity scoring
└── reporter.py      ← JSON / CSV / HTML report generation

app.py              ← Premium Streamlit web UI
cli.py              ← Command-line interface
tests/
├── conftest.py      ← Shared fixtures
├── test_extractor.py
├── test_analyzer.py
├── test_reporter.py
├── test_embedder.py
└── test_cli.py
```

**Data flow:**
```
Files → extractor.py → clean text
                            ↓
               embedder.py → embeddings (cached)
                            ↓
              analyzer.py → PairResult list
              (cosine + TF-IDF + Jaccard → ensemble)
                            ↓
              reporter.py → JSON / CSV / HTML
              app.py      → Interactive Streamlit UI
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/plagiarism-checker.git
cd plagiarism-checker
pip install -r requirements.txt
```

### 2. Launch the Web UI

```bash
streamlit run app.py
```

Then open **http://localhost:8501**, upload 2+ documents, and click Analyze.

### 3. Use the CLI

```bash
# Analyze a folder of documents
python cli.py path/to/docs/ --output-dir reports/

# Specific files with custom threshold
python cli.py essay1.pdf essay2.docx essay3.txt --threshold 0.70

# Choose algorithm weights
python cli.py docs/ --weights 0.6 0.3 0.1

# JSON-only output
python cli.py docs/ --format json --quiet
```

---

## 📐 Algorithm Details

### Ensemble Score

```
ensemble = w₁ × semantic_cosine
         + w₂ × tfidf_ngram
         + w₃ × jaccard_index
```

Default weights: **w₁=0.55, w₂=0.30, w₃=0.15** (configurable in UI and CLI).

### Risk Tiers

| Ensemble Score | Risk Level |
|---|---|
| ≥ 0.85 | 🔴 High |
| 0.65 – 0.85 | 🟡 Medium |
| 0.45 – 0.65 | 🟢 Low |
| < 0.45 | ⚪ None |

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=plagiarism_engine --cov=cli --cov-report=term-missing

# Run specific module
pytest tests/test_analyzer.py -v
```

**Test results (57 tests):**

| Module | Tests | Coverage |
|---|---|---|
| `__init__.py` | — | 100% |
| `analyzer.py` | 18 | 96% |
| `embedder.py` | 12 | 90% |
| `reporter.py` | 12 | 100% |
| `extractor.py` | 11 | 57% |
| **Total** | **57** | **86%** |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PLAGIARISM_CACHE_DIR` | `.plagiarism_cache` | Embedding cache directory |
| `PLAGIARISM_MODEL` | `all-MiniLM-L6-v2` | Default embedding model |
| `PLAGIARISM_THRESHOLD` | `0.75` | Default suspicion threshold |

### CLI Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success — no suspicious pairs |
| `1` | Error (insufficient files, extraction failure) |
| `2` | Success — suspicious pairs detected |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Embedding | `sentence-transformers` (BERT-family) |
| Similarity | `scikit-learn` (cosine, TF-IDF) |
| UI | `streamlit` + `plotly` |
| PDF | `pymupdf` (PyMuPDF) |
| DOCX | `python-docx` |
| Encoding | `chardet` |
| Testing | `pytest` + `pytest-cov` |
| CI/CD | GitHub Actions (multi-OS matrix) |

---

## 📊 Use Cases

- 🎓 **Academic Institutions** — Detect student assignment plagiarism at scale
- ✍️ **Content Creators** — Verify originality before publishing
- 🏢 **Legal & Compliance** — Cross-check documents for copied content
- 🧑‍💻 **Developers** — Verify code documentation and technical writing

---

## 📁 Project Structure

```
Plagirism-Checker/
├── .github/
│   └── workflows/
│       └── ci.yml          ← GitHub Actions CI (lint + test + security)
├── plagiarism_engine/
│   ├── __init__.py
│   ├── extractor.py
│   ├── embedder.py
│   ├── analyzer.py
│   └── reporter.py
├── tests/
│   ├── conftest.py
│   ├── test_extractor.py
│   ├── test_analyzer.py
│   ├── test_reporter.py
│   ├── test_embedder.py
│   └── test_cli.py
├── docs/
│   └── ARCHITECTURE.md
├── app.py                  ← Streamlit web application
├── cli.py                  ← Command-line interface
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

1. Fork → feature branch → PR
2. All PRs must pass CI (lint + 57 tests + security audit)
3. Follow existing code style (type hints, docstrings, class-based tests)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
