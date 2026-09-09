"""Shared test fixtures for all plagiarism engine tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

SAMPLE_ORIGINAL = textwrap.dedent("""\
    Machine learning is a branch of artificial intelligence that focuses
    on building systems that learn from and make decisions based on data.
    Neural networks are a core component of deep learning architectures.
""")

SAMPLE_PARAPHRASE = textwrap.dedent("""\
    Deep learning is a subfield of AI concerned with training computational
    models that automatically improve through experience from datasets.
    Artificial neural networks form the backbone of modern deep learning.
""")

SAMPLE_DIFFERENT = textwrap.dedent("""\
    The stock market experienced significant volatility today as investors
    reacted to new economic data released by the federal reserve.
    Technology stocks led the decline, dropping more than three percent.
""")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_texts():
    return [SAMPLE_ORIGINAL, SAMPLE_PARAPHRASE, SAMPLE_DIFFERENT]


@pytest.fixture
def sample_names():
    return ["original.txt", "paraphrase.txt", "different.txt"]


@pytest.fixture
def sample_embeddings():
    """Deterministic fake embeddings for fast tests (no model loading)."""
    rng = np.random.default_rng(42)
    n, d = 3, 128
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    # Normalise to unit sphere so cosine similarity is well-defined
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def txt_file(tmp_path):
    """Factory fixture: returns a function that creates a .txt file."""
    def _make(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


@pytest.fixture
def pdf_file(tmp_path):
    """Factory fixture: creates a minimal PDF with given text."""
    def _make(name: str, content: str) -> Path:
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), content, fontsize=12)
            dst = tmp_path / name
            doc.save(str(dst))
            doc.close()
            return dst
        except ImportError:
            pytest.skip("PyMuPDF not installed")
    return _make


@pytest.fixture
def docx_file(tmp_path):
    """Factory fixture: creates a minimal DOCX with given text."""
    def _make(name: str, content: str) -> Path:
        try:
            import docx as docx_lib
            doc = docx_lib.Document()
            doc.add_paragraph(content)
            dst = tmp_path / name
            doc.save(str(dst))
            return dst
        except ImportError:
            pytest.skip("python-docx not installed")
    return _make
