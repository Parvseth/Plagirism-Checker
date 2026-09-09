"""Tests for plagiarism_engine.reporter"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from plagiarism_engine.analyzer import PairResult
from plagiarism_engine.reporter import ReportWriter, _diff_snippet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results():
    return [
        PairResult(
            doc_a="essay_a.txt",
            doc_b="essay_b.txt",
            semantic_score=0.92,
            tfidf_score=0.88,
            jaccard_score=0.75,
            ensemble_score=0.89,
            text_a_length=120,
            text_b_length=115,
        ),
        PairResult(
            doc_a="essay_a.txt",
            doc_b="essay_c.txt",
            semantic_score=0.30,
            tfidf_score=0.20,
            jaccard_score=0.10,
            ensemble_score=0.23,
            text_a_length=120,
            text_b_length=200,
        ),
    ]


@pytest.fixture
def sample_matrix():
    return np.array([
        [1.0, 0.89, 0.23],
        [0.89, 1.0, 0.15],
        [0.23, 0.15, 1.0],
    ])


@pytest.fixture
def names():
    return ["essay_a.txt", "essay_b.txt", "essay_c.txt"]


@pytest.fixture
def texts():
    return [
        "Machine learning is a subset of artificial intelligence.",
        "AI and machine learning are closely related fields in computer science.",
        "The stock market rose sharply after positive economic data was released.",
    ]


# ---------------------------------------------------------------------------
# _diff_snippet
# ---------------------------------------------------------------------------

class TestDiffSnippet:
    def test_returns_string(self):
        result = _diff_snippet("hello world", "hello earth")
        assert isinstance(result, str)

    def test_empty_strings(self):
        result = _diff_snippet("", "")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestToJson:
    def test_valid_json(self, sample_results, names, texts):
        writer = ReportWriter()
        out = writer.to_json(sample_results, names, texts)
        data = json.loads(out)  # must not raise
        assert "summary" in data
        assert "pairs" in data
        assert "documents" in data

    def test_summary_fields(self, sample_results, names, texts):
        writer = ReportWriter()
        data = json.loads(writer.to_json(sample_results, names, texts))
        summary = data["summary"]
        assert summary["total_documents"] == 3
        assert summary["total_pairs"] == 2
        assert summary["suspicious_pairs"] == 1  # only ensemble >= 0.75
        assert summary["max_similarity"] == pytest.approx(0.89, abs=0.001)

    def test_pair_count(self, sample_results, names, texts):
        writer = ReportWriter()
        data = json.loads(writer.to_json(sample_results, names, texts))
        assert len(data["pairs"]) == 2


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_has_header(self, sample_results):
        writer = ReportWriter()
        out = writer.to_csv(sample_results)
        reader = csv.reader(io.StringIO(out))
        header = next(reader)
        assert "ensemble_score" in header
        assert "doc_a" in header
        assert "risk_level" in header

    def test_row_count(self, sample_results):
        writer = ReportWriter()
        out = writer.to_csv(sample_results)
        rows = list(csv.reader(io.StringIO(out)))
        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_values(self, sample_results):
        writer = ReportWriter()
        out = writer.to_csv(sample_results)
        assert "essay_a.txt" in out
        assert "essay_b.txt" in out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

class TestToHtml:
    def test_is_html(self, sample_results, names, texts, sample_matrix):
        writer = ReportWriter()
        html = writer.to_html(sample_results, names, texts, sample_matrix)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_document_names(self, sample_results, names, texts, sample_matrix):
        writer = ReportWriter()
        html = writer.to_html(sample_results, names, texts, sample_matrix)
        for name in names:
            assert name in html

    def test_contains_scores(self, sample_results, names, texts, sample_matrix):
        writer = ReportWriter()
        html = writer.to_html(sample_results, names, texts, sample_matrix)
        assert "0.89" in html

    def test_contains_svg(self, sample_results, names, texts, sample_matrix):
        writer = ReportWriter()
        html = writer.to_html(sample_results, names, texts, sample_matrix)
        assert "<svg" in html


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, tmp_path):
        writer = ReportWriter()
        out_path = writer.save("hello", tmp_path / "sub" / "report.txt")
        assert out_path.exists()
        assert out_path.read_text() == "hello"

    def test_creates_parent_dirs(self, tmp_path):
        writer = ReportWriter()
        deep = tmp_path / "a" / "b" / "c" / "report.txt"
        writer.save("content", deep)
        assert deep.exists()
