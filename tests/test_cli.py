"""Tests for cli.py — PlagiarismIQ command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import main, _gather_files


# ---------------------------------------------------------------------------
# Shared mock embedder — returns deterministic fake vectors without a model
# ---------------------------------------------------------------------------

def _fake_encode(texts):
    rng = np.random.default_rng(42)
    n = len(texts)
    vecs = rng.standard_normal((n, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# _gather_files
# ---------------------------------------------------------------------------

class TestGatherFiles:
    def test_direct_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        result = _gather_files([str(f)])
        assert f in result

    def test_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.pdf").write_text("fake pdf")  # won't extract but gathers
        result = _gather_files([str(tmp_path)])
        names = [p.name for p in result]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_deduplication(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        result = _gather_files([str(f), str(f)])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_requires_at_least_two_files(self, tmp_path):
        f = tmp_path / "only.txt"
        f.write_text("hello")
        ret = main([str(f), "--quiet"])
        assert ret == 1

    def test_successful_run(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Machine learning is a branch of artificial intelligence.")
        f2.write_text("Deep learning is a subset of machine learning and AI.")
        out_dir = tmp_path / "reports"
        with patch("plagiarism_engine.embedder.EmbeddingEngine.encode", side_effect=_fake_encode):
            ret = main([str(f1), str(f2), "--output-dir", str(out_dir), "--format", "json", "--quiet"])
        assert ret in (0, 2)  # 0=clean, 2=suspicious found
        assert (out_dir / "plagiarism_report.json").exists()

    def test_json_report_structure(self, tmp_path):
        f1 = tmp_path / "x.txt"
        f2 = tmp_path / "y.txt"
        f1.write_text("The quick brown fox jumps over the lazy dog.")
        f2.write_text("A fast brown fox leaps across a sleepy dog.")
        out_dir = tmp_path / "out"
        with patch("plagiarism_engine.embedder.EmbeddingEngine.encode", side_effect=_fake_encode):
            main([str(f1), str(f2), "--output-dir", str(out_dir), "--format", "json", "--quiet"])
        data = json.loads((out_dir / "plagiarism_report.json").read_text())
        assert "summary" in data
        assert "pairs" in data

    def test_all_formats_created(self, tmp_path):
        f1 = tmp_path / "p.txt"
        f2 = tmp_path / "q.txt"
        f1.write_text("Artificial intelligence transforms modern software development.")
        f2.write_text("AI is reshaping how modern software systems are built today.")
        out_dir = tmp_path / "all"
        with patch("plagiarism_engine.embedder.EmbeddingEngine.encode", side_effect=_fake_encode):
            main([str(f1), str(f2), "--output-dir", str(out_dir), "--format", "all", "--quiet"])
        assert (out_dir / "plagiarism_report.json").exists()
        assert (out_dir / "plagiarism_report.csv").exists()
        assert (out_dir / "plagiarism_report.html").exists()

    def test_weight_normalisation(self, tmp_path):
        f1 = tmp_path / "r.txt"
        f2 = tmp_path / "s.txt"
        f1.write_text("Natural language processing is fascinating.")
        f2.write_text("NLP is an exciting area of AI research.")
        out_dir = tmp_path / "norm"
        with patch("plagiarism_engine.embedder.EmbeddingEngine.encode", side_effect=_fake_encode):
            ret = main([
                str(f1), str(f2),
                "--weights", "1.0", "1.0", "1.0",  # non-normalised
                "--output-dir", str(out_dir),
                "--format", "json",
                "--quiet",
            ])
        assert ret in (0, 2)
