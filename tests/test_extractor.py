"""Tests for plagiarism_engine.extractor"""

from __future__ import annotations

import pytest
from plagiarism_engine.extractor import extract_text, extract_all, _normalise


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_collapses_whitespace(self):
        assert _normalise("hello   world\n\t  foo") == "hello world foo"

    def test_strips_leading_trailing(self):
        assert _normalise("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalise("") == ""


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

class TestExtractTxt:
    def test_basic_txt(self, txt_file):
        p = txt_file("a.txt", "Hello world, this is a test.")
        result = extract_text(p)
        assert "Hello world" in result

    def test_txt_with_extra_whitespace(self, txt_file):
        p = txt_file("b.txt", "line1\n\n\n   line2   ")
        result = extract_text(p)
        assert result == "line1 line2"

    def test_empty_txt(self, txt_file):
        p = txt_file("empty.txt", "")
        result = extract_text(p)
        assert result == ""


# ---------------------------------------------------------------------------
# File not found / unsupported
# ---------------------------------------------------------------------------

class TestExtractErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "nonexistent.txt")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "doc.xyz"
        p.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(p)


# ---------------------------------------------------------------------------
# extract_all
# ---------------------------------------------------------------------------

class TestExtractAll:
    def test_returns_dict(self, txt_file):
        p1 = txt_file("x.txt", "Content A")
        p2 = txt_file("y.txt", "Content B")
        result = extract_all([p1, p2])
        assert "x.txt" in result
        assert "y.txt" in result
        assert "Content A" in result["x.txt"]

    def test_skips_failed_file(self, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("ok")
        bad = tmp_path / "bad.xyz"
        bad.write_text("bad")
        import warnings
        with warnings.catch_warnings(record=True):
            result = extract_all([good, bad])
        assert "good.txt" in result
        assert "bad.xyz" not in result

    def test_empty_list(self):
        result = extract_all([])
        assert result == {}
