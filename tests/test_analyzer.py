"""Tests for plagiarism_engine.analyzer"""

from __future__ import annotations

import numpy as np
import pytest

from plagiarism_engine.analyzer import SimilarityAnalyzer, PairResult, _jaccard


# ---------------------------------------------------------------------------
# Jaccard helper
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_texts(self):
        score = _jaccard("the quick brown fox", "the quick brown fox")
        assert score == pytest.approx(1.0)

    def test_completely_different(self):
        score = _jaccard("apple orange banana", "car truck bus")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = _jaccard("a b c d", "c d e f")
        # intersection={c,d}, union={a,b,c,d,e,f} → 2/6
        assert score == pytest.approx(2 / 6)

    def test_empty_strings(self):
        assert _jaccard("", "") == 0.0

    def test_case_insensitive(self):
        assert _jaccard("Hello World", "hello world") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SimilarityAnalyzer
# ---------------------------------------------------------------------------

class TestSimilarityAnalyzer:
    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SimilarityAnalyzer(weights=(0.5, 0.5, 0.5))

    def test_analyze_returns_correct_count(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        # 3 documents → 3 pairs (C(3,2)=3)
        assert len(results) == 3

    def test_analyze_sorted_descending(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        scores = [r.ensemble_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_analyze_scores_in_range(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        for r in results:
            assert 0.0 <= r.ensemble_score <= 1.0
            assert 0.0 <= r.semantic_score <= 1.0
            assert 0.0 <= r.tfidf_score <= 1.0
            assert 0.0 <= r.jaccard_score <= 1.0

    def test_analyze_result_names(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        all_docs = {r.doc_a for r in results} | {r.doc_b for r in results}
        assert all_docs == set(sample_names)

    def test_flag_suspicious_threshold(self, sample_names, sample_texts, sample_embeddings):
        # With threshold=0.0, everything is suspicious
        analyzer = SimilarityAnalyzer(threshold=0.0)
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        flagged = analyzer.flag_suspicious(results)
        assert len(flagged) == len(results)

    def test_flag_suspicious_high_threshold(self, sample_names, sample_texts, sample_embeddings):
        # With threshold=1.0, nothing is suspicious
        analyzer = SimilarityAnalyzer(threshold=1.0)
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        flagged = analyzer.flag_suspicious(results)
        assert len(flagged) == 0

    def test_less_than_two_docs(self, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(["only_one.txt"], ["some text"], sample_embeddings[:1])
        assert results == []

    def test_similarity_matrix_shape(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        mat = analyzer.similarity_matrix(sample_names, results)
        assert mat.shape == (len(sample_names), len(sample_names))

    def test_similarity_matrix_diagonal(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        mat = analyzer.similarity_matrix(sample_names, results)
        np.testing.assert_array_almost_equal(np.diag(mat), np.ones(len(sample_names)))

    def test_similarity_matrix_symmetric(self, sample_names, sample_texts, sample_embeddings):
        analyzer = SimilarityAnalyzer()
        results = analyzer.analyze(sample_names, sample_texts, sample_embeddings)
        mat = analyzer.similarity_matrix(sample_names, results)
        np.testing.assert_array_almost_equal(mat, mat.T)


# ---------------------------------------------------------------------------
# PairResult
# ---------------------------------------------------------------------------

class TestPairResult:
    def _make(self, score: float) -> PairResult:
        return PairResult(
            doc_a="a.txt",
            doc_b="b.txt",
            ensemble_score=score,
        )

    def test_risk_high(self):
        assert self._make(0.90).risk_level == "🔴 High"

    def test_risk_medium(self):
        assert self._make(0.70).risk_level == "🟡 Medium"

    def test_risk_low(self):
        assert self._make(0.50).risk_level == "🟢 Low"

    def test_risk_none(self):
        assert self._make(0.30).risk_level == "⚪ None"

    def test_to_dict_keys(self):
        r = self._make(0.80)
        d = r.to_dict()
        assert "ensemble_score" in d
        assert "risk_level" in d
        assert "doc_a" in d
