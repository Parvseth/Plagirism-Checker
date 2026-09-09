"""
analyzer.py
===========
Multi-algorithm similarity engine with ensemble scoring.

Algorithms
----------
1. **Semantic cosine** — sentence-embedding cosine similarity (catches
   paraphrasing, synonym substitution, structural rewrites).
2. **TF-IDF n-gram overlap** — character n-gram TF-IDF cosine similarity
   (catches verbatim copying even when sentences are shuffled).
3. **Jaccard index** — word-level set similarity (fast, interpretable
   baseline; robust to short documents).

The *ensemble* score is a configurable weighted average of the three.

Public API
----------
PairResult       — dataclass holding all per-pair metrics
SimilarityAnalyzer(weights, threshold) — stateful analyzer
  .analyze(names, texts, embeddings) -> list[PairResult]
  .flag_suspicious(results)          -> list[PairResult]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """All similarity metrics for a single document pair."""

    doc_a: str
    doc_b: str

    # Individual metric scores in [0, 1]
    semantic_score: float = 0.0
    tfidf_score: float = 0.0
    jaccard_score: float = 0.0

    # Ensemble
    ensemble_score: float = 0.0

    # Metadata
    text_a_length: int = 0
    text_b_length: int = 0

    # Populated by reporter
    diff_snippet: str = ""

    @property
    def risk_level(self) -> str:
        """Human-readable risk tier based on ensemble score."""
        s = self.ensemble_score
        if s >= 0.85:
            return "🔴 High"
        if s >= 0.65:
            return "🟡 Medium"
        if s >= 0.45:
            return "🟢 Low"
        return "⚪ None"

    def to_dict(self) -> dict:
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "semantic_score": round(self.semantic_score, 4),
            "tfidf_score": round(self.tfidf_score, 4),
            "jaccard_score": round(self.jaccard_score, 4),
            "ensemble_score": round(self.ensemble_score, 4),
            "risk_level": self.risk_level,
            "text_a_length": self.text_a_length,
            "text_b_length": self.text_b_length,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b")


def _word_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _tfidf_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise TF-IDF cosine similarity using char n-grams."""
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(texts)
        return cosine_similarity(tfidf)
    except ValueError:
        # All-empty vocabulary — return zero matrix
        n = len(texts)
        return np.zeros((n, n))


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SimilarityAnalyzer:
    """
    Ensemble similarity scorer combining semantic + TF-IDF + Jaccard.

    Parameters
    ----------
    weights:
        Tuple (w_semantic, w_tfidf, w_jaccard).  Must sum to 1.0.
    threshold:
        Ensemble score above which a pair is considered suspicious.
    """

    def __init__(
        self,
        weights: tuple[float, float, float] = (0.55, 0.30, 0.15),
        threshold: float = 0.75,
    ) -> None:
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")
        self.weights = weights
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        names: list[str],
        texts: list[str],
        embeddings: np.ndarray,
    ) -> list[PairResult]:
        """
        Compute pairwise similarity for all document pairs.

        Parameters
        ----------
        names:
            Document identifiers (e.g. filenames).
        texts:
            Normalised plain-text strings aligned with *names*.
        embeddings:
            Precomputed embedding matrix, shape ``(N, D)``.

        Returns
        -------
        list[PairResult]
            Sorted by ensemble score descending.
        """
        n = len(names)
        if n < 2:
            return []

        w_sem, w_tfidf, w_jac = self.weights

        # Semantic cosine matrix
        sem_matrix = cosine_similarity(embeddings)

        # TF-IDF n-gram matrix
        tfidf_matrix = _tfidf_matrix(texts)

        results: list[PairResult] = []
        for i in range(n):
            for j in range(i + 1, n):
                sem = float(np.clip(sem_matrix[i, j], 0.0, 1.0))
                tfidf = float(np.clip(tfidf_matrix[i, j], 0.0, 1.0))
                jac = _jaccard(texts[i], texts[j])
                ensemble = w_sem * sem + w_tfidf * tfidf + w_jac * jac

                results.append(
                    PairResult(
                        doc_a=names[i],
                        doc_b=names[j],
                        semantic_score=sem,
                        tfidf_score=tfidf,
                        jaccard_score=jac,
                        ensemble_score=ensemble,
                        text_a_length=len(texts[i].split()),
                        text_b_length=len(texts[j].split()),
                    )
                )

        results.sort(key=lambda r: r.ensemble_score, reverse=True)
        return results

    def flag_suspicious(self, results: list[PairResult]) -> list[PairResult]:
        """Return only pairs whose ensemble score exceeds the threshold."""
        return [r for r in results if r.ensemble_score >= self.threshold]

    # ------------------------------------------------------------------
    # Full similarity matrix (for heatmap)
    # ------------------------------------------------------------------

    def similarity_matrix(
        self,
        names: list[str],
        results: list[PairResult],
    ) -> np.ndarray:
        """
        Reconstruct the N×N ensemble similarity matrix from *results*.

        The diagonal is set to 1.0 (self-similarity).
        """
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        mat = np.eye(n, dtype=float)

        for r in results:
            i, j = idx[r.doc_a], idx[r.doc_b]
            mat[i, j] = r.ensemble_score
            mat[j, i] = r.ensemble_score

        return mat
