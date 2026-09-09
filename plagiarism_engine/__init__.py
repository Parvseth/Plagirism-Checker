"""
plagiarism_engine
=================
Core package for multi-algorithm academic plagiarism detection.

Sub-modules
-----------
extractor  — Multi-format text extraction (TXT / PDF / DOCX)
embedder   — Sentence-level semantic embedding with local cache
analyzer   — Ensemble similarity scoring (cosine + TF-IDF + Jaccard)
reporter   — JSON / CSV / HTML report generation
"""

from .extractor import extract_text, extract_all
from .embedder import EmbeddingEngine
from .analyzer import SimilarityAnalyzer, PairResult
from .reporter import ReportWriter

__all__ = [
    "extract_text",
    "extract_all",
    "EmbeddingEngine",
    "SimilarityAnalyzer",
    "PairResult",
    "ReportWriter",
]

__version__ = "2.0.0"
