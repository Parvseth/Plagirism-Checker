"""
extractor.py
============
Robust multi-format text extraction with encoding auto-detection.

Supported formats
-----------------
* .txt  — UTF-8 with chardet fallback
* .pdf  — PyMuPDF (fitz) page-by-page extraction
* .docx — python-docx paragraph + table cell extraction

Public API
----------
extract_text(path: str | Path) -> str
extract_all(paths: list[str | Path]) -> dict[str, str]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so unit-tests can mock them
# ---------------------------------------------------------------------------

def _load_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install it with:  pip install pymupdf"
        ) from exc


def _load_docx():
    try:
        import docx
        return docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install it with:  pip install python-docx"
        ) from exc


def _detect_encoding(raw: bytes) -> str:
    """Best-effort encoding detection; falls back to utf-8."""
    try:
        import chardet
        result = chardet.detect(raw)
        return result.get("encoding") or "utf-8"
    except ImportError:
        return "utf-8"


# ---------------------------------------------------------------------------
# Core extractors
# ---------------------------------------------------------------------------

def _extract_txt(path: Path) -> str:
    raw = path.read_bytes()
    encoding = _detect_encoding(raw)
    return raw.decode(encoding, errors="replace")


def _extract_pdf(path: Path) -> str:
    fitz = _load_fitz()
    doc = fitz.open(str(path))
    pages: list[str] = []
    try:
        for page in doc:
            pages.append(page.get_text("text"))
    finally:
        doc.close()
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    docx = _load_docx()
    document = docx.Document(str(path))
    parts: list[str] = []

    # Paragraphs
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # Table cells (often missed by naive extractors)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse excess whitespace and strip leading/trailing space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


def extract_text(path: Union[str, Path]) -> str:
    """
    Extract and normalise plain text from *path*.

    Parameters
    ----------
    path:
        Absolute or relative path to a .txt, .pdf, or .docx file.

    Returns
    -------
    str
        Normalised plain-text content.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(_EXTRACTORS)}"
        )

    raw = extractor(p)
    return _normalise(raw)


def extract_all(
    paths: list[Union[str, Path]],
) -> dict[str, str]:
    """
    Extract text from multiple files.

    Parameters
    ----------
    paths:
        List of file paths to extract.

    Returns
    -------
    dict[filename, extracted_text]
        Files that fail extraction are logged but excluded from the result.
    """
    results: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        try:
            results[p.name] = extract_text(p)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"Skipping {p.name}: {exc}", stacklevel=2)
    return results
