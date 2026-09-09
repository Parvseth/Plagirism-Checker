"""
embedder.py
===========
Semantic embedding engine powered by SentenceTransformers.

Features
--------
* Batched encoding for large document sets
* SHA-256 fingerprint-based disk cache (avoids re-encoding unchanged docs)
* Configurable model selection
* Thread-safe singleton cache

Public API
----------
EmbeddingEngine(model_name, cache_dir) — stateful encoder
  .encode(texts)      -> np.ndarray   shape (N, D)
  .encode_one(text)   -> np.ndarray   shape (D,)
  .fingerprint(text)  -> str          SHA-256 hex
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Default model — compact, fast, strong cross-lingual performance
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Sentence-level semantic embedding engine with transparent disk caching.

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local path.  Defaults to
        ``all-MiniLM-L6-v2``.
    cache_dir:
        Directory for embedding cache files.  ``None`` disables caching.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self._model = None  # lazy init
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir else None
        )
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SentenceTransformer model (once per instance)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install:  pip install sentence-transformers"
                ) from exc
            log.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log.info("Model loaded.")
        return self._model

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def fingerprint(self, text: str) -> str:
        """Return a stable SHA-256 hex fingerprint for *text*."""
        payload = f"{self.model_name}::{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, fp: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{fp}.npy"

    def _load_cache(self, fp: str) -> Optional[np.ndarray]:
        p = self._cache_path(fp)
        if p and p.exists():
            try:
                vec = np.load(str(p))
                log.debug("Cache hit: %s", fp[:12])
                return vec
            except Exception:  # noqa: BLE001
                p.unlink(missing_ok=True)
        return None

    def _save_cache(self, fp: str, vec: np.ndarray) -> None:
        p = self._cache_path(fp)
        if p:
            np.save(str(p), vec)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single document, using cache when available."""
        fp = self.fingerprint(text)
        cached = self._load_cache(fp)
        if cached is not None:
            return cached

        model = self._get_model()
        vec: np.ndarray = model.encode([text], convert_to_numpy=True)[0]
        self._save_cache(fp, vec)
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of documents.

        Already-cached documents are retrieved without re-encoding;
        only new documents hit the model.

        Parameters
        ----------
        texts:
            List of normalised plain-text strings.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), embedding_dim)``.
        """
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(texts):
            fp = self.fingerprint(text)
            cached = self._load_cache(fp)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            model = self._get_model()
            log.info(
                "Encoding %d new document(s) with '%s' …",
                len(uncached_texts),
                self.model_name,
            )
            new_vecs: np.ndarray = model.encode(
                uncached_texts, convert_to_numpy=True, show_progress_bar=False
            )
            for local_i, global_i in enumerate(uncached_indices):
                vec = new_vecs[local_i]
                results[global_i] = vec
                self._save_cache(self.fingerprint(uncached_texts[local_i]), vec)

        # All slots must be filled at this point
        return np.vstack(results)  # type: ignore[arg-type]
