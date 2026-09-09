"""Tests for plagiarism_engine.embedder (no heavy model loading)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plagiarism_engine.embedder import EmbeddingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_encode(texts, convert_to_numpy=True, show_progress_bar=False):
    """Deterministic fake encoder: returns normalised random vecs."""
    rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**31))
    vecs = rng.standard_normal((len(texts), 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_deterministic(self):
        eng = EmbeddingEngine()
        assert eng.fingerprint("hello") == eng.fingerprint("hello")

    def test_different_texts(self):
        eng = EmbeddingEngine()
        assert eng.fingerprint("foo") != eng.fingerprint("bar")

    def test_model_name_in_fingerprint(self):
        eng_a = EmbeddingEngine(model_name="model-A")
        eng_b = EmbeddingEngine(model_name="model-B")
        assert eng_a.fingerprint("text") != eng_b.fingerprint("text")

    def test_returns_hex(self):
        eng = EmbeddingEngine()
        fp = eng.fingerprint("test")
        int(fp, 16)  # must not raise


# ---------------------------------------------------------------------------
# Cache behaviour (no model loaded)
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit_avoids_model(self, tmp_path):
        eng = EmbeddingEngine(cache_dir=tmp_path)
        fake_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fp = eng.fingerprint("cached text")
        np.save(str(tmp_path / f"{fp}.npy"), fake_vec)

        # encode_one must return cached vec without touching the model
        result = eng.encode_one("cached text")
        np.testing.assert_array_almost_equal(result, fake_vec)
        assert eng._model is None  # model never loaded

    def test_cache_write_on_miss(self, tmp_path):
        eng = EmbeddingEngine(model_name="all-MiniLM-L6-v2", cache_dir=tmp_path)

        mock_model = MagicMock()
        mock_model.encode.side_effect = _fake_encode
        eng._model = mock_model

        eng.encode_one("new text")

        fp = eng.fingerprint("new text")
        cached_path = tmp_path / f"{fp}.npy"
        assert cached_path.exists()

    def test_no_cache_dir_works(self):
        eng = EmbeddingEngine(cache_dir=None)
        assert eng._cache_dir is None

    def test_corrupt_cache_fallback(self, tmp_path):
        """A corrupt cache file should not crash; it gets re-encoded."""
        eng = EmbeddingEngine(cache_dir=tmp_path)
        fp = eng.fingerprint("text")
        (tmp_path / f"{fp}.npy").write_bytes(b"CORRUPT")

        mock_model = MagicMock()
        mock_model.encode.side_effect = _fake_encode
        eng._model = mock_model

        result = eng.encode_one("text")
        assert result is not None


# ---------------------------------------------------------------------------
# encode() batch
# ---------------------------------------------------------------------------

class TestEncode:
    def _patched_engine(self, tmp_path=None):
        eng = EmbeddingEngine(cache_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.encode.side_effect = _fake_encode
        eng._model = mock_model
        return eng, mock_model

    def test_encode_shape(self, tmp_path):
        eng, _ = self._patched_engine(tmp_path)
        texts = ["one", "two", "three"]
        out = eng.encode(texts)
        assert out.shape[0] == 3
        assert out.ndim == 2

    def test_encode_partial_cache(self, tmp_path):
        eng, mock = self._patched_engine(tmp_path)
        texts = ["alpha", "beta", "gamma"]

        # Pre-cache 'alpha'
        pre_vec = np.array([1.0] * 64, dtype=np.float32)
        fp = eng.fingerprint("alpha")
        np.save(str(tmp_path / f"{fp}.npy"), pre_vec)

        out = eng.encode(texts)
        # Model must have been called only for beta + gamma
        assert mock.encode.call_count == 1
        call_texts = mock.encode.call_args[0][0]
        assert "alpha" not in call_texts
        assert "beta" in call_texts
        assert "gamma" in call_texts

    def test_encode_all_cached(self, tmp_path):
        eng, mock = self._patched_engine(tmp_path)
        texts = ["x", "y"]

        for t in texts:
            fp = eng.fingerprint(t)
            fake_vec = np.ones(64, dtype=np.float32) * 0.5
            np.save(str(tmp_path / f"{fp}.npy"), fake_vec)

        eng.encode(texts)
        # Model never called when everything is cached
        mock.encode.assert_not_called()
