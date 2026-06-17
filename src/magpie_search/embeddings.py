"""embeddings — local text embedding for hybrid search.

Loads a small sentence-transformer model via fastembed (ONNX, CPU). Backs
the semantic side of `transcript.search`. Hard requirements:

  - No external API calls. Pure local inference. Privacy-preserving.
  - Graceful unavailability: if fastembed / onnxruntime / the model file
    cannot be loaded, `available()` returns False and the rest of the
    indexer + search stack falls back to FTS5-only with no error.
  - Model cache lives at $MAGPIE_SEARCH_MODELS_DIR (default ~/.magpie-search/models/;
    legacy AVIARY_MODELS_DIR honored), NOT the fastembed default (%TEMP%)
    which Windows will eventually clean.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80 MB).
Chosen for: small footprint, decent quality on technical text, Apache-2.0
license, runs on CPU.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

EMBED_DIM = 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_dir() -> Path:
    # Resolution: MAGPIE_SEARCH_MODELS_DIR > AVIARY_MODELS_DIR (legacy) > ~/.magpie-search/models/
    # Treat empty-string env vars as unset (else `Path("")` resolves to CWD).
    env = os.environ.get("MAGPIE_SEARCH_MODELS_DIR") or os.environ.get("AVIARY_MODELS_DIR")
    if env and env.strip():
        return Path(env)
    return Path.home() / ".magpie-search" / "models"


_model: Any = None
_model_lock = threading.Lock()
_load_failed = False
_load_error: str = ""


def _try_load() -> Any | None:
    """One-shot, thread-safe model load. Caches success and failure."""
    global _model, _load_failed, _load_error
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            cache = _cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding
            _model = TextEmbedding(
                model_name=DEFAULT_MODEL,
                cache_dir=str(cache),
            )
            return _model
        except Exception as e:
            _load_failed = True
            _load_error = f"{type(e).__name__}: {e}"
            return None


def available() -> bool:
    """True if the embedding model loads. Cached after first call."""
    return _try_load() is not None


def load_error() -> str:
    return _load_error


def embed_batch(texts: list[str]) -> list[bytes] | None:
    """Embed a batch. Returns list of float32-packed bytes ready for vec0,
    or None if the model is unavailable."""
    model = _try_load()
    if model is None:
        return None
    import numpy as np
    out: list[bytes] = []
    for vec in model.embed(texts):
        arr = np.asarray(vec, dtype=np.float32)
        out.append(arr.tobytes())
    return out


def embed_one(text: str) -> bytes | None:
    res = embed_batch([text])
    return res[0] if res else None
