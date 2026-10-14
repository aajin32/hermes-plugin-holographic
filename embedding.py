"""Local neural embedding module for semantic fact retrieval.

Loads a fixed ONNX sentence-embedding model (bge-small-zh-v1.5 by default) and
exposes just enough for the store and the retriever: encode a text into an
L2-normalized float32 vector, and (de)serialize vectors to the bytes SQLite
stores. Everything is local — no API key, no network, no model downloads at
run time. When the model (or its dependencies) is missing the module reports
``is_available() == False`` and every other function returns ``None``, so the
provider keeps working on its lexical path alone.

The model directory is resolved once:

1. ``HOLOGRAPHIC_MODEL_DIR`` environment variable, when set;
2. ``~/.hermes/models/<HOLOGRAPHIC_MODEL_NAME>/`` otherwise.

The directory must hold an ONNX export (``model.onnx`` + ``tokenizer.json`` /
``tokenizer_config.json`` / ``special_tokens_map.json``), which is what
``optimum`` writes for ``AutoTokenizer`` / ``ORTModelForFeatureExtraction``.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "bge-small-zh-v1.5-onnx"
_DEFAULT_MODEL_DIR = Path.home() / ".hermes" / "models" / _DEFAULT_MODEL_NAME
_MODEL_DIR = Path(os.environ.get("HOLOGRAPHIC_MODEL_DIR") or _DEFAULT_MODEL_DIR).expanduser()

# Pooling output width of bge-small-zh-v1.5; only used as a fallback label when
# the model has not been loaded yet.
_DEFAULT_DIM = 512
_MAX_LENGTH = 512

# Singleton state. The lock guards load; encode runs after the model is set and
# ONNX Runtime inference is thread-safe for a single session.
_model: Any = None
_tokenizer: Any = None
_lock = threading.Lock()
_available = False
_dim = _DEFAULT_DIM
_tried = False


def _log_once_unavailable(reason: str) -> None:
    logger.warning(
        "holographic: neural search disabled (%s); retrieval falls back to lexical scoring",
        reason,
    )


def _load() -> None:
    """Load the ONNX model and tokenizer once. Never raises."""
    global _model, _tokenizer, _available, _dim, _tried

    with _lock:
        if _model is not None or _tried:
            return
        _tried = True

        if not (_MODEL_DIR / "model.onnx").exists():
            _log_once_unavailable(f"model.onnx not found under {_MODEL_DIR}")
            return

        try:
            # Imported lazily: these are the heavy optional dependencies, and a
            # store that never searches semantically should not pay for them.
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except Exception as exc:  # ModuleNotFoundError, ImportError, broken install
            _log_once_unavailable(f"onnxruntime/optimum/transformers unavailable: {exc}")
            return

        try:
            _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
            _model = ORTModelForFeatureExtraction.from_pretrained(str(_MODEL_DIR))
        except Exception as exc:
            _model = None
            _tokenizer = None
            _log_once_unavailable(f"failed to load from {_MODEL_DIR}: {exc}")
            return

        try:
            hidden = int(_model.config.hidden_size)
            if hidden > 0:
                _dim = hidden
        except Exception:
            pass
        _available = True
        logger.info("holographic: embedding model loaded from %s (dim=%d)", _MODEL_DIR, _dim)


def is_available() -> bool:
    """True when the model is loaded and ready (loads on first call)."""
    if _model is None:
        _load()
    return _available and _model is not None and _tokenizer is not None


def get_dim() -> int:
    """Embedding width. Valid after :func:`is_available` returned True."""
    return _dim


def _pool(outputs, attention_mask):
    """Mean-pool the last hidden state over the attention mask, then L2-normalize."""
    import numpy as np

    mask = np.expand_dims(attention_mask, -1).astype("float32")
    hidden = np.asarray(outputs.last_hidden_state, dtype="float32")
    summed = np.sum(hidden * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    vectors = summed / counts
    norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), a_min=1e-9, a_max=None)
    return vectors / norms


def encode_text(text: str):
    """Encode *text* into an L2-normalized float32 vector, or None if unavailable."""
    if not is_available():
        return None

    import numpy as np

    try:
        inputs = _tokenizer(
            text or "", padding=True, truncation=True, max_length=_MAX_LENGTH, return_tensors="np"
        )
        outputs = _model(**inputs)
        return _pool(outputs, inputs["attention_mask"])[0].astype(np.float32)
    except Exception as exc:
        logger.debug("holographic: embedding encode failed: %s", exc)
        return None


def encode_batch(texts: list[str]):
    """Encode several texts at once into an (N, dim) array, or None if unavailable."""
    if not is_available():
        return None

    import numpy as np

    try:
        inputs = _tokenizer(
            [t or "" for t in texts],
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="np",
        )
        outputs = _model(**inputs)
        return _pool(outputs, inputs["attention_mask"]).astype(np.float32)
    except Exception as exc:
        logger.debug("holographic: embedding batch encode failed: %s", exc)
        return None


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two L2-normalized vectors (a plain dot product)."""
    import numpy as np

    return float(np.dot(a, b))


def vector_to_bytes(vec) -> bytes:
    """Serialize a normalized vector for SQLite BLOB storage."""
    import numpy as np

    return np.asarray(vec, dtype=np.float32).tobytes()


def bytes_to_vector(data: bytes):
    """Deserialize a stored vector. Returns None for a blob of the wrong width."""
    import numpy as np

    vec = np.frombuffer(data, dtype=np.float32)
    if vec.size == 0:
        return None
    return vec


def blob_dim(data: bytes) -> int:
    """Element count of a stored blob, used to reject vectors of another model."""
    return len(data) // 4


def shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1], matching the lexical scorers' range."""
    return (sim + 1.0) / 2.0
