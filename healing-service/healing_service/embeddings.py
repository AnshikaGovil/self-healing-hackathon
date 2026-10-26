"""Local semantic similarity via an open-source sentence encoder.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0, 22M params) which
runs on CPU in a few milliseconds per batch.  Nothing leaves the machine.

The encoder answers a question the token-overlap signals cannot: "does this
element *mean* what the business step is asking for?"  It is what lets the
agent map the intent 'the button that dismisses the cookie banner' onto an
element whose class names share no tokens with the broken selector at all.

Loading is lazy and every failure is non-fatal - if torch is missing or the
weights are not cached, the engine simply drops the semantic signal and keeps
healing on the remaining five.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

log = logging.getLogger("bis.embeddings")


class SemanticEncoder:
    """Thread-safe lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._failed = False
        self._cache = {}
        self._cache_limit = 2048

    @property
    def available(self) -> bool:
        return not self._failed

    def warmup(self) -> bool:
        """Load the model up-front so the first heal is not billed for it."""
        return self._ensure() is not None

    def _ensure(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                import os
                # Prefer the local cache; never block a run on a network fetch.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer
                log.info("loading semantic encoder %s", self.model_name)
                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:                      # pragma: no cover
                log.warning("semantic encoder unavailable (%s); "
                            "healing will use lexical signals only", exc)
                self._failed = True
        return self._model

    def encode(self, texts: List[str]):
        model = self._ensure()
        if model is None:
            return None
        return model.encode(
            texts, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def similarities(self, intent: str, sentences: List[str]) -> Optional[List[float]]:
        """Cosine similarity of one intent against many candidate sentences.

        Returned values are rescaled from raw cosine (which for this model
        clusters in roughly [0.0, 0.8] for related English text) onto [0, 1],
        so the signal is on the same footing as the lexical ones.
        """
        if not sentences:
            return None
        model = self._ensure()
        if model is None:
            return None
        try:
            import numpy as np
            key = intent
            if key in self._cache:
                qv = self._cache[key]
            else:
                qv = model.encode([intent], normalize_embeddings=True,
                                  convert_to_numpy=True)[0]
                # Bounded: this is a long-lived process on the one host the
                # whole fleet points at, and an intent is per *locator*, so
                # a fleet automating enough sites would otherwise grow this
                # for the life of the service. Oldest out, which for a run
                # working through a page is also least recently wanted.
                if len(self._cache) >= self._cache_limit:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = qv
            mv = model.encode(sentences, normalize_embeddings=True,
                              batch_size=64, show_progress_bar=False,
                              convert_to_numpy=True)
            cos = mv @ qv
            scaled = (cos - 0.05) / 0.65
            return [float(np.clip(v, 0.0, 1.0)) for v in scaled]
        except Exception as exc:                          # pragma: no cover
            log.warning("embedding pass failed: %s", exc)
            return None


_DEFAULT: Optional[SemanticEncoder] = None


def default_encoder(model_name: str) -> SemanticEncoder:
    """Process-wide singleton - the model is far too big to load per heal."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.model_name != model_name:
        _DEFAULT = SemanticEncoder(model_name)
    return _DEFAULT
