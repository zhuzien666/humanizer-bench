"""A dependency-free sentence-restructuring attack.

Merges adjacent sentences (replacing the full stop with a comma) so that uniform,
evenly-paced machine text gains the long/short rhythm of human writing. This is a
real, well-known "humanizer" tactic for evading burstiness-based detectors, and it
needs no model -- so it works offline. The model-backed back-translation and
synonym attacks arrive in Phase 2.
"""

from __future__ import annotations

import random
import re

from .base import BaseAttack

_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceMergeAttack(BaseAttack):
    """Glue adjacent sentences together to increase sentence-length variation.

    Args:
        rate: Probability of merging at each adjacent sentence boundary
            (``1.0`` merges aggressively; ``0.0`` leaves the text unchanged).
        seed: Optional RNG seed for reproducibility.
    """

    name = "sentence_merge"

    def __init__(self, rate: float = 1.0, seed: int | None = 0) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate
        self._rng = random.Random(seed)

    def transform(self, text: str) -> str:
        sentences = _SPLIT.split(text.strip())
        if len(sentences) < 2:
            return text

        out: list[str] = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            if i + 1 < len(sentences) and self._rng.random() < self.rate:
                nxt = sentences[i + 1]
                body = current.rstrip(".!?")
                nxt = (nxt[0].lower() + nxt[1:]) if nxt else nxt
                out.append(f"{body}, {nxt}")
                i += 2
            else:
                out.append(current)
                i += 1
        return " ".join(out)
