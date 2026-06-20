"""Abstract base class for AI-text detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class BaseDetector(ABC):
    """Common interface that every detector must implement.

    A detector maps a text string to a probability in ``[0, 1]`` that the text
    was machine-generated. Subclasses only need to implement :meth:`score`;
    :meth:`predict` and :meth:`score_batch` are provided for convenience.
    """

    #: Short identifier used in result tables. Override in subclasses.
    name: str = "base"

    @abstractmethod
    def score(self, text: str) -> float:
        """Return ``P(text is AI-generated)`` as a float in ``[0, 1]``."""
        raise NotImplementedError

    def score_batch(self, texts: Iterable[str]) -> List[float]:
        """Score many texts. Override for a faster batched implementation."""
        return [self.score(t) for t in texts]

    def predict(self, text: str, threshold: float = 0.5) -> int:
        """Return ``1`` if ``score(text) >= threshold`` else ``0``."""
        return int(self.score(text) >= threshold)
