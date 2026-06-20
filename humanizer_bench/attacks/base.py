"""Abstract base class for humanizer / evasion attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Common interface for a text transformation that tries to evade detection.

    Subclasses implement :meth:`transform`, which rewrites an input text while
    (ideally) preserving its meaning. Instances are callable as a shorthand for
    :meth:`transform`.
    """

    #: Short identifier used in result tables. Override in subclasses.
    name: str = "base"

    @abstractmethod
    def transform(self, text: str) -> str:
        """Return a rewritten version of ``text``."""
        raise NotImplementedError

    def __call__(self, text: str) -> str:
        return self.transform(text)
