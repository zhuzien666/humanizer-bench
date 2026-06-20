"""Abstract base class for datasets and the :class:`Example` record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Example:
    """A single labeled text example.

    Attributes:
        text: The document text.
        label: ``1`` if AI-generated, ``0`` if human-written.
        source: Optional provenance tag (dataset or model name).
    """

    text: str
    label: int
    source: str = ""


class BaseDataset(ABC):
    """An iterable collection of :class:`Example` objects.

    Subclasses implement :meth:`__iter__`. Iterating must be repeatable: each
    call returns a fresh iterator over the same data.
    """

    #: Short identifier used in result tables. Override in subclasses.
    name: str = "base"

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(1 for _ in self)
