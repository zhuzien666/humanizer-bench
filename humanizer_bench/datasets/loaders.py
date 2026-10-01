"""HuggingFace dataset loaders.

Adapts real corpora from the HuggingFace Hub to the :class:`BaseDataset`
interface. The non-native-English writing corpus (JFLEG) -- the project's
differentiator for measuring false-positive bias -- is implemented here. A
general pure-AI / human-control loader (e.g. RAID) is still a Phase 2 stub.
"""

from __future__ import annotations

from typing import Iterator, List

from .._deps import require
from .base import BaseDataset, Example


class NonNativeDataset(BaseDataset):
    """Non-native English writing from the JFLEG corpus, all human-labeled.

    Every example comes from the ``sentence`` field of ``jhu-clsp/jfleg`` --
    sentences written by English-language learners with different first
    languages (Napoles et al., 2017). All rows are human writing, so every
    example gets ``label=0``. Built for the false-positive experiment: run
    detectors over it and compute ``false_positive_rate`` to see how often
    real human writers get flagged as AI.

    The Hub import is deferred to first iteration and the materialized
    examples are cached on the instance, so constructing the dataset is cheap
    (no download) and iterating stays repeatable without re-downloading.

    Args:
        split: Hub split to load (``"test"`` or ``"validation"``).
        limit: Maximum examples to keep (first rows); ``None`` keeps all.
    """

    name = "nonnative"

    #: Hub repo holding the corpus.
    hf_name = "jhu-clsp/jfleg"

    #: Splits the JFLEG repo actually publishes (there is no "train").
    valid_splits = frozenset({"test", "validation"})

    def __init__(self, split: str = "test", limit: int | None = None) -> None:
        if split not in self.valid_splits:
            raise ValueError(
                f"split must be one of {sorted(self.valid_splits)}, got {split!r}"
            )
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit!r}")
        self.split = split
        self.limit = limit
        self._examples: List[Example] | None = None

    def __iter__(self) -> Iterator[Example]:
        if self._examples is None:
            self._examples = self._load()
        return iter(self._examples)

    def _load(self) -> List[Example]:
        datasets = require("datasets")
        hf = datasets.load_dataset(self.hf_name, split=self.split)
        examples: List[Example] = []
        for row in hf:
            text = (row.get("sentence") or "").strip()
            if not text:
                continue
            examples.append(Example(text=text, label=0, source=self.hf_name))
            if self.limit is not None and len(examples) >= self.limit:
                break
        return examples


def load_nonnative_dataset(
    split: str = "test", limit: int | None = None
) -> BaseDataset:
    """Load the JFLEG non-native-English corpus as a :class:`BaseDataset`.

    Convenience wrapper around :class:`NonNativeDataset`.
    """
    return NonNativeDataset(split=split, limit=limit)


def load_hf_dataset(name: str, split: str = "test", limit: int | None = None) -> BaseDataset:  # pragma: no cover - stub
    """Load a HuggingFace dataset and wrap it as a :class:`BaseDataset`."""
    raise NotImplementedError(
        "HF dataset loaders are a Phase 2 stub. Adapt RAID here. "
        "Install extras with `pip install -e '.[models]'`."
    )
