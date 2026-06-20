"""HuggingFace dataset loaders (Phase 2 -- not yet implemented).

This is where real corpora get adapted to the :class:`BaseDataset` interface:
a pure-AI corpus, a human control corpus, and -- the project's differentiator --
a non-native-English writing corpus for measuring false-positive bias.
"""

from __future__ import annotations

from .base import BaseDataset


def load_hf_dataset(name: str, split: str = "test", limit: int | None = None) -> BaseDataset:  # pragma: no cover - stub
    """Load a HuggingFace dataset and wrap it as a :class:`BaseDataset`."""
    raise NotImplementedError(
        "HF dataset loaders are a Phase 2 stub. Adapt RAID and a non-native "
        "English corpus here. Install extras with `pip install -e '.[models]'`."
    )
