"""A tiny bundled dataset so the pipeline runs offline."""

from __future__ import annotations

import json
from importlib import resources
from typing import Iterator

from .base import BaseDataset, Example


class ToyDataset(BaseDataset):
    """A handful of short human/AI snippets shipped inside the package.

    For demonstration and tests only. Real corpora (RAID, non-native-English
    writing, etc.) are loaded via :mod:`humanizer_bench.datasets.loaders` in
    Phase 2.
    """

    name = "toy"

    def __iter__(self) -> Iterator[Example]:
        raw = (
            resources.files("humanizer_bench.data")
            .joinpath("toy.jsonl")
            .read_text(encoding="utf-8")
        )
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield Example(
                text=record["text"],
                label=int(record["label"]),
                source=record.get("source", "toy"),
            )
