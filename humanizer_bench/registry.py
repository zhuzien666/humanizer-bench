"""Name-based registry of built-in components.

Short names (each class's ``name`` attribute) map to component classes so the
CLI — and anything else — can construct components from strings. When you add a
new detector, attack, or dataset, register it here so ``humanizer-bench --list``
can find it. Phase 2 stubs are listed too; they raise ``NotImplementedError``
with pointers when used.
"""

from __future__ import annotations

from typing import Dict, Type

from .attacks import BackTranslationAttack, BaseAttack, NoiseAttack, SentenceMergeAttack
from .datasets import BaseDataset, ToyDataset
from .detectors import BaseDetector, HeuristicDetector, PerplexityDetector

DETECTORS: Dict[str, Type[BaseDetector]] = {
    HeuristicDetector.name: HeuristicDetector,
    PerplexityDetector.name: PerplexityDetector,  # Phase 2 stub
}

ATTACKS: Dict[str, Type[BaseAttack]] = {
    SentenceMergeAttack.name: SentenceMergeAttack,
    NoiseAttack.name: NoiseAttack,
    BackTranslationAttack.name: BackTranslationAttack,  # Phase 2 stub
}

DATASETS: Dict[str, Type[BaseDataset]] = {
    ToyDataset.name: ToyDataset,
}


def _build(kind: str, table: Dict[str, type], name: str):
    try:
        cls = table[name]
    except KeyError:
        available = ", ".join(sorted(table))
        raise ValueError(f"unknown {kind} {name!r} (available: {available})") from None
    return cls()


def get_detector(name: str) -> BaseDetector:
    """Instantiate a registered detector by name."""
    return _build("detector", DETECTORS, name)


def get_attack(name: str) -> BaseAttack:
    """Instantiate a registered attack by name."""
    return _build("attack", ATTACKS, name)


def get_dataset(name: str) -> BaseDataset:
    """Instantiate a registered dataset by name."""
    return _build("dataset", DATASETS, name)
