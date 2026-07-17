"""Tests for the component registry."""

import pytest

from humanizer_bench.attacks import SentenceMergeAttack
from humanizer_bench.detectors import HeuristicDetector
from humanizer_bench.registry import (
    ATTACKS,
    DATASETS,
    DETECTORS,
    get_attack,
    get_dataset,
    get_detector,
)


def test_get_known_components():
    assert isinstance(get_detector("heuristic"), HeuristicDetector)
    assert isinstance(get_attack("sentence_merge"), SentenceMergeAttack)
    assert get_dataset("toy").name == "toy"


def test_unknown_name_lists_available():
    with pytest.raises(ValueError, match="available"):
        get_detector("does-not-exist")


def test_registry_keys_match_component_names():
    for table in (DETECTORS, ATTACKS, DATASETS):
        for key, cls in table.items():
            assert key == cls.name
