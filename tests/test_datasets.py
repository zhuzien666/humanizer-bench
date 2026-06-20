"""Tests for datasets."""

from humanizer_bench.datasets import Example, ToyDataset


def test_toy_loads_examples():
    examples = list(ToyDataset())
    assert len(examples) > 0
    assert all(isinstance(e, Example) for e in examples)


def test_labels_are_binary():
    assert all(e.label in (0, 1) for e in ToyDataset())


def test_has_both_classes():
    labels = {e.label for e in ToyDataset()}
    assert labels == {0, 1}


def test_iteration_is_repeatable():
    dataset = ToyDataset()
    first = [e.text for e in dataset]
    second = [e.text for e in dataset]
    assert first == second


def test_len_matches_iteration():
    dataset = ToyDataset()
    assert len(dataset) == len(list(dataset))
