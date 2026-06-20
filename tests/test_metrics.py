"""Tests for metrics."""

from humanizer_bench.metrics import accuracy, false_positive_rate


def test_accuracy_perfect():
    assert accuracy([0, 1, 1], [0, 1, 1]) == 1.0


def test_accuracy_half():
    assert accuracy([0, 1], [0, 0]) == 0.5


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_false_positive_rate():
    # Human (label 0) examples are at indices 0 and 2; one is flagged -> 0.5.
    assert false_positive_rate([0, 1, 0], [1, 1, 0]) == 0.5


def test_false_positive_rate_no_negatives():
    assert false_positive_rate([1, 1], [1, 0]) == 0.0
