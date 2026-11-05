"""Tests for datasets."""

import pytest

from humanizer_bench.datasets import (
    Example,
    HFDataset,
    HFMixedDataset,
    NonNativeDataset,
    ToyDataset,
    load_nonnative_dataset,
)
from humanizer_bench.datasets.loaders import load_hf_dataset
from humanizer_bench.detectors import HeuristicDetector
from humanizer_bench.metrics import abstention_rate, false_positive_rate


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


# --- HuggingFace loaders -------------------------------------------------------

def test_hf_dataset_constructs_without_downloading():
    # No Hub access until iterated: construction must stay cheap.
    ds = HFDataset("ag_news", label=0, limit=4)
    assert ds.name == "hf"
    assert ds.label == 0
    assert ds.source == "ag_news"


def test_hf_dataset_invalid_label_raises():
    with pytest.raises(ValueError):
        HFDataset("ag_news", label=2)


def test_hf_dataset_invalid_limit_raises():
    with pytest.raises(ValueError):
        HFDataset("ag_news", limit=-1)


def test_hf_mixed_dataset_constructs_without_downloading():
    # No Hub access until iterated: construction must stay cheap.
    ds = HFMixedDataset()
    assert ds.name == "hf"
    assert isinstance(ds, HFMixedDataset)


@pytest.mark.models
def test_load_hf_dataset_tiny_slice():
    ds = load_hf_dataset(limit=4)
    examples = list(ds)
    assert len(examples) == 8
    assert all(isinstance(e, Example) for e in examples)
    assert {e.label for e in examples} == {0, 1}
    assert all(e.text.strip() for e in examples)
    assert {e.source for e in examples} == {"tatsu-lab/alpaca", "ag_news"}


@pytest.mark.models
def test_load_hf_dataset_iteration_repeatable():
    ds = load_hf_dataset(limit=4)
    assert [e.text for e in ds] == [e.text for e in ds]


_FAKE_ROWS = [
    {
        "sentence": "They are moved by solar energy .",
        "corrections": ["They are moving by solar energy ."],
    },
    {
        "sentence": "There are several reason .",
        "corrections": ["There are several reasons ."],
    },
    {"sentence": "   ", "corrections": []},  # blank -> skipped
    {"sentence": "", "corrections": []},  # empty -> skipped
    {
        "sentence": "I want to talk about nocive or bad products like alcohol .",
        "corrections": ["I want to talk about harmful or bad products like alcohol ."],
    },
]


class _FakeDatasetsModule:
    """Stand-in for the ``datasets`` package: records load calls, serves rows."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def load_dataset(self, name, split=None):
        self.calls.append((name, split))
        return list(self._rows)


@pytest.fixture
def fake_hf(monkeypatch):
    fake = _FakeDatasetsModule(_FAKE_ROWS)
    monkeypatch.setattr(
        "humanizer_bench.datasets.loaders.require",
        lambda module, extra="models": fake,
    )
    return fake


def test_nonnative_constructs_without_downloading(fake_hf):
    NonNativeDataset()
    assert fake_hf.calls == []


def test_nonnative_field_mapping(fake_hf):
    examples = list(NonNativeDataset())
    assert [e.text for e in examples] == [
        "They are moved by solar energy .",
        "There are several reason .",
        "I want to talk about nocive or bad products like alcohol .",
    ]
    assert all(isinstance(e, Example) for e in examples)


def test_nonnative_labels_all_human(fake_hf):
    assert {e.label for e in NonNativeDataset()} == {0}


def test_nonnative_source_tag(fake_hf):
    assert {e.source for e in NonNativeDataset()} == {"jhu-clsp/jfleg"}


def test_nonnative_uses_test_split_by_default(fake_hf):
    list(NonNativeDataset())
    assert fake_hf.calls == [("jhu-clsp/jfleg", "test")]


def test_nonnative_limit(fake_hf):
    examples = list(NonNativeDataset(limit=2))
    assert len(examples) == 2


def test_nonnative_negative_limit_raises():
    with pytest.raises(ValueError, match="non-negative"):
        NonNativeDataset(limit=-1)


def test_nonnative_invalid_split_raises():
    with pytest.raises(ValueError, match="split must be one of"):
        NonNativeDataset(split="train")


def test_nonnative_iteration_repeatable_and_cached(fake_hf):
    dataset = NonNativeDataset()
    first = [e.text for e in dataset]
    second = [e.text for e in dataset]
    assert first == second
    assert len(fake_hf.calls) == 1  # downloaded once, then served from cache


def test_load_nonnative_dataset_wrapper(fake_hf):
    dataset = load_nonnative_dataset(limit=1)
    assert isinstance(dataset, NonNativeDataset)
    assert len(list(dataset)) == 1


def test_nonnative_fpr_smoke(fake_hf):
    """The example's core claim: FPR is computable per detector on this corpus."""
    examples = list(NonNativeDataset())
    y_true = [ex.label for ex in examples]
    y_pred = [HeuristicDetector().predict(ex.text) for ex in examples]
    fpr = false_positive_rate(y_true, y_pred)
    assert 0.0 <= fpr <= 1.0

    # An earlier version asserted `fpr == 1.0`, pinning the tie-break where
    # predict() used `>=` and a neutral 0.5 counted as AI. Swapping that for
    # 0.0 after the move to strict `>` would repeat the mistake: neither
    # number is a measurement, and both move when the convention moves.
    # The detector abstains on these rows, so assert the thing that is
    # actually true and that no tie-break convention can flip.
    scores = [HeuristicDetector().score(ex.text) for ex in examples]
    assert abstention_rate(scores) == 1.0, "every fake row is a single sentence"
    assert fpr == 0.0, "an abstention is not an accusation"
