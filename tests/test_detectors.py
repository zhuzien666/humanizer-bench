"""Tests for detectors."""

import pytest

from humanizer_bench.detectors import BaseDetector, HeuristicDetector, PerplexityDetector


def test_heuristic_score_in_range():
    detector = HeuristicDetector()
    score = detector.score("This is a reasonably long sentence with several words in it.")
    assert 0.0 <= score <= 1.0


def test_heuristic_predict_is_binary():
    detector = HeuristicDetector()
    assert detector.predict("hello world this is a short test of the detector") in (0, 1)


def test_heuristic_short_text_is_neutral():
    detector = HeuristicDetector()
    assert detector.score("hi there") == 0.5


def test_score_batch_matches_score():
    detector = HeuristicDetector()
    texts = ["the first example sentence here", "another distinct example sentence"]
    assert detector.score_batch(texts) == [detector.score(t) for t in texts]


@pytest.mark.models
def test_perplexity_score_in_range():
    detector = PerplexityDetector()
    score = detector.score("The quick brown fox jumps over the lazy dog near the river.")
    assert 0.0 <= score <= 1.0


@pytest.mark.models
def test_perplexity_deterministic():
    detector = PerplexityDetector()
    text = "This is a fixed test string used to check determinism."
    assert detector.score(text) == detector.score(text)


@pytest.mark.models
def test_perplexity_short_text_is_neutral():
    assert PerplexityDetector().score("hi there") == 0.5


def test_base_detector_is_abstract():
    with pytest.raises(TypeError):
        BaseDetector()  # cannot instantiate an abstract class
