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


def test_perplexity_stub_raises():
    with pytest.raises(NotImplementedError):
        PerplexityDetector().score("anything")


def test_base_detector_is_abstract():
    with pytest.raises(TypeError):
        BaseDetector()  # cannot instantiate an abstract class
