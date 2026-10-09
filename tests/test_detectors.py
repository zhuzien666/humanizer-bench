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


def test_predict_treats_no_evidence_as_not_ai():
    # A 0.5 score means "I don't know" (see docs/architecture.md); it must not
    # count as AI. Regression test for issue #16.
    detector = HeuristicDetector()
    assert detector.score("hi there") == 0.5
    assert detector.predict("hi there") == 0


def test_predict_threshold_boundary_is_exclusive():
    class FixedDetector(BaseDetector):
        name = "fixed"

        def __init__(self, s: float):
            self._s = s

        def score(self, text: str) -> float:
            return self._s

    assert FixedDetector(0.5).predict("x") == 0
    assert FixedDetector(0.4999).predict("x") == 0
    assert FixedDetector(0.5001).predict("x") == 1


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


@pytest.mark.models
def test_perplexity_ranks_ai_above_human():
    # The range/determinism/neutral tests above all still pass if the logistic
    # is inverted (`perplexity - _REF` instead of `_REF - perplexity`), which
    # silently turns the detector inside out: every human text gets flagged.
    # Only an ordering assertion catches that, so keep this one.
    detector = PerplexityDetector()
    ai_text = (
        "Artificial intelligence has become an essential part of modern technology. "
        "It enables machines to perform tasks that normally require human intelligence."
    )
    human_text = (
        "Honestly? I forgot my umbrella again and got completely soaked walking to "
        "the station. Third time this week."
    )
    assert detector.score(ai_text) > detector.score(human_text)


def test_base_detector_is_abstract():
    with pytest.raises(TypeError):
        BaseDetector()  # cannot instantiate an abstract class
