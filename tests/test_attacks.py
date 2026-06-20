"""Tests for attacks."""

import pytest

from humanizer_bench.attacks import (
    BackTranslationAttack,
    NoiseAttack,
    SentenceMergeAttack,
)


# --- NoiseAttack ---------------------------------------------------------------

def test_noise_transform_returns_str():
    attack = NoiseAttack(rate=0.1, seed=0)
    out = attack.transform("The quick brown fox jumps over the lazy dog.")
    assert isinstance(out, str)


def test_noise_deterministic_with_seed():
    text = "The quick brown fox jumps over the lazy dog."
    a = NoiseAttack(rate=0.3, seed=42).transform(text)
    b = NoiseAttack(rate=0.3, seed=42).transform(text)
    assert a == b


def test_noise_zero_rate_is_identity():
    text = "No changes should happen here."
    assert NoiseAttack(rate=0.0).transform(text) == text


def test_noise_invalid_rate_raises():
    with pytest.raises(ValueError):
        NoiseAttack(rate=1.5)


# --- SentenceMergeAttack -------------------------------------------------------

def test_merge_returns_str():
    attack = SentenceMergeAttack(rate=1.0)
    out = attack.transform("First sentence here. Second sentence here. Third one too.")
    assert isinstance(out, str)


def test_merge_zero_rate_is_identity():
    text = "First sentence here. Second sentence here."
    assert SentenceMergeAttack(rate=0.0).transform(text) == text


def test_merge_reduces_sentence_count():
    text = "One two three four. Five six seven eight. Nine ten eleven twelve."
    merged = SentenceMergeAttack(rate=1.0).transform(text)
    assert merged.count(".") < text.count(".")


def test_merge_single_sentence_unchanged():
    text = "Only one sentence with no merge possible."
    assert SentenceMergeAttack(rate=1.0).transform(text) == text


def test_merge_invalid_rate_raises():
    with pytest.raises(ValueError):
        SentenceMergeAttack(rate=-0.1)


# --- shared / callable ---------------------------------------------------------

def test_callable_alias():
    attack = SentenceMergeAttack(rate=1.0)
    text = "First sentence here. Second sentence here."
    assert attack(text) == SentenceMergeAttack(rate=1.0).transform(text)


def test_back_translation_stub_raises():
    with pytest.raises(NotImplementedError):
        BackTranslationAttack().transform("anything")
