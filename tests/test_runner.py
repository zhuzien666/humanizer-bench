"""Tests for the end-to-end runner."""

from humanizer_bench.runner import RunResult, format_result, run


def test_run_returns_result():
    result = run()
    assert isinstance(result, RunResult)
    assert result.n > 0
    assert 0.0 <= result.acc_clean <= 1.0
    assert 0.0 <= result.acc_attacked <= 1.0
    assert 0.0 <= result.fpr_clean <= 1.0


def test_drop_is_consistent():
    result = run()
    assert result.drop == result.acc_clean - result.acc_attacked


def test_format_is_string():
    text = format_result(run())
    assert isinstance(text, str)
    assert "accuracy (clean)" in text
