"""Tests for the end-to-end runner."""

from humanizer_bench.attacks import NoiseAttack, SentenceMergeAttack
from humanizer_bench.detectors import HeuristicDetector
from humanizer_bench.runner import (
    RunResult,
    format_matrix,
    format_result,
    run,
    run_matrix,
)


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


def test_matrix_shape_and_names():
    results = run_matrix(
        [HeuristicDetector()], [SentenceMergeAttack(seed=0), NoiseAttack(seed=0)]
    )
    assert len(results) == 2
    assert [r.attack for r in results] == ["sentence_merge", "noise"]
    assert all(r.detector == "heuristic" for r in results)


def test_matrix_clean_metrics_shared_within_detector():
    merge_result, noise_result = run_matrix(
        [HeuristicDetector()], [SentenceMergeAttack(seed=0), NoiseAttack(seed=0)]
    )
    assert merge_result.acc_clean == noise_result.acc_clean
    assert merge_result.fpr_clean == noise_result.fpr_clean


def test_attack_effect_is_detector_specific():
    # Structural rewriting fools the burstiness baseline; character noise should not.
    merge_result, noise_result = run_matrix(
        [HeuristicDetector()],
        [SentenceMergeAttack(rate=1.0, seed=0), NoiseAttack(rate=0.05, seed=0)],
    )
    assert merge_result.drop > 0.0
    assert noise_result.drop < merge_result.drop


def test_format_matrix_contains_grid():
    out = format_matrix(
        run_matrix(
            [HeuristicDetector()], [SentenceMergeAttack(seed=0), NoiseAttack(seed=0)]
        )
    )
    assert "heuristic" in out
    assert "sentence_merge" in out and "noise" in out


def test_format_matrix_empty():
    assert format_matrix([]) == "(no results)"


def test_run_matrix_with_ci():
    results = run_matrix(
        [HeuristicDetector()], [SentenceMergeAttack(seed=0)],
        with_ci=True, n_bootstrap=200,
    )
    result = results[0]
    assert result.ci_95 is not None
    for key in ("acc_clean", "acc_attacked", "fpr_clean"):
        lo, hi = result.ci_95[key]
        assert 0.0 <= lo <= hi <= 1.0
    assert "95% bootstrap CIs" in format_result(result)


def test_run_matrix_without_ci_by_default():
    result = run_matrix(
        [HeuristicDetector()], [SentenceMergeAttack(seed=0)]
    )[0]
    assert result.ci_95 is None
    assert "95% bootstrap CIs" not in format_result(result)
