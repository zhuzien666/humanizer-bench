"""Tests for the command-line interface."""

from humanizer_bench.cli import main


def test_default_run_prints_table(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "accuracy (clean)" in out


def test_multiple_attacks_prints_matrix(capsys):
    assert main(["-a", "sentence_merge", "-a", "noise"]) == 0
    out = capsys.readouterr().out
    assert "sentence_merge" in out and "noise" in out


def test_list_components(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "heuristic" in out and "toy" in out


def test_unknown_detector_fails_cleanly(capsys):
    assert main(["--detector", "nope"]) == 2
    assert "unknown detector" in capsys.readouterr().err


def test_stub_component_fails_cleanly(capsys):
    assert main(["--attack", "back_translation"]) == 2
    assert "stub" in capsys.readouterr().err
