"""Tests for the command-line interface."""

import json

import pytest

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


def test_json_output_is_parseable(capsys):
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "toy"
    assert payload["threshold"] == 0.5
    assert len(payload["results"]) == 1


def test_json_matrix_has_one_entry_per_pair(capsys):
    assert main(["-d", "heuristic", "-a", "sentence_merge", "-a", "noise", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert [r["attack"] for r in results] == ["sentence_merge", "noise"]


def test_json_includes_derived_drop(capsys):
    # `drop` is a property, so dataclasses.asdict would silently omit it.
    assert main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["drop"] == pytest.approx(result["acc_clean"] - result["acc_attacked"])
