"""Tests for the ALFWorld Experiment 2 schema-guard entrypoint."""
from __future__ import annotations

import argparse

import pytest

from experiment.alfworld.examples import run_schema_guard_eval


def test_main_raises_for_missing_states_file(tmp_path, monkeypatch):
    missing_states = tmp_path / "missing_states.jsonl"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_schema_guard_eval,
        "parse_args",
        lambda: argparse.Namespace(
            states_file=str(missing_states),
            n_states=10,
            output_dir=str(output_dir),
        ),
    )

    with pytest.raises(FileNotFoundError, match="Captured states file not found"):
        run_schema_guard_eval.main()


def test_main_raises_for_empty_states_file(tmp_path, monkeypatch):
    states_file = tmp_path / "captured_states.jsonl"
    states_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        run_schema_guard_eval,
        "parse_args",
        lambda: argparse.Namespace(
            states_file=str(states_file),
            n_states=10,
            output_dir=str(output_dir),
        ),
    )

    with pytest.raises(ValueError, match="needs at least one captured state"):
        run_schema_guard_eval.main()
