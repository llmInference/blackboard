"""Tests for ALFWorld workflow comparison runner."""
from __future__ import annotations

import json

from examples.run_workflow_compare import run_workflow_compare


def test_run_workflow_compare_reuses_same_gamefiles_and_writes_summary(tmp_path):
    call_log: list[tuple[str, str]] = []
    gamefiles = ["gf1", "gf2", "gf3"]
    workflow_modes = ["single_action", "planner_action", "planner_llm_action"]
    modes = ["full"]

    def fake_run_workflow_suite(workflow_mode: str, workflow_output_dir: str):
        call_log.append((workflow_mode, workflow_output_dir))
        return {
            "workflow_mode": workflow_mode,
            "architect_mode": "llm",
            "modes": modes,
            "n_gamefiles": len(gamefiles),
            "summaries": {
                "full": {
                    "success_rate": 0.25,
                    "mean_steps": 5.0,
                    "circuit_breaker_trigger_rate": 0.0,
                }
            },
        }

    consolidated = run_workflow_compare(
        gamefiles=gamefiles,
        workflow_modes=workflow_modes,
        modes=modes,
        output_dir=str(tmp_path),
        run_workflow_suite=fake_run_workflow_suite,
    )

    assert [workflow_mode for workflow_mode, _ in call_log] == workflow_modes

    selected_gamefiles = json.loads((tmp_path / "selected_gamefiles.json").read_text(encoding="utf-8"))
    assert selected_gamefiles == gamefiles

    loaded = json.loads((tmp_path / "workflow_compare_summary.json").read_text(encoding="utf-8"))
    assert loaded["workflow_modes"] == workflow_modes
    assert loaded["modes"] == modes
    assert loaded["n_gamefiles"] == 3
    assert loaded["architect_mode"] == "llm"
    assert loaded["workflow_summaries"]["planner_llm_action"]["result"]["summaries"]["full"]["mean_steps"] == 5.0
    assert consolidated["workflow_summaries"]["single_action"]["workflow_mode"] == "single_action"
