"""Tests for the ALFWorld ablation runner helpers."""
from __future__ import annotations

import json
from pathlib import Path

from evaluators.metrics import compute_summary
from examples.run_ablation import run_ablation_suite
from utils.ablation_matrix import build_ablation_config, resolve_ablation_modes


def test_resolve_ablation_modes_defaults_and_order():
    assert resolve_ablation_modes() == ["full", "ablate_c2", "ablate_c3"]
    assert resolve_ablation_modes(["ablate_c3", "full"]) == ["full", "ablate_c3"]


def test_build_ablation_config_disables_expected_component():
    full = build_ablation_config("full")
    c2 = build_ablation_config("ablate_c2")
    c3 = build_ablation_config("ablate_c3")

    assert full.use_schema_validation is True
    assert c2.use_schema_validation is False
    assert c3.use_circuit_breaker is False


def test_compute_summary_includes_runtime_breakdowns():
    summary = compute_summary(
        [
            {
                "success": True,
                "steps": 2,
                "total_tokens": 0,
                "goal_condition_rate": 1.0,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "no_update_event_count": 1,
                "repeat_action_count": 1,
                "repeat_observation_count": 1,
                "stagnation_event_count": 1,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "environment_done",
                "final_status": "success",
                "workflow_final_status": "completed",
            },
            {
                "success": False,
                "steps": 1,
                "total_tokens": 0,
                "goal_condition_rate": 0.0,
                "fallback_action_count": 1,
                "patch_error_count": 1,
                "no_update_event_count": 1,
                "repeat_action_count": 1,
                "repeat_observation_count": 1,
                "stagnation_event_count": 1,
                "circuit_breaker_triggered": True,
                "waiting_for_user": False,
                "stop_reason": "circuit_breaker",
                "final_status": "circuit_breaker",
                "workflow_final_status": "completed",
            },
        ]
    )

    assert "mean_no_update_rate" in summary
    assert summary["circuit_breaker_trigger_rate"] == 0.5
    assert summary["stop_reason_breakdown"]["environment_done"] == 1
    assert summary["stop_reason_breakdown"]["circuit_breaker"] == 1
    assert summary["final_status_breakdown"]["success"] == 1
    assert summary["workflow_final_status_breakdown"]["completed"] == 2
    assert summary["mean_repeat_action_rate"] > 0.0
    assert summary["mean_stagnation_rate"] > 0.0


def test_compute_summary_empty_has_stable_shape():
    summary = compute_summary([])
    assert summary["n_episodes"] == 0
    assert summary["mean_no_update_rate"] == 0.0
    assert summary["stop_reason_breakdown"] == {}
    assert summary["workflow_final_status_breakdown"] == {}


class _FakeEvaluator:
    def __init__(self, mode: str, output_dir: str, call_log: list[tuple[str, list[str]]]) -> None:
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_log = call_log

    def run_batch(self, gamefiles, **kwargs):
        self.call_log.append((self.mode, list(gamefiles)))
        episodes_path = self.output_dir / f"{self.mode}_episodes.jsonl"
        episodes_path.write_text('{"episode_id":"ep0"}\n', encoding="utf-8")
        return [{"episode_id": "ep0"}]

    def finalize(self):
        summary_path = self.output_dir / f"{self.mode}_summary.json"
        summary = {
            "n_episodes": 1,
            "success_rate": 0.5,
            "mean_steps": 1.0,
            "circuit_breaker_trigger_rate": 0.0,
            "workflow_final_status_breakdown": {"completed": 1},
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary


def test_run_ablation_suite_reuses_same_gamefiles_and_writes_summary(tmp_path):
    call_log: list[tuple[str, list[str]]] = []
    gamefiles = ["gf1", "gf2", "gf3"]

    def evaluator_factory(mode: str, output_dir: str):
        return _FakeEvaluator(mode, output_dir, call_log)

    consolidated = run_ablation_suite(
        gamefiles=gamefiles,
        modes=["full", "ablate_c2", "ablate_c3"],
        output_dir=str(tmp_path),
        evaluator_factory=evaluator_factory,
        max_steps=5,
        model_name="deterministic_alfworld",
        workflow_mode="planner_action",
        architect_mode="llm",
    )

    assert [mode for mode, _ in call_log] == ["full", "ablate_c2", "ablate_c3"]
    assert all(called_gamefiles == gamefiles for _, called_gamefiles in call_log)

    selected_gamefiles = json.loads((tmp_path / "selected_gamefiles.json").read_text(encoding="utf-8"))
    assert selected_gamefiles == gamefiles

    loaded = json.loads((tmp_path / "ablation_summary.json").read_text(encoding="utf-8"))
    assert loaded["modes"] == ["full", "ablate_c2", "ablate_c3"]
    assert loaded["n_gamefiles"] == 3
    assert loaded["workflow_mode"] == "planner_action"
    assert loaded["architect_mode"] == "llm"
    assert "full" in loaded["summaries"]
    assert consolidated["summaries"]["ablate_c2"]["disabled_components"] == ["C2"]


def test_resolve_ablation_modes_supports_all_component_names():
    assert resolve_ablation_modes(["ablate_c5", "ablate_c1"]) == ["ablate_c1", "ablate_c5"]
