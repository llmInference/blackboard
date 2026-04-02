"""
M1 acceptance tests for ALFWorld experiment infrastructure.

Covers:
- extract_action_with_meta
- adapter.run_episode result fields and counters
- ResultWriter
- DatasetSampler
- ALFWorldEvaluator batch run + finalize
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.state_bridge import extract_action_with_meta
from utils.result_writer import ResultWriter
from utils.dataset_sampler import DatasetSampler


# ---------------------------------------------------------------------------
# extract_action_with_meta
# ---------------------------------------------------------------------------

def test_extract_action_with_meta_valid():
    result = {"domain_state": {"selected_action": "take apple 1"}}
    admissible = ["take apple 1", "look", "go to fridge 1"]
    action, fallback_used = extract_action_with_meta(result, admissible)
    assert action == "take apple 1"
    assert fallback_used is False


def test_extract_action_with_meta_invalid():
    result = {"domain_state": {"selected_action": "fly to moon"}}
    admissible = ["take apple 1", "look"]
    action, fallback_used = extract_action_with_meta(result, admissible)
    assert action == "look"
    assert fallback_used is True


def test_extract_action_with_meta_empty():
    result = {"domain_state": {"selected_action": ""}}
    admissible = ["look", "go to table 1"]
    action, fallback_used = extract_action_with_meta(result, admissible)
    assert action == "look"
    assert fallback_used is True


def test_extract_action_with_meta_no_look_fallback():
    result = {"domain_state": {"selected_action": "invalid"}}
    admissible = ["go to table 1", "take apple 1"]
    action, fallback_used = extract_action_with_meta(result, admissible)
    assert action == "go to table 1"
    assert fallback_used is True


# ---------------------------------------------------------------------------
# ResultWriter
# ---------------------------------------------------------------------------

def test_result_writer_appends_jsonl(tmp_path):
    writer = ResultWriter(str(tmp_path), run_id="test_run")
    writer.write_episode({"episode_id": "ep1", "success": True, "steps": 5})
    writer.write_episode({"episode_id": "ep2", "success": False, "steps": 10})

    lines = writer.episodes_path.read_text(encoding="utf-8").strip().splitlines()
    canonical_lines = writer.standard_episodes_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert len(canonical_lines) == 2
    assert json.loads(lines[0])["episode_id"] == "ep1"
    assert json.loads(lines[1])["episode_id"] == "ep2"
    assert json.loads(canonical_lines[0])["record_type"] == "episode_result"
    assert json.loads(canonical_lines[0])["env_name"] == "alfworld"


def test_result_writer_summary_keys(tmp_path):
    writer = ResultWriter(str(tmp_path), run_id="test_run")
    metrics = {
        "success_rate": 0.8,
        "mean_steps": 12.5,
        "mean_total_tokens": 0,
        "mean_fallback_rate": 0.1,
        "mean_patch_error_rate": 0.05,
        "mean_goal_condition_rate": 0.9,
        "n_episodes": 10,
    }
    writer.write_summary(metrics)
    loaded = json.loads(writer.summary_path.read_text(encoding="utf-8"))
    canonical_loaded = json.loads(writer.standard_summary_path.read_text(encoding="utf-8"))
    for key in ("success_rate", "mean_steps", "mean_total_tokens"):
        assert key in loaded
        assert key in canonical_loaded
    assert canonical_loaded["record_type"] == "summary"


def test_result_writer_creates_output_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    writer = ResultWriter(str(nested), run_id="r1")
    writer.write_episode({"x": 1})
    assert writer.episodes_path.exists()
    assert writer.standard_episodes_path.exists()


def test_result_writer_writes_captured_states_to_separate_jsonl(tmp_path):
    writer = ResultWriter(str(tmp_path), run_id="test_run")
    writer.write_episode(
        {
            "episode_id": "ep1",
            "success": True,
            "captured_states": [
                {
                    "episode_id": "ep1",
                    "gamefile": "/fake/game.tw-pddl",
                    "step_id": 0,
                    "domain_state": {"workflow_status": "deciding"},
                    "data_schema": {"type": "object"},
                }
            ],
        }
    )

    episode_line = json.loads(writer.episodes_path.read_text(encoding="utf-8").strip())
    state_line = json.loads(writer.states_path.read_text(encoding="utf-8").strip())
    canonical_state_line = json.loads(writer.standard_states_path.read_text(encoding="utf-8").strip())

    assert "captured_states" not in episode_line
    assert state_line["episode_id"] == "ep1"
    assert state_line["step_id"] == 0
    assert canonical_state_line["record_type"] == "captured_state"


def test_result_writer_writes_config_snapshot(tmp_path):
    writer = ResultWriter(str(tmp_path), run_id="test_run")
    writer.write_config_snapshot({"model_name": "demo-model", "max_steps": 5})

    loaded = json.loads(writer.config_snapshot_path.read_text(encoding="utf-8"))
    assert loaded["record_type"] == "config_snapshot"
    assert loaded["run_id"] == "test_run"
    assert loaded["config"]["max_steps"] == 5


# ---------------------------------------------------------------------------
# DatasetSampler
# ---------------------------------------------------------------------------

def _make_fake_data_root(tmp_path: Path) -> Path:
    """Create a fake ALFWorld data directory with 4 tasks per type (24 total)."""
    from utils.dataset_sampler import _TASK_TYPES
    for task_type in _TASK_TYPES:
        for i in range(4):
            trial_dir = tmp_path / task_type / f"trial_{i}" / "game"
            trial_dir.mkdir(parents=True)
            (trial_dir / "game.tw-pddl").write_text("{}", encoding="utf-8")
    return tmp_path


def test_dataset_sampler_debug_size(tmp_path):
    data_root = _make_fake_data_root(tmp_path)
    sampler = DatasetSampler(str(data_root))
    result = sampler.debug_split()
    assert len(result) == 12


def test_dataset_sampler_formal_size(tmp_path):
    data_root = _make_fake_data_root(tmp_path)
    # formal_split wants 10 per type but we only have 4 — should return all available
    sampler = DatasetSampler(str(data_root))
    result = sampler.formal_split()
    # 4 per type * 6 types = 24 (capped at available)
    assert len(result) == 24


def test_dataset_sampler_reproducible(tmp_path):
    data_root = _make_fake_data_root(tmp_path)
    sampler = DatasetSampler(str(data_root))
    r1 = sampler.debug_split(seed=42)
    r2 = sampler.debug_split(seed=42)
    assert r1 == r2


def test_dataset_sampler_different_seeds(tmp_path):
    data_root = _make_fake_data_root(tmp_path)
    sampler = DatasetSampler(str(data_root))
    r1 = sampler.sample(n_per_type=2, seed=1)
    r2 = sampler.sample(n_per_type=2, seed=99)
    # With only 4 files per type and n=2, different seeds may or may not differ,
    # but both should return 12 items
    assert len(r1) == 12
    assert len(r2) == 12


# ---------------------------------------------------------------------------
# Adapter run_episode result fields (mocked env + kernel)
# ---------------------------------------------------------------------------

def _make_mock_env(done_on_step=1):
    """Return a mock ALFWorldEnvWrapper that completes after done_on_step steps."""
    env = MagicMock()
    env.env = MagicMock()  # already initialized

    obs0 = ["You are in the kitchen."]
    infos0 = {
        "extra.goal": ["Put the apple in the fridge"],
        "admissible_commands": [["look", "take apple 1"]],
        "extra.gamefile": ["/fake/game.tw-pddl"],
        "won": [False],
        "goal_condition_success_rate": [0.0],
    }
    env.reset.return_value = (obs0, infos0)

    step_infos = {
        "admissible_commands": [["look"]],
        "won": [True],
        "goal_condition_success_rate": [1.0],
        "extra.gamefile": ["/fake/game.tw-pddl"],
    }
    env.step.return_value = (["Done!"], [1.0], [True], step_infos)
    return env


def _make_mock_kernel_graph(selected_action="look", patch_error="", **overrides):
    graph = MagicMock()
    result = {
        "domain_state": {
            "task_goal": "Put the apple in the fridge",
            "current_observation": "You are in the kitchen.",
            "available_actions": ["look", "take apple 1"],
            "selected_action": selected_action,
            "decision_reason": "test reason",
            "action_history": [],
            "observation_history": ["You are in the kitchen."],
            "current_gamefile": "/fake/game.tw-pddl",
            "workflow_status": "completed",
        },
        "data_schema": {
            "type": "object",
            "properties": {
                "task_goal": {"type": "string"},
                "current_observation": {"type": "string"},
                "available_actions": {"type": "array"},
                "selected_action": {"type": "string"},
                "decision_reason": {"type": "string"},
                "workflow_status": {"type": "string"},
            },
        },
        "patch_error": patch_error,
        "retry_count": 0,
    }
    result.update(overrides)
    graph.invoke.return_value = result
    return graph


def _make_adapter(env, kernel_graph):
    from core.adapter import ALFWorldKernelAdapter
    adapter = ALFWorldKernelAdapter(env, llm=None, config={})
    adapter.kernel_graph = kernel_graph
    return adapter


def test_adapter_result_has_required_m1_fields():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(selected_action="look")
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode(episode_id="ep1", run_id="run1", experiment_id="full", model_name="test-model")

    required = [
        "episode_id", "run_id", "experiment_id", "model_name",
        "gamefile", "task_goal", "ablation_config",
        "success", "steps", "goal_condition_rate",
        "fallback_action_count", "patch_error_count",
        "no_update_event_count", "stop_reason", "waiting_for_user",
        "circuit_breaker_triggered", "circuit_breaker_reason", "final_status",
        "workflow_final_status", "repeat_action_count", "repeat_observation_count",
        "stagnation_event_count",
        "worker_input_tokens", "worker_output_tokens", "total_tokens",
        "trajectory",
    ]
    for field in required:
        assert field in result, f"Missing field: {field}"

    assert result["episode_id"] == "ep1"
    assert result["model_name"] == "test-model"


def test_adapter_trajectory_uses_step_id():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(selected_action="look")
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    assert len(result["trajectory"]) > 0
    assert "step_id" in result["trajectory"][0]
    assert "step" not in result["trajectory"][0]
    assert "observation_before" in result["trajectory"][0]
    assert "available_actions_before" in result["trajectory"][0]
    assert "planner_state" in result["trajectory"][0]
    assert "canonical_task" in result["trajectory"][0]


def test_adapter_trajectory_records_planner_and_canonical_diagnostics():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(
        selected_action="take apple 1",
        domain_state={
            "task_goal": "Put the apple in the fridge",
            "current_observation": "You are in the kitchen.",
            "available_actions": ["look", "take apple 1"],
            "selected_action": "take apple 1",
            "decision_reason": "take the visible apple",
            "action_history": [],
            "observation_history": ["You are in the kitchen."],
            "current_gamefile": "/fake/game.tw-pddl",
            "workflow_status": "completed",
            "planner_summary": "Acquire the goal object.",
            "subgoal": "acquire_goal_object",
            "focus_object": "Apple",
            "focus_receptacle": "Fridge",
            "required_transform": "",
            "goal_object_guidance": "Canonical goal object: Apple.",
            "search_guidance": "Prefer visible goal objects.",
            "recommended_actions": ["take apple 1"],
            "searched_locations": ["table 1"],
            "failed_search_locations": [],
            "canonical_task_type": "pick_and_place_simple",
            "canonical_goal_object": "Apple",
            "canonical_target_receptacle": "Fridge",
        },
    )
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    step = result["trajectory"][0]

    assert step["observation_before"] == "You are in the kitchen."
    assert step["available_actions_before"] == ["look", "take apple 1"]
    assert step["planner_state"]["subgoal"] == "acquire_goal_object"
    assert step["planner_state"]["recommended_actions"] == ["take apple 1"]
    assert step["canonical_task"]["canonical_goal_object"] == "Apple"
    assert step["canonical_task"]["canonical_target_receptacle"] == "Fridge"


def test_adapter_counts_fallback():
    env = _make_mock_env()
    # selected_action not in admissible_commands → fallback
    graph = _make_mock_kernel_graph(selected_action="fly to moon")
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    assert result["fallback_action_count"] >= 1


def test_adapter_counts_patch_error():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(selected_action="look", patch_error="schema_type_error")
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    assert result["patch_error_count"] >= 1


def test_adapter_records_runtime_fields_on_environment_completion():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(selected_action="look", no_update_count=1)
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    assert result["stop_reason"] == "environment_done"
    assert result["final_status"] == "success"
    assert result["workflow_final_status"] == "completed"
    assert result["no_update_event_count"] == 1
    assert result["circuit_breaker_triggered"] is False


def test_adapter_stops_on_circuit_breaker_before_env_step():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(
        selected_action="look",
        circuit_breaker_triggered=True,
        circuit_breaker_reason="timeout",
    )
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode()
    assert result["stop_reason"] == "circuit_breaker"
    assert result["circuit_breaker_triggered"] is True
    assert result["circuit_breaker_reason"] == "timeout"
    assert result["final_status"] == "circuit_breaker"
    assert result["workflow_final_status"] == "completed"
    assert result["steps"] == 0
    env.step.assert_not_called()


def test_adapter_episode_stagnation_breaker_stops_when_c3_enabled():
    from core.adapter import ALFWorldKernelAdapter

    env = MagicMock()
    env.env = MagicMock()
    env.reset.return_value = (
        ["You are in the room."],
        {
            "extra.goal": ["Put the apple in the fridge"],
            "admissible_commands": [["look"]],
            "extra.gamefile": ["/fake/game.tw-pddl"],
            "won": [False],
            "goal_condition_success_rate": [0.0],
        },
    )
    step_infos = {
        "admissible_commands": [["look"]],
        "won": [False],
        "goal_condition_success_rate": [0.0],
        "extra.gamefile": ["/fake/game.tw-pddl"],
    }
    env.step.return_value = (["You are still in the room."], [0.0], [False], step_infos)

    graph = _make_mock_kernel_graph(selected_action="look")
    adapter = ALFWorldKernelAdapter(
        env,
        llm=None,
        config={"episode_breaker": {"repeat_pair_threshold": 2, "repeat_observation_threshold": 2, "repeat_action_threshold": 2}},
    )
    adapter.kernel_graph = graph

    result = adapter.run_episode(max_steps=5)
    assert result["stop_reason"] == "stagnation_detected"
    assert result["final_status"] == "stagnation"
    assert result["circuit_breaker_triggered"] is True
    assert result["repeat_action_count"] >= 1
    assert result["stagnation_event_count"] >= 1
    assert result["steps"] < 5


def test_adapter_episode_stagnation_breaker_disabled_with_ablate_c3():
    from langgraph_kernel.ablation import AblationConfig
    from core.adapter import ALFWorldKernelAdapter

    env = MagicMock()
    env.env = MagicMock()
    env.reset.return_value = (
        ["You are in the room."],
        {
            "extra.goal": ["Put the apple in the fridge"],
            "admissible_commands": [["look"]],
            "extra.gamefile": ["/fake/game.tw-pddl"],
            "won": [False],
            "goal_condition_success_rate": [0.0],
        },
    )
    step_infos = {
        "admissible_commands": [["look"]],
        "won": [False],
        "goal_condition_success_rate": [0.0],
        "extra.gamefile": ["/fake/game.tw-pddl"],
    }
    env.step.return_value = (["You are still in the room."], [0.0], [False], step_infos)

    graph = _make_mock_kernel_graph(selected_action="look")
    adapter = ALFWorldKernelAdapter(
        env,
        llm=None,
        config={
            "ablation": AblationConfig.ablate("C3"),
            "episode_breaker": {"repeat_pair_threshold": 2, "repeat_observation_threshold": 2, "repeat_action_threshold": 2},
        },
    )
    adapter.kernel_graph = graph

    result = adapter.run_episode(max_steps=3)
    assert result["stop_reason"] == "max_steps_reached"
    assert result["final_status"] == "incomplete"
    assert result["circuit_breaker_triggered"] is False
    assert result["steps"] == 3


def test_adapter_capture_states_returns_runtime_snapshots():
    env = _make_mock_env()
    graph = _make_mock_kernel_graph(selected_action="fly to moon")
    adapter = _make_adapter(env, graph)

    result = adapter.run_episode(episode_id="ep_capture", capture_states=True)

    assert len(result["captured_states"]) == 1
    captured = result["captured_states"][0]
    assert captured["episode_id"] == "ep_capture"
    assert captured["gamefile"] == "/fake/game.tw-pddl"
    assert captured["step_id"] == 0
    assert captured["data_schema"]["type"] == "object"
    assert captured["domain_state"]["fallback_used"] is True
    assert captured["domain_state"]["fallback_action"] == "look"


# ---------------------------------------------------------------------------
# ALFWorldEvaluator batch run + finalize
# ---------------------------------------------------------------------------

def test_evaluator_batch_writes_episodes_and_finalizes(tmp_path):
    from evaluators.alfworld_evaluator import ALFWorldEvaluator
    from utils.result_writer import ResultWriter

    # Build two mock adapters that return canned results
    fake_result = {
        "episode_id": "x", "run_id": "", "experiment_id": "", "model_name": "",
        "gamefile": "/fake/game.tw-pddl", "task_goal": "test",
        "ablation_config": {}, "trajectory": [],
        "success": True, "steps": 3, "goal_condition_rate": 1.0,
        "fallback_action_count": 0, "patch_error_count": 0,
        "no_update_event_count": 0, "stop_reason": "environment_done",
        "waiting_for_user": False, "circuit_breaker_triggered": False,
        "circuit_breaker_reason": "", "final_status": "success",
        "workflow_final_status": "completed",
        "repeat_action_count": 0, "repeat_observation_count": 0, "stagnation_event_count": 0,
        "worker_input_tokens": 0, "worker_output_tokens": 0, "total_tokens": 0,
    }

    mock_adapter = MagicMock()
    mock_adapter.run_episode.return_value = fake_result

    writer = ResultWriter(str(tmp_path), run_id="eval_run")
    evaluator = ALFWorldEvaluator(adapter=mock_adapter, result_writer=writer)

    results = evaluator.run_batch(["gf1", "gf2"], run_id="eval_run")
    assert len(results) == 2

    lines = writer.episodes_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    summary = evaluator.finalize()
    assert "success_rate" in summary
    assert "mean_steps" in summary
    assert "mean_total_tokens" in summary
    assert "mean_no_update_rate" in summary
    assert "mean_repeat_action_rate" in summary
    assert "mean_repeat_observation_rate" in summary
    assert "mean_stagnation_rate" in summary
    assert "circuit_breaker_trigger_rate" in summary
    assert "workflow_final_status_breakdown" in summary
    assert summary["n_episodes"] == 2


def test_evaluator_batch_writes_states_when_capture_enabled(tmp_path):
    from evaluators.alfworld_evaluator import ALFWorldEvaluator

    fake_result = {
        "episode_id": "x",
        "run_id": "",
        "experiment_id": "",
        "model_name": "",
        "gamefile": "/fake/game.tw-pddl",
        "task_goal": "test",
        "ablation_config": {},
        "trajectory": [],
        "success": True,
        "steps": 1,
        "goal_condition_rate": 1.0,
        "fallback_action_count": 0,
        "patch_error_count": 0,
        "no_update_event_count": 0,
        "worker_input_tokens": 0,
        "worker_output_tokens": 0,
        "total_tokens": 0,
        "stop_reason": "environment_done",
        "waiting_for_user": False,
        "circuit_breaker_triggered": False,
        "circuit_breaker_reason": "",
        "final_status": "success",
        "workflow_final_status": "completed",
        "repeat_action_count": 0,
        "repeat_observation_count": 0,
        "stagnation_event_count": 0,
        "captured_states": [
            {
                "episode_id": "x",
                "gamefile": "/fake/game.tw-pddl",
                "step_id": 0,
                "domain_state": {"workflow_status": "deciding"},
                "data_schema": {"type": "object"},
            }
        ],
    }

    mock_adapter = MagicMock()
    mock_adapter.run_episode.return_value = fake_result

    writer = ResultWriter(str(tmp_path), run_id="eval_run")
    evaluator = ALFWorldEvaluator(adapter=mock_adapter, result_writer=writer)
    evaluator.run_batch(["gf1", "gf2"], run_id="eval_run", capture_states=True)

    state_lines = writer.states_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(state_lines) == 2
    assert mock_adapter.run_episode.call_args.kwargs["capture_states"] is True
