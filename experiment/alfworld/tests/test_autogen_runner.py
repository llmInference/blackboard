"""Tests for the AutoGen ALFWorld baseline runner."""
from __future__ import annotations

from unittest.mock import MagicMock

from systems.autogen_runner import ALFWorldAutoGenRunner
from utils.autogen_factory import AutoGenALFWorldMockClient, parse_autogen_turn_prompt, parse_executor_response


def _make_mock_env():
    env = MagicMock()
    env.env = MagicMock()

    obs0 = ["You are in the kitchen."]
    infos0 = {
        "extra.goal": ["Put the apple in the fridge"],
        "admissible_commands": [["look", "take apple 1", "go to fridge 1"]],
        "extra.gamefile": ["/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial_1/game.tw-pddl"],
        "won": [False],
        "goal_condition_success_rate": [0.0],
    }
    env.reset.return_value = (obs0, infos0)

    step_infos = {
        "admissible_commands": [["look"]],
        "won": [True],
        "goal_condition_success_rate": [1.0],
        "extra.gamefile": ["/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial_1/game.tw-pddl"],
    }
    env.step.return_value = (["Done!"], [1.0], [True], step_infos)
    return env


def test_parse_autogen_turn_prompt_extracts_core_fields():
    prompt = (
        "ALFWORLD TURN CONTEXT\n\n"
        "TASK GOAL:\nPut the apple in the fridge\n\n"
        "CURRENT OBSERVATION:\nYou are in the kitchen.\n\n"
        "AVAILABLE ACTIONS:\n- look\n- take apple 1\n\n"
        "PREVIOUS ACTIONS:\n- look\n\n"
        "GAMEFILE:\n/fake/game.tw-pddl\n\n"
        "CANONICAL TASK TYPE:\npick_and_place_simple\n\n"
        "CANONICAL GOAL OBJECT:\nApple\n\n"
        "CANONICAL TARGET RECEPTACLE:\nFridge\n"
    )
    parsed = parse_autogen_turn_prompt(prompt)
    assert parsed["task_goal"] == "Put the apple in the fridge"
    assert parsed["current_observation"] == "You are in the kitchen."
    assert parsed["available_actions"] == ["look", "take apple 1"]
    assert parsed["action_history"] == ["look"]


def test_parse_executor_response_prefers_explicit_action_line():
    action, reason = parse_executor_response(
        "ACTION: take apple 1\nREASON: The apple is visible.",
        ["look", "take apple 1"],
    )
    assert action == "take apple 1"
    assert reason == "The apple is visible."


def test_autogen_mock_client_returns_role_specific_text():
    planner_client = AutoGenALFWorldMockClient("planner")
    executor_client = AutoGenALFWorldMockClient("executor")
    assert planner_client.role == "planner"
    assert executor_client.role == "executor"


def test_autogen_runner_run_episode_has_expected_fields():
    env = _make_mock_env()
    runner = ALFWorldAutoGenRunner(
        env,
        model_client_factory=lambda role: AutoGenALFWorldMockClient(role),
        config={"workflow_mode": "planner_action"},
    )
    result = runner.run_episode(
        episode_id="ep1",
        run_id="run1",
        experiment_id="autogen_baseline",
        model_name="autogen_mock",
    )

    assert result["episode_id"] == "ep1"
    assert result["run_id"] == "run1"
    assert result["experiment_id"] == "autogen_baseline"
    assert result["success"] is True
    assert result["steps"] == 1
    assert result["stop_reason"] == "environment_done"
    assert result["worker_input_tokens"] > 0
    assert len(result["trajectory"]) == 1
    assert len(result["trajectory"][0]["communication_trace"]) >= 2


def test_autogen_runner_single_action_mode_runs():
    env = _make_mock_env()
    runner = ALFWorldAutoGenRunner(
        env,
        model_client_factory=lambda role: AutoGenALFWorldMockClient(role),
        config={"workflow_mode": "single_action"},
    )
    result = runner.run_episode(model_name="autogen_single")
    assert result["success"] is True
    assert result["trajectory"][0]["communication_trace"][-1]["source"] == "executor"
