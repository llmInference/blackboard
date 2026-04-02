"""Tests for the plain LangGraph ALFWorld baseline runner."""
from __future__ import annotations

from unittest.mock import MagicMock

from core.state_bridge import obs_to_kernel_state
from systems.langgraph_runner import ALFWorldLangGraphRunner
from utils.mock_llm import ALFWorldMockActionLLM


class _UsageReportingMockLLM(ALFWorldMockActionLLM):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        content = result.generations[0].message.content
        result.generations[0] = ChatGeneration(
            message=AIMessage(
                content=content,
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 5,
                        "total_tokens": 22,
                    }
                },
            )
        )
        return ChatResult(generations=result.generations)


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


def test_langgraph_runner_builds_turn_graph_for_single_action():
    runner = ALFWorldLangGraphRunner(MagicMock(), None, config={"workflow_mode": "single_action"})
    graph = runner.build_turn_graph()
    result = graph.invoke(
        obs_to_kernel_state(
            "You are in the kitchen.",
            {
                "extra.goal": ["Put the apple in the fridge"],
                "admissible_commands": [["look", "take apple 1", "go to fridge 1"]],
            },
        )
    )

    assert result["domain_state"]["selected_action"] in {"take apple 1", "go to fridge 1", "look"}
    assert result["domain_state"]["workflow_status"] == "completed"


def test_langgraph_runner_builds_turn_graph_for_planner_llm_action():
    runner = ALFWorldLangGraphRunner(
        MagicMock(),
        ALFWorldMockActionLLM(),
        config={"workflow_mode": "planner_llm_action"},
    )
    graph = runner.build_turn_graph()
    result = graph.invoke(
        obs_to_kernel_state(
            "You are in the kitchen.",
            {
                "extra.goal": ["Put the apple in the fridge"],
                "extra.gamefile": ["/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial_1/game.tw-pddl"],
                "admissible_commands": [["look", "take apple 1", "go to fridge 1"]],
            },
        )
    )

    assert result["domain_state"]["subgoal"] == "acquire_goal_object"
    assert result["domain_state"]["selected_action"] == "take apple 1"
    assert result["turn_worker_input_tokens"] == 0
    assert result["turn_worker_output_tokens"] == 0


def test_langgraph_runner_collects_worker_token_usage():
    runner = ALFWorldLangGraphRunner(
        MagicMock(),
        _UsageReportingMockLLM(),
        config={"workflow_mode": "planner_llm_action"},
    )
    graph = runner.build_turn_graph()
    result = graph.invoke(
        obs_to_kernel_state(
            "You are in the kitchen.",
            {
                "extra.goal": ["Put the apple in the fridge"],
                "extra.gamefile": ["/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial_1/game.tw-pddl"],
                "admissible_commands": [["look", "take apple 1", "go to fridge 1"]],
            },
        )
    )

    assert result["turn_worker_input_tokens"] == 17
    assert result["turn_worker_output_tokens"] == 5


def test_langgraph_runner_run_episode_has_expected_fields():
    env = _make_mock_env()
    runner = ALFWorldLangGraphRunner(env, None, config={"workflow_mode": "single_action"})

    result = runner.run_episode(
        episode_id="ep1",
        run_id="run1",
        experiment_id="langgraph_full",
        model_name="langgraph_plain",
    )

    assert result["episode_id"] == "ep1"
    assert result["run_id"] == "run1"
    assert result["experiment_id"] == "langgraph_full"
    assert result["model_name"] == "langgraph_plain"
    assert result["success"] is True
    assert result["steps"] == 1
    assert result["stop_reason"] == "environment_done"
    assert result["workflow_final_status"] == "completed"
    assert len(result["trajectory"]) == 1
