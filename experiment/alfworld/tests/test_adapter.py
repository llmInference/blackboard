"""
Unit tests for ALFWorldKernelAdapter.

Tests the core adapter functionality.
"""
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest
from core.adapter import ALFWorldKernelAdapter
from environments.env_wrapper import ALFWorldEnvWrapper
from core.state_bridge import obs_to_kernel_state
from utils.mock_llm import ALFWorldMockActionLLM
from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.worker.base import RuleWorkerAgent


class _CaptureContextWorker(RuleWorkerAgent):
    last_context = None

    def __init__(self, llm=None, instruction=None) -> None:
        self.llm = llm
        self.instruction = instruction

    def _think(self, context):
        _CaptureContextWorker.last_context = context
        return [
            {"op": "replace", "path": "/selected_action", "value": "look"},
            {"op": "replace", "path": "/decision_reason", "value": "capture"},
        ]


class _UsageReportingMockLLM(ALFWorldMockActionLLM):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
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


def test_adapter_init():
    """Test adapter initialization."""
    # Create mock environment
    env = ALFWorldEnvWrapper(config_dict={
        "env": {
            "type": "AlfredTWEnv",
            "data_path": "/test/path"
        }
    })

    # Create mock LLM (we'll use None for now since we're just testing init)
    llm = None

    adapter = ALFWorldKernelAdapter(env, llm, config={"max_steps": 10})

    assert adapter.env == env
    assert adapter.llm == llm
    assert adapter.config["max_steps"] == 10
    assert adapter.kernel_graph is None


def test_adapter_default_config():
    """Test adapter with default config."""
    env = ALFWorldEnvWrapper()
    llm = None

    adapter = ALFWorldKernelAdapter(env, llm)

    assert adapter.config == {}


# Note: Full integration tests with actual Kernel System graph
# require LLM and registered workers. These would be added in
# integration test suite after Phase 3 (Workers implementation).


def test_adapter_builds_kernel_graph():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(env, None)

    graph = adapter.build_kernel_graph()
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


def test_adapter_builds_kernel_graph_for_planner_action_mode():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(env, None, config={"workflow_mode": "planner_action"})

    graph = adapter.build_kernel_graph()
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

    assert result["domain_state"]["subgoal"] in {
        "acquire_goal_object",
        "place_goal_object",
        "task_completed",
    }
    assert result["domain_state"]["selected_action"] in {"take apple 1", "go to fridge 1", "look"}


def test_adapter_builds_kernel_graph_for_planner_llm_action_mode():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(
        env,
        ALFWorldMockActionLLM(),
        config={"workflow_mode": "planner_llm_action"},
    )

    graph = adapter.build_kernel_graph()
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
    assert "Mock LLM followed subgoal" in result["domain_state"]["decision_reason"]


def test_adapter_llm_workflow_mode_requires_llm():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(env, None, config={"workflow_mode": "planner_llm_action"})

    with pytest.raises(ValueError, match="requires a configured llm"):
        adapter.build_kernel_graph()


def test_adapter_llm_architect_mode_requires_llm():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(env, None, config={"workflow_mode": "single_action", "architect_mode": "llm"})

    with pytest.raises(ValueError, match="requires a configured llm"):
        adapter.build_kernel_graph()


def test_adapter_ablate_c4_passes_full_state_to_worker():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(
        env,
        None,
        config={
            "workflow_mode": "single_action",
            "ablation": AblationConfig.ablate("C4"),
        },
    )
    graph = adapter.build_kernel_graph()

    from langgraph_kernel.worker.registry import get_registry

    registry = get_registry()
    original_worker = registry.get("action_worker")
    registry.register("action_worker", _CaptureContextWorker)
    try:
        state = obs_to_kernel_state(
            "You are in the kitchen.",
            {
                "extra.goal": ["Put the apple in the fridge"],
                "admissible_commands": [["look", "take apple 1", "go to fridge 1"]],
            },
        )
        state["status_history"] = ["deciding"]
        state["task_flow"] = [{"subtask": "Select action", "worker": "action_worker"}]
        state["worker_instructions"] = {"action_worker": "Use full state"}
        graph.invoke(state)
    finally:
        registry.register("action_worker", original_worker)

    assert _CaptureContextWorker.last_context is not None
    assert "status_history" in _CaptureContextWorker.last_context
    assert "task_flow" in _CaptureContextWorker.last_context
    assert "worker_instructions" in _CaptureContextWorker.last_context


def test_adapter_collects_llm_token_usage_into_turn_result():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(
        env,
        _UsageReportingMockLLM(),
        config={"workflow_mode": "planner_llm_action"},
    )

    graph = adapter.build_kernel_graph()
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


def test_adapter_builds_kernel_graph_for_llm_architect_mode():
    env = ALFWorldEnvWrapper()
    adapter = ALFWorldKernelAdapter(
        env,
        ALFWorldMockActionLLM(),
        config={"workflow_mode": "planner_llm_action", "architect_mode": "llm"},
    )

    graph = adapter.build_kernel_graph()
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

    assert result["domain_state"]["architect_decision"]
    assert "apple" in result["worker_instructions"]["llm_action_worker"].lower()
