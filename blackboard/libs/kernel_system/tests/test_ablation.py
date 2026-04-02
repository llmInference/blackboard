from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph_kernel.ablation import AblationConfig
import langgraph_kernel.graph as graph_module
from langgraph_kernel.architect.fixed_pipeline import FixedPipelineArchitect
from langgraph_kernel.graph import build_kernel_graph
from langgraph_kernel.kernel.node import kernel_node
from langgraph_kernel.worker.base import LLMWorkerAgent, RuleWorkerAgent


class _MockLLM(BaseChatModel):
    architect_response: dict[str, Any] | None = None
    response_content: str | None = None
    last_messages: list[Any] | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import json

        object.__setattr__(self, "last_messages", messages)
        if self.architect_response is not None:
            content = json.dumps(self.architect_response)
        else:
            content = self.response_content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock"


class _StubCircuitBreaker:
    def __init__(self, should_break: bool = False, reason: str = "", details: dict[str, Any] | None = None):
        self.should_break = should_break
        self.reason = reason
        self.details = details or {}
        self.check_calls = 0

    def check(self, state: dict[str, Any]):
        self.check_calls += 1
        return self.should_break, self.reason, self.details

    def generate_diagnostic_report(self, reason: str, details: dict[str, Any]) -> str:
        return f"diagnostic:{reason}:{details.get('message', '')}"


class _CaptureWorker(RuleWorkerAgent):
    def __init__(self) -> None:
        self.last_context: dict[str, Any] | None = None

    def _think(self, context: dict[str, Any]):
        self.last_context = context
        return []


class _FakeStateGraph:
    def __init__(self, _schema) -> None:
        self.node_kwargs: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, Any] = {}

    def add_node(self, name: str, node, **kwargs):
        self.nodes[name] = node
        self.node_kwargs[name] = kwargs

    def add_conditional_edges(self, *args, **kwargs):
        return None

    def add_edge(self, *args, **kwargs):
        return None

    def compile(self, checkpointer=None):
        return self


class _ArchitectStub:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def test_ablation_config_ablate_disables_requested_components():
    config = AblationConfig.ablate("C1", "C2", "C4")

    assert config.use_json_patch is False
    assert config.use_schema_validation is False
    assert config.use_circuit_breaker is False
    assert config.use_context_slicing is True
    assert config.use_architect_agent is False
    assert config.hard_max_steps == 50


def test_build_kernel_graph_injects_ablation_config():
    ablation = AblationConfig.ablate("C1", "C3")
    llm = _MockLLM(architect_response={"should_continue": False})
    graph = build_kernel_graph(llm, ablation=ablation)

    result = graph.invoke(
        {
            "domain_state": {"user_prompt": "结束当前任务"},
            "task_flow": [],
            "data_schema": {},
            "workflow_rules": {},
            "worker_instructions": {},
            "selected_workers": [],
            "pending_patch": [],
            "patch_error": "",
            "step_count": 0,
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
            "conversation_history": [],
            "pending_user_question": "",
            "user_response": "",
            "waiting_for_user": False,
        }
    )

    assert result["ablation_config"] == ablation
    assert result["ablation_config"].use_json_patch is False
    assert result["ablation_config"].use_schema_validation is False
    assert result["ablation_config"].use_context_slicing is False
    assert result["domain_state"]["status"] == "done"


def test_llm_worker_returns_plain_text_when_json_patch_disabled():
    llm = _MockLLM(response_content="这是自然语言分析结果")
    worker = LLMWorkerAgent(llm)

    result = worker(
        {
            "domain_state": {"user_prompt": "分析这个任务"},
            "data_schema": {},
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.ablate("C1"),
        }
    )

    assert result["pending_patch"] == "这是自然语言分析结果"
    assert llm.last_messages is not None
    assert "Output plain text only." in llm.last_messages[0].content


def test_kernel_stores_worker_message_when_json_patch_disabled():
    result = kernel_node(
        {
            "ablation_config": AblationConfig.ablate("C1"),
            "domain_state": {
                "status": "planning",
                "messages": [{"worker": "seed", "content": "已有内容"}],
            },
            "data_schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
            "workflow_rules": {"status": {"planning": "planner_worker", "done": None}},
            "pending_patch": "新增的自然语言输出",
            "patch_error": "",
            "step_count": 0,
            "current_worker": "planner_worker",
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
    )

    assert result["domain_state"]["messages"] == [
        {"worker": "seed", "content": "已有内容"},
        {"worker": "planner_worker", "content": "新增的自然语言输出"},
    ]
    assert result["domain_state"]["status"] == "done"
    assert result["step_count"] == 1


def test_json_patch_path_still_works_when_c1_enabled():
    llm = _MockLLM(
        response_content='[{"op": "add", "path": "/analysis", "value": "结构化结果"}]'
    )
    worker = LLMWorkerAgent(llm)

    worker_result = worker(
        {
            "domain_state": {"status": "planning", "user_prompt": "分析任务"},
            "data_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["planning", "done"]},
                    "analysis": {"type": "string"},
                },
                "required": ["status"],
            },
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.full(),
        }
    )

    assert worker_result["pending_patch"] == [
        {"op": "add", "path": "/analysis", "value": "结构化结果"}
    ]
    assert llm.last_messages is not None
    assert "must return a JSON Patch" in llm.last_messages[0].content

    kernel_result = kernel_node(
        {
            "ablation_config": AblationConfig.full(),
            "domain_state": {"status": "planning"},
            "data_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["planning", "done"]},
                    "analysis": {"type": "string"},
                },
                "required": ["status"],
            },
            "workflow_rules": {"status": {"planning": "planner_worker", "done": None}},
            "pending_patch": worker_result["pending_patch"],
            "patch_error": "",
            "step_count": 0,
            "current_worker": "planner_worker",
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
    )

    assert kernel_result["domain_state"]["analysis"] == "结构化结果"
    assert kernel_result["domain_state"]["status"] == "done"
    assert kernel_result["patch_error"] == ""


def test_llm_worker_converts_object_output_to_json_patch():
    llm = _MockLLM(
        response_content='{"selected_action": "go to armchair 1", "decision_reason": "The chair is the target receptacle."}'
    )
    worker = LLMWorkerAgent(llm)

    worker_result = worker(
        {
            "domain_state": {"status": "acting", "user_prompt": "Place the credit card on a chair"},
            "data_schema": {"type": "object", "properties": {}},
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.full(),
        }
    )

    assert worker_result["pending_patch"] == [
        {"op": "replace", "path": "/selected_action", "value": "go to armchair 1"},
        {"op": "replace", "path": "/decision_reason", "value": "The chair is the target receptacle."},
    ]


def test_llm_worker_strips_reasoning_wrapper_before_json_parse():
    llm = _MockLLM(
        response_content=(
            "<think>Need to reason about the next action first.</think>\n"
            '{"selected_action": "go to drawer 1"}'
        )
    )
    worker = LLMWorkerAgent(llm)

    worker_result = worker(
        {
            "domain_state": {"status": "acting", "user_prompt": "Find the credit card"},
            "data_schema": {"type": "object", "properties": {}},
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.full(),
        }
    )

    assert worker_result["pending_patch"] == [
        {"op": "replace", "path": "/selected_action", "value": "go to drawer 1"},
        {
            "op": "replace",
            "path": "/decision_reason",
            "value": "Selected the next action from the current state slice.",
        },
    ]


def test_llm_worker_recovers_action_from_natural_language_response():
    llm = _MockLLM(
        response_content=(
            "Based on the current state slice, the next logical step is to search drawer 1 "
            "for the credit card before placing it on the armchair."
        )
    )
    worker = LLMWorkerAgent(llm)

    worker_result = worker(
        {
            "domain_state": {
                "status": "acting",
                "user_prompt": "Find the credit card and place it on the chair",
                "available_actions": ["go to drawer 1", "go to drawer 2", "go to armchair 1", "look"],
                "selected_action": "",
                "decision_reason": "",
                "recommended_actions": ["go to drawer 1", "go to drawer 2"],
            },
            "data_schema": {"type": "object", "properties": {}},
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.full(),
        }
    )

    assert worker_result["pending_patch"][0]["value"] == "go to drawer 1"
    assert worker_result["pending_patch"][1]["path"] == "/decision_reason"


def test_llm_worker_recovers_first_recommended_action_from_summary_text():
    llm = _MockLLM(
        response_content=(
            "Based on the current state slice, the next logical step would be to search for the credit card "
            "in the recommended locations: drawer 1 and drawer 2."
        )
    )
    worker = LLMWorkerAgent(llm)

    worker_result = worker(
        {
            "domain_state": {
                "status": "acting",
                "user_prompt": "Find the credit card and place it on the chair",
                "available_actions": ["go to drawer 1", "go to drawer 2", "go to armchair 1", "look"],
                "selected_action": "",
                "decision_reason": "",
                "recommended_actions": ["go to drawer 1", "go to drawer 2"],
            },
            "data_schema": {"type": "object", "properties": {}},
            "error_feedback": "",
            "retry_count": 0,
            "ablation_config": AblationConfig.full(),
        }
    )

    assert worker_result["pending_patch"][0]["value"] == "go to drawer 1"


def test_schema_validation_enabled_still_requests_retry_on_invalid_patch():
    result = kernel_node(
        {
            "ablation_config": AblationConfig.full(),
            "domain_state": {"status": "planning"},
            "data_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["planning", "done"]},
                    "analysis": {"type": "string"},
                },
                "required": ["status"],
            },
            "workflow_rules": {"status": {"planning": "planner_worker", "done": None}},
            "pending_patch": [{"path": "/analysis", "value": "missing op"}],
            "patch_error": "",
            "step_count": 0,
            "current_worker": "planner_worker",
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
    )

    assert result["step_count"] == 0
    assert result["retry_count"] == 1
    assert result["patch_error"] == ""
    assert result["error_feedback"] != ""
    assert result["domain_state"] == {"status": "planning"}


def test_schema_validation_override_applies_patch_without_type_check():
    result = kernel_node(
        {
            "ablation_config": AblationConfig(use_schema_validation=False),
            "domain_state": {"status": "planning"},
            "data_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["planning", "done"]},
                    "analysis": {"type": "string"},
                },
                "required": ["status"],
            },
            "workflow_rules": {"status": {"planning": "planner_worker", "done": None}},
            "pending_patch": [{"op": "add", "path": "/analysis", "value": {"bad": "type"}}],
            "patch_error": "",
            "step_count": 0,
            "current_worker": "planner_worker",
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
    )

    assert result["domain_state"]["analysis"] == {"bad": "type"}
    assert result["domain_state"]["status"] == "done"
    assert result["step_count"] == 1
    assert result["retry_count"] == 0
    assert result["patch_error"] == ""


def test_schema_validation_override_does_not_retry_on_apply_error():
    result = kernel_node(
        {
            "ablation_config": AblationConfig(use_schema_validation=False),
            "domain_state": {"status": "planning"},
            "data_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["planning", "done"]},
                    "analysis": {"type": "string"},
                },
                "required": ["status"],
            },
            "workflow_rules": {"status": {"planning": "planner_worker", "done": None}},
            "pending_patch": [{"op": "replace", "path": "/analysis", "value": "missing field"}],
            "patch_error": "",
            "step_count": 0,
            "current_worker": "planner_worker",
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
    )

    assert result["domain_state"] == {"status": "done"}
    assert result["step_count"] == 1
    assert result["retry_count"] == 0
    assert result["patch_error"] == ""
    assert result["no_update_count"] == 1


def test_c2_enabled_still_uses_circuit_breaker_checks():
    breaker = _StubCircuitBreaker(
        should_break=True,
        reason="timeout",
        details={"message": "breaker triggered"},
    )

    result = kernel_node(
        {
            "ablation_config": AblationConfig.full(),
            "domain_state": {"status": "planning"},
            "pending_patch": [],
            "patch_error": "",
            "step_count": 3,
            "circuit_breaker": breaker,
        }
    )

    assert breaker.check_calls == 1
    assert result["circuit_breaker_triggered"] is True
    assert result["circuit_breaker_reason"] == "timeout"
    assert result["circuit_breaker_details"] == {"message": "breaker triggered"}
    assert result["patch_error"] == "系统熔断: timeout"


def test_c2_disabled_skips_circuit_breaker_before_hard_limit():
    breaker = _StubCircuitBreaker(should_break=True, reason="timeout", details={"message": "should not run"})

    result = kernel_node(
        {
            "ablation_config": AblationConfig.ablate("C2"),
            "domain_state": {"status": "planning"},
            "pending_patch": [],
            "patch_error": "",
            "step_count": 2,
            "retry_count": 0,
            "error_feedback": "",
            "circuit_breaker": breaker,
        }
    )

    assert breaker.check_calls == 0
    assert result["patch_error"] == ""
    assert result["step_count"] == 2
    assert result["retry_count"] == 0


def test_c2_disabled_uses_hard_max_steps_only():
    breaker = _StubCircuitBreaker(should_break=True, reason="timeout", details={"message": "should not run"})
    ablation = AblationConfig.ablate("C2")
    ablation.hard_max_steps = 3

    result = kernel_node(
        {
            "ablation_config": ablation,
            "domain_state": {"status": "planning"},
            "pending_patch": [],
            "patch_error": "",
            "step_count": 3,
            "circuit_breaker": breaker,
        }
    )

    assert breaker.check_calls == 0
    assert result["circuit_breaker_triggered"] is True
    assert result["circuit_breaker_reason"] == "hard_limit"
    assert result["circuit_breaker_details"]["hard_max_steps"] == 3
    assert result["patch_error"] == "系统熔断: hard_limit"


def test_context_slicing_enabled_keeps_worker_context_minimal():
    worker = _CaptureWorker()

    worker(
        {
            "ablation_config": AblationConfig.full(),
            "domain_state": {
                "status": "planning",
                "user_prompt": "写一份计划",
                "analysis": "已有分析",
            },
            "data_schema": {"type": "object"},
            "error_feedback": "fix previous patch",
            "retry_count": 1,
            "step_count": 7,
            "status_history": ["planning", "reviewing"],
            "task_flow": [{"subtask": "plan", "worker": "planner"}],
            "selected_workers": ["planner"],
            "worker_instructions": {"planner": "focus on budget"},
        }
    )

    assert worker.last_context == {
        "status": "planning",
        "user_prompt": "写一份计划",
        "analysis": "已有分析",
        "error_feedback": "fix previous patch",
        "retry_count": 1,
        "_data_schema": {"type": "object"},
    }


def test_context_slicing_disabled_passes_full_runtime_context():
    worker = _CaptureWorker()

    worker(
        {
            "ablation_config": AblationConfig.ablate("C3"),
            "domain_state": {
                "status": "planning",
                "user_prompt": "写一份计划",
                "analysis": "已有分析",
            },
            "data_schema": {"type": "object"},
            "error_feedback": "fix previous patch",
            "retry_count": 1,
            "step_count": 7,
            "status_history": ["planning", "reviewing"],
            "task_flow": [{"subtask": "plan", "worker": "planner"}],
            "selected_workers": ["planner"],
            "worker_instructions": {"planner": "focus on budget"},
        }
    )

    assert worker.last_context == {
        "status": "planning",
        "user_prompt": "写一份计划",
        "analysis": "已有分析",
        "error_feedback": "fix previous patch",
        "retry_count": 1,
        "_data_schema": {"type": "object"},
        "step_count": 7,
        "status_history": ["planning", "reviewing"],
        "task_flow": [{"subtask": "plan", "worker": "planner"}],
        "selected_workers": ["planner"],
        "worker_instructions": {"planner": "focus on budget"},
    }


def test_build_graph_uses_worker_input_schema_when_context_slicing_enabled(monkeypatch):
    monkeypatch.setattr(graph_module, "StateGraph", _FakeStateGraph)

    builder = build_kernel_graph(_MockLLM(architect_response={"should_continue": False}))

    assert "input_schema" in builder.node_kwargs["worker"]
    assert builder.node_kwargs["worker"]["input_schema"] is not None


def test_build_graph_omits_worker_input_schema_when_context_slicing_disabled(monkeypatch):
    monkeypatch.setattr(graph_module, "StateGraph", _FakeStateGraph)

    builder = build_kernel_graph(
        _MockLLM(architect_response={"should_continue": False}),
        ablation=AblationConfig.ablate("C3"),
    )

    assert "input_schema" not in builder.node_kwargs["worker"]


def test_fixed_pipeline_architect_returns_expected_static_workflow():
    architect = FixedPipelineArchitect()

    result = architect(
        {
            "domain_state": {"user_prompt": "帮我完成这个任务"},
        }
    )

    assert result["task_flow"] == [
        {"subtask": "analyze", "worker": "analyzer"},
        {"subtask": "plan", "worker": "planner"},
        {"subtask": "execute", "worker": "executor"},
        {"subtask": "review", "worker": "reviewer"},
    ]
    assert result["workflow_rules"] == {
        "status": {
            "analyzing": "analyzer",
            "planning": "planner",
            "executing": "executor",
            "reviewing": "reviewer",
            "done": None,
        }
    }
    assert result["selected_workers"] == ["analyzer", "planner", "executor", "reviewer"]
    assert result["worker_instructions"] == {}
    assert result["domain_state"] == {"user_prompt": "帮我完成这个任务", "status": "analyzing"}
    assert result["data_schema"]["required"] == ["status", "user_prompt"]


def test_build_graph_uses_orchestrator_architect_when_c4_enabled(monkeypatch):
    monkeypatch.setattr(graph_module, "StateGraph", _FakeStateGraph)
    monkeypatch.setattr(graph_module, "OrchestratorArchitect", _ArchitectStub)
    monkeypatch.setattr(graph_module, "FixedPipelineArchitect", _ArchitectStub)

    llm = _MockLLM(architect_response={"should_continue": False})
    builder = build_kernel_graph(llm, ablation=AblationConfig.full())

    architect_node = builder.nodes["architect"]
    assert isinstance(architect_node, _ArchitectStub)
    assert architect_node.args == (llm,)


def test_build_graph_uses_fixed_pipeline_architect_when_c4_disabled(monkeypatch):
    monkeypatch.setattr(graph_module, "StateGraph", _FakeStateGraph)
    monkeypatch.setattr(graph_module, "OrchestratorArchitect", _ArchitectStub)
    monkeypatch.setattr(graph_module, "FixedPipelineArchitect", _ArchitectStub)

    builder = build_kernel_graph(
        _MockLLM(architect_response={"should_continue": False}),
        ablation=AblationConfig.ablate("C4"),
    )

    architect_node = builder.nodes["architect"]
    assert isinstance(architect_node, _ArchitectStub)
    assert architect_node.args == ()
