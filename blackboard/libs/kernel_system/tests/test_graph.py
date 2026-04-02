"""
集成测试：使用 mock LLM 验证完整图执行流程。
"""
from typing import Any
from typing_extensions import TypedDict

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from langgraph_kernel.graph import build_kernel_graph
from langgraph_kernel.worker.base import RuleWorkerAgent
from langgraph_kernel.types import JsonPatch


# ── Mock LLM ─────────────────────────────────────────────────────────────────

class _MockLLM(BaseChatModel):
    """返回固定 structured output 的 mock LLM，用于测试 ArchitectAgent。"""

    architect_response: dict[str, Any]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import json
        content = json.dumps(self.architect_response)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock"

    def with_structured_output(self, schema):
        """返回一个直接输出 architect_response 的 runnable。"""
        from langchain_core.runnables import RunnableLambda
        data = self.architect_response

        def _parse(_):
            return schema(**data)

        return RunnableLambda(_parse)


# ── Mock Worker ───────────────────────────────────────────────────────────────

class _PlannerWorker(RuleWorkerAgent):
    """将 status 从 planning 改为 done。"""

    class InputSchema(TypedDict):
        domain_state: dict

    input_schema = InputSchema

    def _think(self, context: dict[str, Any]) -> JsonPatch:
        return [{"op": "replace", "path": "/status", "value": "done"}]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_full_flow():
    """完整流程：跳过 architect，直接测试 kernel + worker 循环"""
    llm = _MockLLM(architect_response={})  # 不使用 architect

    workers = {"planner_worker": _PlannerWorker()}

    # 手动构建图，跳过 architect
    from langgraph.graph import START, StateGraph
    from langgraph_kernel.types import KernelState
    from langgraph_kernel.kernel.node import kernel_node
    from langgraph_kernel.kernel.router import WorkflowRouter

    builder = StateGraph(KernelState)
    builder.add_node("kernel", kernel_node)
    builder.add_node("planner_worker", _PlannerWorker(), input_schema=_PlannerWorker.InputSchema)

    router = WorkflowRouter(["planner_worker"])
    builder.add_edge(START, "kernel")
    builder.add_conditional_edges("kernel", router.route)
    builder.add_edge("planner_worker", "kernel")

    graph = builder.compile()

    result = graph.invoke({
        "domain_state": {"status": "planning"},
        "data_schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["planning", "done"]}},
        },
        "workflow_rules": {"status": {"planning": "planner_worker"}},
        "pending_patch": [],
        "patch_error": "",
        "step_count": 0,
    })

    assert result["domain_state"]["status"] == "done"
    assert result["patch_error"] == ""


def test_patch_error_terminates():
    """patch 验证失败时图应终止，patch_error 非空。"""
    class _BadWorker(RuleWorkerAgent):
        class InputSchema(TypedDict):
            domain_state: dict
        input_schema = InputSchema

        def _think(self, context):
            # 提交一个违反 schema 的 patch
            return [{"op": "replace", "path": "/status", "value": "INVALID"}]

    # 手动构建图
    from langgraph.graph import START, StateGraph
    from langgraph_kernel.types import KernelState
    from langgraph_kernel.kernel.node import kernel_node
    from langgraph_kernel.kernel.router import WorkflowRouter

    builder = StateGraph(KernelState)
    builder.add_node("kernel", kernel_node)
    builder.add_node("bad_worker", _BadWorker(), input_schema=_BadWorker.InputSchema)

    router = WorkflowRouter(["bad_worker"])
    builder.add_edge(START, "kernel")
    builder.add_conditional_edges("kernel", router.route)
    builder.add_edge("bad_worker", "kernel")

    graph = builder.compile()

    result = graph.invoke({
        "domain_state": {"status": "planning"},
        "data_schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["planning", "done"]}},
        },
        "workflow_rules": {"status": {"planning": "bad_worker"}},
        "pending_patch": [],
        "patch_error": "",
        "step_count": 0,
    })

    assert result["patch_error"] != ""
    assert "schema validation error" in result["patch_error"]
