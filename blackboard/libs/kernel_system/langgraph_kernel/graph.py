from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.architect.fixed_pipeline import FixedPipelineArchitect
from langgraph_kernel.architect.orchestrator import OrchestratorArchitect
from langgraph_kernel.kernel.node import kernel_node
from langgraph_kernel.kernel.router import WorkflowRouter
from langgraph_kernel.types import KernelState
from langgraph_kernel.worker.registry import get_registry
from langgraph_kernel.workflow import extract_worker_name, get_matching_workflow_entry


def build_kernel_graph(
    llm: BaseChatModel,
    max_steps: int = 50,
    checkpointer: BaseCheckpointSaver | None = None,
    ablation: AblationConfig | None = None,
) -> Any:
    """
    构建编排型 Architect-Kernel-Worker 图（使用固定 Worker 类型）。

    工作流程：
    1. Orchestrator Architect 分析用户 prompt，拆分任务
    2. 从 Worker Registry 中选择合适的 workers
    3. 生成 data_schema 和 workflow_rules 来编排这些 workers
    4. Kernel 根据 workflow_rules 路由到相应的 worker
    5. Workers 使用预定义的 system_prompt 执行任务

    参数：
        llm: 兼容 OpenAI 格式的第三方模型
        max_steps: 最大执行步数
        checkpointer: 可选的检查点保存器
        ablation: 消融实验配置，默认开启全部组件

    示例：
        graph = build_kernel_graph(llm)
        result = graph.invoke({
            "domain_state": {"user_prompt": "帮我分析这个问题"},
            "task_flow": [],
            "data_schema": {},
            "workflow_rules": {},
            "selected_workers": [],
            "pending_patch": [],
            "patch_error": "",
            "step_count": 0,
        })
    """
    registry = get_registry()
    ablation_config = ablation or AblationConfig.full()

    # 创建通用 worker 节点的 input schema
    class WorkerInputSchema(TypedDict):
        domain_state: dict
        selected_workers: list[str]
        current_worker: str
        data_schema: dict
        ablation_config: AblationConfig

    # 通用 worker 节点
    def worker_node(state: dict[str, Any]) -> dict[str, Any]:
        """通用 worker 节点，根据 current_worker 从 registry 创建 worker"""
        worker_name = state.get("current_worker", "")
        worker_instructions = state.get("worker_instructions", {})

        # 获取该 worker 的特定指令
        instruction = worker_instructions.get(worker_name, None)

        # 从 registry 创建 worker，传递指令
        worker = registry.create_worker(worker_name, llm, instruction)
        if worker is None:
            return {
                "pending_patch": [],
                "patch_error": f"Unknown worker type: {worker_name}",
            }

        worker.input_schema = WorkerInputSchema if ablation_config.use_context_slicing else None

        # 执行 worker
        return worker(state)

    # 动态路由
    router = WorkflowRouter(
        worker_names=[],  # 动态 worker，不预定义
        max_steps=max_steps,
        max_no_update=2,
        loop_detection_window=3,
    )

    def route_to_worker(state: KernelState) -> str:
        """根据 workflow_rules 动态路由到 worker"""
        # 1. 检查熔断
        if state.get("circuit_breaker_triggered", False):
            print("\n🔴 系统已熔断，终止执行")
            return END

        # 2. 检查是否正在等待用户输入
        if state.get("waiting_for_user", False):
            return END

        # 3. 正常路由
        result = router.route(state)

        if result != END:
            return "worker"
        return END

    # 设置 current_worker 的中间节点
    def set_current_worker(state: KernelState) -> dict[str, Any]:
        """在路由到 worker 之前，设置 current_worker 字段"""
        workflow_rules = state.get("workflow_rules", {})
        domain_state = state.get("domain_state", {})

        _, _, entry = get_matching_workflow_entry(domain_state, workflow_rules)
        return {"current_worker": extract_worker_name(entry) or ""}

    # 处理用户响应的节点
    def handle_user_response(state: KernelState) -> dict[str, Any]:
        """处理用户的响应，更新对话历史"""
        user_response = state.get("user_response", "")
        pending_question = state.get("pending_user_question", "")
        conversation_history = state.get("conversation_history", [])
        domain_state = state.get("domain_state", {})

        if user_response:
            # 添加问题和回答到对话历史
            new_history = conversation_history.copy()
            if pending_question:
                new_history.append({"role": "system", "content": pending_question})
            new_history.append({"role": "user", "content": user_response})

            # 更新 domain_state，将用户反馈添加到 user_prompt 中
            # 这样 Architect 可以看到完整的上下文
            new_domain_state = domain_state.copy()
            original_prompt = new_domain_state.get("user_prompt", "")
            new_domain_state["user_prompt"] = f"{original_prompt}\n\n用户反馈: {user_response}"
            new_domain_state["user_feedback"] = user_response

            return {
                "conversation_history": new_history,
                "pending_user_question": "",
                "user_response": "",  # 清空 user_response，避免重复处理
                "waiting_for_user": False,  # 清空等待标志
                "domain_state": new_domain_state,
            }

        return {}

    def initialize_ablation(_: KernelState) -> dict[str, AblationConfig]:
        """Inject the configured ablation switches into the runtime state."""
        return {"ablation_config": ablation_config}

    architect = (
        OrchestratorArchitect(llm)
        if ablation_config.use_architect_agent
        else FixedPipelineArchitect()
    )

    builder = StateGraph(KernelState)

    # 添加节点
    builder.add_node("initialize_ablation", initialize_ablation)
    builder.add_node("handle_user_response", handle_user_response)
    builder.add_node("architect", architect)
    builder.add_node("kernel", kernel_node)
    builder.add_node("set_worker", set_current_worker)
    if ablation_config.use_context_slicing:
        builder.add_node("worker", worker_node, input_schema=WorkerInputSchema)
    else:
        builder.add_node("worker", worker_node)

    # 条件边：检查是否是继续对话（有 user_response）还是新对话
    def check_conversation_mode(state: KernelState) -> str:
        """检查是继续对话还是新对话"""
        # 如果有 user_response（非空字符串），说明是继续对话
        if state.get("user_response", "").strip():
            return "continue"
        # 否则是新对话，需要 architect
        return "new"

    # 添加边
    builder.add_conditional_edges(
        "initialize_ablation",
        check_conversation_mode,
        {"continue": "handle_user_response", "new": "architect"}
    )
    builder.add_edge(START, "initialize_ablation")
    builder.add_edge("handle_user_response", "architect")  # 用户响应后重新从 Architect 开始
    builder.add_edge("architect", "kernel")

    # 条件边：从 kernel 到 set_worker 或 END
    builder.add_conditional_edges(
        "kernel",
        route_to_worker,
        {"worker": "set_worker", END: END}
    )

    builder.add_edge("set_worker", "worker")
    builder.add_edge("worker", "kernel")

    return builder.compile(checkpointer=checkpointer)

