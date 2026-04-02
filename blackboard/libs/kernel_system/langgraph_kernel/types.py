from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from typing_extensions import Annotated, NotRequired, TypedDict

if TYPE_CHECKING:
    from langgraph_kernel.circuit_breaker import CircuitBreaker
    from langgraph_kernel.state_tree import StateTree

# ── 基础类型别名 ──────────────────────────────────────────────────────────────

# JSON Schema dict，由 Architect 输出，约束 domain_state 的结构
DataSchema = Dict[str, Any]

# Workflow Rules：字段名 -> {状态值 -> worker或结构化路由配置}
# 例：{"status": {"planning": "planner_worker", "executing": {"worker": "executor_worker", "next": "done"}}}
WorkflowRules = Dict[str, Dict[str, Any]]

# RFC 6902 JSON Patch 操作列表
# 例：[{"op": "replace", "path": "/status", "value": "planning"}]
JsonPatch = List[Dict[str, Any]]

# Worker 到 Kernel 的通信载荷：
# 默认使用 JSON Patch；C1 关闭时退化为自然语言字符串。
WorkerPayload = JsonPatch | str


# ── Reducer ───────────────────────────────────────────────────────────────────

def _overwrite(current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """kernel_node 验证后直接覆盖 domain_state，不做合并。"""
    return new if new is not None else current


# ── 全局状态 ──────────────────────────────────────────────────────────────────

class KernelState(TypedDict):
    """贯穿整个图执行的全局状态。"""

    # Architect 输出（只写一次）
    task_flow: List[Dict[str, Any]]  # 任务流：可包含 worker、subtask、depends_on、branch 等元数据
    data_schema: DataSchema
    workflow_rules: WorkflowRules
    worker_instructions: Dict[str, str]  # 新增：worker 指令 {"worker_name": "instruction"}
    selected_workers: List[str]  # 选中的 worker 列表

    # 业务状态（受 data_schema 约束，由 kernel_node 验证后更新）
    domain_state: Annotated[Dict[str, Any], _overwrite]

    # 运行时控制
    ablation_config: NotRequired[Any]  # 消融实验配置，由 graph 入口注入
    pending_patch: WorkerPayload   # worker 提交的载荷，等待 kernel 处理
    patch_error: str           # 验证失败时的错误信息，空字符串表示无错误
    step_count: int            # 已执行步数，防止无限循环
    current_worker: str        # 当前要执行的 worker 名称
    turn_worker_input_tokens: NotRequired[int]  # 当前 ALFWorld turn 内累计输入 token
    turn_worker_output_tokens: NotRequired[int]  # 当前 ALFWorld turn 内累计输出 token
    turn_architect_input_tokens: NotRequired[int]  # 当前 ALFWorld turn 内 architect 输入 token
    turn_architect_output_tokens: NotRequired[int]  # 当前 ALFWorld turn 内 architect 输出 token

    # Layer 5: 反馈重试
    retry_count: int           # 当前 worker 的重试次数
    error_feedback: str        # 反馈给 worker 的错误信息

    # 终止检测
    no_update_count: int       # 连续无有效更新的次数（用于检测 worker 无实质工作）
    status_history: List[str]  # 最近的状态历史（用于检测循环）

    # 中间状态持久化：使用状态树替代简单列表
    state_tree: Any  # StateTree 实例（使用 Any 避免循环导入）
    current_node_id: str  # 当前节点 ID

    # 熔断器
    circuit_breaker: Any  # CircuitBreaker 实例（使用 Any 避免循环导入）
    circuit_breaker_triggered: bool  # 是否已触发熔断
    circuit_breaker_reason: str  # 熔断原因
    circuit_breaker_details: Dict[str, Any]  # 熔断详情

    # 多轮对话支持
    conversation_history: List[Dict[str, str]]  # 对话历史：[{"role": "user"/"system", "content": "..."}]
    pending_user_question: str  # 等待用户回答的问题
    user_response: str  # 用户的最新回复
    waiting_for_user: bool  # 是否正在等待用户输入
    trace_events: NotRequired[List[Dict[str, Any]]]  # 调试时间线事件
