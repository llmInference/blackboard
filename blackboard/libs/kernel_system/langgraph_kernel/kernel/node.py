from __future__ import annotations

import copy
import json

import jsonpatch

from langgraph_kernel.kernel.patch_fixer import PatchFixer, PatchFixerConfig
from langgraph_kernel.types import KernelState
from langgraph_kernel.workflow import determine_next_status as workflow_determine_next_status
from langgraph_kernel.workflow import get_status_field as workflow_get_status_field

# 创建修复器实例（启用所有自动修复功能）
_fixer = PatchFixer(
    config=PatchFixerConfig(
        fix_structure=True,
        fix_spelling=True,
        fix_path_format=True,
        auto_switch_add_replace=True,
        create_missing_parents=False,  # 保守策略
        fix_type_mismatch=True,
        fuzzy_match_enum=True,
        fuzzy_match_threshold=0.8,
        skip_invalid_operations=False,
    )
)

# Layer 5 配置
MAX_RETRIES = 2  # 最多重试次数


def _determine_next_status(
    current_state: dict,
    workflow_rules: dict,
    current_worker: str,
) -> str | None:
    """Delegate workflow progression to the shared workflow parser."""
    return workflow_determine_next_status(current_state, workflow_rules, current_worker)


def _filter_status_operations(patch: list, workflow_rules: dict) -> tuple[list, list]:
    """
    过滤掉 patch 中所有针对 status 字段的操作

    返回：(过滤后的 patch, 被移除的操作列表)
    """
    if not workflow_rules:
        return patch, []

    # 找到 status 字段名
    status_field = workflow_get_status_field(workflow_rules)
    if not status_field:
        return patch, []

    # 过滤操作
    filtered_patch = []
    removed_ops = []

    for op in patch:
        path = op.get("path", "")
        # 检查是否是针对 status 字段的操作
        # path 可能是 "/status" 或 "/status/xxx"
        if path == f"/{status_field}" or path.startswith(f"/{status_field}/"):
            removed_ops.append(op)
        else:
            filtered_patch.append(op)

    return filtered_patch, removed_ops


def _get_status_field(workflow_rules: dict) -> str | None:
    """Return the workflow field that drives worker routing."""
    return workflow_get_status_field(workflow_rules)


def kernel_node(state: KernelState) -> dict:
    """
    Kernel 核心节点：
    0. **检查熔断条件**（新增）
    1. 若无 pending_patch，直接跳过（首次进入）
    2. **过滤掉 worker 提交的所有 status 更新操作**
    3. 使用 PatchFixer 自动修复常见错误（Layer 1-4）
    4. 验证 patch 是否符合 data_schema
    5. 如果失败且未超过重试次数，生成错误反馈并请求重试（Layer 5）
    6. 应用 patch 更新 domain_state
    7. **自动更新 status 到下一个状态**（由 Kernel 控制）
    8. **更新状态树**（新增）
    9. 清空 pending_patch，写入 patch_error，递增 step_count
    """

    ablation = state.get("ablation_config")
    # ===== 0. 检查熔断条件 =====
    circuit_breaker = state.get("circuit_breaker")
    if ablation and not ablation.use_circuit_breaker:
        step_count = state.get("step_count", 0)
        if step_count >= ablation.hard_max_steps:
            reason = "hard_limit"
            details = {
                "step_count": step_count,
                "hard_max_steps": ablation.hard_max_steps,
                "message": f"系统执行步数 ({step_count}) 超过硬限制 ({ablation.hard_max_steps})",
            }
            print(f"\n🔴 系统达到硬步数上限: {ablation.hard_max_steps}")

            state_tree = state.get("state_tree")
            if state_tree and state.get("current_node_id"):
                state_tree.mark_terminal(state["current_node_id"], reason)

            return {
                "circuit_breaker_triggered": True,
                "circuit_breaker_reason": reason,
                "circuit_breaker_details": details,
                "domain_state": state.get("domain_state"),
                "patch_error": f"系统熔断: {reason}",
            }
    elif circuit_breaker:
        should_break, reason, details = circuit_breaker.check(state)

        if should_break:
            # 触发熔断
            print(f"\n🔴 系统熔断: {reason}")
            print(f"   {details.get('message', '')}")

            # 生成诊断报告
            report = circuit_breaker.generate_diagnostic_report(reason, details)
            print(f"\n{report}")

            # 标记状态树的当前节点为终止节点
            state_tree = state.get("state_tree")
            if state_tree and state.get("current_node_id"):
                state_tree.mark_terminal(state["current_node_id"], reason)

            return {
                "circuit_breaker_triggered": True,
                "circuit_breaker_reason": reason,
                "circuit_breaker_details": details,
                "domain_state": state.get("domain_state"),
                "patch_error": f"系统熔断: {reason}",
            }

    patch = state.get("pending_patch")
    if not patch:
        return {
            "patch_error": "",
            "step_count": state.get("step_count", 0),
            "retry_count": 0,
            "error_feedback": "",
        }

    if ablation and not ablation.use_json_patch:
        new_state = copy.deepcopy(state["domain_state"])
        current_worker = state.get("current_worker", "")
        workflow_rules = state.get("workflow_rules", {})
        message_content = patch if isinstance(patch, str) else json.dumps(patch, ensure_ascii=False)

        # 兼容直接产出 action patch 的退化运行时场景
        if isinstance(patch, list) and any(
            isinstance(op, dict) and op.get("path", "").startswith("/selected_action")
            for op in patch
        ):
            try:
                new_state = jsonpatch.apply_patch(new_state, patch)
                print(f"\n📝 Kernel 应用了 {current_worker or 'worker'} 的 action patch（自然语言模式）")
            except Exception as exc:
                print(f"\n⚠️  Kernel 应用 action patch 失败: {exc}，回退到消息模式")

        messages = list(new_state.get("messages", []))
        messages.append({"worker": current_worker, "content": message_content})
        new_state["messages"] = messages

        print(f"\n📝 Kernel 已将 {current_worker or 'worker'} 的自然语言输出写入 domain_state.messages")

        # 确定下一个状态
        next_status = _determine_next_status(new_state, workflow_rules, current_worker)

        # 找到 status 字段名
        status_field = _get_status_field(workflow_rules)

        if next_status:
            if status_field and status_field in new_state:
                old_status = new_state[status_field]
                new_state[status_field] = next_status
                print(f"\n🔄 Kernel 自动更新状态: {old_status} → {next_status}")

        # 更新状态历史（用于循环检测）
        status_history = state.get("status_history", [])
        current_status = new_state.get(status_field, "") if status_field else ""
        if current_status:
            status_history = (status_history + [current_status])[-5:]

        state_tree = state.get("state_tree")
        current_node_id = state.get("current_node_id")
        worker_instructions = state.get("worker_instructions", {})

        if state_tree and current_node_id:
            new_node_id = state_tree.add_node(
                parent_id=current_node_id,
                worker_name=current_worker,
                domain_state=new_state,
                patch=[],
                patch_valid=True,
                patch_error="",
                worker_instruction=worker_instructions.get(current_worker),
                metadata={
                    "step_number": state.get("step_count", 0) + 1,
                    "retry_count": state.get("retry_count", 0),
                    "message_output": message_content,
                }
            )
        else:
            new_node_id = current_node_id

        if circuit_breaker:
            circuit_breaker.record_state({
                "step_count": state.get("step_count", 0) + 1,
                "domain_state": new_state,
                "current_worker": current_worker,
                "pending_patch": patch,
                "patch_error": "",
            })

        return {
            "domain_state": new_state,
            "pending_patch": [],
            "patch_error": "",
            "step_count": state.get("step_count", 0) + 1,
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": status_history,
            "current_node_id": new_node_id,
        }

    # 过滤掉 status 相关的操作
    workflow_rules = state.get("workflow_rules", {})
    filtered_patch, removed_ops = _filter_status_operations(patch, workflow_rules)

    if removed_ops:
        print(f"\n🚫 Kernel 已过滤 {len(removed_ops)} 个 status 更新操作（由 Kernel 控制状态转换）")
        for op in removed_ops:
            print(f"   - {op.get('op')} {op.get('path')}")

    # 如果过滤后没有操作了，增加无更新计数
    if not filtered_patch:
        print(f"\n⚠️  Worker 提交的 patch 全部被过滤，没有有效操作")
        no_update_count = state.get("no_update_count", 0) + 1
        return {
            "domain_state": state["domain_state"],
            "pending_patch": [],
            "patch_error": "",
            "step_count": state.get("step_count", 0),
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": no_update_count,
        }

    use_schema_validation = True if ablation is None else ablation.use_schema_validation
    retry_count = state.get("retry_count", 0)
    validation_error = ""
    patch_valid = True

    if use_schema_validation:
        # 使用修复器进行自动修复和验证（Layer 1-4）
        new_state, error, logs = _fixer.fix_and_validate(
            filtered_patch,
            state["domain_state"],
            state["data_schema"],
        )

        # 打印修复日志（如果有）
        if logs:
            print(f"\n🔧 Patch 自动修复 (Layer 1-4):")
            for log in logs:
                print(f"  {log}")

        # Layer 5: 如果修复失败，检查是否可以重试
        if error and retry_count < MAX_RETRIES:
            # 生成详细的错误反馈
            error_feedback = _fixer.generate_error_report()
            if not error_feedback or error_feedback == "No errors":
                error_feedback = f"Error: {error}"

            print(f"\n🔄 Layer 5: 自动修复失败，请求 worker 重试 (尝试 {retry_count + 1}/{MAX_RETRIES})")
            print(f"📋 错误反馈:\n{error_feedback}")

            # 返回状态，触发重试
            return {
                "domain_state": state["domain_state"],  # 保持原状态
                "pending_patch": [],  # 清空 patch
                "patch_error": "",  # 清空错误（允许继续）
                "step_count": state.get("step_count", 0),  # 不增加步数
                "retry_count": retry_count + 1,  # 增加重试计数
                "error_feedback": error_feedback,  # 设置错误反馈
            }

        # 成功或重试次数用尽
        if error:
            print(f"\n❌ Layer 5: 已达到最大重试次数 ({MAX_RETRIES})，放弃修复")
            return {
                "domain_state": new_state,
                "pending_patch": [],
                "patch_error": error or "",
                "step_count": state.get("step_count", 0) + 1,
                "retry_count": 0,
                "error_feedback": "",
            }

        no_update_count = 0
    else:
        try:
            new_state = jsonpatch.apply_patch(copy.deepcopy(state["domain_state"]), filtered_patch)
            no_update_count = 0 if new_state != state["domain_state"] else state.get("no_update_count", 0) + 1
            print("\n🧪 Schema guard 已关闭：跳过 PatchFixer/PatchValidator，直接应用 JSON Patch")
        except Exception as exc:
            validation_error = str(exc)
            patch_valid = False
            new_state = copy.deepcopy(state["domain_state"])
            no_update_count = state.get("no_update_count", 0) + 1
            print(f"\n⚠️  Schema guard 已关闭：直接应用 patch 失败，记录错误后继续执行: {validation_error}")

    # 成功应用 patch，自动更新状态
    current_worker = state.get("current_worker", "")
    workflow_rules = state.get("workflow_rules", {})

    # 确定下一个状态
    next_status = _determine_next_status(new_state, workflow_rules, current_worker)

    # 找到 status 字段名
    status_field = _get_status_field(workflow_rules)

    if next_status:
        # 自动更新 status 字段
        new_state = copy.deepcopy(new_state)

        if status_field and status_field in new_state:
            old_status = new_state[status_field]
            new_state[status_field] = next_status
            print(f"\n🔄 Kernel 自动更新状态: {old_status} → {next_status}")

    # 更新状态历史（用于循环检测）
    status_history = state.get("status_history", [])
    current_status = new_state.get(status_field, "") if status_field else ""
    if current_status:
        # 保留最近 5 个状态
        status_history = (status_history + [current_status])[-5:]

    # ===== 8. 更新状态树 =====
    state_tree = state.get("state_tree")
    current_node_id = state.get("current_node_id")
    worker_instructions = state.get("worker_instructions", {})

    if state_tree and current_node_id:
        # 添加新节点到状态树
        new_node_id = state_tree.add_node(
            parent_id=current_node_id,
            worker_name=current_worker,
            domain_state=new_state,
            patch=filtered_patch,
            patch_valid=patch_valid,
            patch_error=validation_error,
            worker_instruction=worker_instructions.get(current_worker),
            metadata={
                "step_number": state.get("step_count", 0) + 1,
                "retry_count": state.get("retry_count", 0),
            }
        )
    else:
        new_node_id = current_node_id

    # ===== 记录到熔断器 =====
    if circuit_breaker:
        circuit_breaker.record_state({
            "step_count": state.get("step_count", 0) + 1,
            "domain_state": new_state,
            "current_worker": current_worker,
            "pending_patch": filtered_patch,
            "patch_error": validation_error,
        })

    result = {
        "domain_state": new_state,
        "pending_patch": [],
        "patch_error": "",
        "step_count": state.get("step_count", 0) + 1,
        "retry_count": 0,  # 重置重试计数
        "error_feedback": "",  # 清空错误反馈
        "no_update_count": no_update_count,
        "status_history": status_history,
        "current_node_id": new_node_id,  # 更新当前节点 ID
    }

    # 提升多轮对话控制字段到顶层状态
    # 如果 domain_state 中有 waiting_for_user 或 pending_user_question，
    # 将它们提升到顶层状态，以便路由逻辑可以检测到
    if "waiting_for_user" in new_state:
        result["waiting_for_user"] = new_state["waiting_for_user"]
        print(f"\n💬 检测到用户交互请求: waiting_for_user={new_state['waiting_for_user']}")

    if "pending_user_question" in new_state:
        result["pending_user_question"] = new_state["pending_user_question"]
        if new_state.get("waiting_for_user"):
            question = new_state["pending_user_question"]
            if len(question) > 100:
                question = question[:100] + "..."
            print(f"   问题: {question}")

    return result
