from __future__ import annotations

from langgraph_kernel.kernel.patch_fixer import PatchFixer, PatchFixerConfig
from langgraph_kernel.types import KernelState

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
    """
    根据 workflow_rules 和当前 worker 确定下一个状态

    逻辑：
    1. 找到当前 worker 在 workflow_rules 中对应的状态
    2. 查找该状态的下一个状态（按顺序）
    3. 如果没有下一个状态，返回 None（保持当前状态）
    """
    if not workflow_rules or not current_worker:
        return None

    # 找到当前 worker 对应的状态
    current_status = None
    for field_name, rules in workflow_rules.items():
        for status_value, worker_name in rules.items():
            if worker_name == current_worker:
                current_status = status_value
                break
        if current_status:
            break

    if not current_status:
        return None

    # 获取所有状态的顺序
    status_field = None
    for field_name, rules in workflow_rules.items():
        if current_status in rules:
            status_field = field_name
            break

    if not status_field:
        return None

    # 获取状态列表（按照 workflow_rules 的顺序）
    status_list = list(workflow_rules[status_field].keys())

    # 找到当前状态的索引
    try:
        current_index = status_list.index(current_status)
    except ValueError:
        return None

    # 返回下一个状态（如果存在）
    if current_index + 1 < len(status_list):
        return status_list[current_index + 1]

    # 如果是最后一个状态，检查是否有 "done" 状态
    # 如果没有，返回 None（保持当前状态）
    return None


def _filter_status_operations(patch: list, workflow_rules: dict) -> tuple[list, list]:
    """
    过滤掉 patch 中所有针对 status 字段的操作

    返回：(过滤后的 patch, 被移除的操作列表)
    """
    if not workflow_rules:
        return patch, []

    # 找到 status 字段名
    status_field = None
    for field_name, rules in workflow_rules.items():
        if isinstance(rules, dict) and any(isinstance(v, str) for v in rules.values()):
            status_field = field_name
            break

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


def kernel_node(state: KernelState) -> dict:
    """
    Kernel 核心节点：
    1. 若无 pending_patch，直接跳过（首次进入）
    2. **过滤掉 worker 提交的所有 status 更新操作**（新增）
    3. 使用 PatchFixer 自动修复常见错误（Layer 1-4）
    4. 验证 patch 是否符合 data_schema
    5. 如果失败且未超过重试次数，生成错误反馈并请求重试（Layer 5）
    6. 应用 patch 更新 domain_state
    7. **自动更新 status 到下一个状态**（由 Kernel 控制）
    8. 清空 pending_patch，写入 patch_error，递增 step_count
    """
    patch = state.get("pending_patch") or []
    if not patch:
        return {
            "patch_error": "",
            "step_count": state.get("step_count", 0),
            "retry_count": 0,
            "error_feedback": "",
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
    retry_count = state.get("retry_count", 0)

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

    # 成功应用 patch，自动更新状态
    current_worker = state.get("current_worker", "")
    workflow_rules = state.get("workflow_rules", {})

    # 确定下一个状态
    next_status = _determine_next_status(new_state, workflow_rules, current_worker)

    # 找到 status 字段名
    status_field = None
    for field_name, rules in workflow_rules.items():
        if isinstance(rules, dict) and any(isinstance(v, str) for v in rules.values()):
            status_field = field_name
            break

    if next_status:
        # 自动更新 status 字段
        import copy
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

    return {
        "domain_state": new_state,
        "pending_patch": [],
        "patch_error": "",
        "step_count": state.get("step_count", 0) + 1,
        "retry_count": 0,  # 重置重试计数
        "error_feedback": "",  # 清空错误反馈
        "no_update_count": 0,  # 重置无更新计数（因为有有效更新）
        "status_history": status_history,
    }
