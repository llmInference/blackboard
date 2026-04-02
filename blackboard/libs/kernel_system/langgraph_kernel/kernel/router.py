from __future__ import annotations

from langgraph.graph import END

from langgraph_kernel.types import KernelState
from langgraph_kernel.workflow import extract_worker_name, get_matching_workflow_entry


class WorkflowRouter:
    """
    根据 workflow_rules 和当前 domain_state 决定下一个节点。

    workflow_rules 兼容两种格式：
        1. 旧格式：{"field_name": {"value": "worker_name", ...}}
        2. 新格式：{"field_name": {"value": {"worker": "worker_name", "next": ...}, ...}}

    路由优先级（三层终止机制）：
        1. patch_error 非空 → END（错误终止）
        2. step_count >= max_steps → END（防止无限循环）
        3. 显式终止状态：workflow_rules 中状态映射到 None/"END"/"" → END
        4. 无效更新检测：连续 N 次无有效业务数据更新 → END
        5. 状态循环检测：状态在短期内重复出现 → END
        6. 匹配 workflow_rules 中当前状态对应的 worker
        7. 无匹配 → END
    """

    def __init__(
        self,
        worker_names: list[str],
        max_steps: int = 50,
        max_no_update: int = 2,
        loop_detection_window: int = 3,
    ) -> None:
        self.worker_names = worker_names
        self.max_steps = max_steps
        self.max_no_update = max_no_update
        self.loop_detection_window = loop_detection_window

    def route(self, state: KernelState) -> str:
        if state.get("patch_error"):
            print("\n🛑 终止原因: patch 验证错误")
            return END

        if state.get("step_count", 0) >= self.max_steps:
            print(f"\n🛑 终止原因: 达到最大步数限制 ({self.max_steps})")
            return END

        domain = state.get("domain_state") or {}
        rules = state.get("workflow_rules") or {}

        _, current_status, entry = get_matching_workflow_entry(domain, rules)
        if current_status is not None:
            worker = extract_worker_name(entry)
            if worker is None:
                print(f"\n🛑 终止原因: 到达显式终止状态 ({current_status})")
                return END

        no_update_count = state.get("no_update_count", 0)
        if no_update_count >= self.max_no_update:
            print(f"\n🛑 终止原因: 连续 {no_update_count} 次无有效业务数据更新")
            return END

        status_history = state.get("status_history", [])
        if len(status_history) >= self.loop_detection_window:
            recent = status_history[-self.loop_detection_window:]
            if len(recent) != len(set(recent)):
                print(f"\n🛑 终止原因: 检测到状态循环 {recent}")
                return END

        _, _, entry = get_matching_workflow_entry(domain, rules)
        worker = extract_worker_name(entry)
        if worker and (not self.worker_names or worker in self.worker_names):
            return worker

        print("\n🛑 终止原因: 当前状态无匹配的 worker")
        return END
