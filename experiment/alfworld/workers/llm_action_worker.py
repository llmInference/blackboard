"""LLM-backed action worker for ALFWorld multi-worker experiments."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_kernel.worker.base import LLMWorkerAgent


class LLMActionWorker(LLMWorkerAgent):
    """Select the next ALFWorld action from planner-provided structured hints."""

    system_prompt = """You are an ALFWorld action worker.

Your input contains:
- the current text observation
- admissible ALFWorld actions
- planner outputs such as subgoal, focus_object, focus_receptacle, required_transform
- canonical object guidance and search memory fields such as goal_object_guidance, searched_locations, failed_search_locations, recommended_actions, and search_guidance

Your job:
1. Choose exactly one admissible action that best advances the current subgoal
2. Explain the decision briefly in decision_reason
3. Return ONLY JSON Patch operations

Required output fields:
- /selected_action
- /decision_reason

Hard rules:
- The action MUST be one of available_actions
- Prefer planner subgoal completion over broad exploration
- If task_goal wording conflicts with planner focus_object or goal_object_guidance, follow the planner fields as the canonical ground truth
- If recommended_actions is non-empty, strongly prefer its first admissible option unless the current observation exposes the goal object directly
- Do not substitute semantically similar objects when focus_object identifies a specific goal object
- Treat failed_search_locations as negative evidence; do not keep opening the same kind of empty location when a different unexplored type is available
- After multiple empty searches of the same location type, pivot to a different type instead of extending the same pattern
- Do not update workflow_status
- Keep the patch minimal

Example:
[
  {"op": "replace", "path": "/selected_action", "value": "go to microwave 1"},
  {"op": "replace", "path": "/decision_reason", "value": "Planner subgoal is heat_goal_object, so move toward the microwave."}
]
"""

    _NL_ACTION_SUFFIX = """

CRITICAL: End your response with exactly this line (copy the action verbatim from the list above):
ACTION: <one exact action from available_actions>

Example final line: ACTION: go to microwave 1"""

    def _think(self, context: dict[str, Any], *, use_json_patch: bool = True) -> Any:
        # JSON Patch 模式：走父类标准逻辑，不做任何改动
        if use_json_patch:
            return super()._think(context, use_json_patch=True)

        # 自然语言模式（C1 ablated）：强制 LLM 在末尾输出 ACTION: <action>
        available_actions = list(context.get("available_actions", []) or [])
        action_list = "\n".join(f"  - {a}" for a in available_actions)
        nl_prompt = (
            self.system_prompt
            + f"\n\nAvailable actions for this step:\n{action_list}"
            + self._NL_ACTION_SUFFIX
        )
        response = self._llm.invoke([
            SystemMessage(content=nl_prompt),
            HumanMessage(content=f"Current state:\n{json.dumps(context, indent=2, ensure_ascii=False)}"),
        ])
        self._record_token_usage(response)
        content = self._normalize_response_content(response.content)

        # 从末尾的 "ACTION: ..." 行提取动作
        for line in reversed(content.splitlines()):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                candidate = line[len("ACTION:"):].strip()
                if candidate in available_actions:
                    return [
                        {"op": "replace", "path": "/selected_action", "value": candidate},
                        {"op": "add", "path": "/decision_reason", "value": content.splitlines()[0][:240]},
                    ]

        # 兜底1：heuristic token 匹配
        patch = self._heuristic_action_patch_from_text(content=content, context=context)
        if patch:
            return patch

        # 兜底2：选第一个可用动作，避免 fallback "look"
        if available_actions:
            return [
                {"op": "replace", "path": "/selected_action", "value": available_actions[0]},
            ]
        return content
