"""Deterministic mock chat models for ALFWorld LLM workflow smoke tests."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _parse_context(messages: list[Any]) -> dict[str, Any]:
    prefixes = ("Current state slice:\n", "Architect context:\n")
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        for prefix in prefixes:
            if prefix not in content:
                continue
            payload = content.split(prefix, 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                continue
    return {}


class ALFWorldMockActionLLM(BaseChatModel):
    """Small deterministic chat model for planner_llm_action infra validation."""

    @staticmethod
    def _base_messages(messages: list[Any]) -> tuple[str, str]:
        system_content = ""
        human_content = ""
        if messages:
            system_content = str(getattr(messages[0], "content", "") or "")
            human_content = str(getattr(messages[-1], "content", "") or "")
        return system_content, human_content

    @staticmethod
    def _score_action(action: str, context: dict[str, Any], recent_actions: list[str]) -> float:
        action_lower = action.lower()
        subgoal = str(context.get("subgoal", "")).lower()
        focus_object = _normalize_token(str(context.get("focus_object", "")))
        focus_receptacle = _normalize_token(str(context.get("focus_receptacle", "")))
        required_transform = str(context.get("required_transform", "")).lower()

        score = 0.0
        if action_lower == "look":
            score -= 5.0

        if action in recent_actions:
            score -= 3.0

        action_token = _normalize_token(action_lower)
        if focus_object and focus_object in action_token:
            score += 5.0
        if focus_receptacle and focus_receptacle in action_token:
            score += 4.0

        if subgoal == "activate_light" and action_lower.startswith(("use ", "toggle ")):
            score += 8.0
        if subgoal == "inspect_goal_object" and action_lower.startswith("examine "):
            score += 8.0
        if subgoal == "acquire_goal_object" and action_lower.startswith("take "):
            score += 6.0
        if subgoal == "place_goal_object" and action_lower.startswith(("move ", "put ")):
            score += 8.0

        if required_transform and action_lower.startswith(f"{required_transform} "):
            score += 10.0
        if required_transform and action_lower.startswith("go to ") and focus_receptacle and focus_receptacle in action_token:
            score += 6.0
        if required_transform and action_lower.startswith("open ") and focus_receptacle and focus_receptacle in action_token:
            score += 4.0

        if action_lower.startswith("go to "):
            score += 1.0
        if action_lower.startswith("open "):
            score += 0.5

        return score

    def _build_patch(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        available_actions = list(context.get("available_actions") or [])
        recent_actions = list(context.get("action_history") or [])[-4:]

        selected_action = ""
        if available_actions:
            selected_action = max(
                available_actions,
                key=lambda action: self._score_action(action, context, recent_actions),
            )

        subgoal = context.get("subgoal", "") or "unknown"
        decision_reason = (
            f"Mock LLM followed subgoal '{subgoal}' and selected admissible action '{selected_action or 'none'}'."
        )
        return [
            {"op": "replace", "path": "/selected_action", "value": selected_action},
            {"op": "replace", "path": "/decision_reason", "value": decision_reason},
        ]

    @staticmethod
    def _build_architect_payload(context: dict[str, Any]) -> dict[str, Any]:
        allowed_workers = list(context.get("allowed_workers") or context.get("selected_workers") or [])
        canonical_goal = str(context.get("canonical_goal_object", "") or "").strip()
        canonical_target = str(context.get("canonical_target_receptacle", "") or "").strip()
        task_goal = str(context.get("task_goal", "") or "").strip()
        base_worker_instructions = dict(context.get("base_worker_instructions") or {})
        base_workflow_rules = dict(context.get("base_workflow_rules") or {})

        goal_clause = canonical_goal or task_goal or "the task object"
        target_clause = canonical_target or "the target receptacle"
        status_field = next(iter(base_workflow_rules.keys()), "workflow_status")

        action_worker_name = ""
        for candidate in ("llm_action_worker", "action_worker"):
            if candidate in allowed_workers:
                action_worker_name = candidate
                break

        if "planner_worker" in allowed_workers and action_worker_name:
            workflow_rules = {
                status_field: {
                    "planning": {
                        "worker": "planner_worker",
                        "next": {
                            "cases": [
                                {
                                    "when": {"field": "failed_search_locations", "truthy": True},
                                    "target": "replanning",
                                }
                            ],
                            "default": "acting",
                        },
                    },
                    "replanning": {"worker": "planner_worker", "next": "acting"},
                    "acting": {"worker": action_worker_name, "next": "completed"},
                    "completed": None,
                }
            }
            task_flow = [
                {
                    "id": "planning",
                    "state": "planning",
                    "subtask": f"Track the canonical object {goal_clause} and derive the next structured subgoal.",
                    "worker": "planner_worker",
                },
                {
                    "id": "replanning",
                    "state": "replanning",
                    "depends_on": ["planning"],
                    "branch_reason": "Retry planning only when previous search signals suggest the first plan is stale.",
                    "subtask": f"Re-plan around failed search paths for {goal_clause} before acting again.",
                    "worker": "planner_worker",
                },
                {
                    "id": "acting",
                    "state": "acting",
                    "depends_on": ["planning"],
                    "subtask": f"Choose the next admissible action that advances {goal_clause} toward {target_clause}.",
                    "worker": action_worker_name,
                },
            ]
        elif action_worker_name:
            workflow_rules = {
                status_field: {
                    "acting": {"worker": action_worker_name, "next": "completed"},
                    "completed": None,
                }
            }
            task_flow = [
                {
                    "id": "acting",
                    "state": "acting",
                    "subtask": f"Choose the next admissible action that advances {goal_clause} toward {target_clause}.",
                    "worker": action_worker_name,
                }
            ]
        else:
            workflow_rules = base_workflow_rules
            task_flow = list(context.get("base_task_flow") or [])

        worker_instructions = dict(base_worker_instructions)
        if action_worker_name and action_worker_name in worker_instructions:
            worker_instructions[action_worker_name] = (
                f"{worker_instructions[action_worker_name]}\n"
                f"- Canonical object for this turn: {goal_clause}\n"
                f"- Target receptacle or appliance: {target_clause}\n"
                "- If search stalls, pivot to an unexplored location type."
            )
        if "planner_worker" in allowed_workers and "planner_worker" in worker_instructions:
            worker_instructions["planner_worker"] = (
                f"{worker_instructions['planner_worker']} "
                f"Prioritize canonical object {goal_clause} and target {target_clause}."
            )

        return {
            "architect_decision": (
                f"Use a dynamic {context.get('workflow_mode', 'configured')} workflow with canonical focus on "
                f"{goal_clause} and target {target_clause}."
            ),
            "task_flow": task_flow,
            "workflow_rules": workflow_rules,
            "worker_instructions": worker_instructions,
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message_list = list(messages)
        system_content, _ = self._base_messages(message_list)
        context = _parse_context(message_list)
        if "You are an ALFWorld architect." in system_content:
            content = json.dumps(self._build_architect_payload(context), ensure_ascii=False)
        else:
            patch = self._build_patch(context)
            content = json.dumps(patch, ensure_ascii=False)
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "alfworld_mock_action"
