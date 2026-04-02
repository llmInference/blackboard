"""ALFWorld-specific architect node for Kernel System."""
from __future__ import annotations

from typing import Any

from langgraph_kernel.types import KernelState


class ALFWorldArchitect:
    """Build a one-step action-selection workflow for each ALFWorld turn."""

    def __init__(self, llm: Any = None, workflow_mode: str = "single_action") -> None:
        self._llm = llm
        self._workflow_mode = workflow_mode

    def _base_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_prompt": {"type": "string"},
                "task_goal": {"type": "string"},
                "current_observation": {"type": "string"},
                "available_actions": {"type": "array", "items": {"type": "string"}},
                "selected_action": {"type": "string"},
                "decision_reason": {"type": "string"},
                "action_history": {"type": "array", "items": {"type": "string"}},
                "observation_history": {"type": "array", "items": {"type": "string"}},
                "current_gamefile": {"type": "string"},
                "architect_decision": {"type": "string"},
                "canonical_task_type": {"type": "string"},
                "canonical_goal_object": {"type": "string"},
                "canonical_target_receptacle": {"type": "string"},
                "workflow_status": {"type": "string"},
                "workflow_entry_status": {"type": "string"},
                "planner_summary": {"type": "string"},
                "subgoal": {"type": "string"},
                "focus_object": {"type": "string"},
                "focus_receptacle": {"type": "string"},
                "required_transform": {"type": "string"},
                "goal_object_guidance": {"type": "string"},
                "searched_locations": {"type": "array", "items": {"type": "string"}},
                "failed_search_locations": {"type": "array", "items": {"type": "string"}},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
                "search_guidance": {"type": "string"},
                "transform_completed": {"type": "boolean"},
                "light_activated": {"type": "boolean"},
                "delivered_goal_count": {"type": "integer"},
                "goal_delivery_target_count": {"type": "integer"},
            },
        }

    @staticmethod
    def _build_action_instruction(worker_name: str) -> str:
        if worker_name == "llm_action_worker":
            return (
                "You are an ALFWorld action worker.\n"
                "Your input includes the current observation, admissible ALFWorld actions, and planner fields such as "
                "subgoal, focus_object, focus_receptacle, required_transform, goal_object_guidance, searched_locations, "
                "failed_search_locations, recommended_actions, search_guidance, transform_completed, and "
                "light_activated.\n"
                "Choose exactly one admissible action that best advances the current subgoal.\n"
                "Return JSON Patch operations that update ONLY /selected_action and /decision_reason.\n"
                "Hard rules:\n"
                "- /selected_action MUST be exactly one entry from available_actions\n"
                "- Use planner outputs before broad exploration\n"
                "- Treat focus_object and goal_object_guidance as the canonical object identity when task wording is ambiguous\n"
                "- Treat recommended_actions and search_guidance as the default search policy when the goal object is not directly visible\n"
                "- After repeated empty searches of one location type, pivot to a different admissible location type when possible\n"
                "- Avoid repeated no-progress actions when a better admissible action exists\n"
                "- Do not update workflow_status or any other control field\n"
                "- Keep the patch minimal"
            )

        return (
            "Choose the best admissible ALFWorld action using the planner outputs. "
            "Advance the current subgoal and avoid repetition."
        )

    def build_workflow_spec(self) -> dict[str, Any]:
        data_schema = self._base_schema()
        if self._workflow_mode == "single_action":
            data_schema["properties"]["workflow_status"]["enum"] = ["deciding", "completed"]
            data_schema["required"] = ["workflow_status", "task_goal", "current_observation", "available_actions"]
            workflow_rules = {
                "workflow_status": {
                    "deciding": "action_worker",
                    "completed": None,
                }
            }
            selected_workers = ["action_worker"]
            task_flow = [
                {
                    "subtask": "Select the next admissible ALFWorld action.",
                    "worker": "action_worker",
                }
            ]
            worker_instructions = {
                "action_worker": (
                    "Choose the best admissible ALFWorld action for the current turn. "
                    "Prefer actions that make progress toward the goal and avoid pointless repetition."
                )
            }
            workflow_entry_status = "deciding"
        elif self._workflow_mode in {"planner_action", "planner_llm_action"}:
            action_worker_name = (
                "action_worker"
                if self._workflow_mode == "planner_action"
                else "llm_action_worker"
            )
            data_schema["properties"]["workflow_status"]["enum"] = ["planning", "acting", "completed"]
            data_schema["required"] = ["workflow_status", "task_goal", "current_observation", "available_actions"]
            workflow_rules = {
                "workflow_status": {
                    "planning": "planner_worker",
                    "acting": action_worker_name,
                    "completed": None,
                }
            }
            selected_workers = ["planner_worker", action_worker_name]
            task_flow = [
                {
                    "subtask": "Summarize the current ALFWorld turn into a compact structured subgoal.",
                    "worker": "planner_worker",
                },
                {
                    "subtask": "Choose the next admissible ALFWorld action from the planner state.",
                    "worker": action_worker_name,
                },
            ]
            worker_instructions = {
                "planner_worker": (
                    "Produce a concise structured plan for the current turn. "
                    "Focus on the immediate subgoal, target object, target receptacle, and required transform."
                ),
                action_worker_name: self._build_action_instruction(action_worker_name),
            }
            workflow_entry_status = "planning"
        else:
            raise ValueError(f"Unsupported ALFWorld workflow_mode: {self._workflow_mode}")

        return {
            "data_schema": data_schema,
            "workflow_rules": workflow_rules,
            "selected_workers": selected_workers,
            "task_flow": task_flow,
            "worker_instructions": worker_instructions,
            "workflow_entry_status": workflow_entry_status,
        }

    def build_domain_state(self, state: KernelState, *, workflow_entry_status: str) -> dict[str, Any]:
        current_domain_state = dict(state.get("domain_state") or {})
        root_prompt = state.get("user_prompt", "")
        current_prompt = current_domain_state.get("user_prompt", root_prompt)

        domain_state = {
            "user_prompt": current_prompt,
            "task_goal": current_domain_state.get("task_goal", ""),
            "current_observation": current_domain_state.get("current_observation", ""),
            "available_actions": list(current_domain_state.get("available_actions", [])),
            "selected_action": current_domain_state.get("selected_action", ""),
            "decision_reason": current_domain_state.get("decision_reason", ""),
            "action_history": list(current_domain_state.get("action_history", [])),
            "observation_history": list(current_domain_state.get("observation_history", [])),
            "current_gamefile": current_domain_state.get("current_gamefile", ""),
            "architect_decision": current_domain_state.get("architect_decision", ""),
            "canonical_task_type": current_domain_state.get("canonical_task_type", ""),
            "canonical_goal_object": current_domain_state.get("canonical_goal_object", ""),
            "canonical_target_receptacle": current_domain_state.get("canonical_target_receptacle", ""),
            "planner_summary": current_domain_state.get("planner_summary", ""),
            "subgoal": current_domain_state.get("subgoal", ""),
            "focus_object": current_domain_state.get("focus_object", ""),
            "focus_receptacle": current_domain_state.get("focus_receptacle", ""),
            "required_transform": current_domain_state.get("required_transform", ""),
            "goal_object_guidance": current_domain_state.get("goal_object_guidance", ""),
            "searched_locations": list(current_domain_state.get("searched_locations", [])),
            "failed_search_locations": list(current_domain_state.get("failed_search_locations", [])),
            "recommended_actions": list(current_domain_state.get("recommended_actions", [])),
            "search_guidance": current_domain_state.get("search_guidance", ""),
            "transform_completed": current_domain_state.get("transform_completed", False),
            "light_activated": current_domain_state.get("light_activated", False),
            "delivered_goal_count": current_domain_state.get("delivered_goal_count", 0),
            "goal_delivery_target_count": current_domain_state.get("goal_delivery_target_count", 1),
            "workflow_status": workflow_entry_status,
            "workflow_entry_status": workflow_entry_status,
        }

        return domain_state

    def __call__(self, state: KernelState) -> dict[str, Any]:
        workflow_spec = self.build_workflow_spec()
        domain_state = self.build_domain_state(
            state,
            workflow_entry_status=workflow_spec["workflow_entry_status"],
        )

        return {
            "task_flow": workflow_spec["task_flow"],
            "data_schema": workflow_spec["data_schema"],
            "workflow_rules": workflow_spec["workflow_rules"],
            "worker_instructions": workflow_spec["worker_instructions"],
            "selected_workers": workflow_spec["selected_workers"],
            "domain_state": domain_state,
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
