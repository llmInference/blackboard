"""LLM-backed ALFWorld architect for C5 experiments."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_kernel.types import KernelState
from langgraph_kernel.workflow import (
    derive_selected_workers,
    extract_worker_name,
    sync_workflow_state_enums,
)

from .alfworld_architect import ALFWorldArchitect


class LLMALFWorldArchitect:
    """Use an LLM to specialize the ALFWorld workflow metadata for each turn."""

    system_prompt = """You are an ALFWorld architect.

You receive the current ALFWorld turn state plus a deterministic workflow template.
Your job is to specialize the workflow metadata for this specific task while preserving
all hard action-validity and JSON-Patch constraints from the template.

Return ONLY a JSON object with:
- architect_decision: short string summarizing why this workflow is appropriate now
- task_flow: array of workflow nodes; each node must contain subtask and worker, and may include id / state / depends_on / branch_reason
- workflow_rules: object describing the runtime routing state machine
- worker_instructions: object keyed by worker name

Hard rules:
- Use ONLY the workers listed in allowed_workers
- Do not invent new worker names
- You MAY keep, skip, repeat, or branch between allowed workers when justified by the current turn
- Preserve all hard action-validity / JSON-Patch constraints from the base instructions
- Keep control flow inside workflow_rules; workers should not edit workflow_status directly
- If the next action is obvious, you may route directly to an action worker
- If search or transform uncertainty is high, you may insert planner_worker re-entry or a conditional branch before the action worker
- If the task text conflicts with canonical metadata, prefer the canonical metadata
"""

    def __init__(self, llm: Any, workflow_mode: str = "single_action") -> None:
        self._llm = llm
        self._workflow_mode = workflow_mode
        self._fallback = ALFWorldArchitect(workflow_mode=workflow_mode)
        self._last_token_usage = {
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
        }

    @staticmethod
    def _normalize_response_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)

    def _record_token_usage(self, response: Any) -> None:
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        self._last_token_usage = {
            "architect_input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "architect_output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        candidate = content.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _sanitize_task_flow(
        proposed: Any,
        fallback: list[dict[str, Any]],
        allowed_workers: list[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(proposed, list):
            return fallback

        sanitized: list[dict[str, Any]] = []
        for index, item in enumerate(proposed, start=1):
            if not isinstance(item, dict):
                return fallback
            worker_name = str(item.get("worker", "")).strip()
            if worker_name not in allowed_workers:
                return fallback

            subtask = str(item.get("subtask", "")).strip() or f"{worker_name} step {index}"
            node: dict[str, Any] = {"subtask": subtask, "worker": worker_name}

            for key in ("id", "state", "branch_reason"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    node[key] = value.strip()

            depends_on = item.get("depends_on")
            if isinstance(depends_on, list):
                deps = [str(dep).strip() for dep in depends_on if str(dep).strip()]
                if deps:
                    node["depends_on"] = deps

            sanitized.append(node)

        return sanitized or fallback

    @staticmethod
    def _sanitize_worker_instructions(
        proposed: Any,
        fallback: dict[str, str],
        allowed_workers: list[str],
    ) -> dict[str, str]:
        instructions = dict(fallback)
        if not isinstance(proposed, dict):
            return instructions

        for worker_name in allowed_workers:
            value = proposed.get(worker_name)
            if isinstance(value, str) and value.strip():
                instructions[worker_name] = value.strip()
        return instructions

    @classmethod
    def _sanitize_transition_spec(cls, proposed: Any, allowed_states: set[str]) -> Any:
        if proposed is None:
            return None

        if isinstance(proposed, str):
            target = proposed.strip()
            return target if target in allowed_states else None

        if not isinstance(proposed, dict):
            return None

        if "cases" in proposed and isinstance(proposed["cases"], list):
            cases: list[dict[str, Any]] = []
            for case in proposed["cases"]:
                if not isinstance(case, dict):
                    continue
                when = case.get("when", case.get("condition"))
                if not isinstance(when, dict):
                    continue
                target = cls._sanitize_transition_spec(case.get("target", case.get("next")), allowed_states)
                if target:
                    cases.append({"when": when, "target": target})

            result: dict[str, Any] = {}
            if cases:
                result["cases"] = cases
            default = cls._sanitize_transition_spec(proposed.get("default"), allowed_states)
            if default:
                result["default"] = default
            return result or None

        if "when" in proposed or "condition" in proposed:
            when = proposed.get("when", proposed.get("condition"))
            if not isinstance(when, dict):
                return None
            then_target = cls._sanitize_transition_spec(
                proposed.get("then", proposed.get("target", proposed.get("next"))),
                allowed_states,
            )
            else_target = cls._sanitize_transition_spec(
                proposed.get("else", proposed.get("default")),
                allowed_states,
            )
            result = {"when": when}
            if then_target:
                result["then"] = then_target
            if else_target:
                result["else"] = else_target
            return result if len(result) > 1 else None

        for key in ("target", "next", "default"):
            if key in proposed:
                return cls._sanitize_transition_spec(proposed.get(key), allowed_states)

        return None

    @classmethod
    def _sanitize_workflow_rules(
        cls,
        proposed: Any,
        fallback: dict[str, Any],
        allowed_workers: list[str],
    ) -> dict[str, Any]:
        if not isinstance(proposed, dict) or not proposed:
            return fallback

        sanitized: dict[str, dict[str, Any]] = {}
        state_sets: dict[str, set[str]] = {}

        for field_name, rules in proposed.items():
            if not isinstance(field_name, str) or not field_name.strip() or not isinstance(rules, dict) or not rules:
                return fallback

            normalized_field = field_name.strip()
            normalized_rules: dict[str, Any] = {}
            for state_name, entry in rules.items():
                normalized_state = str(state_name).strip()
                if not normalized_state:
                    return fallback

                if entry is None:
                    normalized_rules[normalized_state] = None
                    continue

                if isinstance(entry, str):
                    worker_name = entry.strip()
                    if worker_name in {"", "END"}:
                        normalized_rules[normalized_state] = None
                        continue
                    if worker_name not in allowed_workers:
                        return fallback
                    normalized_rules[normalized_state] = worker_name
                    continue

                if not isinstance(entry, dict):
                    return fallback

                worker_name = extract_worker_name(entry)
                if worker_name not in allowed_workers:
                    return fallback

                normalized_entry: dict[str, Any] = {"worker": worker_name}
                for key in ("next", "next_state", "transition"):
                    if key in entry:
                        normalized_entry["next"] = entry[key]
                        break
                normalized_rules[normalized_state] = normalized_entry

            sanitized[normalized_field] = normalized_rules
            state_sets[normalized_field] = set(normalized_rules.keys())

        for field_name, rules in sanitized.items():
            allowed_states = state_sets[field_name]
            for state_name, entry in list(rules.items()):
                if not isinstance(entry, dict) or "next" not in entry:
                    continue
                next_spec = cls._sanitize_transition_spec(entry.get("next"), allowed_states)
                if next_spec:
                    entry["next"] = next_spec
                else:
                    entry.pop("next", None)

        if not any(
            extract_worker_name(entry)
            for rules in sanitized.values()
            for entry in rules.values()
        ):
            return fallback

        if not any(
            extract_worker_name(entry) is None
            for rules in sanitized.values()
            for entry in rules.values()
        ):
            first_field = next(iter(sanitized))
            terminal_state = "completed"
            fallback_rules = fallback.get(first_field, {}) if isinstance(fallback, dict) else {}
            if isinstance(fallback_rules, dict):
                for state_name, entry in fallback_rules.items():
                    if extract_worker_name(entry) is None:
                        terminal_state = state_name
                        break
            sanitized[first_field][terminal_state] = None

        return sanitized

    @staticmethod
    def _build_task_flow_from_rules(
        workflow_rules: dict[str, Any],
        fallback: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fallback_subtasks: dict[str, str] = {}
        for item in fallback:
            if not isinstance(item, dict):
                continue
            worker_name = str(item.get("worker", "")).strip()
            subtask = str(item.get("subtask", "")).strip()
            if worker_name and subtask and worker_name not in fallback_subtasks:
                fallback_subtasks[worker_name] = subtask

        task_flow: list[dict[str, Any]] = []
        for _, rules in (workflow_rules or {}).items():
            if not isinstance(rules, dict):
                continue
            for state_name, entry in rules.items():
                worker_name = extract_worker_name(entry)
                if not worker_name:
                    continue
                task_flow.append(
                    {
                        "subtask": fallback_subtasks.get(worker_name, f"{worker_name} handles {state_name}."),
                        "worker": worker_name,
                        "state": state_name,
                    }
                )

        return task_flow or fallback

    @staticmethod
    def _build_context(
        *,
        workflow_mode: str,
        fallback_result: dict[str, Any],
    ) -> dict[str, Any]:
        domain_state = fallback_result.get("domain_state", {})
        return {
            "workflow_mode": workflow_mode,
            "task_goal": domain_state.get("task_goal", ""),
            "current_observation": domain_state.get("current_observation", ""),
            "available_actions": list(domain_state.get("available_actions", [])),
            "action_history": list(domain_state.get("action_history", []))[-5:],
            "observation_history": list(domain_state.get("observation_history", []))[-5:],
            "canonical_task_type": domain_state.get("canonical_task_type", ""),
            "canonical_goal_object": domain_state.get("canonical_goal_object", ""),
            "canonical_target_receptacle": domain_state.get("canonical_target_receptacle", ""),
            "planner_summary": domain_state.get("planner_summary", ""),
            "subgoal": domain_state.get("subgoal", ""),
            "focus_object": domain_state.get("focus_object", ""),
            "focus_receptacle": domain_state.get("focus_receptacle", ""),
            "required_transform": domain_state.get("required_transform", ""),
            "goal_object_guidance": domain_state.get("goal_object_guidance", ""),
            "recommended_actions": list(domain_state.get("recommended_actions", [])),
            "search_guidance": domain_state.get("search_guidance", ""),
            "searched_locations": list(domain_state.get("searched_locations", [])),
            "failed_search_locations": list(domain_state.get("failed_search_locations", [])),
            "base_task_flow": fallback_result.get("task_flow", []),
            "base_workflow_rules": fallback_result.get("workflow_rules", {}),
            "base_worker_instructions": fallback_result.get("worker_instructions", {}),
            "allowed_workers": fallback_result.get("selected_workers", []),
        }

    def __call__(self, state: KernelState) -> dict[str, Any]:
        fallback_result = self._fallback(state)
        self._last_token_usage = {
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
        }
        context = self._build_context(
            workflow_mode=self._workflow_mode,
            fallback_result=fallback_result,
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"Architect context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"),
                ]
            )
            self._record_token_usage(response)
            payload = self._extract_json_object(self._normalize_response_content(response.content))
        except Exception:
            payload = {}

        allowed_workers = list(fallback_result.get("selected_workers", []))
        workflow_rules = self._sanitize_workflow_rules(
            payload.get("workflow_rules"),
            fallback_result["workflow_rules"],
            allowed_workers,
        )
        generated_task_flow = self._build_task_flow_from_rules(workflow_rules, fallback_result["task_flow"])
        task_flow = self._sanitize_task_flow(
            payload.get("task_flow"),
            generated_task_flow,
            allowed_workers,
        )
        worker_instructions = self._sanitize_worker_instructions(
            payload.get("worker_instructions"),
            fallback_result["worker_instructions"],
            allowed_workers,
        )
        architect_decision = str(payload.get("architect_decision", "")).strip()
        selected_workers = derive_selected_workers(task_flow, workflow_rules) or allowed_workers
        data_schema = sync_workflow_state_enums(fallback_result["data_schema"], workflow_rules)

        domain_state = {
            **fallback_result["domain_state"],
            "architect_decision": architect_decision,
        }
        for field_name, transitions in workflow_rules.items():
            if transitions:
                domain_state[field_name] = next(iter(transitions.keys()))
                break

        merged = dict(fallback_result)
        merged["task_flow"] = task_flow
        merged["workflow_rules"] = workflow_rules
        merged["worker_instructions"] = worker_instructions
        merged["selected_workers"] = selected_workers
        merged["data_schema"] = data_schema
        merged["domain_state"] = domain_state
        merged["turn_architect_input_tokens"] = self._last_token_usage["architect_input_tokens"]
        merged["turn_architect_output_tokens"] = self._last_token_usage["architect_output_tokens"]
        return merged
