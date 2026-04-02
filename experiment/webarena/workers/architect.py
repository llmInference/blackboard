"""Task-scoped architect for WebArena blackboard runs."""
from __future__ import annotations

import json
import os
from typing import Any

from experiment.common.neutral import TaskSpec, ToolSpec
from experiment.webarena.core.target_urls import extract_expected_target_urls, normalize_navigable_url
from experiment.webarena.workers.base import WorkerSpec


def _schema_entry(
    *,
    field_type: str,
    required: bool,
    producers: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "type": field_type,
        "required": required,
        "producer": producers,
        "description": description,
    }


def _expected_target_urls(task: TaskSpec, *, use_target_urls: bool) -> list[str]:
    if not use_target_urls:
        return []
    rendered = [
        normalize_navigable_url(url)
        for url in list(task.metadata.get("rendered_target_urls", []) or [])
    ]
    rendered = [url for url in rendered if url]
    if rendered:
        return rendered
    return extract_expected_target_urls(list(task.metadata.get("eval", []) or []))


def _stage_profile(*, requires_final_response: bool, has_target_urls: bool) -> tuple[list[str], str]:
    if requires_final_response:
        phase_order = ["navigate", "evidence", "respond"] if has_target_urls else ["evidence", "respond"]
        initial_phase = "navigate" if has_target_urls else "evidence"
        return phase_order, initial_phase
    return (["navigate", "complete"], "navigate")


def _token_usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "architect_input_tokens": prompt_tokens,
        "architect_output_tokens": completion_tokens,
    }


def _normalize_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or item))
        return "".join(parts)
    return str(content)


class WebArenaArchitect:
    """LLM-first architect that generates task-scoped workflow and data schema."""

    ROUTE_CONDITIONS = {
        "always",
        "answer_ready",
        "goal_reached",
        "goal_reached_and_requires_response",
        "needs_action",
        "has_grounded_action",
        "tool_call_available",
        "finished",
    }

    def __init__(self, worker_specs: tuple[WorkerSpec, ...], llm: Any | None = None, *, use_target_urls: bool = False) -> None:
        self.worker_specs = {worker.name: worker for worker in worker_specs}
        self.llm = llm
        self.use_target_urls = bool(use_target_urls)

    @staticmethod
    def _retry_attempts() -> int:
        raw = str(os.environ.get("WEBARENA_ARCHITECT_RETRY_ATTEMPTS", "3") or "3").strip()
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 3
        return max(parsed, 1)

    def _worker_instruction(self, worker_name: str) -> str:
        spec = self.worker_specs[worker_name]
        tool_note = " It may request one browser action." if spec.can_use_tools else ""
        return (
            f"You are {worker_name}. "
            f"Only handle: {spec.description}. "
            f"Read only {', '.join(spec.reads)}. "
            f"Write only {', '.join(spec.writes)}.{tool_note}"
        )

    def _invoke_llm_json(self, *, constraints: dict[str, Any], tools_by_name: dict[str, ToolSpec], fallback: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        if self.llm is None:
            raise ValueError("architect llm is not configured")
        from langchain_core.messages import HumanMessage, SystemMessage

        worker_catalog = {name: spec.to_dict() for name, spec in self.worker_specs.items()}
        system_prompt = """You are the WebArena architect worker.

You must generate a task-scoped multi-worker execution plan.
Return ONLY one JSON object with keys:
- workflow
- data_schema
- selected_workers
- worker_instructions
- warnings

Hard constraints:
- Only use workers from worker_catalog.
- Every workflow step must include: id, worker, purpose, reads, writes, exit_conditions, next.
- next must be a list where each item is {condition, step}.
- condition must be one of:
  always, answer_ready, goal_reached, goal_reached_and_requires_response, needs_action, has_grounded_action, tool_call_available, finished
- step is either another step id or empty string for terminal.
- Architect chooses orchestration; workers do not re-plan orchestration.
- Prefer plans that first gather page state, then decide whether to act, and loop as needed.
- Ensure target_urls/start_urls/task_category influence routing.
- Respect task_constraints.phase_order and initial_phase as the stage policy.
- Keep workflow concise (3-6 steps).
- data_schema must define fields produced/consumed by the workflow and workers.
"""
        human_context = {
            "task_constraints": constraints,
            "tool_names": sorted(tools_by_name),
            "worker_catalog": worker_catalog,
            "fallback_reference": {
                "workflow": fallback.get("workflow"),
                "data_schema": fallback.get("data_schema"),
                "selected_workers": fallback.get("selected_workers"),
            },
        }
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(human_context, ensure_ascii=False, indent=2)),
            ]
        )
        raw = _normalize_response_content(getattr(response, "content", ""))
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("architect llm must return a JSON object")
        return payload, _token_usage(response)

    def _task_constraints(self, task: TaskSpec) -> dict[str, Any]:
        instruction = str(task.instruction or "").strip()
        eval_items = list(task.metadata.get("eval", []) or [])
        target_urls = _expected_target_urls(task, use_target_urls=self.use_target_urls)
        requires_final_response = bool(task.metadata.get("requires_final_response", False))
        phase_order, initial_phase = _stage_profile(
            requires_final_response=requires_final_response,
            has_target_urls=bool(target_urls),
        )
        response_schema: dict[str, Any] = {}
        expected_response: Any = None
        for item in eval_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("evaluator", "") or "") != "AgentResponseEvaluator":
                continue
            candidate_schema = item.get("results_schema")
            if isinstance(candidate_schema, dict):
                response_schema = dict(candidate_schema)
            expected = item.get("expected")
            if isinstance(expected, dict):
                expected_response = expected.get("retrieved_data")
            break
        return {
            "task_id": str(task.task_id or ""),
            "goal": instruction,
            "sites": list(task.metadata.get("sites", []) or []),
            "start_urls": list(task.metadata.get("start_urls", []) or []),
            "target_urls": target_urls,
            "enable_target_urls": bool(self.use_target_urls),
            "requires_final_response": requires_final_response,
            "task_category": str(task.metadata.get("task_category", "") or ""),
            "evaluators": list(task.metadata.get("evaluators", []) or []),
            "response_schema": response_schema,
            "expected_response_template": expected_response,
            "phase_order": phase_order,
            "initial_phase": initial_phase,
        }

    def _fallback_plan(self, constraints: dict[str, Any], tools_by_name: dict[str, ToolSpec]) -> dict[str, Any]:
        requires_final_response = bool(constraints["requires_final_response"])
        selected_workers = [
            "page_state_worker",
            "argument_grounding_worker",
            "browser_action_worker",
        ]
        if requires_final_response:
            selected_workers.append("response_worker")

        workflow_steps = [
            {
                "id": "inspect_current_page",
                "worker": "page_state_worker",
                "purpose": "Normalize latest browser observation into shared page state.",
                "reads": ["last_observation"],
                "writes": ["current_page", "page_evidence", "open_tabs", "verification", "finish", "finish_reason"],
                "exit_conditions": ["page state extracted"],
                "next": (
                    [
                        {"condition": "answer_ready", "step": "compose_final_answer"},
                        {"condition": "needs_action", "step": "bind_action_arguments"},
                    ]
                    if requires_final_response
                    else [
                        {"condition": "goal_reached", "step": ""},
                        {"condition": "needs_action", "step": "bind_action_arguments"},
                    ]
                ),
            },
            {
                "id": "bind_action_arguments",
                "worker": "argument_grounding_worker",
                "purpose": "Ground the next browser action from current state and task constraints.",
                "reads": ["task_constraints", "current_page", "page_evidence", "verification"],
                "writes": ["grounded_action", "action_arguments"],
                "exit_conditions": ["grounded action updated"],
                "next": [
                    {"condition": "has_grounded_action", "step": "execute_browser_action"},
                    {"condition": "always", "step": "inspect_current_page"},
                ],
            },
            {
                "id": "execute_browser_action",
                "worker": "browser_action_worker",
                "purpose": "Request one browser tool call for the grounded action.",
                "reads": ["grounded_action"],
                "writes": ["last_action", "action_history", "execution_error"],
                "exit_conditions": ["browser action requested or action rejected"],
                "next": [
                    {"condition": "tool_call_available", "step": "inspect_current_page"},
                    {"condition": "always", "step": "inspect_current_page"},
                ],
            },
        ]
        if requires_final_response:
            workflow_steps.append(
                {
                    "id": "compose_final_answer",
                    "worker": "response_worker",
                    "purpose": "Compose final answer from verified page evidence.",
                    "reads": ["verification", "page_evidence", "current_page"],
                    "writes": ["assistant_response"],
                    "exit_conditions": ["assistant response ready"],
                    "next": [],
                }
            )

        data_schema = {
            "task_constraints": _schema_entry(
                field_type="object",
                required=True,
                producers=["architect"],
                description="Task-scoped goal summary and routing constraints generated by architect.",
            ),
            "current_page": _schema_entry(
                field_type="object",
                required=False,
                producers=["page_state_worker"],
                description="Normalized summary of current active page.",
            ),
            "page_evidence": _schema_entry(
                field_type="object",
                required=False,
                producers=["page_state_worker"],
                description="Task-relevant evidence extracted from current page.",
            ),
            "open_tabs": _schema_entry(
                field_type="array",
                required=False,
                producers=["page_state_worker"],
                description="Current tab inventory.",
            ),
            "grounded_action": _schema_entry(
                field_type="object",
                required=False,
                producers=["argument_grounding_worker"],
                description="Next structured browser action proposal.",
            ),
            "action_arguments": _schema_entry(
                field_type="object",
                required=False,
                producers=["argument_grounding_worker"],
                description="Arguments bound for grounded browser action.",
            ),
            "last_action": _schema_entry(
                field_type="object",
                required=False,
                producers=["browser_action_worker"],
                description="Most recent requested browser action.",
            ),
            "last_observation": _schema_entry(
                field_type="object",
                required=False,
                producers=["runtime"],
                description="Most recent browser observation from runtime.",
            ),
            "action_history": _schema_entry(
                field_type="array",
                required=False,
                producers=["browser_action_worker"],
                description="Requested browser actions in chronological order.",
            ),
            "execution_error": _schema_entry(
                field_type="string",
                required=False,
                producers=["browser_action_worker"],
                description="Latest action-level execution error.",
            ),
            "verification": _schema_entry(
                field_type="object",
                required=False,
                producers=["page_state_worker"],
                description="LLM-derived completion and progress status for current task.",
            ),
            "assistant_response": _schema_entry(
                field_type="string",
                required=requires_final_response,
                producers=["response_worker"],
                description="Final textual answer when task requires one.",
            ),
            "finish": _schema_entry(
                field_type="boolean",
                required=True,
                producers=["page_state_worker"],
                description="Worker-managed episode finish flag.",
            ),
            "finish_reason": _schema_entry(
                field_type="string",
                required=False,
                producers=["page_state_worker"],
                description="Worker-managed terminal reason when workflow stops.",
            ),
        }
        return {
            "workflow": {
                "workflow_id": f"webarena_{constraints['task_category'] or 'generic'}",
                "goal": constraints["goal"],
                "steps": workflow_steps,
            },
            "data_schema": data_schema,
            "task_constraints": constraints,
            "selected_workers": selected_workers,
            "worker_instructions": {
                worker_name: self._worker_instruction(worker_name)
                for worker_name in selected_workers
            },
            "warnings": [],
            "fallback_reason": "heuristic_architect",
            "used_fallback_architect": True,
            "architect_debug": {
                "tool_names": sorted(tools_by_name),
                "task_category": constraints["task_category"],
                "requires_final_response": requires_final_response,
                "planner": "heuristic",
                "stage_policy": {
                    "phase_order": list(constraints.get("phase_order", []) or []),
                    "initial_phase": str(constraints.get("initial_phase", "") or ""),
                },
            },
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
        }

    def _sanitize_next(self, raw_next: Any) -> list[dict[str, str]]:
        routes: list[dict[str, str]] = []
        if isinstance(raw_next, str):
            text = str(raw_next or "").strip()
            if text:
                routes.append({"condition": "always", "step": text})
            return routes
        if isinstance(raw_next, dict):
            raw_next = [raw_next]
        if not isinstance(raw_next, list):
            return routes
        for item in raw_next:
            if isinstance(item, str):
                text = str(item or "").strip()
                if text:
                    routes.append({"condition": "always", "step": text})
                continue
            if not isinstance(item, dict):
                continue
            condition = str(item.get("condition", item.get("if", "always")) or "always").strip()
            if condition not in self.ROUTE_CONDITIONS:
                condition = "always"
            step_id = str(item.get("step", "") or "").strip()
            routes.append({"condition": condition, "step": step_id})
        return routes

    def _sanitize_data_schema(self, raw_schema: Any, fallback_schema: dict[str, Any]) -> dict[str, Any]:
        schema = {str(k): dict(v) for k, v in dict(fallback_schema or {}).items()}
        if not isinstance(raw_schema, dict):
            return schema
        for key, value in raw_schema.items():
            slot = str(key or "").strip()
            if slot not in schema or not isinstance(value, dict):
                continue
            merged = dict(schema[slot])
            if "required" in value:
                merged["required"] = bool(value.get("required"))
            if "description" in value:
                merged["description"] = str(value.get("description", merged.get("description", "")) or "")
            schema[slot] = merged
        return schema

    def _sanitize_workflow(self, raw_workflow: Any, *, fallback_workflow: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
        warnings: list[str] = []
        if not isinstance(raw_workflow, dict):
            raw_workflow = {}
        fallback_steps = list(dict(fallback_workflow).get("steps", []) or [])
        raw_steps = list(raw_workflow.get("steps", []) or [])
        sanitized_steps: list[dict[str, Any]] = []
        used_workers: list[str] = []
        seen_ids: set[str] = set()

        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                continue
            step_id = str(raw_step.get("id", "") or "").strip() or f"step_{index + 1}"
            if step_id in seen_ids:
                step_id = f"{step_id}_{index + 1}"
            worker_name = str(raw_step.get("worker", "") or "").strip()
            if worker_name not in self.worker_specs:
                warnings.append(f"invalid worker in architect plan: {worker_name}")
                continue
            worker_spec = self.worker_specs[worker_name]
            step = {
                "id": step_id,
                "worker": worker_name,
                "purpose": str(raw_step.get("purpose", "") or worker_spec.description),
                "reads": list(worker_spec.reads),
                "writes": list(worker_spec.writes),
                "exit_conditions": [str(item or "") for item in list(raw_step.get("exit_conditions", []) or []) if str(item or "").strip()],
                "next": self._sanitize_next(raw_step.get("next")),
            }
            sanitized_steps.append(step)
            used_workers.append(worker_name)
            seen_ids.add(step_id)

        if not sanitized_steps:
            return dict(fallback_workflow), [step.get("worker", "") for step in fallback_steps], warnings + ["architect workflow invalid; fallback used"]

        valid_ids = {str(step.get("id", "") or "") for step in sanitized_steps}
        for step in sanitized_steps:
            normalized_routes: list[dict[str, str]] = []
            for route in list(step.get("next", []) or []):
                if not isinstance(route, dict):
                    continue
                next_step = str(route.get("step", "") or "")
                if next_step and next_step not in valid_ids:
                    warnings.append(f"invalid next step id dropped: {next_step}")
                    continue
                normalized_routes.append(
                    {
                        "condition": str(route.get("condition", "always") or "always"),
                        "step": next_step,
                    }
                )
            step["next"] = normalized_routes

        workflow = {
            "workflow_id": str(raw_workflow.get("workflow_id", "") or "webarena_llm"),
            "goal": str(raw_workflow.get("goal", "") or ""),
            "steps": sanitized_steps,
        }
        if not workflow["goal"]:
            workflow["goal"] = str(fallback_workflow.get("goal", "") or "")
        return workflow, used_workers, warnings

    def _sanitize_plan(
        self,
        payload: dict[str, Any],
        *,
        fallback: dict[str, Any],
        constraints: dict[str, Any],
        tools_by_name: dict[str, ToolSpec],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        workflow, used_workers, warnings = self._sanitize_workflow(
            payload.get("workflow"),
            fallback_workflow=dict(fallback.get("workflow") or {}),
        )
        selected_raw = list(payload.get("selected_workers", []) or [])
        selected_workers = [name for name in [str(item or "") for item in selected_raw] if name in self.worker_specs]
        if not selected_workers:
            selected_workers = [name for name in used_workers if name in self.worker_specs]
        for name in used_workers:
            if name not in selected_workers:
                selected_workers.append(name)
        if not selected_workers:
            selected_workers = list(fallback.get("selected_workers", []) or [])

        raw_instructions_payload = payload.get("worker_instructions")
        raw_instructions = raw_instructions_payload if isinstance(raw_instructions_payload, dict) else {}
        if raw_instructions_payload is not None and not isinstance(raw_instructions_payload, dict):
            warnings.append("architect worker_instructions is not an object; defaults applied")
        worker_instructions = {
            worker_name: str(raw_instructions.get(worker_name, "") or self._worker_instruction(worker_name))
            for worker_name in selected_workers
        }

        raw_warnings = [str(item or "") for item in list(payload.get("warnings", []) or []) if str(item or "").strip()]
        data_schema = self._sanitize_data_schema(payload.get("data_schema"), dict(fallback.get("data_schema") or {}))

        return {
            "workflow": workflow,
            "data_schema": data_schema,
            "task_constraints": constraints,
            "selected_workers": selected_workers,
            "worker_instructions": worker_instructions,
            "warnings": warnings + raw_warnings,
            "fallback_reason": "",
            "used_fallback_architect": False,
            "architect_debug": {
                "tool_names": sorted(tools_by_name),
                "task_category": str(constraints.get("task_category", "") or ""),
                "requires_final_response": bool(constraints.get("requires_final_response", False)),
                "planner": "llm",
                "stage_policy": {
                    "phase_order": list(constraints.get("phase_order", []) or []),
                    "initial_phase": str(constraints.get("initial_phase", "") or ""),
                },
            },
            "architect_input_tokens": int(usage.get("architect_input_tokens", 0) or 0),
            "architect_output_tokens": int(usage.get("architect_output_tokens", 0) or 0),
        }

    def build(self, task: TaskSpec, tools_by_name: dict[str, ToolSpec]) -> dict[str, Any]:
        constraints = self._task_constraints(task)
        fallback = self._fallback_plan(constraints, tools_by_name)
        if self.llm is None:
            return fallback

        attempt_errors: list[str] = []
        max_attempts = self._retry_attempts()
        for attempt in range(1, max_attempts + 1):
            try:
                payload, usage = self._invoke_llm_json(
                    constraints=constraints,
                    tools_by_name=tools_by_name,
                    fallback=fallback,
                )
                plan = self._sanitize_plan(
                    payload,
                    fallback=fallback,
                    constraints=constraints,
                    tools_by_name=tools_by_name,
                    usage=usage,
                )
                warnings = list(plan.get("warnings") or [])
                if any("architect workflow invalid; fallback used" in item for item in warnings):
                    raise ValueError("architect_workflow_invalid")
                debug = dict(plan.get("architect_debug") or {})
                debug["retry_attempt"] = attempt
                debug["retry_max"] = max_attempts
                plan["architect_debug"] = debug
                return plan
            except Exception as exc:
                attempt_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < max_attempts:
                    continue

        plan = dict(fallback)
        plan["fallback_reason"] = "architect_llm_retry_exhausted"
        debug = dict(plan.get("architect_debug") or {})
        debug["planner"] = "heuristic"
        debug["llm_error"] = attempt_errors[-1] if attempt_errors else "unknown architect llm error"
        debug["retry_max"] = max_attempts
        plan["architect_debug"] = debug
        warnings = list(plan.get("warnings") or [])
        warnings.append("architect llm planning failed after retries; heuristic fallback applied")
        warnings.extend(attempt_errors)
        plan["warnings"] = warnings
        return plan
