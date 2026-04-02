"""Minimal Blackboard system for WebArena tasks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any

from experiment.common.neutral import EmitMessage, Finish, MultiAgentSession, MultiAgentSystem, TaskSpec, ToolSpec, TurnInput, TurnOutput
from experiment.webarena.workers import WebArenaArchitect, WorkerResult, WorkerRuntime, build_default_workers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pointer_segments(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON pointer path: {path!r}")
    if path == "/":
        return []
    return [segment.replace("~1", "/").replace("~0", "~") for segment in path[1:].split("/")]


def _apply_patch_operation(state: dict[str, Any], operation: dict[str, Any]) -> None:
    op = str(operation.get("op", "") or "").strip()
    path = str(operation.get("path", "") or "").strip()
    segments = _pointer_segments(path)
    parent: Any = state
    for segment in segments[:-1]:
        if isinstance(parent, dict):
            parent = parent.setdefault(segment, {})
        else:
            raise ValueError(f"Unsupported patch parent for path {path!r}")
    key = segments[-1] if segments else ""
    if op in {"add", "replace"}:
        if not segments:
            raise ValueError("Root replacement is not supported.")
        if not isinstance(parent, dict):
            raise ValueError(f"Unsupported patch target for path {path!r}")
        parent[key] = operation.get("value")
        return
    if op == "remove":
        if not isinstance(parent, dict):
            raise ValueError(f"Unsupported remove target for path {path!r}")
        parent.pop(key, None)
        return
    raise ValueError(f"Unsupported patch op {op!r}")


def _normalized_url(value: str) -> str:
    text = str(value or "").strip()
    if text.endswith("/"):
        return text[:-1]
    return text


def _text_excerpt(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _action_label(tool_name: str, arguments: dict[str, Any]) -> str:
    normalized_args = dict(arguments or {})
    if not normalized_args:
        return str(tool_name or "")
    return f"{tool_name} {json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)}"


def _available_actions(current_page: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for element in list(current_page.get("interactive_elements", []) or [])[:12]:
        if not isinstance(element, dict):
            continue
        tag = str(element.get("tag", "") or "").strip()
        element_id = str(element.get("element_id", "") or "").strip()
        text = _text_excerpt(element.get("text", ""), limit=48)
        label = ":".join(part for part in (tag, element_id or text) if part)
        if label:
            actions.append(label)
    return actions


def _context_fragment_counts(worker_input: dict[str, Any]) -> tuple[int, int, int]:
    shared = dict(worker_input.get("shared_data") or {})
    count = len(shared)
    if worker_input.get("task_constraints"):
        count += 1
    if worker_input.get("execution"):
        count += 1
    return count, count, 0


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "over",
    "under",
    "all",
    "among",
    "any",
    "exactly",
}


def _intent_tokens(goal: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(goal or "").lower())
    return [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]


@dataclass(slots=True)
class WebArenaBlackboardSession(MultiAgentSession):
    """Task-scoped Blackboard session for one WebArena task."""

    task: TaskSpec
    tools: tuple[ToolSpec, ...]
    tool_executor: Any | None = None
    llm: Any | None = None
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.config = dict(self.config or {})
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.workers = {worker.spec.name: worker for worker in build_default_workers(llm=self.llm)}
        self.worker_specs = tuple(worker.spec for worker in self.workers.values())
        self.architect = WebArenaArchitect(
            self.worker_specs,
            llm=self.llm,
            use_target_urls=bool(self.config.get("enable_target_urls", False)),
        )
        self.state: dict[str, Any] | None = None
        self.finished = False

    def _initial_observation(self) -> dict[str, Any]:
        if self.tool_executor is None or not hasattr(self.tool_executor, "current_observation"):
            return {}
        try:
            observation = self.tool_executor.current_observation()
        except Exception:
            observation = {}
        return dict(observation or {})

    def _initial_state(self) -> dict[str, Any]:
        initial_observation = self._initial_observation()
        navigation_history: list[str] = []
        current_url = _normalized_url(initial_observation.get("url", ""))
        if current_url:
            navigation_history.append(current_url)
        return {
            "task_id": str(self.task.task_id or ""),
            "workflow": {},
            "data_schema": {},
            "task_constraints": {},
            "shared_data": {
                "current_page": {},
                "page_evidence": {},
                "open_tabs": list(initial_observation.get("tabs", []) or []),
                "grounded_action": {},
                "action_arguments": {},
                "last_action": {},
                "last_observation": initial_observation,
                "action_history": [],
                "navigation_history": navigation_history,
                "execution_error": "",
                "verification": {},
                "assistant_response": "",
                "finish": False,
                "finish_reason": "",
            },
            "execution": {
                "status": "uninitialized",
                "current_step_id": "",
                "current_step": {},
                "completed_steps": [],
                "active_worker": "",
                "last_worker": "",
                "selected_workers": [],
                "worker_instructions": {},
                "warnings": [],
                "fallback_reason": "",
                "used_fallback_architect": False,
                "architect_debug": {},
                "tool_call_counter": 0,
                "trajectory": [],
                "pending_trajectory_index": None,
                "worker_call_count": 0,
                "context_fragment_total": 0,
                "relevant_fragment_total": 0,
                "irrelevant_fragment_total": 0,
                "worker_input_tokens": 0,
                "worker_output_tokens": 0,
                "architect_input_tokens": 0,
                "architect_output_tokens": 0,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "worker_retry_count": 0,
                "worker_retry_by_step": {},
            },
            "circuit_breaker": {
                "no_progress_count": 0,
                "tripped": False,
                "reason": "",
                "last_error": "",
            },
            "communication_trace": [],
        }

    def _record_trace(self, event: dict[str, Any]) -> None:
        assert self.state is not None
        trace = list(self.state.get("communication_trace") or [])
        trace.append({"timestamp": _utc_now(), **event})
        self.state["communication_trace"] = trace[-100:]

    def _step_by_id(self, step_id: str) -> dict[str, Any]:
        assert self.state is not None
        workflow = dict(self.state.get("workflow") or {})
        for step in list(workflow.get("steps", []) or []):
            if str(step.get("id", "") or "") == step_id:
                return dict(step)
        return {}

    def _current_step(self) -> dict[str, Any]:
        assert self.state is not None
        execution = dict(self.state.get("execution") or {})
        step_id = str(execution.get("current_step_id", "") or "")
        if not step_id:
            return {}
        return self._step_by_id(step_id)

    def _ensure_initialized(self) -> None:
        if self.state is None:
            self.state = self._initial_state()
        execution = dict(self.state.get("execution") or {})
        if execution.get("status") != "uninitialized":
            return

        use_architect = bool(self.config.get("use_architect", True))
        plan = self.architect.build(self.task, self.tools_by_name)
        if not use_architect:
            task_constraints = dict(plan.get("task_constraints") or {})
            task_constraints["target_urls"] = []
            plan["task_constraints"] = task_constraints
            plan["fallback_reason"] = "architect_disabled"
            plan["used_fallback_architect"] = True
            debug = dict(plan.get("architect_debug") or {})
            debug["architect_disabled"] = True
            plan["architect_debug"] = debug
            workflow = dict(plan.get("workflow") or {})
            workflow["workflow_id"] = "webarena_fixed_no_architect"
            plan["workflow"] = workflow
        self.state["workflow"] = dict(plan["workflow"])
        self.state["data_schema"] = dict(plan["data_schema"])
        self.state["task_constraints"] = dict(plan["task_constraints"])

        first_step = dict(plan["workflow"]["steps"][0]) if plan["workflow"]["steps"] else {}
        execution["status"] = "running"
        execution["current_step_id"] = str(first_step.get("id", "") or "")
        execution["current_step"] = first_step
        execution["selected_workers"] = list(plan["selected_workers"])
        execution["worker_instructions"] = dict(plan["worker_instructions"])
        execution["warnings"] = list(plan.get("warnings") or [])
        execution["fallback_reason"] = str(plan.get("fallback_reason", "") or "")
        execution["used_fallback_architect"] = bool(plan.get("used_fallback_architect", False))
        execution["architect_debug"] = dict(plan.get("architect_debug") or {})
        self.state["execution"] = execution
        self._record_trace(
            {
                "event_type": "architect",
                "task_id": str(self.task.task_id or ""),
                "selected_workers": list(plan["selected_workers"]),
                "workflow_id": str(plan["workflow"].get("workflow_id", "") or ""),
                "used_fallback_architect": bool(plan.get("used_fallback_architect", False)),
            }
        )
        execution = dict(self.state.get("execution") or {})
        execution["architect_input_tokens"] = int(execution.get("architect_input_tokens", 0) or 0) + int(plan.get("architect_input_tokens", 0) or 0)
        execution["architect_output_tokens"] = int(execution.get("architect_output_tokens", 0) or 0) + int(plan.get("architect_output_tokens", 0) or 0)
        self.state["execution"] = execution

    def _merge_turn_input(self, turn_input: TurnInput) -> None:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        if turn_input.new_tool_results:
            latest_tool_result = turn_input.new_tool_results[-1]
            shared["last_tool_result"] = {
                "tool_name": latest_tool_result.tool_name,
                "status": str(latest_tool_result.status.value),
                "content": latest_tool_result.content,
                "payload": dict(latest_tool_result.payload),
                "error_message": latest_tool_result.error_message,
            }
            shared["execution_error"] = str(latest_tool_result.error_message or "")
        if turn_input.new_observations:
            latest_observation = turn_input.new_observations[-1]
            payload = dict(latest_observation.payload)
            shared["last_observation"] = payload
            url = _normalized_url(payload.get("url", ""))
            navigation_history = list(shared.get("navigation_history", []) or [])
            if url and (not navigation_history or navigation_history[-1] != url):
                navigation_history.append(url)
            shared["navigation_history"] = navigation_history[-20:]
        self.state["shared_data"] = shared

        execution = dict(self.state.get("execution") or {})
        pending_index = execution.get("pending_trajectory_index")
        if pending_index is None:
            self.state["execution"] = execution
            return

        trajectory = list(execution.get("trajectory") or [])
        if not isinstance(pending_index, int) or pending_index < 0 or pending_index >= len(trajectory):
            execution["pending_trajectory_index"] = None
            self.state["execution"] = execution
            return

        step_record = dict(trajectory[pending_index] or {})
        if turn_input.new_tool_results:
            latest_tool_result = turn_input.new_tool_results[-1]
            step_record["tool_result_status"] = str(latest_tool_result.status.value)
            step_record["tool_result_content"] = str(latest_tool_result.content or "")
            step_record["tool_error"] = str(latest_tool_result.error_message or "")
        if turn_input.new_observations:
            latest_observation = turn_input.new_observations[-1]
            step_record["observation_after"] = dict(latest_observation.payload or {})
        trajectory[pending_index] = step_record
        execution["trajectory"] = trajectory
        execution["pending_trajectory_index"] = None
        self.state["execution"] = execution

    def _allowed_patch_paths(self, step: dict[str, Any]) -> set[str]:
        allowed = {"/execution/tool_call_counter"}
        for slot in list(step.get("writes", []) or []):
            allowed.add(f"/shared_data/{slot}")
        return allowed

    def _build_worker_input_state(self, worker_name: str) -> dict[str, Any]:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        execution = dict(self.state.get("execution") or {})
        base = {
            "task_constraints": dict(self.state.get("task_constraints") or {}),
            "execution": {
                "current_step_id": str(execution.get("current_step_id", "") or ""),
                "current_step": dict(execution.get("current_step") or {}),
                "worker_instruction": str(dict(execution.get("worker_instructions") or {}).get(worker_name, "") or ""),
                "used_fallback_architect": bool(execution.get("used_fallback_architect", False)),
                "tool_call_counter": int(execution.get("tool_call_counter", 0) or 0),
            },
            "shared_data": {},
        }
        if worker_name == "page_state_worker":
            base["shared_data"] = {
                "last_observation": dict(shared.get("last_observation") or {}),
                "current_page": dict(shared.get("current_page") or {}),
                "page_evidence": dict(shared.get("page_evidence") or {}),
                "verification": dict(shared.get("verification") or {}),
                "action_history": list(shared.get("action_history", []) or []),
                "finish": bool(shared.get("finish", False)),
                "finish_reason": str(shared.get("finish_reason", "") or ""),
            }
        elif worker_name == "argument_grounding_worker":
            base["shared_data"] = {
                "current_page": dict(shared.get("current_page") or {}),
                "page_evidence": dict(shared.get("page_evidence") or {}),
                "verification": dict(shared.get("verification") or {}),
                "action_history": list(shared.get("action_history", []) or []),
            }
        elif worker_name == "browser_action_worker":
            base["shared_data"] = {
                "grounded_action": dict(shared.get("grounded_action") or {}),
                "action_history": list(shared.get("action_history", []) or []),
            }
        elif worker_name == "response_worker":
            base["shared_data"] = {
                "verification": dict(shared.get("verification") or {}),
                "page_evidence": dict(shared.get("page_evidence") or {}),
                "current_page": dict(shared.get("current_page") or {}),
            }
        return base

    def _refresh_kernel_verification(self) -> None:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        verification = dict(shared.get("verification") or {})
        phase = str(verification.get("phase", "") or "").strip().lower()
        if phase and phase not in {"navigate", "evidence", "respond", "complete"}:
            verification["phase"] = "evidence"
        shared["verification"] = verification
        if bool(shared.get("finish", False)) and not str(shared.get("finish_reason", "") or "").strip():
            shared["finish_reason"] = "task_completed"
        self.state["shared_data"] = shared

    def _schedule_worker_retry(self, *, step_id: str, worker_name: str, reason: str, error: str) -> bool:
        assert self.state is not None
        execution = dict(self.state.get("execution") or {})
        retries_by_step = dict(execution.get("worker_retry_by_step") or {})
        retry_count = int(retries_by_step.get(step_id, 0) or 0) + 1
        retries_by_step[step_id] = retry_count
        execution["worker_retry_by_step"] = retries_by_step
        execution["worker_retry_count"] = int(execution.get("worker_retry_count", 0) or 0) + 1
        self.state["execution"] = execution

        shared = dict(self.state.get("shared_data") or {})
        shared["execution_error"] = str(error or "")
        self.state["shared_data"] = shared

        max_worker_retries = int(self.config.get("max_worker_retries", 2) or 2)
        self._record_trace(
            {
                "event_type": "worker_retry",
                "worker": worker_name,
                "step_id": step_id,
                "reason": reason,
                "error": str(error or ""),
                "retry_attempt": retry_count,
                "retry_max": max_worker_retries,
            }
        )
        return retry_count <= max_worker_retries

    def _apply_worker_result(self, result: WorkerResult, *, step: dict[str, Any]) -> str:
        assert self.state is not None
        allowed_paths = self._allowed_patch_paths(step)
        for operation in result.patch:
            path = str(operation.get("path", "") or "")
            if path not in allowed_paths:
                return f"Patch path {path} is outside the worker write scope."
            _apply_patch_operation(self.state, operation)
        return ""

    def _record_context_stats(self, worker_input: dict[str, Any]) -> None:
        assert self.state is not None
        execution = dict(self.state.get("execution") or {})
        context_count, relevant_count, irrelevant_count = _context_fragment_counts(worker_input)
        execution["worker_call_count"] = int(execution.get("worker_call_count", 0) or 0) + 1
        execution["context_fragment_total"] = int(execution.get("context_fragment_total", 0) or 0) + context_count
        execution["relevant_fragment_total"] = int(execution.get("relevant_fragment_total", 0) or 0) + relevant_count
        execution["irrelevant_fragment_total"] = int(execution.get("irrelevant_fragment_total", 0) or 0) + irrelevant_count
        self.state["execution"] = execution

    def _append_action_trajectory(self, *, step: dict[str, Any], result: WorkerResult) -> None:
        assert self.state is not None
        if result.tool_call is None:
            return

        execution = dict(self.state.get("execution") or {})
        shared = dict(self.state.get("shared_data") or {})
        constraints = dict(self.state.get("task_constraints") or {})
        current_page = dict(shared.get("current_page") or {})
        grounded_action = dict(shared.get("grounded_action") or {})
        action_label = _action_label(result.tool_call.tool_name, result.tool_call.arguments)
        trajectory = list(execution.get("trajectory") or [])
        trajectory_index = len(trajectory)
        planner_state = {
            "intent": str(constraints.get("goal", "") or ""),
            "next_action": action_label,
            "recommended_actions": [action_label] if action_label else [],
        }
        trajectory.append(
            {
                "step_id": trajectory_index,
                "workflow_step_id": str(step.get("id", "") or ""),
                "worker": str(step.get("worker", "") or ""),
                "action": action_label,
                "expected_action": action_label,
                "decision_reason": str(result.tool_call.rationale or grounded_action.get("rationale", "") or ""),
                "fallback_used": False,
                "patch_error": "",
                "action_matched": True,
                "context_mode": "filtered_blackboard",
                "planner_state": planner_state,
                "observation_before": dict(shared.get("last_observation") or {}),
                "available_actions_before": _available_actions(current_page),
                "communication_trace": [
                    {
                        "source": "architect",
                        "channel": "structured",
                        "content": {
                            "intent": planner_state["intent"],
                            "task_category": str(constraints.get("task_category", "") or ""),
                            "next_action": action_label,
                        },
                    },
                    {
                        "source": "argument_grounding_worker",
                        "channel": "structured",
                        "content": grounded_action,
                    },
                    {
                        "source": "browser_action_worker",
                        "channel": "structured",
                        "content": {
                            "selected_action": action_label,
                            "rationale": str(result.tool_call.rationale or ""),
                        },
                    },
                ],
            }
        )
        execution["trajectory"] = trajectory
        execution["pending_trajectory_index"] = trajectory_index
        self.state["execution"] = execution

    def _mark_finished(self, reason: str) -> None:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        shared["finish"] = True
        shared["finish_reason"] = str(reason or shared.get("finish_reason", "") or "")
        self.state["shared_data"] = shared
        execution = dict(self.state.get("execution") or {})
        execution["status"] = "finished"
        self.state["execution"] = execution
        self.finished = True

    def _normalize_routes(self, raw_next: Any) -> list[dict[str, str]]:
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
            condition = str(item.get("condition", item.get("if", "always")) or "always").strip() or "always"
            step_id = str(item.get("step", "") or "").strip()
            routes.append({"condition": condition, "step": step_id})
        return routes

    def _route_condition_met(self, condition: str, *, result: WorkerResult) -> bool:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        verification = dict(shared.get("verification") or {})
        constraints = dict(self.state.get("task_constraints") or {})
        grounded_action = dict(shared.get("grounded_action") or {})

        goal_reached = bool(verification.get("goal_reached", False))
        answer_ready = bool(verification.get("answer_ready", False))
        requires_response = bool(constraints.get("requires_final_response", False))
        has_grounded_action = bool(str(grounded_action.get("tool_name", "") or "").strip())
        finished = bool(shared.get("finish", False)) or result.finished

        normalized = str(condition or "always").strip() or "always"
        if normalized == "always":
            return True
        if normalized == "answer_ready":
            return answer_ready
        if normalized == "goal_reached":
            return goal_reached
        if normalized == "goal_reached_and_requires_response":
            return goal_reached and requires_response
        if normalized == "needs_action":
            return (not goal_reached) and (not answer_ready) and (not finished)
        if normalized == "has_grounded_action":
            return has_grounded_action
        if normalized == "tool_call_available":
            return result.tool_call is not None
        if normalized == "finished":
            return finished
        return False

    def _legacy_next_step_id(self, step: dict[str, Any], result: WorkerResult) -> str:
        worker_name = str(step.get("worker", "") or "")
        if worker_name == "page_state_worker":
            return "bind_action_arguments"
        if worker_name == "argument_grounding_worker":
            grounded = dict(dict(self.state.get("shared_data") or {}).get("grounded_action") or {})
            return "execute_browser_action" if str(grounded.get("tool_name", "") or "").strip() else ""
        if worker_name == "browser_action_worker":
            if result.tool_call is not None:
                return "inspect_current_page"
            return ""
        if worker_name == "response_worker":
            return ""
        return ""

    def _next_step_id(self, step: dict[str, Any], result: WorkerResult) -> str:
        routes = self._normalize_routes(step.get("next"))
        if not routes:
            return self._legacy_next_step_id(step, result)
        for route in routes:
            condition = str(route.get("condition", "always") or "always")
            if not self._route_condition_met(condition, result=result):
                continue
            step_id = str(route.get("step", "") or "")
            if not step_id:
                return ""
            if self._step_by_id(step_id):
                return step_id
        return ""

    def _update_no_progress(self) -> None:
        assert self.state is not None
        shared = dict(self.state.get("shared_data") or {})
        breaker = dict(self.state.get("circuit_breaker") or {})
        verification = dict(shared.get("verification") or {})
        if bool(verification.get("goal_reached", False)) or bool(verification.get("progress_made", False)):
            breaker["no_progress_count"] = 0
        else:
            breaker["no_progress_count"] = int(breaker.get("no_progress_count", 0) or 0) + 1
        self.state["circuit_breaker"] = breaker

    def _metadata(self) -> dict[str, Any]:
        assert self.state is not None
        execution = dict(self.state.get("execution") or {})
        breaker = dict(self.state.get("circuit_breaker") or {})
        shared = dict(self.state.get("shared_data") or {})
        verification = dict(shared.get("verification") or {})
        worker_calls = max(int(execution.get("worker_call_count", 0) or 0), 1)
        goal_condition_rate = 1.0 if bool(verification.get("goal_reached", False)) else 0.0
        return {
            "kernel_status": str(execution.get("status", "") or ""),
            "current_step_id": str(execution.get("current_step_id", "") or ""),
            "current_step": dict(execution.get("current_step") or {}),
            "active_worker": str(execution.get("active_worker", "") or ""),
            "finish_reason": str(shared.get("finish_reason", "") or ""),
            "verification": verification,
            "used_fallback_architect": bool(execution.get("used_fallback_architect", False)),
            "fallback_reason": str(execution.get("fallback_reason", "") or ""),
            "architect_debug": dict(execution.get("architect_debug") or {}),
            "warnings": list(execution.get("warnings") or []),
            "communication_trace": list(self.state.get("communication_trace") or []),
            "trajectory": list(execution.get("trajectory") or []),
            "goal_condition_rate": goal_condition_rate,
            "worker_input_tokens": int(execution.get("worker_input_tokens", 0) or 0),
            "worker_output_tokens": int(execution.get("worker_output_tokens", 0) or 0),
            "architect_input_tokens": int(execution.get("architect_input_tokens", 0) or 0),
            "architect_output_tokens": int(execution.get("architect_output_tokens", 0) or 0),
            "total_tokens": int(execution.get("worker_input_tokens", 0) or 0)
            + int(execution.get("worker_output_tokens", 0) or 0)
            + int(execution.get("architect_input_tokens", 0) or 0)
            + int(execution.get("architect_output_tokens", 0) or 0),
            "fallback_action_count": int(execution.get("fallback_action_count", 0) or 0),
            "patch_error_count": int(execution.get("patch_error_count", 0) or 0),
            "worker_retry_count": int(execution.get("worker_retry_count", 0) or 0),
            "retrieval_precision": goal_condition_rate,
            "context_fragment_count": float(int(execution.get("context_fragment_total", 0) or 0) / worker_calls),
            "relevant_fragment_count": float(int(execution.get("relevant_fragment_total", 0) or 0) / worker_calls),
            "irrelevant_fragment_count": float(int(execution.get("irrelevant_fragment_total", 0) or 0) / worker_calls),
            "circuit_breaker": breaker,
        }

    def _advance(self) -> TurnOutput:
        assert self.state is not None
        max_kernel_steps = int(self.config.get("max_kernel_steps", 8) or 8)
        max_no_progress = int(self.config.get("max_no_progress", 2) or 2)

        for _ in range(max_kernel_steps):
            step = self._current_step()
            if not step:
                self._mark_finished("workflow_exhausted")
                break

            worker_name = str(step.get("worker", "") or "")
            worker = self.workers[worker_name]
            execution = dict(self.state.get("execution") or {})
            execution["active_worker"] = worker_name
            execution["last_worker"] = worker_name
            execution["current_step"] = dict(step)
            self.state["execution"] = execution

            worker_input = self._build_worker_input_state(worker_name)
            self._record_context_stats(worker_input)
            try:
                result = worker.run(WorkerRuntime(state=worker_input, tools_by_name=self.tools_by_name))
            except Exception as exc:
                step_id = str(step.get("id", "") or "")
                should_retry = self._schedule_worker_retry(
                    step_id=step_id,
                    worker_name=worker_name,
                    reason="worker_exception",
                    error=str(exc),
                )
                if should_retry:
                    continue
                breaker = dict(self.state.get("circuit_breaker") or {})
                breaker["tripped"] = True
                breaker["reason"] = "worker_error"
                breaker["last_error"] = str(exc)
                self.state["circuit_breaker"] = breaker
                self._mark_finished("worker_error")
                break
            execution = dict(self.state.get("execution") or {})
            result_metadata = dict(getattr(result, "metadata", {}) or {})
            execution["worker_input_tokens"] = int(execution.get("worker_input_tokens", 0) or 0) + int(result_metadata.get("worker_input_tokens", 0) or 0)
            execution["worker_output_tokens"] = int(execution.get("worker_output_tokens", 0) or 0) + int(result_metadata.get("worker_output_tokens", 0) or 0)
            self.state["execution"] = execution
            patch_error = self._apply_worker_result(result, step=step)
            if patch_error:
                execution = dict(self.state.get("execution") or {})
                execution["patch_error_count"] = int(execution.get("patch_error_count", 0) or 0) + 1
                self.state["execution"] = execution
                step_id = str(step.get("id", "") or "")
                should_retry = self._schedule_worker_retry(
                    step_id=step_id,
                    worker_name=worker_name,
                    reason="patch_error",
                    error=patch_error,
                )
                if should_retry:
                    continue
                breaker = dict(self.state.get("circuit_breaker") or {})
                breaker["tripped"] = True
                breaker["reason"] = "patch_error"
                breaker["last_error"] = patch_error
                self.state["circuit_breaker"] = breaker
                self._record_trace(
                    {
                        "event_type": "worker",
                        "worker": worker_name,
                        "step_id": step_id,
                        "patch_error": patch_error,
                    }
                )
                self._mark_finished("patch_error")
                break

            self._record_trace(
                {
                    "event_type": "worker",
                    "worker": worker_name,
                    "step_id": str(step.get("id", "") or ""),
                    "writes": list(step.get("writes", []) or []),
                    "tool_name": result.tool_call.tool_name if result.tool_call is not None else "",
                    "finished": result.finished,
                }
            )

            if worker_name == "browser_action_worker" and result.tool_call is not None:
                self._append_action_trajectory(step=step, result=result)

            if worker_name == "page_state_worker":
                self._update_no_progress()
                breaker = dict(self.state.get("circuit_breaker") or {})
                if int(breaker.get("no_progress_count", 0) or 0) > max_no_progress:
                    breaker["tripped"] = True
                    breaker["reason"] = "no_progress_threshold_exceeded"
                    self.state["circuit_breaker"] = breaker
                    self._mark_finished("no_progress_threshold_exceeded")
                    break

            next_step_id = self._next_step_id(step, result)
            execution = dict(self.state.get("execution") or {})
            execution["current_step_id"] = next_step_id
            execution["current_step"] = self._step_by_id(next_step_id) if next_step_id else {}
            self.state["execution"] = execution

            if result.tool_call is not None:
                tool_output = TurnOutput.request_tool(
                    result.tool_call.tool_name,
                    result.tool_call.arguments,
                    call_id=result.tool_call.call_id,
                    rationale=result.tool_call.rationale,
                    metadata=self._metadata(),
                )
                return TurnOutput(actions=tool_output.actions, metadata=self._metadata())

            shared = dict(self.state.get("shared_data") or {})
            if result.message is not None:
                reason = result.finish_reason or str(shared.get("finish_reason", "") or "task_completed")
                self._mark_finished(reason)
                return TurnOutput(
                    actions=(
                        EmitMessage(result.message),
                        Finish(reason=reason, metadata=self._metadata()),
                    ),
                    metadata=self._metadata(),
                )

            if bool(shared.get("finish", False)):
                reason = str(shared.get("finish_reason", "") or "task_completed")
                self._mark_finished(reason)
                return TurnOutput(
                    actions=(Finish(reason=reason, metadata=self._metadata()),),
                    metadata=self._metadata(),
                )

        if not self.finished:
            breaker = dict(self.state.get("circuit_breaker") or {})
            breaker["tripped"] = True
            breaker["reason"] = "max_kernel_steps_exceeded"
            self.state["circuit_breaker"] = breaker
            self._mark_finished("max_kernel_steps_exceeded")
        return TurnOutput(
            actions=(Finish(reason=str(dict(self.state.get("shared_data") or {}).get("finish_reason", "") or "max_kernel_steps_exceeded"), metadata=self._metadata()),),
            metadata=self._metadata(),
        )

    def step(self, turn_input: TurnInput) -> TurnOutput:
        self._ensure_initialized()
        self._merge_turn_input(turn_input)
        return self._advance()


class WebArenaBlackboardSystem(MultiAgentSystem):
    """Factory for WebArena blackboard sessions."""

    def __init__(self, llm: Any | None = None, *, config: dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.config = dict(config or {})

    def create_session(
        self,
        task: TaskSpec,
        tools: tuple[ToolSpec, ...] = (),
        native_tools: tuple[Any, ...] = (),
    ) -> MultiAgentSession:
        tool_executor = next(
            (native for native in native_tools if hasattr(native, "execute_tool_call") or hasattr(native, "current_observation")),
            None,
        )
        return WebArenaBlackboardSession(
            task=task,
            tools=tools,
            tool_executor=tool_executor,
            llm=self.llm,
            config=self.config,
        )
