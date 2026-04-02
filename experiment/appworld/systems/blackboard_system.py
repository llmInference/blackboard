"""AppWorld blackboard system aligned with the task-scoped architect/kernel split."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiment.appworld.core.api_runtime import AppWorldApiRuntime
from experiment.appworld.core.state_bridge import (
    merge_turn_input_into_state,
    turn_input_to_blackboard_state,
)
from experiment.appworld.workers import AppWorldArchitect, WorkerRuntime, build_default_workers
from experiment.common.neutral import (
    EmitMessage,
    Finish,
    Message,
    MultiAgentSession,
    MultiAgentSystem,
    TaskSpec,
    ToolSpec,
    TurnInput,
    TurnOutput,
)


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
        value = operation.get("value")
        if segments:
            if isinstance(parent, dict):
                parent[key] = value
            else:
                raise ValueError(f"Unsupported patch target for path {path!r}")
        else:
            raise ValueError("Root replacement is not supported.")
    elif op == "remove":
        if isinstance(parent, dict):
            parent.pop(key, None)
        else:
            raise ValueError(f"Unsupported remove target for path {path!r}")
    else:
        raise ValueError(f"Unsupported patch op {op!r}")


def _slice_state_by_schema(source: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return source
    properties = dict(schema.get("properties") or {})
    if not properties:
        return source
    if not isinstance(source, dict):
        return {}
    sliced: dict[str, Any] = {}
    for key, child_schema in properties.items():
        if key not in source:
            continue
        child_value = source[key]
        child_properties = dict((child_schema or {}).get("properties") or {})
        if child_properties and isinstance(child_value, dict):
            sliced[key] = _slice_state_by_schema(child_value, dict(child_schema))
        else:
            sliced[key] = child_value
    return sliced


@dataclass(slots=True)
class AppWorldBlackboardSession(MultiAgentSession):
    """Task-scoped AppWorld blackboard session."""

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
        self.architect = AppWorldArchitect(self.worker_specs, llm=self.llm)
        self.state: dict[str, Any] | None = None
        self.finished = False

    def _initial_state(self, turn_input: TurnInput) -> dict[str, Any]:
        return turn_input_to_blackboard_state(
            turn_input,
            available_workers=tuple(spec.to_dict() for spec in self.worker_specs),
        )

    def _ensure_initialized(self, turn_input: TurnInput) -> None:
        if self.state is None:
            self.state = self._initial_state(turn_input)
        else:
            self.state = merge_turn_input_into_state(self.state, turn_input)

        execution = dict(self.state.get("execution") or {})
        if execution.get("status") != "uninitialized":
            return
        plan = self.architect.build(self.state, self.tools_by_name)
        self.state["workflow"] = dict(plan["workflow"])
        self.state["data_schema"] = dict(plan["data_schema"])
        execution["status"] = "running"
        first_step = dict(plan["workflow"]["steps"][0]) if plan["workflow"]["steps"] else {}
        execution["current_step_id"] = str(first_step.get("id", "") or "")
        execution["current_step"] = first_step
        execution["selected_workers"] = list(plan["selected_workers"])
        execution["worker_instructions"] = dict(plan["worker_instructions"])
        execution["warnings"] = list(plan.get("warnings") or [])
        execution["fallback_reason"] = str(plan.get("fallback_reason", "") or "")
        execution["used_fallback_architect"] = bool(plan.get("used_fallback_architect", False))
        execution["architect_debug"] = dict(plan.get("architect_debug") or {})
        if execution["used_fallback_architect"]:
            history = list(execution.get("history") or [])
            history.append(
                {
                    "worker": "architect",
                    "status": "warning",
                    "message": execution["fallback_reason"] or "Architect fallback was used.",
                }
            )
            execution["history"] = history[-20:]
        self.state["execution"] = execution

    def _allowed_patch_paths(self, step: dict[str, Any]) -> set[str]:
        allowed = {"/execution/history"}
        for slot in list(step.get("writes") or []):
            allowed.add(f"/shared_data/{slot}")
        return allowed

    def _filtered_tool_names(self) -> list[str]:
        assert self.state is not None
        task = dict(self.state.get("task") or {})
        metadata = dict(task.get("metadata") or {})
        allowed_apps = set(str(app) for app in list(metadata.get("allowed_apps") or []))
        oracle_required_apis = [str(item) for item in list(metadata.get("oracle_required_apis") or []) if str(item).strip()]
        instruction = str(task.get("instruction", "") or "").lower()
        ranked: list[tuple[int, str]] = []
        for tool_name, tool in self.tools_by_name.items():
            app_name = str(tool.metadata.get("app_name", "") or "")
            if app_name and allowed_apps and app_name not in allowed_apps and app_name != "supervisor":
                continue
            api_name = tool_name.partition("__")[2].replace("_", " ").lower()
            description = str(tool.description or "").lower()
            score = 0
            if app_name in allowed_apps:
                score += 3
            for token in instruction.split():
                token = token.strip(",.?!:")
                if token and token in api_name:
                    score += 3
                if token and token in description:
                    score += 1
            if not bool(tool.metadata.get("mutates_state", True)):
                score += 1
            ranked.append((score, tool_name))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [tool_name for _, tool_name in ranked[:12]]
        protected_apps = {
            str(self.tools_by_name[tool_name].metadata.get("app_name", "") or "")
            for tool_name in selected
            if tool_name in self.tools_by_name
        }
        for app_name in sorted(protected_apps):
            login_tool = f"{app_name}__login"
            if login_tool in self.tools_by_name and login_tool not in selected:
                selected.append(login_tool)
        for api_name in oracle_required_apis:
            if "." not in api_name:
                continue
            app_name, _, api_leaf = api_name.partition(".")
            tool_name = f"{app_name}__{api_leaf}"
            if tool_name in self.tools_by_name and tool_name not in selected:
                selected.append(tool_name)
        if "supervisor__show_account_passwords" in self.tools_by_name and "supervisor__show_account_passwords" not in selected:
            selected.append("supervisor__show_account_passwords")
        return list(dict.fromkeys(selected))

    def _prepare_state_for_worker(self, worker_name: str) -> None:
        assert self.state is not None
        if worker_name != "tool_worker":
            return
        shared_data = dict(self.state.get("shared_data") or {})
        shared_data["candidate_tools"] = self._filtered_tool_names()
        self.state["shared_data"] = shared_data

    def _build_worker_input_state(self, worker_name: str) -> dict[str, Any]:
        assert self.state is not None
        worker = self.workers[worker_name]
        input_schema = dict(worker.spec.input_schema or {})
        if not input_schema:
            return dict(self.state)
        worker_state = _slice_state_by_schema(self.state, input_schema)
        if "task" not in worker_state and "task" in self.state:
            worker_state["task"] = dict(self.state.get("task") or {})
        if "execution" not in worker_state and "execution" in self.state:
            worker_state["execution"] = {
                "current_step_id": str(dict(self.state.get("execution") or {}).get("current_step_id", "") or ""),
                "current_step": dict(dict(self.state.get("execution") or {}).get("current_step") or {}),
                "history": list(dict(self.state.get("execution") or {}).get("history") or []),
            }
        return worker_state

    def _apply_worker_patch(self, patch: list[dict[str, Any]], allowed_paths: set[str]) -> tuple[bool, str]:
        assert self.state is not None
        changed = False
        for operation in patch:
            path = str(operation.get("path", "") or "")
            if path not in allowed_paths:
                return False, f"Patch path {path} is outside the worker write scope."
            _apply_patch_operation(self.state, operation)
            changed = True
        return changed, ""

    def _mark_circuit_breaker(self, reason: str) -> None:
        assert self.state is not None
        breaker = dict(self.state.get("circuit_breaker") or {})
        breaker["tripped"] = True
        breaker["reason"] = reason
        breaker["last_error"] = reason
        self.state["circuit_breaker"] = breaker
        shared_data = dict(self.state.get("shared_data") or {})
        if not str(shared_data.get("assistant_response", "") or "").strip():
            shared_data["assistant_response"] = "I could not complete the AppWorld workflow."
        shared_data["finish"] = True
        shared_data["finish_reason"] = reason
        self.state["shared_data"] = shared_data
        execution = dict(self.state.get("execution") or {})
        execution["status"] = "circuit_break"
        self.state["execution"] = execution

    def _choose_next_step(self, step: dict[str, Any]) -> str:
        assert self.state is not None
        shared_data = dict(self.state.get("shared_data") or {})
        next_steps = [step_id for step_id in list(step.get("next") or []) if step_id]
        if step.get("worker") == "state_worker":
            progress_state = dict(shared_data.get("progress_state") or {})
            if bool(progress_state.get("needs_more_tooling", False)):
                for step_id in next_steps:
                    worker = self._step_by_id(step_id).get("worker")
                    if worker == "tool_worker":
                        return step_id
                tool_step_id = self._first_step_for_worker("tool_worker")
                if tool_step_id:
                    return tool_step_id
            if bool(progress_state.get("ready_for_response", False)):
                for step_id in next_steps:
                    worker = self._step_by_id(step_id).get("worker")
                    if worker in {"analysis_worker", "response_worker"}:
                        return step_id
                analysis_step_id = self._first_step_for_worker("analysis_worker")
                if analysis_step_id:
                    return analysis_step_id
                response_step_id = self._first_step_for_worker("response_worker")
                if response_step_id:
                    return response_step_id
        if step.get("worker") == "analysis_worker":
            progress_state = dict(shared_data.get("progress_state") or {})
            if bool(progress_state.get("needs_more_tooling", False)):
                for step_id in next_steps:
                    worker = self._step_by_id(step_id).get("worker")
                    if worker == "tool_worker":
                        return step_id
                tool_step_id = self._first_step_for_worker("tool_worker")
                if tool_step_id:
                    return tool_step_id
            if bool(progress_state.get("ready_for_response", False)) or bool(progress_state.get("blocked", False)):
                for step_id in next_steps:
                    worker = self._step_by_id(step_id).get("worker")
                    if worker == "response_worker":
                        return step_id
                response_step_id = self._first_step_for_worker("response_worker")
                if response_step_id:
                    return response_step_id
        if step.get("worker") == "response_worker":
            tool_status = dict(shared_data.get("tool_status") or {})
            required_api_status = dict(tool_status.get("required_api_status") or {})
            pending_deferred = [
                str(item)
                for item in list(required_api_status.get("actionable_pending_deferred_tools") or [])
                if str(item).strip()
            ]
            if pending_deferred:
                for step_id in next_steps:
                    worker = self._step_by_id(step_id).get("worker")
                    if worker == "tool_worker":
                        return step_id
                tool_step_id = self._first_step_for_worker("tool_worker")
                if tool_step_id:
                    return tool_step_id
        if not next_steps:
            return ""
        return next_steps[0]

    def _step_by_id(self, step_id: str) -> dict[str, Any]:
        assert self.state is not None
        for step in list(dict(self.state.get("workflow") or {}).get("steps") or []):
            if step.get("id") == step_id:
                return dict(step)
        return {}

    def _first_step_for_worker(self, worker_name: str) -> str:
        assert self.state is not None
        for step in list(dict(self.state.get("workflow") or {}).get("steps") or []):
            if str(step.get("worker", "") or "") == worker_name:
                return str(step.get("id", "") or "")
        return ""

    def _advance(self) -> None:
        assert self.state is not None
        max_kernel_steps = int(self.config.get("max_kernel_steps", 12) or 12)
        breaker_limit = int(self.config.get("max_no_progress", 2) or 2)
        execution = dict(self.state.get("execution") or {})
        circuit_breaker = dict(self.state.get("circuit_breaker") or {})

        for _ in range(max_kernel_steps):
            if bool(dict(self.state.get("shared_data") or {}).get("finish", False)):
                execution["status"] = "completed"
                self.state["execution"] = execution
                self.finished = True
                return

            current_step_id = str(execution.get("current_step_id", "") or "")
            if not current_step_id:
                self._mark_circuit_breaker("workflow_exhausted_without_finish")
                self.finished = True
                return

            step = self._step_by_id(current_step_id)
            execution["current_step"] = dict(step)
            worker_name = str(step.get("worker", "") or "")
            worker = self.workers.get(worker_name)
            if worker is None:
                self._mark_circuit_breaker(f"unknown_worker:{worker_name}")
                self.finished = True
                return

            self._prepare_state_for_worker(worker_name)

            attempts = dict(execution.get("step_attempts") or {})
            attempts[current_step_id] = int(attempts.get(current_step_id, 0) or 0) + 1
            execution["step_attempts"] = attempts
            execution["active_worker"] = worker_name
            execution["last_worker"] = worker_name
            worker_input_state = self._build_worker_input_state(worker_name)
            execution["last_worker_input"] = worker_input_state
            worker_input_history = list(execution.get("worker_input_history") or [])
            worker_input_history.append({"worker": worker_name, "input": worker_input_state})
            execution["worker_input_history"] = worker_input_history[-10:]
            self.state["execution"] = execution

            runtime = WorkerRuntime(
                state=worker_input_state,
                tools_by_name=self.tools_by_name,
                tool_executor=self.tool_executor if worker.spec.can_use_tools else None,
            )
            patch = worker.run(runtime)
            execution["last_patch"] = list(patch)
            allowed_paths = self._allowed_patch_paths(step)
            changed, patch_error = self._apply_worker_patch(patch, allowed_paths)
            execution["last_patch_error"] = patch_error

            if patch_error:
                circuit_breaker["retry_count"] = int(circuit_breaker.get("retry_count", 0) or 0) + 1
                circuit_breaker["last_error"] = patch_error
                self.state["circuit_breaker"] = circuit_breaker
                self._mark_circuit_breaker(patch_error)
                self.finished = True
                return

            if changed:
                completed = list(execution.get("completed_steps") or [])
                if current_step_id not in completed:
                    completed.append(current_step_id)
                execution["completed_steps"] = completed
                circuit_breaker["no_progress_count"] = 0
            else:
                circuit_breaker["no_progress_count"] = int(circuit_breaker.get("no_progress_count", 0) or 0) + 1
                if int(circuit_breaker["no_progress_count"]) > breaker_limit:
                    self.state["circuit_breaker"] = circuit_breaker
                    self._mark_circuit_breaker("no_progress_threshold_exceeded")
                    self.finished = True
                    return

            next_step_id = self._choose_next_step(step)
            execution["current_step_id"] = next_step_id
            execution["current_step"] = self._step_by_id(next_step_id) if next_step_id else {}
            self.state["execution"] = execution
            self.state["circuit_breaker"] = circuit_breaker

        self._mark_circuit_breaker("max_kernel_steps_exceeded")
        self.finished = True

    def step(self, turn_input: TurnInput) -> TurnOutput:
        self._ensure_initialized(turn_input)
        self._advance()
        assert self.state is not None
        shared_data = dict(self.state.get("shared_data") or {})
        execution = dict(self.state.get("execution") or {})
        breaker = dict(self.state.get("circuit_breaker") or {})
        metadata = {
            "kernel_status": str(execution.get("status", "") or ""),
            "current_step_id": str(execution.get("current_step_id", "") or ""),
            "current_step": dict(execution.get("current_step") or {}),
            "completed_steps": list(execution.get("completed_steps") or []),
            "active_worker": str(execution.get("active_worker", "") or ""),
            "last_patch_error": str(execution.get("last_patch_error", "") or ""),
            "circuit_breaker": dict(breaker),
            "warnings": list(execution.get("warnings") or []),
            "fallback_reason": str(execution.get("fallback_reason", "") or ""),
            "used_fallback_architect": bool(execution.get("used_fallback_architect", False)),
            "architect_debug": dict(execution.get("architect_debug") or {}),
        }
        if bool(shared_data.get("finish", False)):
            actions = []
            response_text = str(shared_data.get("assistant_response", "") or "").strip()
            if response_text:
                actions.append(
                    EmitMessage(
                        Message(
                            role="assistant",
                            content=response_text,
                            metadata={
                                "finish_reason": str(shared_data.get("finish_reason", "") or ""),
                                "warnings": list(execution.get("warnings") or []),
                                "used_fallback_architect": bool(execution.get("used_fallback_architect", False)),
                                "architect_debug": dict(execution.get("architect_debug") or {}),
                            },
                        )
                    )
                )
            actions.append(
                Finish(
                    reason=str(shared_data.get("finish_reason", "") or ""),
                    metadata=metadata,
                )
            )
            return TurnOutput(actions=tuple(actions), metadata=metadata)

        return TurnOutput.emit_text(
            "The AppWorld blackboard session did not reach a terminal state.",
            metadata=metadata,
        )


class AppWorldBlackboardSystem(MultiAgentSystem):
    """Factory for AppWorld blackboard sessions."""

    def __init__(self, llm: Any | None = None, *, config: dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.config = dict(config or {})

    def create_session(
        self,
        task: TaskSpec,
        tools: tuple[ToolSpec, ...] = (),
        native_tools: tuple[Any, ...] = (),
    ) -> MultiAgentSession:
        tool_executor = None
        for native in native_tools:
            if isinstance(native, AppWorldApiRuntime) or hasattr(native, "execute_tool_call"):
                tool_executor = native
                break
        return AppWorldBlackboardSession(
            task=task,
            tools=tools,
            tool_executor=tool_executor,
            llm=self.llm,
            config=self.config,
        )
