"""Thin AppWorld task runner and neutral turn-loop wrapper."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from appworld import AppWorld, update_root

from experiment.appworld.bridge import task_to_spec, tool_call_to_request, tools_to_specs
from experiment.appworld.core.api_runtime import AppWorldApiRuntime
from experiment.common.neutral import (
    Message,
    MultiAgentSystem,
    TaskSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnInput,
)


@dataclass(frozen=True, slots=True)
class AppWorldRunResult:
    """Neutralized output of one generic AppWorld session run."""

    task: TaskSpec
    messages: tuple[Message, ...]
    tool_results: tuple[ToolResult, ...]
    evaluation: dict[str, Any]
    finished: bool
    steps: int
    run_metadata: dict[str, Any]


class AppWorldTaskRunner:
    """Wrap one AppWorld task as neutral task/tool/runtime objects."""

    def __init__(
        self,
        *,
        task_id: str,
        experiment_name: str,
        dataset_name: str = "",
        remote_apis_url: str | None = None,
        appworld_root: str | None = None,
        load_ground_truth: bool = True,
        ground_truth_mode: str = "minimal",
        world_factory: Callable[..., AppWorld] | None = None,
        world_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.remote_apis_url = remote_apis_url
        self.appworld_root = appworld_root
        self.load_ground_truth = bool(load_ground_truth)
        self.ground_truth_mode = str(ground_truth_mode or "minimal")
        self.world_factory = world_factory or AppWorld
        self.world_kwargs = dict(world_kwargs or {})
        self.world: AppWorld | None = None
        self.runtime: AppWorldApiRuntime | None = None
        self._task_spec: TaskSpec | None = None
        self._tool_specs: tuple[ToolSpec, ...] | None = None

    def _load_oracle_required_apis(self) -> list[str]:
        if not self.appworld_root:
            return []
        required_apis_path = (
            Path(self.appworld_root)
            / "data"
            / "tasks"
            / self.task_id
            / "ground_truth"
            / "required_apis.json"
        )
        if not required_apis_path.exists():
            return []
        try:
            payload = json.loads(required_apis_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload if str(item).strip()]

    def open(self) -> "AppWorldTaskRunner":
        """Initialize the underlying AppWorld world if needed."""
        if self.world is not None:
            return self
        if self.appworld_root:
            update_root(self.appworld_root)
        world_kwargs = dict(self.world_kwargs)
        world_kwargs.setdefault("task_id", self.task_id)
        world_kwargs.setdefault("experiment_name", self.experiment_name)
        world_kwargs.setdefault("load_ground_truth", self.load_ground_truth)
        world_kwargs.setdefault("ground_truth_mode", self.ground_truth_mode)
        if self.remote_apis_url:
            world_kwargs.setdefault("remote_apis_url", self.remote_apis_url)
        self.world = self.world_factory(**world_kwargs)
        self.runtime = AppWorldApiRuntime(self.world.requester)
        task_metadata = {
            "oracle_required_apis": self._load_oracle_required_apis(),
        }
        self._task_spec = task_to_spec(
            self.world.task,
            dataset_name=self.dataset_name,
            task_metadata=task_metadata,
        )
        self._tool_specs = tools_to_specs(self.world.task.api_docs.keep_apps(self.world.task.allowed_apps))
        return self

    @property
    def task_spec(self) -> TaskSpec:
        """Return the neutral task spec for the current world."""
        self.open()
        assert self._task_spec is not None
        return self._task_spec

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        """Return the neutral tool registry for the current world."""
        self.open()
        assert self._tool_specs is not None
        return self._tool_specs

    def build_turn_input(
        self,
        *,
        message_history: tuple[Message, ...] = (),
        new_tool_results: tuple[ToolResult, ...] = (),
        step_index: int = 0,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnInput:
        """Build one neutral turn snapshot from the current AppWorld task."""
        return TurnInput(
            task=self.task_spec,
            message_history=message_history,
            available_tools=self.tool_specs,
            new_tool_results=new_tool_results,
            step_index=step_index,
            max_steps=max_steps,
            metadata={
                "dataset_name": self.dataset_name,
                "appworld_task_id": self.task_id,
                **dict(metadata or {}),
            },
        )

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute one neutral tool request against the wrapped AppWorld world."""
        self.open()
        assert self.runtime is not None
        return self.runtime.execute(tool_call_to_request(tool_call))

    def evaluate(self, *, suppress_errors: bool = True) -> dict[str, Any]:
        """Run official AppWorld evaluation and return a serializable dict."""
        self.open()
        assert self.world is not None
        try:
            if hasattr(self.world, "save"):
                self.world.save()
            tracker = self.world.evaluate(suppress_errors=suppress_errors)
        except Exception as exc:
            return {
                "success": False,
                "evaluation_error": str(exc),
            }
        if hasattr(tracker, "to_dict"):
            return dict(tracker.to_dict(stats_only=False))
        return {"raw_evaluation": tracker}

    def run_session(
        self,
        system: MultiAgentSystem,
        *,
        max_steps: int = 20,
        metadata: dict[str, Any] | None = None,
    ) -> AppWorldRunResult:
        """Drive a generic neutral system over one AppWorld task."""
        self.open()
        native_tools: tuple[Any, ...] = ()
        if self.runtime is not None:
            native_tools = (self.runtime,)
        session = system.create_session(self.task_spec, self.tool_specs, native_tools=native_tools)
        message_history: list[Message] = []
        tool_results: list[ToolResult] = []
        pending_tool_results: tuple[ToolResult, ...] = ()
        finished = False
        steps_taken = 0
        run_metadata: dict[str, Any] = {}

        try:
            for step_index in range(max_steps):
                steps_taken = step_index + 1
                turn_input = self.build_turn_input(
                    message_history=tuple(message_history),
                    new_tool_results=pending_tool_results,
                    step_index=step_index,
                    max_steps=max_steps,
                    metadata=metadata,
                )
                pending_tool_results = ()
                turn_output = session.step(turn_input)
                run_metadata = dict(turn_output.metadata)
                for message in turn_output.emitted_messages:
                    message_history.append(message)
                if turn_output.requested_tool_calls:
                    batch: list[ToolResult] = []
                    for tool_call in turn_output.requested_tool_calls:
                        result = self.execute_tool_call(tool_call)
                        batch.append(result)
                        tool_results.append(result)
                        message_history.append(
                            Message(
                                role="tool",
                                name=result.tool_name,
                                tool_call_id=result.call_id,
                                content=result.content,
                                metadata=dict(result.metadata),
                            )
                        )
                    pending_tool_results = tuple(batch)
                if turn_output.finished:
                    finished = True
                    break
                if not turn_output.actions:
                    break
        finally:
            if hasattr(session, "state"):
                state = getattr(session, "state", None) or {}
                execution = dict(state.get("execution") or {})
                shared_data = dict(state.get("shared_data") or {})
                run_metadata = {
                    **run_metadata,
                    "warnings": list(execution.get("warnings") or run_metadata.get("warnings") or []),
                    "fallback_reason": str(
                        execution.get("fallback_reason", "") or run_metadata.get("fallback_reason", "") or ""
                    ),
                    "used_fallback_architect": bool(
                        execution.get("used_fallback_architect", run_metadata.get("used_fallback_architect", False))
                    ),
                    "architect_debug": dict(execution.get("architect_debug") or run_metadata.get("architect_debug") or {}),
                    "last_worker": str(execution.get("last_worker", "") or ""),
                    "history_tail": list(execution.get("history") or [])[-10:],
                    "tool_status": dict(shared_data.get("tool_status") or {}),
                    "progress_state": dict(shared_data.get("progress_state") or {}),
                    "finish_reason": str(shared_data.get("finish_reason", "") or run_metadata.get("finish_reason", "") or ""),
                }
            system.close_session(session)

        return AppWorldRunResult(
            task=self.task_spec,
            messages=tuple(message_history),
            tool_results=tuple(tool_results),
            evaluation=self.evaluate(),
            finished=finished,
            steps=steps_taken,
            run_metadata=run_metadata,
        )

    def close(self) -> None:
        """Release the wrapped AppWorld world."""
        if self.world is None:
            return
        self.world.close()
        self.world = None
        self.runtime = None
        self._task_spec = None
        self._tool_specs = None

    def __enter__(self) -> "AppWorldTaskRunner":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
