"""Thin WebArena task runner and neutral turn-loop wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from experiment.common.neutral import (
    Message,
    MultiAgentSystem,
    Observation,
    TaskSpec,
    ToolResult,
    ToolSpec,
    TurnInput,
)
from experiment.webarena.bridge import browser_tools_to_specs, task_to_spec
from experiment.webarena.core.browser_runtime import WebArenaBrowserRuntime, default_ui_login
from experiment.webarena.core.target_urls import extract_expected_target_urls


@dataclass(frozen=True, slots=True)
class WebArenaRunResult:
    """Neutralized output of one generic WebArena session run."""

    task: TaskSpec
    messages: tuple[Message, ...]
    tool_results: tuple[ToolResult, ...]
    observations: tuple[Observation, ...]
    evaluation: dict[str, Any]
    finished: bool
    steps: int
    run_metadata: dict[str, Any]


class WebArenaTaskRunner:
    """Wrap one WebArena task as neutral task/tool/runtime objects."""

    def __init__(
        self,
        *,
        task_id: int,
        config_path: str | Path,
        output_dir: str | Path,
        task_config: dict[str, Any] | None = None,
        dataset_name: str = "webarena-verified",
        headless: bool = True,
        slow_mo_ms: int = 0,
        webarena_factory: Callable[..., Any] | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        ui_login_func: Any | None = None,
    ) -> None:
        self.task_id = int(task_id)
        self.config_path = Path(config_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_config = dict(task_config or {})
        self.dataset_name = str(dataset_name or "webarena-verified")
        self.headless = bool(headless)
        self.slow_mo_ms = int(slow_mo_ms)
        self.webarena_factory = webarena_factory
        self.runtime_factory = runtime_factory
        self.ui_login_func = ui_login_func

        self.webarena: Any | None = None
        self.task: Any | None = None
        self.runtime: Any | None = None
        self._task_spec: TaskSpec | None = None
        self._tool_specs: tuple[ToolSpec, ...] | None = None

    def open(self) -> "WebArenaTaskRunner":
        """Initialize the underlying WebArena runtime if needed."""
        if self.runtime is not None:
            return self

        if self.webarena_factory is None:
            from webarena_verified.api import WebArenaVerified

            self.webarena_factory = WebArenaVerified
            if self.ui_login_func is None:
                self.ui_login_func = default_ui_login

        self.webarena = self.webarena_factory(config=self.config_path)
        self.task = self.webarena.get_task(self.task_id)

        if self.task_config:
            task_config = dict(self.task_config)
        elif hasattr(self.task, "model_dump"):
            task_config = dict(self.task.model_dump(mode="json"))
        else:
            task_config = {}

        self._task_spec = task_to_spec(
            task_config,
            config_path=str(self.config_path),
            dataset_name=self.dataset_name,
            config_template=str(self.config_path),
        )
        rendered_target_urls = extract_expected_target_urls(
            list(task_config.get("eval", []) or []),
            render_urls=lambda urls: self.webarena.config.render_url(urls, self.task.sites),
        )
        if rendered_target_urls:
            task_metadata = dict(self._task_spec.metadata)
            task_metadata["rendered_target_urls"] = rendered_target_urls
            self._task_spec = TaskSpec(
                task_id=self._task_spec.task_id,
                instruction=self._task_spec.instruction,
                domain=self._task_spec.domain,
                title=self._task_spec.title,
                policy=self._task_spec.policy,
                context=self._task_spec.context,
                metadata=task_metadata,
            )
        self._tool_specs = browser_tools_to_specs()

        runtime_factory = self.runtime_factory or WebArenaBrowserRuntime
        self.runtime = runtime_factory(
            task=self.task,
            config=self.webarena.config,
            evaluator=self.webarena,
            output_dir=self.output_dir,
            headless=self.headless,
            slow_mo_ms=self.slow_mo_ms,
            ui_login_func=self.ui_login_func,
        )
        self.runtime.open()
        return self

    @property
    def task_spec(self) -> TaskSpec:
        self.open()
        assert self._task_spec is not None
        return self._task_spec

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        self.open()
        assert self._tool_specs is not None
        return self._tool_specs

    def build_turn_input(
        self,
        *,
        message_history: tuple[Message, ...] = (),
        new_observations: tuple[Observation, ...] = (),
        new_tool_results: tuple[ToolResult, ...] = (),
        step_index: int = 0,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnInput:
        """Build one neutral turn snapshot from the current WebArena task."""
        return TurnInput(
            task=self.task_spec,
            message_history=message_history,
            available_tools=self.tool_specs,
            new_observations=new_observations,
            new_tool_results=new_tool_results,
            step_index=step_index,
            max_steps=max_steps,
            metadata={
                "dataset_name": self.dataset_name,
                "webarena_task_id": self.task_id,
                **dict(metadata or {}),
            },
        )

    def execute_tool_call(self, tool_call: Any) -> tuple[ToolResult, Observation]:
        self.open()
        return self.runtime.execute_tool_call(tool_call)

    def evaluate(self, *, agent_response: Any | None = None) -> dict[str, Any]:
        self.open()
        return dict(self.runtime.evaluate(agent_response=agent_response))

    def _default_agent_response(
        self,
        *,
        observations: list[Observation],
        run_metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        task = self.task_spec
        if bool(task.metadata.get("requires_final_response", False)):
            return None

        category = str(task.metadata.get("task_category", "") or "").strip().lower()
        task_type = "NAVIGATE" if category == "navigation" else "MUTATE"
        rendered_targets = {
            str(url).rstrip("/")
            for url in list(task.metadata.get("rendered_target_urls", []) or [])
            if str(url).strip()
        }
        last_url = ""
        if observations:
            last_url = str(observations[-1].payload.get("url", "") or "").rstrip("/")

        finish_reason = str(run_metadata.get("finish_reason", "") or "")
        success = bool(last_url and last_url in rendered_targets) or finish_reason == "task_completed"
        status = "SUCCESS" if success else "UNKNOWN_ERROR"
        response = {
            "task_type": task_type,
            "status": status,
            "retrieved_data": None,
        }
        if not success:
            response["error_details"] = (
                f"Expected one of {sorted(rendered_targets)} but the final URL was {last_url or '<empty>'}."
            )
        return response

    def run_session(
        self,
        system: MultiAgentSystem,
        *,
        max_steps: int = 20,
        metadata: dict[str, Any] | None = None,
    ) -> WebArenaRunResult:
        """Drive a generic neutral system over one WebArena task."""
        self.open()
        native_tools: tuple[Any, ...] = (self.runtime,)
        session = system.create_session(self.task_spec, self.tool_specs, native_tools=native_tools)
        message_history: list[Message] = []
        tool_results: list[ToolResult] = []
        observations: list[Observation] = []
        pending_tool_results: tuple[ToolResult, ...] = ()
        pending_observations: tuple[Observation, ...] = ()
        finished = False
        steps_taken = 0
        run_metadata: dict[str, Any] = {}

        try:
            for step_index in range(max_steps):
                steps_taken = step_index + 1
                turn_input = self.build_turn_input(
                    message_history=tuple(message_history),
                    new_observations=pending_observations,
                    new_tool_results=pending_tool_results,
                    step_index=step_index,
                    max_steps=max_steps,
                    metadata=metadata,
                )
                pending_tool_results = ()
                pending_observations = ()
                turn_output = session.step(turn_input)
                run_metadata = dict(turn_output.metadata)

                for message in turn_output.emitted_messages:
                    message_history.append(message)

                if turn_output.requested_tool_calls:
                    result_batch: list[ToolResult] = []
                    observation_batch: list[Observation] = []
                    for tool_call in turn_output.requested_tool_calls:
                        result, observation = self.execute_tool_call(tool_call)
                        result_batch.append(result)
                        observation_batch.append(observation)
                        tool_results.append(result)
                        observations.append(observation)
                        message_history.append(
                            Message(
                                role="tool",
                                name=result.tool_name,
                                tool_call_id=result.call_id,
                                content=result.content,
                                metadata=dict(result.metadata),
                            )
                        )
                    pending_tool_results = tuple(result_batch)
                    pending_observations = tuple(observation_batch)

                if turn_output.finished:
                    finished = True
                    break
                if not turn_output.actions:
                    break
        finally:
            system.close_session(session)

        agent_response: Any | None = None
        for message in reversed(message_history):
            if message.role == "assistant" and str(message.content or "").strip():
                agent_response = str(message.content)
                break
        if agent_response is None:
            agent_response = self._default_agent_response(
                observations=observations,
                run_metadata=run_metadata,
            )

        evaluation = self.evaluate(agent_response=agent_response)

        return WebArenaRunResult(
            task=self.task_spec,
            messages=tuple(message_history),
            tool_results=tuple(tool_results),
            observations=tuple(observations),
            evaluation=evaluation,
            finished=finished,
            steps=steps_taken,
            run_metadata=run_metadata,
        )

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.webarena = None
        self.task = None
        self._task_spec = None
        self._tool_specs = None

    def __enter__(self) -> "WebArenaTaskRunner":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
