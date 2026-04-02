"""Base worker primitives for the AppWorld blackboard runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from experiment.common.neutral import ToolCall, ToolSpec


PatchOp = dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Static description of one AppWorld capability worker."""

    name: str
    description: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    can_use_tools: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "can_use_tools": self.can_use_tools,
            "input_schema": dict(self.input_schema),
        }


@dataclass(slots=True)
class WorkerRuntime:
    """Runtime context exposed to one worker invocation."""

    state: dict[str, Any]
    tools_by_name: dict[str, ToolSpec]
    tool_executor: Any | None = None

    def execute_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        rationale: str = "",
    ) -> Any:
        if self.tool_executor is None:
            raise RuntimeError("No tool executor is available for this worker.")
        if not hasattr(self.tool_executor, "execute_tool_call"):
            raise RuntimeError("Tool executor does not support execute_tool_call().")
        return self.tool_executor.execute_tool_call(
            ToolCall(
                tool_name=tool_name,
                arguments=dict(arguments or {}),
                rationale=rationale,
            )
        )


class AppWorldWorker(ABC):
    """Capability worker that reads shared state and emits JSON Patch updates."""

    spec: WorkerSpec

    @abstractmethod
    def run(self, runtime: WorkerRuntime) -> list[PatchOp]:
        raise NotImplementedError


def replace(path: str, value: Any) -> PatchOp:
    return {"op": "replace", "path": path, "value": value}


def add(path: str, value: Any) -> PatchOp:
    return {"op": "add", "path": path, "value": value}
