"""Base worker primitives for the WebArena blackboard runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any

from experiment.common.neutral import Message, ToolCall, ToolSpec


PatchOp = dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Static description of one WebArena worker."""

    name: str
    description: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    is_llm: bool = False
    can_use_tools: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "is_llm": self.is_llm,
            "can_use_tools": self.can_use_tools,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """One bounded worker execution result."""

    patch: tuple[PatchOp, ...] = ()
    tool_call: ToolCall | None = None
    message: Message | None = None
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return bool(str(self.finish_reason or "").strip())


@dataclass(slots=True)
class WorkerRuntime:
    """Task-scoped worker runtime view."""

    state: dict[str, Any]
    tools_by_name: dict[str, ToolSpec]


class WebArenaWorker(ABC):
    """Capability worker operating on filtered blackboard state."""

    spec: WorkerSpec

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None and bool(self.spec.is_llm)

    @staticmethod
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

    def _record_token_usage(self, response: Any) -> dict[str, int]:
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return {
            "worker_input_tokens": prompt_tokens,
            "worker_output_tokens": completion_tokens,
        }

    def _invoke_json(self, *, system_prompt: str, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        if self._llm is None:
            raise ValueError(f"{self.spec.name} received no llm instance")
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(context, ensure_ascii=False, indent=2)),
            ]
        )
        raw_text = self._normalize_response_content(getattr(response, "content", ""))
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError(f"{self.spec.name} expected a JSON object response")
        return payload, self._record_token_usage(response)

    @abstractmethod
    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        raise NotImplementedError


def replace(path: str, value: Any) -> PatchOp:
    return {"op": "replace", "path": path, "value": value}
