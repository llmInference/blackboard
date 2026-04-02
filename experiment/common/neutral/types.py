"""Benchmark-neutral protocol types shared across experiment integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias

JSONDict: TypeAlias = dict[str, Any]
MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool", "environment"]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Benchmark-neutral description of one task."""

    task_id: str
    instruction: str
    domain: str = ""
    title: str = ""
    policy: str = ""
    context: tuple[str, ...] = ()
    metadata: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Neutral tool definition exposed to a multi-agent system."""

    name: str
    description: str = ""
    parameters_json_schema: JSONDict = field(default_factory=dict)
    returns_json_schema: JSONDict = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized conversation message."""

    role: MessageRole
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    metadata: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    """Non-conversational environment feedback for the current turn."""

    source: str
    content: str = ""
    payload: JSONDict = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Neutral request to invoke one tool."""

    tool_name: str
    arguments: JSONDict = field(default_factory=dict)
    call_id: str = ""
    rationale: str = ""
    metadata: JSONDict = field(default_factory=dict)


class ToolResultStatus(str, Enum):
    """Execution status of a tool call."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Neutral result returned from one tool invocation."""

    tool_name: str
    call_id: str = ""
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    content: str = ""
    payload: JSONDict = field(default_factory=dict)
    error_message: str = ""
    metadata: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmitMessage:
    """Action instructing the adapter to emit one assistant message."""

    message: Message


@dataclass(frozen=True, slots=True)
class RequestToolCall:
    """Action instructing the adapter to execute one tool call."""

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class Finish:
    """Action instructing the adapter to stop the current episode or turn loop."""

    reason: str = ""
    metadata: JSONDict = field(default_factory=dict)


AgentAction: TypeAlias = EmitMessage | RequestToolCall | Finish


@dataclass(frozen=True, slots=True)
class TurnInput:
    """All neutral information a system needs to decide the next step."""

    task: TaskSpec
    message_history: tuple[Message, ...] = ()
    available_tools: tuple[ToolSpec, ...] = ()
    new_observations: tuple[Observation, ...] = ()
    new_tool_results: tuple[ToolResult, ...] = ()
    step_index: int = 0
    max_steps: int | None = None
    metadata: JSONDict = field(default_factory=dict)

    @property
    def latest_message(self) -> Message | None:
        """Return the latest conversation message when available."""
        if not self.message_history:
            return None
        return self.message_history[-1]


@dataclass(frozen=True, slots=True)
class TurnOutput:
    """Neutral decision emitted by a multi-agent system for one turn."""

    actions: tuple[AgentAction, ...] = ()
    state_delta: JSONDict = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)

    @property
    def emitted_messages(self) -> tuple[Message, ...]:
        """Return assistant messages requested by the system."""
        return tuple(
            action.message for action in self.actions if isinstance(action, EmitMessage)
        )

    @property
    def requested_tool_calls(self) -> tuple[ToolCall, ...]:
        """Return requested tool calls for this turn."""
        return tuple(
            action.tool_call
            for action in self.actions
            if isinstance(action, RequestToolCall)
        )

    @property
    def finished(self) -> bool:
        """Return whether the system wants to terminate the session."""
        return any(isinstance(action, Finish) for action in self.actions)

    @classmethod
    def emit_text(
        cls,
        content: str,
        *,
        metadata: JSONDict | None = None,
        message_metadata: JSONDict | None = None,
    ) -> "TurnOutput":
        """Convenience constructor for a plain assistant text response."""
        return cls(
            actions=(
                EmitMessage(
                    Message(
                        role="assistant",
                        content=content,
                        metadata=dict(message_metadata or {}),
                    )
                ),
            ),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def request_tool(
        cls,
        tool_name: str,
        arguments: JSONDict | None = None,
        *,
        call_id: str = "",
        rationale: str = "",
        metadata: JSONDict | None = None,
    ) -> "TurnOutput":
        """Convenience constructor for a single tool request."""
        return cls(
            actions=(
                RequestToolCall(
                    ToolCall(
                        tool_name=tool_name,
                        arguments=dict(arguments or {}),
                        call_id=call_id,
                        rationale=rationale,
                    )
                ),
            ),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def finish_turn(
        cls,
        reason: str = "",
        *,
        metadata: JSONDict | None = None,
    ) -> "TurnOutput":
        """Convenience constructor for a finish action."""
        return cls(
            actions=(Finish(reason=reason, metadata=dict(metadata or {})),),
        )
