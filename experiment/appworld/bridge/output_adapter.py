"""Output conversion helpers for AppWorld neutral tool execution."""
from __future__ import annotations

from dataclasses import dataclass

from experiment.common.neutral import ToolCall


def split_tool_name(tool_name: str) -> tuple[str, str]:
    """Split `app__api` tool names into AppWorld app/api parts."""
    app_name, separator, api_name = str(tool_name or "").partition("__")
    if not separator or not app_name or not api_name:
        raise ValueError(
            f"Invalid AppWorld tool name {tool_name!r}. Expected the form 'app__api_name'."
        )
    return app_name, api_name


@dataclass(frozen=True, slots=True)
class AppWorldToolRequest:
    """AppWorld-ready view of a neutral tool call."""

    tool_name: str
    app_name: str
    api_name: str
    arguments: dict[str, object]
    call_id: str = ""
    rationale: str = ""
    metadata: dict[str, object] | None = None


def tool_call_to_request(tool_call: ToolCall) -> AppWorldToolRequest:
    """Convert a neutral tool call into an AppWorld request payload."""
    app_name, api_name = split_tool_name(tool_call.tool_name)
    return AppWorldToolRequest(
        tool_name=tool_call.tool_name,
        app_name=app_name,
        api_name=api_name,
        arguments=dict(tool_call.arguments),
        call_id=tool_call.call_id,
        rationale=tool_call.rationale,
        metadata=dict(tool_call.metadata),
    )
