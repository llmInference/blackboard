"""Tool conversion helpers for the WebArena neutral bridge."""
from __future__ import annotations

from typing import Any

from experiment.common.neutral import ToolSpec


def _tool_spec(
    *,
    name: str,
    description: str,
    required: list[str] | None = None,
    properties: dict[str, Any] | None = None,
    mutates_state: bool = True,
    action_kind: str = "",
) -> ToolSpec:
    parameters_json_schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties or {}),
    }
    if required:
        parameters_json_schema["required"] = list(required)

    return ToolSpec(
        name=name,
        description=description,
        parameters_json_schema=parameters_json_schema,
        returns_json_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "observation": {"type": "object"},
            },
        },
        metadata={
            "tool_family": "browser",
            "mutates_state": bool(mutates_state),
            "action_kind": action_kind or name.removeprefix("browser__"),
        },
    )


def browser_tools_to_specs() -> tuple[ToolSpec, ...]:
    """Return the minimal explicit browser tool surface for WebArena."""
    return (
        _tool_spec(
            name="browser__goto",
            description="Navigate the active tab to a target URL.",
            required=["url"],
            properties={"url": {"type": "string"}},
            action_kind="goto",
        ),
        _tool_spec(
            name="browser__click",
            description="Click an interactive element on the current page.",
            required=["element_id"],
            properties={"element_id": {"type": "string"}},
            action_kind="click",
        ),
        _tool_spec(
            name="browser__type",
            description="Type text into an input element on the current page.",
            required=["element_id", "text"],
            properties={
                "element_id": {"type": "string"},
                "text": {"type": "string"},
                "clear_first": {"type": "boolean"},
            },
            action_kind="type",
        ),
        _tool_spec(
            name="browser__select_option",
            description="Select an option in a dropdown or select element.",
            required=["element_id", "value"],
            properties={
                "element_id": {"type": "string"},
                "value": {"type": "string"},
            },
            action_kind="select_option",
        ),
        _tool_spec(
            name="browser__press",
            description="Send a keyboard key press to the active page or element.",
            required=["key"],
            properties={
                "key": {"type": "string"},
                "element_id": {"type": "string"},
            },
            action_kind="press",
        ),
        _tool_spec(
            name="browser__scroll",
            description="Scroll the active page by a specified offset.",
            properties={
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            action_kind="scroll",
        ),
        _tool_spec(
            name="browser__go_back",
            description="Navigate back in the active tab history.",
            mutates_state=False,
            action_kind="go_back",
        ),
        _tool_spec(
            name="browser__new_tab",
            description="Open a new browser tab, optionally at a URL.",
            properties={"url": {"type": "string"}},
            action_kind="new_tab",
        ),
        _tool_spec(
            name="browser__switch_tab",
            description="Switch to an existing browser tab by index.",
            required=["tab_index"],
            properties={"tab_index": {"type": "integer"}},
            mutates_state=False,
            action_kind="switch_tab",
        ),
        _tool_spec(
            name="browser__close_tab",
            description="Close a browser tab by index or the active tab if omitted.",
            properties={"tab_index": {"type": "integer"}},
            action_kind="close_tab",
        ),
        _tool_spec(
            name="browser__finish",
            description="Mark browser-side execution as complete, optionally attaching a final response.",
            properties={
                "response": {"type": "string"},
                "stop_reason": {"type": "string"},
            },
            mutates_state=False,
            action_kind="finish",
        ),
    )
