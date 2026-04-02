"""Output conversion helpers for the WebArena neutral bridge."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiment.common.neutral import Observation, ToolResult, ToolResultStatus


def _text_excerpt(value: Any, *, limit: int = 400) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_tabs(raw_tabs: Any) -> list[dict[str, Any]]:
    tabs: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw_tabs or [])):
        if isinstance(item, Mapping):
            tabs.append(
                {
                    "index": int(item.get("index", index)),
                    "url": str(item.get("url", "") or ""),
                    "title": str(item.get("title", "") or ""),
                    "active": bool(item.get("active", False)),
                }
            )
        else:
            tabs.append({"index": index, "url": str(item or ""), "title": "", "active": index == 0})
    return tabs


def _normalize_elements(raw_elements: Any) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for item in list(raw_elements or []):
        if isinstance(item, Mapping):
            elements.append(
                {
                    "element_id": str(
                        item.get("element_id")
                        or item.get("id")
                        or item.get("bid")
                        or ""
                    ),
                    "tag": str(item.get("tag", "") or ""),
                    "role": str(item.get("role", "") or ""),
                    "text": _text_excerpt(item.get("text") or item.get("label") or "", limit=160),
                    "enabled": bool(item.get("enabled", True)),
                    "visible": bool(item.get("visible", True)),
                }
            )
    return elements


def standardize_browser_observation(raw_observation: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a browser observation payload into a stable prompt-friendly schema."""
    raw_observation = dict(raw_observation or {})
    return {
        "url": str(raw_observation.get("url", "") or ""),
        "page_title": str(raw_observation.get("page_title") or raw_observation.get("title") or ""),
        "visible_text": _text_excerpt(
            raw_observation.get("visible_text")
            or raw_observation.get("text")
            or raw_observation.get("content")
            or "",
            limit=2000,
        ),
        "interactive_elements": _normalize_elements(
            raw_observation.get("interactive_elements") or raw_observation.get("elements")
        ),
        "open_tabs": _normalize_tabs(raw_observation.get("open_tabs") or raw_observation.get("tabs")),
        "active_tab_index": int(raw_observation.get("active_tab_index", 0) or 0),
        "last_action": dict(raw_observation.get("last_action") or {}),
        "raw": raw_observation,
    }


def observation_to_neutral(
    raw_observation: Mapping[str, Any] | None,
    *,
    source: str = "webarena_browser",
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Convert a runtime observation into the shared neutral Observation type."""
    standardized = standardize_browser_observation(raw_observation)
    summary_parts = [
        f"URL: {standardized['url']}" if standardized["url"] else "",
        f"Title: {standardized['page_title']}" if standardized["page_title"] else "",
        standardized["visible_text"],
    ]
    content = "\n".join(part for part in summary_parts if part)
    return Observation(
        source=source,
        content=content,
        payload=standardized,
        metadata=dict(metadata or {}),
    )


def browser_result_to_tool_result(
    tool_name: str,
    raw_result: Mapping[str, Any] | None,
    *,
    call_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """Convert one browser action result into a neutral tool result."""
    raw_result = dict(raw_result or {})
    error_message = str(raw_result.get("error_message") or raw_result.get("error") or "").strip()
    status = ToolResultStatus.ERROR if error_message else ToolResultStatus.SUCCESS
    standardized_observation = standardize_browser_observation(raw_result.get("observation"))

    content_parts = [
        f"tool={tool_name}",
        f"status={status.value}",
    ]
    if standardized_observation.get("url"):
        content_parts.append(f"url={standardized_observation['url']}")
    if standardized_observation.get("page_title"):
        content_parts.append(f"title={standardized_observation['page_title']}")
    if error_message:
        content_parts.append(f"error={error_message}")

    return ToolResult(
        tool_name=tool_name,
        call_id=call_id,
        status=status,
        content=" | ".join(content_parts),
        payload={
            "observation": standardized_observation,
            "reward": raw_result.get("reward"),
            "done": bool(raw_result.get("done", False)),
            "raw_result": raw_result,
        },
        error_message=error_message,
        metadata=dict(metadata or {}),
    )


def adapt_browser_result(
    tool_name: str,
    raw_result: Mapping[str, Any] | None,
    *,
    call_id: str = "",
    metadata: dict[str, Any] | None = None,
    observation_source: str = "webarena_browser",
) -> tuple[ToolResult, Observation]:
    """Convert one runtime browser result into neutral tool + observation outputs."""
    result = browser_result_to_tool_result(
        tool_name,
        raw_result,
        call_id=call_id,
        metadata=metadata,
    )
    observation = observation_to_neutral(
        dict(raw_result or {}).get("observation"),
        source=observation_source,
        metadata={
            **dict(metadata or {}),
            "tool_name": tool_name,
            "call_id": call_id,
        },
    )
    return result, observation
