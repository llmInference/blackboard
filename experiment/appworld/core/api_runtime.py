"""Runtime execution helpers for AppWorld API-backed tool calls."""
from __future__ import annotations

import json
from typing import Any

from experiment.appworld.bridge.output_adapter import AppWorldToolRequest, tool_call_to_request
from experiment.common.neutral import ToolCall, ToolResult, ToolResultStatus


def _response_payload_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    return {"response": response}


def _response_payload_to_text(response: Any) -> str:
    if isinstance(response, (dict, list)):
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    return str(response)


class AppWorldApiRuntime:
    """Execute AppWorld API requests through a world's requester object."""

    def __init__(self, requester: Any) -> None:
        self.requester = requester

    def execute(self, request: AppWorldToolRequest) -> ToolResult:
        """Run one prepared AppWorld API request and normalize the result."""
        try:
            response = self.requester.request(
                request.app_name,
                request.api_name,
                **dict(request.arguments),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=request.tool_name,
                call_id=request.call_id,
                status=ToolResultStatus.ERROR,
                content=str(exc),
                error_message=str(exc),
                metadata={
                    "app_name": request.app_name,
                    "api_name": request.api_name,
                },
            )

        return ToolResult(
            tool_name=request.tool_name,
            call_id=request.call_id,
            status=ToolResultStatus.SUCCESS,
            content=_response_payload_to_text(response),
            payload=_response_payload_to_dict(response),
            metadata={
                "app_name": request.app_name,
                "api_name": request.api_name,
            },
        )

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a neutral tool call directly."""
        return self.execute(tool_call_to_request(tool_call))
