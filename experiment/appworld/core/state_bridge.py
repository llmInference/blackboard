"""State bridge between neutral AppWorld turns and the AppWorld blackboard runtime."""
from __future__ import annotations

import json
from typing import Any

from experiment.common.neutral import Message, ToolResult, TurnInput


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "metadata": dict(message.metadata),
    }


def _tool_result_to_dict(tool_result: ToolResult) -> dict[str, Any]:
    return {
        "tool_name": tool_result.tool_name,
        "call_id": tool_result.call_id,
        "status": tool_result.status.value,
        "content": tool_result.content,
        "payload": dict(tool_result.payload),
        "error_message": tool_result.error_message,
        "metadata": dict(tool_result.metadata),
    }


def _tool_spec_to_dict(tool_spec: Any) -> dict[str, Any]:
    return {
        "name": tool_spec.name,
        "description": tool_spec.description,
        "parameters_json_schema": dict(tool_spec.parameters_json_schema),
        "returns_json_schema": dict(tool_spec.returns_json_schema),
        "metadata": dict(tool_spec.metadata),
    }


def _build_task_payload(turn_input: TurnInput) -> dict[str, Any]:
    return {
        "task_id": turn_input.task.task_id,
        "title": turn_input.task.title,
        "instruction": turn_input.task.instruction,
        "domain": turn_input.task.domain,
        "policy": turn_input.task.policy,
        "context": list(turn_input.task.context),
        "metadata": dict(turn_input.task.metadata),
    }


def _build_tools_catalog(turn_input: TurnInput) -> str:
    blocks: list[str] = []
    for tool in turn_input.available_tools:
        blocks.append(
            "\n".join(
                [
                    f"Tool: {tool.name}",
                    f"Description: {tool.description}",
                    "Parameters: "
                    + json.dumps(tool.parameters_json_schema, ensure_ascii=False, sort_keys=True),
                    "Metadata: "
                    + json.dumps(dict(tool.metadata), ensure_ascii=False, sort_keys=True),
                ]
            )
        )
    return "\n\n".join(blocks)


def turn_input_to_blackboard_state(
    turn_input: TurnInput,
    *,
    available_workers: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Convert one neutral turn snapshot into AppWorld blackboard state."""
    task_payload = _build_task_payload(turn_input)
    tool_payloads = [_tool_spec_to_dict(tool) for tool in turn_input.available_tools]
    worker_payloads = [dict(worker) for worker in available_workers]
    return {
        "task": task_payload,
        "workflow": {
            "workflow_id": "",
            "goal": "",
            "steps": [],
        },
        "data_schema": {},
        "shared_data": {
            "candidate_tools": [],
            "auth": {},
            "raw_results": [],
            "entities": [],
            "selected_entities": [],
            "analysis_results": {},
            "evidence": {},
            "intermediate_results": {},
            "progress_state": {},
            "tool_status": {},
            "action_history": [],
            "assistant_response": "",
            "response_confidence": "",
            "finish": False,
            "finish_reason": "",
        },
        "execution": {
            "status": "uninitialized",
            "current_step_id": "",
            "current_step": {},
            "completed_steps": [],
            "active_worker": "",
            "step_attempts": {},
            "last_worker": "",
            "last_worker_input": {},
            "last_patch": [],
            "last_patch_error": "",
            "history": [],
            "worker_input_history": [],
            "warnings": [],
            "fallback_reason": "",
            "used_fallback_architect": False,
            "architect_debug": {
                "raw_output": "",
                "invoke_error": "",
                "parse_error": "",
                "validation_errors": [],
            },
        },
        "circuit_breaker": {
            "retry_count": 0,
            "no_progress_count": 0,
            "last_error": "",
            "tripped": False,
            "reason": "",
        },
        "environment": {
            "step_index": int(turn_input.step_index),
            "max_steps": turn_input.max_steps,
            "message_history": [_message_to_dict(message) for message in turn_input.message_history],
            "new_tool_results": [_tool_result_to_dict(result) for result in turn_input.new_tool_results],
            "tools_catalog": tool_payloads,
            "workers_catalog": worker_payloads,
            "tools_catalog_text": _build_tools_catalog(turn_input),
            "turn_metadata": dict(turn_input.metadata),
        },
    }


def merge_turn_input_into_state(state: dict[str, Any], turn_input: TurnInput) -> dict[str, Any]:
    """Refresh turn-scoped environment data without discarding shared state."""
    environment = dict(state.get("environment") or {})
    environment.update(
        {
            "step_index": int(turn_input.step_index),
            "max_steps": turn_input.max_steps,
            "message_history": [_message_to_dict(message) for message in turn_input.message_history],
            "new_tool_results": [_tool_result_to_dict(result) for result in turn_input.new_tool_results],
            "turn_metadata": dict(turn_input.metadata),
        }
    )
    next_state = dict(state)
    next_state["environment"] = environment
    return next_state
