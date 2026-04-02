"""Task-scoped architect for AppWorld blackboard runs."""
from __future__ import annotations

import json
from typing import Any

from experiment.appworld.workers.base import WorkerSpec


def _stringify_response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    return content


def _truncate_text(value: str, *, limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _architect_debug(
    *,
    raw_output: str = "",
    invoke_error: str = "",
    parse_error: str = "",
    repair_error: str = "",
    repaired_output: str = "",
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "raw_output": _truncate_text(raw_output),
        "invoke_error": str(invoke_error or ""),
        "parse_error": str(parse_error or ""),
        "repair_error": str(repair_error or ""),
        "repaired_output": _truncate_text(repaired_output),
        "validation_errors": [str(item) for item in list(validation_errors or []) if str(item).strip()],
    }


def _extract_json_dict(content: str) -> dict[str, Any]:
    raw = str(content or "").strip()
    if not raw:
        raise ValueError("Architect returned empty content.")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Architect response is not valid JSON: {raw[:200]}")
    candidate = raw[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError(f"Top-level JSON must be an object, got {type(parsed).__name__}.")
    return parsed


def _load_llm_json(llm: Any, prompt: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if llm is None or not hasattr(llm, "invoke"):
        return None, _architect_debug(invoke_error="No architect LLM configured.")
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        return None, _architect_debug(invoke_error=f"{type(exc).__name__}: {exc}")
    content = _stringify_response_content(response)
    try:
        parsed = _extract_json_dict(content)
    except Exception as exc:
        return None, _architect_debug(raw_output=content, parse_error=f"{type(exc).__name__}: {exc}")
    return parsed, _architect_debug(raw_output=content)


class AppWorldArchitect:
    """Generate a task-scoped workflow and shared-state schema."""

    def __init__(self, worker_specs: tuple[WorkerSpec, ...], llm: Any | None = None) -> None:
        self.worker_specs = {worker.name: worker for worker in worker_specs}
        self._llm = llm

    def _schema_entry(
        self,
        *,
        field_type: str,
        required: bool,
        producers: list[str],
        description: str,
    ) -> dict[str, Any]:
        return {
            "type": field_type,
            "required": required,
            "producer": producers,
            "description": description,
        }

    def _iteration_count(self, task: dict[str, Any]) -> int:
        instruction = str(task.get("instruction", "") or "").lower()
        metadata = dict(task.get("metadata") or {})
        allowed_apps = list(metadata.get("allowed_apps") or [])
        if len(allowed_apps) > 2:
            return 3
        if any(token in instruction for token in ("return", "buy", "pay", "send", "update", "change")):
            return 3
        return 2

    def _fallback_data_schema(self, instruction: str) -> dict[str, Any]:
        return {
            "working_memory": self._schema_entry(
                field_type="object",
                required=False,
                producers=["tool_worker", "state_worker", "analysis_worker"],
                description=f"Fallback generic working memory for task: {instruction[:120]}",
            ),
            "schema_extensions": self._schema_entry(
                field_type="object",
                required=False,
                producers=["tool_worker", "state_worker", "analysis_worker", "response_worker"],
                description="Fallback extension area when the architect could not generate a task-specific schema.",
            ),
            "analysis_results": self._schema_entry(
                field_type="object",
                required=False,
                producers=["analysis_worker"],
                description="Fallback task-specific analysis outputs and final answer candidates.",
            ),
            "assistant_response": self._schema_entry(
                field_type="string",
                required=True,
                producers=["response_worker"],
                description=f"Fallback final user-facing answer for: {instruction[:120]}",
            ),
            "finish": self._schema_entry(
                field_type="boolean",
                required=True,
                producers=["response_worker"],
                description="Fallback terminal completion flag.",
            ),
            "finish_reason": self._schema_entry(
                field_type="string",
                required=True,
                producers=["response_worker"],
                description="Why the fallback run finished.",
            ),
        }

    def _worker_instruction(self, worker_name: str) -> str:
        spec = self.worker_specs[worker_name]
        tool_note = " This worker may directly use tools within its bounded role." if spec.can_use_tools else ""
        return (
            f"You are {worker_name}. Your only responsibility is: {spec.description}. "
            f"Read only {', '.join(spec.reads)}. "
            f"Write only {', '.join(spec.writes)}.{tool_note}"
        )

    def _coerce_string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _coerce_next_steps(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, dict):
            ordered_keys = ("on_success", "success", "default", "next", "on_failure", "failure")
            collected: list[str] = []
            for key in ordered_keys:
                item = value.get(key)
                if isinstance(item, str) and item.strip() and item not in collected:
                    collected.append(item)
                elif isinstance(item, list):
                    for child in item:
                        child_text = str(child)
                        if child_text.strip() and child_text not in collected:
                            collected.append(child_text)
            return collected
        return []

    def _coerce_instruction_text(self, value: Any, worker_name: str) -> str:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return "\n".join(parts)
        return self._worker_instruction(worker_name)

    def _fallback_build(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._fallback_with_reason(state)

    def _fallback_with_reason(
        self,
        state: dict[str, Any],
        *,
        reason: str = "",
        architect_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = dict(state.get("task") or {})
        instruction = str(task.get("instruction", "") or "")
        iterations = self._iteration_count(task)
        selected_workers = ["tool_worker", "state_worker", "analysis_worker", "response_worker"]
        warning = (
            "Architect fallback was used because no valid LLM-generated workflow/data schema was available. "
            "The current data schema is a minimal generic fallback, not a task-specific schema."
        )
        fallback_reason = warning
        if str(reason or "").strip():
            fallback_reason = f"{warning} Cause: {str(reason).strip()}"
        debug_payload = dict(architect_debug or {})

        workflow_steps: list[dict[str, Any]] = []
        step_index = 1
        for _ in range(iterations):
            workflow_steps.append(
                {
                    "id": f"step_{step_index}",
                    "worker": "tool_worker",
                    "purpose": self.worker_specs["tool_worker"].description,
                    "api_goal": "Make task-relevant AppWorld tool progress for the current subgoal.",
                    "preferred_tools": [],
                    "expected_outputs": ["raw_results", "tool_status", "auth"],
                    "reads": list(self.worker_specs["tool_worker"].reads),
                    "writes": list(self.worker_specs["tool_worker"].writes),
                    "exit_conditions": ["A bounded tool sequence was executed or skipped."],
                    "next": [f"step_{step_index + 1}"],
                }
            )
            step_index += 1
            workflow_steps.append(
                {
                    "id": f"step_{step_index}",
                    "worker": "state_worker",
                    "purpose": self.worker_specs["state_worker"].description,
                    "api_goal": "",
                    "preferred_tools": [],
                    "expected_outputs": ["entities", "progress_state"],
                    "reads": list(self.worker_specs["state_worker"].reads),
                    "writes": list(self.worker_specs["state_worker"].writes),
                    "exit_conditions": ["Shared blackboard state has been updated from raw tool outputs."],
                    "next": [f"step_{step_index + 1}"],
                }
            )
            step_index += 1
            workflow_steps.append(
                {
                    "id": f"step_{step_index}",
                    "worker": "analysis_worker",
                    "purpose": self.worker_specs["analysis_worker"].description,
                    "api_goal": "",
                    "preferred_tools": [],
                    "expected_outputs": ["analysis_results", "selected_entities", "progress_state"],
                    "reads": list(self.worker_specs["analysis_worker"].reads),
                    "writes": list(self.worker_specs["analysis_worker"].writes),
                    "exit_conditions": ["Task-specific analysis has been updated from the current shared state."],
                    "next": [f"step_{step_index + 1}"],
                }
            )
            step_index += 1

        workflow_steps.append(
            {
                "id": f"step_{step_index}",
                "worker": "response_worker",
                "purpose": self.worker_specs["response_worker"].description,
                "api_goal": "",
                "preferred_tools": [],
                "expected_outputs": ["assistant_response", "finish", "finish_reason"],
                "reads": list(self.worker_specs["response_worker"].reads),
                "writes": list(self.worker_specs["response_worker"].writes),
                "exit_conditions": ["A final response has been generated."],
                "next": [],
            }
        )

        return {
            "workflow": {
                "workflow_id": f"appworld_{task.get('task_id', 'task')}",
                "goal": instruction,
                "steps": workflow_steps,
            },
            "data_schema": self._fallback_data_schema(instruction),
            "selected_workers": selected_workers,
            "worker_instructions": {
                worker_name: self._worker_instruction(worker_name)
                for worker_name in selected_workers
            },
            "warnings": [warning] + ([fallback_reason] if fallback_reason != warning else []),
            "fallback_reason": fallback_reason,
            "used_fallback_architect": True,
            "architect_debug": debug_payload,
        }

    def _architect_prompt(self, state: dict[str, Any], tools_by_name: dict[str, Any]) -> str:
        task = dict(state.get("task") or {})
        tools_summary = [
            {
                "name": tool.name,
                "app_name": str(tool.metadata.get("app_name", "") or ""),
                "api_name": str(tool.metadata.get("api_name", "") or ""),
                "required_args": list(dict(tool.parameters_json_schema or {}).get("required") or []),
                "mutates_state": bool(tool.metadata.get("mutates_state", True)),
            }
            for tool in tools_by_name.values()
        ]
        worker_summary = [
            {
                "name": spec.name,
                "description": spec.description,
                "reads": list(spec.reads),
                "writes": list(spec.writes),
                "can_use_tools": bool(spec.can_use_tools),
            }
            for spec in self.worker_specs.values()
        ]
        state_summary = {
            "task": {
                "task_id": str(task.get("task_id", "") or ""),
                "instruction": str(task.get("instruction", "") or ""),
                "metadata": {
                    "allowed_apps": list(dict(task.get("metadata") or {}).get("allowed_apps") or []),
                    "oracle_required_apis": list(dict(task.get("metadata") or {}).get("oracle_required_apis") or []),
                },
            },
            "execution": {
                "status": str(dict(state.get("execution") or {}).get("status", "") or ""),
            },
        }
        return (
            "You are the AppWorld architect worker.\n"
            "Your job is to dynamically decompose the task using the available workers and tools.\n"
            "Return exactly one valid JSON object. No markdown, no prose, no code fences.\n"
            "You must output strict JSON with keys: workflow, data_schema, selected_workers, worker_instructions.\n"
            "Rules:\n"
            "- workflow must be task-specific, not a fixed template\n"
            "- use only available worker names\n"
            "- workflow.steps entries must include id, worker, purpose, reads, writes, exit_conditions, next\n"
            "- For repeated tool_worker steps, include api_goal, preferred_tools, expected_outputs so each step has a distinct API objective\n"
            "- Use analysis_worker when the task requires counting, ranking, filtering, aggregation, cross-tool synthesis, or exact answer formatting\n"
            "- workflow.steps[].next must be an array of next step ids, not an object\n"
            "- data_schema entries must include type, required, producer, description\n"
            "- data_schema.*.producer must be an array of worker names, for example [\"tool_worker\"]\n"
            "- selected_workers must be the worker names actually used in the workflow\n"
            "- worker_instructions must only contain selected_workers keys and each value must be a string\n"
            "- Prefer the smallest workflow that still solves the task\n"
            "- It is valid to repeat tool_worker multiple times, and to interleave tool_worker -> state_worker -> analysis_worker before response_worker\n\n"
            "Output shape reminder:\n"
            "{\"workflow\":{\"workflow_id\":\"...\",\"goal\":\"...\",\"steps\":[...]},"
            "\"data_schema\":{\"field\":{\"type\":\"string\",\"required\":false,\"producer\":[\"tool_worker\"],\"description\":\"...\"}},"
            "\"selected_workers\":[\"tool_worker\",\"state_worker\",\"analysis_worker\",\"response_worker\"],"
            "\"worker_instructions\":{\"tool_worker\":\"...\",\"state_worker\":\"...\",\"analysis_worker\":\"...\",\"response_worker\":\"...\"}}\n\n"
            f"Task and current state:\n{json.dumps(state_summary, ensure_ascii=False)}\n\n"
            f"Available workers:\n{json.dumps(worker_summary, ensure_ascii=False)}\n\n"
            f"Available tools:\n{json.dumps(tools_summary[:80], ensure_ascii=False)}"
        )

    def _repair_prompt(self, invalid_output: str, reason: str) -> str:
        return (
            "Repair the following AppWorld architect output into one valid JSON object.\n"
            "Keep the original intent, but fix JSON syntax and schema compliance.\n"
            "Return exactly one JSON object with keys: workflow, data_schema, selected_workers, worker_instructions.\n"
            "Hard requirements:\n"
            "- workflow.steps[].next must be an array of step ids\n"
            "- data_schema.*.producer must be an array of valid worker names\n"
            "- worker_instructions values must be strings\n"
            "- final workflow step must be response_worker\n\n"
            f"Failure reason:\n{reason}\n\n"
            f"Broken output:\n{invalid_output}"
        )

    def _normalize_step(self, step: dict[str, Any], index: int) -> dict[str, Any] | None:
        worker_name = str(step.get("worker", "") or "").strip()
        if worker_name not in self.worker_specs:
            return None
        spec = self.worker_specs[worker_name]
        reads = self._coerce_string_list(step.get("reads"))
        writes = self._coerce_string_list(step.get("writes"))
        if not reads:
            reads = list(spec.reads)
        else:
            reads = list(dict.fromkeys([*(str(item) for item in reads), *spec.reads]))
        if not writes:
            writes = list(spec.writes)
        else:
            writes = list(dict.fromkeys([*(str(item) for item in writes), *spec.writes]))
        return {
            "id": str(step.get("id", "") or f"step_{index}"),
            "worker": worker_name,
            "purpose": str(step.get("purpose", "") or spec.description),
            "api_goal": str(step.get("api_goal", "") or ""),
            "preferred_tools": self._coerce_string_list(step.get("preferred_tools")),
            "expected_outputs": self._coerce_string_list(step.get("expected_outputs")),
            "reads": reads,
            "writes": writes,
            "exit_conditions": self._coerce_string_list(step.get("exit_conditions")) or ["Worker produced a bounded patch."],
            "next": self._coerce_next_steps(step.get("next")),
        }

    def _normalize_plan(self, raw: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        workflow = dict(raw.get("workflow") or {})
        raw_steps = list(workflow.get("steps") or [])
        if not workflow:
            errors.append("Missing workflow object.")
        if not raw_steps:
            errors.append("Workflow contains no steps.")
        steps: list[dict[str, Any]] = []
        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                errors.append(f"workflow.steps[{index - 1}] is not an object.")
                continue
            normalized = self._normalize_step(item, index)
            if normalized is not None:
                steps.append(normalized)
            else:
                errors.append(
                    f"workflow.steps[{index - 1}] references unknown worker {str(item.get('worker', '') or '')!r}."
                )
        if not steps:
            errors.append("No valid workflow steps remained after normalization.")
            return None, errors

        valid_step_ids = {str(step["id"]) for step in steps}
        for index, step in enumerate(steps):
            valid_next = [step_id for step_id in list(step.get("next") or []) if step_id in valid_step_ids]
            if valid_next:
                step["next"] = valid_next
                continue
            if index + 1 < len(steps):
                step["next"] = [str(steps[index + 1]["id"])]
            else:
                step["next"] = []

        selected_workers = []
        for step in steps:
            worker_name = step["worker"]
            if worker_name not in selected_workers:
                selected_workers.append(worker_name)
        if "response_worker" not in selected_workers:
            errors.append("Workflow does not include response_worker.")
        if steps and steps[-1].get("worker") != "response_worker":
            errors.append("Final workflow step must be response_worker.")

        raw_schema = dict(raw.get("data_schema") or {})
        if not raw_schema:
            errors.append("Missing data_schema object.")
        normalized_schema: dict[str, Any] = {}
        for field_name, schema_entry in raw_schema.items():
            if not isinstance(schema_entry, dict):
                errors.append(f"data_schema.{field_name} is not an object.")
                continue
            producers = [item for item in self._coerce_string_list(schema_entry.get("producer")) if item in self.worker_specs]
            if not producers:
                errors.append(f"data_schema.{field_name} has no valid producer.")
            normalized_schema[str(field_name)] = {
                "type": str(schema_entry.get("type", "object") or "object"),
                "required": bool(schema_entry.get("required", False)),
                "producer": producers,
                "description": str(schema_entry.get("description", "") or ""),
            }
        if not normalized_schema:
            errors.append("No valid task-specific data schema entries remained after normalization.")

        raw_instructions = dict(raw.get("worker_instructions") or {})
        worker_instructions = {
            worker_name: self._coerce_instruction_text(raw_instructions.get(worker_name, ""), worker_name)
            for worker_name in selected_workers
        }
        if not worker_instructions:
            errors.append("worker_instructions did not contain any selected workers.")
        if errors:
            return None, errors
        workflow_id = str(workflow.get("workflow_id", "") or fallback["workflow"]["workflow_id"])
        goal = str(workflow.get("goal", "") or fallback["workflow"]["goal"])
        return {
            "workflow": {
                "workflow_id": workflow_id,
                "goal": goal,
                "steps": steps,
            },
            "data_schema": normalized_schema,
            "selected_workers": selected_workers,
            "worker_instructions": worker_instructions,
            "warnings": [str(item) for item in list(raw.get("warnings") or []) if str(item).strip()],
            "fallback_reason": "",
            "used_fallback_architect": False,
            "architect_debug": {},
        }, []

    def build(self, state: dict[str, Any], tools_by_name: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback_build(state)
        prompt = self._architect_prompt(state, tools_by_name)
        raw, debug_payload = _load_llm_json(self._llm, prompt)
        if raw is None and debug_payload.get("raw_output"):
            repaired, repair_debug = _load_llm_json(
                self._llm,
                self._repair_prompt(
                    str(debug_payload.get("raw_output", "") or ""),
                    str(debug_payload.get("parse_error") or debug_payload.get("invoke_error") or "Invalid architect output."),
                ),
            )
            if repaired is not None:
                debug_payload["repair_error"] = ""
                debug_payload["repaired_output"] = str(repair_debug.get("raw_output", "") or "")
                raw = repaired
            else:
                debug_payload["repair_error"] = str(
                    repair_debug.get("parse_error") or repair_debug.get("invoke_error") or "Architect repair failed."
                )
        if raw is None:
            reason = debug_payload.get("invoke_error") or debug_payload.get("parse_error") or "Architect returned no valid JSON."
            return self._fallback_with_reason(state, reason=reason, architect_debug=debug_payload)
        normalized, validation_errors = self._normalize_plan(raw, fallback)
        if normalized is None and debug_payload.get("raw_output"):
            repaired, repair_debug = _load_llm_json(
                self._llm,
                self._repair_prompt(
                    str(debug_payload.get("raw_output", "") or ""),
                    "; ".join(validation_errors[:8]) if validation_errors else "Architect plan validation failed.",
                ),
            )
            if repaired is not None:
                debug_payload["repair_error"] = ""
                debug_payload["repaired_output"] = str(repair_debug.get("raw_output", "") or "")
                normalized, validation_errors = self._normalize_plan(repaired, fallback)
            else:
                debug_payload["repair_error"] = str(
                    repair_debug.get("parse_error") or repair_debug.get("invoke_error") or "Architect repair failed."
                )
        if normalized is None:
            debug_payload["validation_errors"] = list(validation_errors)
            reason = "; ".join(validation_errors[:6]) if validation_errors else "Architect plan validation failed."
            return self._fallback_with_reason(state, reason=reason, architect_debug=debug_payload)
        normalized["architect_debug"] = debug_payload
        return normalized
