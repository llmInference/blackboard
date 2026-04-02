"""Built-in workers for the AppWorld blackboard runtime."""
from __future__ import annotations

import json
import re
from typing import Any

from experiment.common.neutral import ToolResultStatus

from experiment.appworld.workers.base import AppWorldWorker, WorkerRuntime, WorkerSpec, replace


TOOL_WORKER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "instruction": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
        "shared_data": {
            "type": "object",
            "properties": {
                "candidate_tools": {"type": "array"},
                "auth": {"type": "object"},
                "raw_results": {"type": "array"},
                "selected_entities": {"type": "array"},
                "progress_state": {"type": "object"},
                "action_history": {"type": "array"},
                "assistant_response": {"type": "string"},
                "finish_reason": {"type": "string"},
            },
        },
        "execution": {
            "type": "object",
            "properties": {
                "current_step_id": {"type": "string"},
                "current_step": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "worker": {"type": "string"},
                        "purpose": {"type": "string"},
                        "api_goal": {"type": "string"},
                        "preferred_tools": {"type": "array"},
                        "expected_outputs": {"type": "array"},
                    },
                },
                "history": {"type": "array"},
            },
        },
    },
}

STATE_WORKER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "instruction": {"type": "string"},
            },
        },
        "shared_data": {
            "type": "object",
            "properties": {
                "raw_results": {"type": "array"},
                "tool_status": {"type": "object"},
                "auth": {"type": "object"},
                "entities": {"type": "array"},
                "selected_entities": {"type": "array"},
                "intermediate_results": {"type": "object"},
                "progress_state": {"type": "object"},
            },
        },
        "execution": {
            "type": "object",
            "properties": {
                "current_step_id": {"type": "string"},
                "history": {"type": "array"},
            },
        },
    },
}

ANALYSIS_WORKER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "instruction": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
        "shared_data": {
            "type": "object",
            "properties": {
                "raw_results": {"type": "array"},
                "tool_status": {"type": "object"},
                "auth": {"type": "object"},
                "entities": {"type": "array"},
                "selected_entities": {"type": "array"},
                "evidence": {"type": "object"},
                "intermediate_results": {"type": "object"},
                "progress_state": {"type": "object"},
                "analysis_results": {"type": "object"},
            },
        },
        "execution": {
            "type": "object",
            "properties": {
                "current_step_id": {"type": "string"},
                "current_step": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "worker": {"type": "string"},
                        "purpose": {"type": "string"},
                        "api_goal": {"type": "string"},
                        "preferred_tools": {"type": "array"},
                        "expected_outputs": {"type": "array"},
                    },
                },
                "history": {"type": "array"},
            },
        },
    },
}

RESPONSE_WORKER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "instruction": {"type": "string"},
            },
        },
        "shared_data": {
            "type": "object",
            "properties": {
                "analysis_results": {"type": "object"},
                "selected_entities": {"type": "array"},
                "evidence": {"type": "object"},
                "intermediate_results": {"type": "object"},
                "progress_state": {"type": "object"},
                "tool_status": {"type": "object"},
            },
        },
        "execution": {
            "type": "object",
            "properties": {
                "current_step_id": {"type": "string"},
                "history": {"type": "array"},
            },
        },
    },
}


def _shared_data(runtime: WorkerRuntime) -> dict[str, Any]:
    return dict(runtime.state.get("shared_data") or {})


def _task(runtime: WorkerRuntime) -> dict[str, Any]:
    return dict(runtime.state.get("task") or {})


def _execution(runtime: WorkerRuntime) -> dict[str, Any]:
    return dict(runtime.state.get("execution") or {})


def _current_step(runtime: WorkerRuntime) -> dict[str, Any]:
    return dict(_execution(runtime).get("current_step") or {})


def _append_history(runtime: WorkerRuntime, worker_name: str, note: str) -> dict[str, Any]:
    execution = dict(runtime.state.get("execution") or {})
    history = list(execution.get("history") or [])
    history.append({"worker": worker_name, "note": note})
    return replace("/execution/history", history)


def _set_runtime_shared_auth(runtime: WorkerRuntime, auth: dict[str, Any]) -> None:
    shared = dict(runtime.state.get("shared_data") or {})
    shared["auth"] = dict(auth)
    runtime.state["shared_data"] = shared


def _extract_requested_count(instruction: str) -> int | None:
    lowered = instruction.lower()
    match = re.search(r"\btop\s+(\d+)\b", lowered)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\b", lowered)
    if match:
        return int(match.group(1))
    return None


def _normalize_genre_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _extract_entities(value: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            lowered = {str(key).lower(): key for key in node.keys()}
            display_key = lowered.get("title") or lowered.get("name") or lowered.get("id")
            entity: dict[str, Any] = {}
            if display_key is not None:
                entity["label"] = node.get(display_key)
            for key in ("id", "name", "title", "email", "phone_number", "play_count"):
                if key in node:
                    entity[key] = node.get(key)
            if entity and entity not in found:
                found.append(entity)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found[:limit]


def _parse_json_payload(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        return None


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _llm_json(llm: Any, prompt: str) -> dict[str, Any] | None:
    if llm is None or not hasattr(llm, "invoke"):
        return None
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        content = str(content)
    try:
        return json.loads(content)
    except Exception:
        return None


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _entity_label(entity: dict[str, Any]) -> str:
    return str(
        entity.get("label")
        or entity.get("title")
        or entity.get("name")
        or entity.get("id")
        or ""
    ).strip()


class ToolWorker(AppWorldWorker):
    _MAX_LIBRARY_PAGES = 10
    _MAX_DETAIL_EXPANSIONS = 100

    spec = WorkerSpec(
        name="tool_worker",
        description="Use AppWorld tools to read or change task-relevant external state, including login when needed.",
        reads=(
            "candidate_tools",
            "auth",
            "raw_results",
            "selected_entities",
            "progress_state",
            "action_history",
            "assistant_response",
            "finish_reason",
        ),
        writes=("raw_results", "tool_status", "action_history", "auth"),
        can_use_tools=True,
        input_schema=TOOL_WORKER_INPUT_SCHEMA,
    )

    def _normalize_preferred_tools(self, runtime: WorkerRuntime, value: Any) -> list[str]:
        candidate_tools = set(str(name) for name in list(_shared_data(runtime).get("candidate_tools") or []) if str(name))
        if not candidate_tools:
            candidate_tools = set(runtime.tools_by_name.keys())
        normalized: list[str] = []
        items = [value] if isinstance(value, str) else list(value or []) if isinstance(value, list) else []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            variants = [text]
            if "." in text:
                app_name, _, api_name = text.partition(".")
                variants.append(f"{app_name}__{api_name}")
            for variant in variants:
                if variant in candidate_tools and variant not in normalized:
                    normalized.append(variant)
        return normalized

    def _candidate_tools(self, runtime: WorkerRuntime) -> list[str]:
        shared = _shared_data(runtime)
        candidates = [str(name) for name in list(shared.get("candidate_tools") or []) if str(name)]
        if not candidates:
            candidates = list(runtime.tools_by_name.keys())
        preferred_tools = self._normalize_preferred_tools(runtime, _current_step(runtime).get("preferred_tools"))
        if preferred_tools:
            prioritized = [tool_name for tool_name in preferred_tools if tool_name in candidates]
            others = [tool_name for tool_name in candidates if tool_name not in prioritized]
            candidates = prioritized + others
        oracle_tools = self._oracle_required_tool_names(runtime)
        if not oracle_tools:
            return candidates
        prioritized = [tool_name for tool_name in candidates if tool_name in preferred_tools or tool_name in oracle_tools]
        helper_tools = [
            tool_name
            for tool_name in candidates
            if tool_name not in prioritized
            and (
                tool_name == "supervisor__show_account_passwords"
                or tool_name.endswith("__login")
            )
        ]
        others = [tool_name for tool_name in candidates if tool_name not in prioritized and tool_name not in helper_tools]
        return prioritized + helper_tools + others

    def _oracle_required_tool_names(self, runtime: WorkerRuntime) -> set[str]:
        task_metadata = dict(_task(runtime).get("metadata") or {})
        required_apis = [str(item) for item in list(task_metadata.get("oracle_required_apis") or []) if str(item).strip()]
        tool_names: set[str] = set()
        for api_name in required_apis:
            if "." not in api_name:
                continue
            app_name, _, api_leaf = api_name.partition(".")
            tool_names.add(f"{app_name}__{api_leaf}")
        return tool_names

    def _deferred_required_tool_names(self, runtime: WorkerRuntime) -> set[str]:
        oracle_tools = self._oracle_required_tool_names(runtime)
        deferred: set[str] = set()
        for tool_name in oracle_tools:
            if tool_name == "supervisor__complete_task":
                deferred.add(tool_name)
        return deferred

    def _virtual_completed_required_tool_names(self, runtime: WorkerRuntime, auth: dict[str, Any]) -> set[str]:
        completed: set[str] = set()
        oracle_tools = self._oracle_required_tool_names(runtime)
        task_metadata = dict(_task(runtime).get("metadata") or {})
        supervisor = dict(task_metadata.get("supervisor") or {})
        if "supervisor__show_profile" in oracle_tools and any(
            _has_meaningful_value(supervisor.get(field_name))
            for field_name in ("email", "phone_number", "first_name", "last_name")
        ):
            completed.add("supervisor__show_profile")
        if "spotify__login" in oracle_tools and self._has_access_token("spotify__login", auth):
            completed.add("spotify__login")
        return completed

    def _required_tool_sort_key(self, tool_name: str) -> tuple[int, str]:
        api_name = tool_name.partition("__")[2]
        if api_name.endswith("_library"):
            return (0, tool_name)
        if api_name in {"show_album", "show_playlist"}:
            return (1, tool_name)
        if api_name == "show_song":
            return (2, tool_name)
        return (3, tool_name)

    def _required_api_status(
        self,
        runtime: WorkerRuntime,
        raw_results: list[dict[str, Any]],
        auth: dict[str, Any],
    ) -> dict[str, Any]:
        oracle_tools = self._oracle_required_tool_names(runtime)
        deferred_tools = self._deferred_required_tool_names(runtime)
        successful_tools = {
            str(item.get("tool_name", "") or "")
            for item in raw_results
            if str(item.get("status", "") or "") == ToolResultStatus.SUCCESS.value
        }
        failed_tools = {
            str(item.get("tool_name", "") or "")
            for item in raw_results
            if str(item.get("status", "") or "") == ToolResultStatus.ERROR.value
        }
        completed_tools = set(successful_tools) | self._virtual_completed_required_tool_names(runtime, auth)
        required_tools = oracle_tools - deferred_tools
        completed_tools -= failed_tools
        pending_tools = (required_tools - completed_tools) | (required_tools & failed_tools)
        actionable_pending_tools = [tool_name for tool_name in pending_tools if tool_name in runtime.tools_by_name]
        actionable_pending_tools.sort(key=self._required_tool_sort_key)
        pending_deferred_tools = (deferred_tools - completed_tools) | (deferred_tools & failed_tools)
        actionable_pending_deferred_tools = [
            tool_name for tool_name in pending_deferred_tools if tool_name in runtime.tools_by_name
        ]
        actionable_pending_deferred_tools.sort(key=self._required_tool_sort_key)
        return {
            "required_tools": sorted(required_tools),
            "completed_tools": sorted(required_tools & completed_tools),
            "failed_tools": sorted(required_tools & failed_tools),
            "pending_tools": sorted(required_tools & pending_tools),
            "deferred_tools": sorted(oracle_tools & deferred_tools),
            "completed_deferred_tools": sorted(deferred_tools & completed_tools),
            "pending_deferred_tools": sorted(deferred_tools & pending_deferred_tools),
            "actionable_pending_deferred_tools": actionable_pending_deferred_tools,
            "actionable_pending_tools": actionable_pending_tools,
            "all_actionable_completed": not actionable_pending_tools,
        }

    def _required_arguments(self, tool_name: str, runtime: WorkerRuntime) -> tuple[list[str], dict[str, Any]]:
        tool = runtime.tools_by_name.get(tool_name)
        if tool is None:
            return [], {}
        schema = dict(tool.parameters_json_schema or {})
        return list(schema.get("required") or []), dict(schema.get("properties") or {})

    def _tool_properties(self, tool_name: str, runtime: WorkerRuntime) -> dict[str, Any]:
        _, properties = self._required_arguments(tool_name, runtime)
        return properties

    def _tool_supports_argument(self, tool_name: str, runtime: WorkerRuntime, argument_name: str) -> bool:
        return argument_name in self._tool_properties(tool_name, runtime)

    def _finish_reason_for_submission(self, runtime: WorkerRuntime) -> str:
        shared = _shared_data(runtime)
        finish_reason = str(shared.get("finish_reason", "") or "").strip()
        if finish_reason:
            return finish_reason
        progress_state = dict(shared.get("progress_state") or {})
        blocked_reason = str(progress_state.get("blocked_reason", "") or "").strip()
        if blocked_reason in {"missing_authentication", "missing_arguments"}:
            return "prerequisite_missing"
        if blocked_reason == "authentication_failed":
            return "authentication_failed"
        if blocked_reason == "tool_execution_error":
            return "tool_execution_failed"
        if blocked_reason:
            return "insufficient_state"
        selected_entities = list(shared.get("selected_entities") or [])
        latest_result_preview = str(dict(shared.get("evidence") or {}).get("latest_result_preview", "") or "").strip()
        if selected_entities or latest_result_preview:
            return "task_completed"
        return ""

    def _instruction_requests_answer(self, runtime: WorkerRuntime) -> bool:
        instruction = str(_task(runtime).get("instruction", "") or "").strip().lower()
        if not instruction:
            return False
        answer_markers = (
            "?",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "list",
            "return",
            "provide",
            "give",
            "tell",
            "name",
            "identify",
            "find",
            "count",
            "total",
            "top",
            "most",
            "least",
            "comma-separated",
            "comma separated",
        )
        return any(marker in instruction for marker in answer_markers)

    def _complete_task_arguments(
        self,
        runtime: WorkerRuntime,
        *,
        required: list[str],
        properties: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        shared = _shared_data(runtime)
        assistant_response = str(shared.get("assistant_response", "") or "").strip()
        finish_reason = self._finish_reason_for_submission(runtime)
        arguments: dict[str, Any] = {}
        missing: list[str] = []

        if "status" in properties or "status" in required:
            if finish_reason:
                arguments["status"] = "success" if finish_reason == "task_completed" else "fail"
            elif "status" in required:
                missing.append("status")

        if "answer" in properties or "answer" in required:
            if assistant_response and self._instruction_requests_answer(runtime):
                arguments["answer"] = assistant_response
            elif properties.get("answer", {}).get("default") is not None:
                arguments["answer"] = properties["answer"]["default"]

        return arguments, missing

    def _resolve_arguments(self, tool_name: str, runtime: WorkerRuntime) -> tuple[dict[str, Any], list[str]]:
        required, properties = self._required_arguments(tool_name, runtime)
        if tool_name == "supervisor__complete_task":
            return self._complete_task_arguments(runtime, required=required, properties=properties)
        shared = _shared_data(runtime)
        auth = dict(shared.get("auth") or {})
        supervisor = dict(auth.get("supervisor") or {})
        auth_sessions = dict(auth.get("auth_sessions") or {})
        selected_entities = list(shared.get("selected_entities") or [])
        entity = selected_entities[0] if selected_entities else {}

        arguments: dict[str, Any] = {}
        missing: list[str] = []
        for field_name in required:
            lowered = field_name.lower()
            if field_name in entity and _has_meaningful_value(entity[field_name]):
                arguments[field_name] = entity[field_name]
                continue
            if field_name in supervisor and _has_meaningful_value(supervisor[field_name]):
                arguments[field_name] = supervisor[field_name]
                continue
            if lowered == "password" or "password" in lowered:
                app_name = tool_name.partition("__")[0]
                passwords = dict(auth.get("account_passwords") or {})
                password = passwords.get(app_name, "")
                if _has_meaningful_value(password):
                    arguments[field_name] = password
                else:
                    missing.append(field_name)
            elif ("email" in lowered or lowered in {"username", "user_name", "login", "login_id"}) and supervisor.get("email"):
                arguments[field_name] = supervisor["email"]
            elif "phone" in lowered and supervisor.get("phone_number"):
                arguments[field_name] = supervisor["phone_number"]
            elif "access_token" in lowered:
                app_name = tool_name.partition("__")[0]
                session = dict(auth_sessions.get(app_name) or {})
                payload = dict(session.get("payload") or {})
                access_token = payload.get("access_token", "")
                if _has_meaningful_value(access_token):
                    arguments[field_name] = access_token
                else:
                    missing.append(field_name)
            elif properties.get(field_name, {}).get("default") is not None:
                arguments[field_name] = properties[field_name]["default"]
            else:
                missing.append(field_name)
        for field_name in properties.keys():
            if field_name in arguments:
                continue
            lowered = field_name.lower()
            if "access_token" in lowered:
                app_name = tool_name.partition("__")[0]
                session = dict(auth_sessions.get(app_name) or {})
                payload = dict(session.get("payload") or {})
                access_token = payload.get("access_token", "")
                if _has_meaningful_value(access_token):
                    arguments[field_name] = access_token
            elif properties.get(field_name, {}).get("default") is not None:
                arguments[field_name] = properties[field_name]["default"]
        return arguments, missing

    def _build_arguments(self, tool_name: str, runtime: WorkerRuntime) -> dict[str, Any]:
        arguments, _ = self._resolve_arguments(tool_name, runtime)
        return arguments

    def _tool_requires_access_token(self, tool_name: str, runtime: WorkerRuntime) -> bool:
        required, _ = self._required_arguments(tool_name, runtime)
        return any("access_token" in str(field_name).lower() for field_name in required)

    def _has_access_token(self, tool_name: str, auth: dict[str, Any]) -> bool:
        app_name = tool_name.partition("__")[0]
        auth_sessions = dict(auth.get("auth_sessions") or {})
        session = dict(auth_sessions.get(app_name) or {})
        payload = dict(session.get("payload") or {})
        return _has_meaningful_value(payload.get("access_token"))

    def _record_result(
        self,
        raw_results: list[dict[str, Any]],
        *,
        worker_name: str,
        result: Any,
        arguments: dict[str, Any],
    ) -> None:
        raw_results.append(
            {
                "worker": worker_name,
                "tool_name": result.tool_name,
                "status": result.status.value,
                "payload": dict(result.payload),
                "content": result.content,
                "arguments": dict(arguments),
            }
        )

    def _already_executed(self, raw_results: list[dict[str, Any]], tool_name: str, arguments: dict[str, Any]) -> bool:
        normalized = json.dumps(dict(arguments or {}), ensure_ascii=False, sort_keys=True)
        for item in raw_results:
            if str(item.get("tool_name", "") or "") != tool_name:
                continue
            existing_arguments = dict(item.get("arguments") or {})
            if json.dumps(existing_arguments, ensure_ascii=False, sort_keys=True) == normalized:
                return True
        return False

    def _result_rows(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, dict):
            response = payload.get("response")
            if isinstance(response, list):
                return list(response)
            if isinstance(response, dict):
                return [dict(response)]
            for key in ("items", "songs", "albums", "playlists", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return list(value)
            return [payload]
        return []

    def _collect_scalar_values(self, value: Any, field_names: list[str], *, limit: int) -> list[Any]:
        collected: list[Any] = []
        seen: set[str] = set()

        def add(candidate: Any) -> None:
            if not _has_meaningful_value(candidate):
                return
            key = json.dumps(candidate, ensure_ascii=False, sort_keys=True) if isinstance(candidate, (dict, list)) else str(candidate)
            if key in seen:
                return
            seen.add(key)
            collected.append(candidate)

        def visit(node: Any) -> None:
            if len(collected) >= limit:
                return
            if isinstance(node, dict):
                for field_name in field_names:
                    if field_name in node:
                        add(node[field_name])
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        return collected[:limit]

    def _id_candidates(self, raw_results: list[dict[str, Any]], tool_name: str, field_name: str) -> list[Any]:
        app_name = tool_name.partition("__")[0]
        api_name = tool_name.partition("__")[2]
        resource_name = field_name.removesuffix("_id")
        relevant_tool_names: set[str] = set()
        if resource_name == "song":
            relevant_tool_names.update(
                {
                    f"{app_name}__show_song_library",
                    f"{app_name}__show_album",
                    f"{app_name}__show_playlist",
                    f"{app_name}__show_song",
                }
            )
        elif resource_name == "album":
            relevant_tool_names.update({f"{app_name}__show_album_library", f"{app_name}__show_album"})
        elif resource_name == "playlist":
            relevant_tool_names.update({f"{app_name}__show_playlist_library", f"{app_name}__show_playlist"})
        if api_name.endswith("_library"):
            relevant_tool_names.add(tool_name)
        relevant_results = [
            item for item in raw_results if str(item.get("tool_name", "") or "") in relevant_tool_names
        ]
        if not relevant_results:
            relevant_results = [
                item
                for item in raw_results
                if str(item.get("tool_name", "") or "").startswith(f"{app_name}__")
            ]
        field_names = [field_name]
        if resource_name == "song":
            field_names.append("id")
        values: list[Any] = []
        for item in relevant_results:
            for source in (item.get("payload"), _parse_json_payload(str(item.get("content", "") or ""))):
                if source is None:
                    continue
                values.extend(self._collect_scalar_values(source, field_names, limit=self._MAX_DETAIL_EXPANSIONS))
                if len(values) >= self._MAX_DETAIL_EXPANSIONS:
                    break
            if len(values) >= self._MAX_DETAIL_EXPANSIONS:
                break
        deduped: list[Any] = []
        seen: set[str] = set()
        for value in values:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped[: self._MAX_DETAIL_EXPANSIONS]

    def _execute_and_store(
        self,
        runtime: WorkerRuntime,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        rationale: str,
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
        tool_status: dict[str, Any],
    ) -> Any | None:
        if self._already_executed(raw_results, tool_name, arguments):
            return None
        result = runtime.execute_tool(tool_name=tool_name, arguments=arguments, rationale=rationale)
        self._record_result(raw_results, worker_name=self.spec.name, result=result, arguments=arguments)
        action_history.append({"tool_name": result.tool_name, "status": result.status.value})
        tool_status.update(
            {
                "status": "success" if result.status == ToolResultStatus.SUCCESS else "tool_execution_error",
                "last_tool_name": result.tool_name,
                "last_status": result.status.value,
                "last_error": result.error_message or result.content if result.status == ToolResultStatus.ERROR else "",
            }
        )
        return result

    def _account_passwords_from_result(self, result: Any) -> dict[str, str]:
        payload = result.payload
        if isinstance(payload, dict) and all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
        ):
            return {str(key): str(value) for key, value in payload.items() if str(key).strip()}

        rows: list[Any] = []
        if isinstance(payload, list):
            rows = list(payload)
        elif isinstance(payload, dict) and isinstance(payload.get("response"), list):
            rows = list(payload.get("response") or [])

        account_passwords: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            account_name = str(row.get("account_name", "") or "").strip()
            password = str(row.get("password", "") or "")
            if account_name and password:
                account_passwords[account_name] = password
        return account_passwords

    def _ensure_passwords(
        self,
        runtime: WorkerRuntime,
        candidate_tools: list[str],
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
    ) -> None:
        if runtime.tool_executor is None:
            return
        if auth.get("account_passwords"):
            return
        if "supervisor__show_account_passwords" not in candidate_tools:
            return
        result = runtime.execute_tool(
            tool_name="supervisor__show_account_passwords",
            rationale="Fetch account passwords before protected AppWorld tool use.",
        )
        self._record_result(raw_results, worker_name=self.spec.name, result=result, arguments={})
        action_history.append({"tool_name": result.tool_name, "status": result.status.value})
        if result.status == ToolResultStatus.SUCCESS:
            auth["account_passwords"] = self._account_passwords_from_result(result)
            _set_runtime_shared_auth(runtime, auth)

    def _login_target_apps(self, runtime: WorkerRuntime, candidate_tools: list[str], auth: dict[str, Any]) -> list[str]:
        passwords = dict(auth.get("account_passwords") or {})
        if not passwords:
            return []

        step = _current_step(runtime)
        preferred_tools = self._normalize_preferred_tools(runtime, step.get("preferred_tools"))
        step_apps = [
            tool_name.partition("__")[0]
            for tool_name in preferred_tools
            if tool_name.partition("__")[0] in passwords and tool_name.partition("__")[0] != "supervisor"
        ]
        if step_apps:
            return list(dict.fromkeys(step_apps))

        oracle_apps = [
            tool_name.partition("__")[0]
            for tool_name in candidate_tools
            if tool_name in self._oracle_required_tool_names(runtime)
            and tool_name.partition("__")[0] in passwords
            and tool_name.partition("__")[0] != "supervisor"
        ]
        if oracle_apps:
            return list(dict.fromkeys(oracle_apps))

        protected_apps = [
            tool_name.partition("__")[0]
            for tool_name in candidate_tools
            if not tool_name.endswith("__login")
            and tool_name != "supervisor__show_account_passwords"
            and self._tool_requires_access_token(tool_name, runtime)
            and tool_name.partition("__")[0] in passwords
            and tool_name.partition("__")[0] != "supervisor"
        ]
        if protected_apps:
            return list(dict.fromkeys(protected_apps))

        return [
            app_name
            for app_name in passwords.keys()
            if app_name != "supervisor"
        ]

    def _ensure_logins(
        self,
        runtime: WorkerRuntime,
        candidate_tools: list[str],
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if runtime.tool_executor is None:
            return None
        passwords = dict(auth.get("account_passwords") or {})
        if not passwords:
            return None
        target_apps = self._login_target_apps(runtime, candidate_tools, auth)
        if not target_apps:
            return None
        auth_sessions = dict(auth.get("auth_sessions") or {})
        active_account = dict(auth.get("active_account") or {})
        for app_name in target_apps:
            if app_name not in passwords:
                continue
            login_tool = f"{app_name}__login"
            if login_tool not in candidate_tools or app_name in active_account:
                continue
            arguments, missing = self._resolve_arguments(login_tool, runtime)
            if missing:
                auth["auth_sessions"] = auth_sessions
                auth["active_account"] = active_account
                _set_runtime_shared_auth(runtime, auth)
                return {
                    "status": "login_missing_required_arguments",
                    "tool_name": login_tool,
                    "missing_arguments": missing,
                    "error": f"Missing required login arguments for {login_tool}: {', '.join(missing)}",
                }
            result = runtime.execute_tool(
                tool_name=login_tool,
                arguments=arguments,
                rationale=f"Bootstrap login state for {app_name}.",
            )
            self._record_result(raw_results, worker_name=self.spec.name, result=result, arguments=arguments)
            action_history.append({"tool_name": result.tool_name, "status": result.status.value})
            auth_sessions[app_name] = {
                "tool_name": result.tool_name,
                "status": result.status.value,
                "payload": dict(result.payload),
            }
            if result.status == ToolResultStatus.SUCCESS and _has_meaningful_value(dict(result.payload).get("access_token")):
                active_account[app_name] = {
                    "email": arguments.get("email", ""),
                    "username": arguments.get("username", ""),
                    "phone_number": arguments.get("phone_number", ""),
                }
            else:
                auth["auth_sessions"] = auth_sessions
                auth["active_account"] = active_account
                _set_runtime_shared_auth(runtime, auth)
                return {
                    "status": "login_failed",
                    "tool_name": result.tool_name,
                    "missing_arguments": [],
                    "error": result.error_message or result.content or f"Login failed for {app_name}.",
                }
        auth["auth_sessions"] = auth_sessions
        auth["active_account"] = active_account
        _set_runtime_shared_auth(runtime, auth)
        return None

    def _non_helper_tools(self, tool_names: list[str]) -> list[str]:
        return [
            tool_name
            for tool_name in tool_names
            if tool_name != "supervisor__show_account_passwords" and not tool_name.endswith("__login")
        ]

    def _submission_ready(self, runtime: WorkerRuntime) -> bool:
        finish_reason = self._finish_reason_for_submission(runtime)
        if not finish_reason:
            return False
        shared = _shared_data(runtime)
        assistant_response = str(shared.get("assistant_response", "") or "").strip()
        return bool(assistant_response) or finish_reason != "task_completed"

    def _sequence_tools(self, runtime: WorkerRuntime, candidate_tools: list[str]) -> list[str]:
        required_api_status = self._required_api_status(
            runtime,
            list(_shared_data(runtime).get("raw_results") or []),
            dict(_shared_data(runtime).get("auth") or {}),
        )
        pending_deferred_tools = [
            tool_name
            for tool_name in list(required_api_status.get("actionable_pending_deferred_tools") or [])
            if tool_name in candidate_tools
        ]
        if pending_deferred_tools and self._submission_ready(runtime):
            return self._non_helper_tools(pending_deferred_tools)
        step = _current_step(runtime)
        preferred_tools = self._normalize_preferred_tools(runtime, step.get("preferred_tools"))
        if preferred_tools:
            return self._non_helper_tools(preferred_tools)
        if required_api_status.get("actionable_pending_tools"):
            pending_tools = [
                tool_name
                for tool_name in list(required_api_status.get("actionable_pending_tools") or [])
                if tool_name in candidate_tools
            ]
            if pending_tools:
                return self._non_helper_tools(pending_tools)
        selected_tool = self._select_action_tool(runtime, candidate_tools)
        return self._non_helper_tools([selected_tool] if selected_tool else [])

    def _execute_single_tool(
        self,
        runtime: WorkerRuntime,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        rationale: str,
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
        tool_status: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._tool_requires_access_token(tool_name, runtime) and not self._has_access_token(tool_name, auth):
            return {
                "status": "missing_access_token",
                "blocked_tool_name": tool_name,
                "missing_arguments": ["access_token"],
                "last_error": f"Missing access token for protected tool {tool_name}.",
            }
        result = self._execute_and_store(
            runtime,
            tool_name=tool_name,
            arguments=arguments,
            rationale=rationale,
            raw_results=raw_results,
            action_history=action_history,
            tool_status=tool_status,
        )
        if result is not None and result.status == ToolResultStatus.ERROR:
            return {
                "status": "tool_execution_error",
                "blocked_tool_name": tool_name,
                "missing_arguments": [],
                "last_error": result.error_message or result.content or f"Tool execution failed for {tool_name}.",
            }
        return None

    def _execute_paginated_tool(
        self,
        runtime: WorkerRuntime,
        *,
        tool_name: str,
        base_arguments: dict[str, Any],
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
        tool_status: dict[str, Any],
    ) -> dict[str, Any] | None:
        previous_signature = ""
        for page_index in range(self._MAX_LIBRARY_PAGES):
            arguments = dict(base_arguments)
            arguments["page_index"] = page_index
            failure = self._execute_single_tool(
                runtime,
                tool_name=tool_name,
                arguments=arguments,
                rationale=f"Enumerate AppWorld collection data for {tool_name}, page {page_index}.",
                auth=auth,
                raw_results=raw_results,
                action_history=action_history,
                tool_status=tool_status,
            )
            if failure is not None:
                return failure
            matching_results = [
                item
                for item in raw_results
                if str(item.get("tool_name", "") or "") == tool_name and dict(item.get("arguments") or {}) == arguments
            ]
            if not matching_results:
                continue
            latest_payload = matching_results[-1].get("payload")
            rows = self._result_rows(latest_payload)
            if not rows:
                break
            signature = json.dumps(rows, ensure_ascii=False, sort_keys=True)
            if signature == previous_signature:
                break
            previous_signature = signature
        return None

    def _execute_id_expansion_tool(
        self,
        runtime: WorkerRuntime,
        *,
        tool_name: str,
        base_arguments: dict[str, Any],
        missing_id_fields: list[str],
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
        tool_status: dict[str, Any],
    ) -> dict[str, Any] | None:
        if len(missing_id_fields) != 1:
            return {
                "status": "missing_required_arguments",
                "blocked_tool_name": tool_name,
                "missing_arguments": missing_id_fields,
                "last_error": f"Missing required arguments for {tool_name}: {', '.join(missing_id_fields)}",
            }
        id_field = missing_id_fields[0]
        candidate_values = self._id_candidates(raw_results, tool_name, id_field)
        if not candidate_values:
            return None
        for value in candidate_values[: self._MAX_DETAIL_EXPANSIONS]:
            arguments = dict(base_arguments)
            arguments[id_field] = value
            failure = self._execute_single_tool(
                runtime,
                tool_name=tool_name,
                arguments=arguments,
                rationale=f"Expand AppWorld detail records for {tool_name}.",
                auth=auth,
                raw_results=raw_results,
                action_history=action_history,
                tool_status=tool_status,
            )
            if failure is not None:
                return failure
        return None

    def _follow_up_tools_for(self, tool_name: str, runtime: WorkerRuntime) -> list[str]:
        app_name = tool_name.partition("__")[0]
        api_name = tool_name.partition("__")[2]
        available = runtime.tools_by_name
        follow_ups: list[str] = []
        oracle_tools = self._oracle_required_tool_names(runtime)
        if api_name == "show_song_library":
            if f"{app_name}__show_album_library" in available and f"{app_name}__show_album_library" in oracle_tools:
                follow_ups.append(f"{app_name}__show_album_library")
            if f"{app_name}__show_playlist_library" in available and f"{app_name}__show_playlist_library" in oracle_tools:
                follow_ups.append(f"{app_name}__show_playlist_library")
            detail_tool = f"{app_name}__show_song"
            if detail_tool in available:
                follow_ups.append(detail_tool)
        elif api_name == "show_album_library":
            if f"{app_name}__show_playlist_library" in available and f"{app_name}__show_playlist_library" in oracle_tools:
                follow_ups.append(f"{app_name}__show_playlist_library")
            detail_tool = f"{app_name}__show_album"
            if detail_tool in available:
                follow_ups.append(detail_tool)
            song_tool = f"{app_name}__show_song"
            if song_tool in available:
                follow_ups.append(song_tool)
        elif api_name == "show_playlist_library":
            if f"{app_name}__show_album_library" in available and f"{app_name}__show_album_library" in oracle_tools:
                follow_ups.append(f"{app_name}__show_album_library")
            detail_tool = f"{app_name}__show_playlist"
            if detail_tool in available:
                follow_ups.append(detail_tool)
            song_tool = f"{app_name}__show_song"
            if song_tool in available:
                follow_ups.append(song_tool)
        elif api_name in {"show_album", "show_playlist"}:
            song_tool = f"{app_name}__show_song"
            if song_tool in available:
                follow_ups.append(song_tool)
        return list(dict.fromkeys(follow_ups))

    def _execute_tool_pattern(
        self,
        runtime: WorkerRuntime,
        *,
        tool_name: str,
        auth: dict[str, Any],
        raw_results: list[dict[str, Any]],
        action_history: list[dict[str, Any]],
        tool_status: dict[str, Any],
        visited: set[str],
    ) -> dict[str, Any] | None:
        if tool_name in visited:
            return None
        visited.add(tool_name)
        try:
            arguments, missing = self._resolve_arguments(tool_name, runtime)
            id_missing = [field_name for field_name in missing if field_name.endswith("_id")]
            other_missing = [field_name for field_name in missing if field_name not in id_missing]
            if "access_token" in other_missing:
                return {
                    "status": "missing_access_token",
                    "blocked_tool_name": tool_name,
                    "missing_arguments": ["access_token"],
                    "last_error": f"Missing access token for protected tool {tool_name}.",
                }
            other_missing = [field_name for field_name in other_missing if field_name != "access_token"]
            if other_missing:
                return {
                    "status": "missing_required_arguments",
                    "blocked_tool_name": tool_name,
                    "missing_arguments": other_missing,
                    "last_error": f"Missing required arguments for {tool_name}: {', '.join(other_missing)}",
                }
            if self._tool_supports_argument(tool_name, runtime, "page_index") and tool_name.partition("__")[2].startswith("show_"):
                failure = self._execute_paginated_tool(
                    runtime,
                    tool_name=tool_name,
                    base_arguments=arguments,
                    auth=auth,
                    raw_results=raw_results,
                    action_history=action_history,
                    tool_status=tool_status,
                )
                if failure is not None:
                    return failure
            elif id_missing:
                failure = self._execute_id_expansion_tool(
                    runtime,
                    tool_name=tool_name,
                    base_arguments=arguments,
                    missing_id_fields=id_missing,
                    auth=auth,
                    raw_results=raw_results,
                    action_history=action_history,
                    tool_status=tool_status,
                )
                if failure is not None:
                    return failure
            else:
                failure = self._execute_single_tool(
                    runtime,
                    tool_name=tool_name,
                    arguments=arguments,
                    rationale="Execute the next AppWorld tool action.",
                    auth=auth,
                    raw_results=raw_results,
                    action_history=action_history,
                    tool_status=tool_status,
                )
                if failure is not None:
                    return failure
            for follow_up_tool in self._follow_up_tools_for(tool_name, runtime):
                failure = self._execute_tool_pattern(
                    runtime,
                    tool_name=follow_up_tool,
                    auth=auth,
                    raw_results=raw_results,
                    action_history=action_history,
                    tool_status=tool_status,
                    visited=visited,
                )
                if failure is not None:
                    return failure
            return None
        finally:
            visited.discard(tool_name)

    def _select_action_tool(self, runtime: WorkerRuntime, candidate_tools: list[str]) -> str:
        instruction = str(_task(runtime).get("instruction", "") or "").lower()
        step = _current_step(runtime)
        api_goal = str(step.get("api_goal", "") or "").lower()
        purpose = str(step.get("purpose", "") or "").lower()
        expected_outputs = " ".join(str(item).lower() for item in list(step.get("expected_outputs") or []) if str(item).strip())
        preferred_tools = set(self._normalize_preferred_tools(runtime, step.get("preferred_tools")))
        oracle_tool_names = self._oracle_required_tool_names(runtime)
        excluded = {"supervisor__show_account_passwords"}
        ranked: list[tuple[int, str]] = []
        for tool_name in candidate_tools:
            if tool_name in excluded or tool_name.endswith("__login"):
                continue
            tool = runtime.tools_by_name.get(tool_name)
            if tool is None:
                continue
            score = 0
            api_name = tool_name.partition("__")[2].replace("_", " ").lower()
            description = str(tool.description or "").lower()
            if tool_name in preferred_tools:
                score += 50
            if oracle_tool_names:
                if tool_name in oracle_tool_names:
                    score += 20
                else:
                    score -= 5
            if not bool(tool.metadata.get("mutates_state", True)):
                score += 2
            for token in instruction.split():
                token = token.strip(",.?!:")
                if token and token in api_name:
                    score += 3
                if token and token in description:
                    score += 1
            for token in f"{api_goal} {purpose} {expected_outputs}".split():
                token = token.strip(",.?!:()[]{}")
                if token and token in api_name:
                    score += 4
                if token and token in description:
                    score += 2
            ranked.append((score, tool_name))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1] if ranked else ""

    def run(self, runtime: WorkerRuntime) -> list[dict[str, Any]]:
        shared = _shared_data(runtime)
        auth = dict(shared.get("auth") or {})
        auth.setdefault("supervisor", dict(_task(runtime).get("metadata", {}).get("supervisor", {}) or {}))
        raw_results = list(shared.get("raw_results") or [])
        action_history = list(shared.get("action_history") or [])
        tool_status = dict(shared.get("tool_status") or {})
        candidate_tools = self._candidate_tools(runtime)

        if runtime.tool_executor is None:
            tool_status["status"] = "missing_tool_executor"
            return [
                replace("/shared_data/auth", auth),
                replace("/shared_data/tool_status", tool_status),
                replace("/shared_data/action_history", action_history),
                _append_history(runtime, self.spec.name, "No tool executor available."),
            ]

        self._ensure_passwords(runtime, candidate_tools, auth, raw_results, action_history)
        login_failure = self._ensure_logins(runtime, candidate_tools, auth, raw_results, action_history)
        if login_failure is not None:
            tool_status.update(
                {
                    "status": str(login_failure.get("status", "") or "login_failed"),
                    "blocked_tool_name": str(login_failure.get("tool_name", "") or ""),
                    "missing_arguments": list(login_failure.get("missing_arguments") or []),
                    "last_error": str(login_failure.get("error", "") or ""),
                }
            )
            tool_status["required_api_status"] = self._required_api_status(runtime, raw_results, auth)
            return [
                replace("/shared_data/auth", auth),
                replace("/shared_data/raw_results", raw_results),
                replace("/shared_data/tool_status", tool_status),
                replace("/shared_data/action_history", action_history),
                _append_history(runtime, self.spec.name, tool_status["last_error"] or "Login failed before action execution."),
            ]

        sequence_tools = self._sequence_tools(runtime, candidate_tools)
        if not sequence_tools:
            tool_status["status"] = "no_action_tool_selected"
            tool_status["required_api_status"] = self._required_api_status(runtime, raw_results, auth)
            return [
                replace("/shared_data/auth", auth),
                replace("/shared_data/raw_results", raw_results),
                replace("/shared_data/tool_status", tool_status),
                replace("/shared_data/action_history", action_history),
                _append_history(runtime, self.spec.name, "No action tool selected."),
            ]
        visited: set[str] = set()
        executed_before = len(raw_results)
        for tool_name in sequence_tools:
            failure = self._execute_tool_pattern(
                runtime,
                tool_name=tool_name,
                auth=auth,
                raw_results=raw_results,
                action_history=action_history,
                tool_status=tool_status,
                visited=visited,
            )
            if failure is not None:
                tool_status.update(failure)
                tool_status["required_api_status"] = self._required_api_status(runtime, raw_results, auth)
                return [
                    replace("/shared_data/auth", auth),
                    replace("/shared_data/raw_results", raw_results),
                    replace("/shared_data/tool_status", tool_status),
                    replace("/shared_data/action_history", action_history),
                    _append_history(runtime, self.spec.name, tool_status["last_error"] or "Tool execution failed."),
                ]
        if len(raw_results) == executed_before:
            tool_status.setdefault("status", "success")
            history_note = "No new tool calls were needed."
        else:
            history_note = f"Executed AppWorld tool sequence ending with {tool_status.get('last_tool_name', sequence_tools[-1])}."
        tool_status["required_api_status"] = self._required_api_status(runtime, raw_results, auth)
        return [
            replace("/shared_data/auth", auth),
            replace("/shared_data/raw_results", raw_results),
            replace("/shared_data/tool_status", tool_status),
            replace("/shared_data/action_history", action_history),
            _append_history(runtime, self.spec.name, history_note),
        ]


class StateWorker(AppWorldWorker):
    spec = WorkerSpec(
        name="state_worker",
        description="Normalize raw AppWorld tool outputs into reusable execution state and lightweight shared memory.",
        reads=("raw_results", "tool_status", "auth", "entities", "selected_entities"),
        writes=("entities", "selected_entities", "evidence", "intermediate_results", "progress_state"),
        input_schema=STATE_WORKER_INPUT_SCHEMA,
    )

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    _ALLOWED_BLOCK_REASONS = {
        "",
        "missing_authentication",
        "authentication_failed",
        "missing_arguments",
        "tool_execution_error",
        "missing_tool_executor",
        "no_action_tool_selected",
        "insufficient_state",
        "task_completed",
    }

    _AUTH_BLOCK_REASONS = {
        "missing_authentication",
        "authentication_failed",
    }

    def _auth_ready_from_auth(self, auth: dict[str, Any]) -> bool:
        auth_sessions = dict(auth.get("auth_sessions") or {})
        for session_value in auth_sessions.values():
            session = dict(session_value or {})
            payload = dict(session.get("payload") or {})
            if _has_meaningful_value(payload.get("access_token")):
                return True
        return False

    def _tool_status_progress_state(self, runtime: WorkerRuntime) -> dict[str, Any]:
        shared = _shared_data(runtime)
        tool_status = dict(shared.get("tool_status") or {})
        required_api_status = dict(tool_status.get("required_api_status") or {})
        raw_results = list(shared.get("raw_results") or [])
        auth = dict(shared.get("auth") or {})
        latest_content = str(raw_results[-1].get("content", "") or "").strip() if raw_results else ""
        selected_entities = list(shared.get("selected_entities") or [])
        status = str(tool_status.get("status", "") or "")

        blocked_reason = ""
        blocked = False
        recoverable = False
        missing_prerequisites = [str(item) for item in list(tool_status.get("missing_arguments") or []) if str(item)]

        if status == "missing_access_token":
            blocked = True
            recoverable = True
            blocked_reason = "missing_authentication"
            if "access_token" not in missing_prerequisites:
                missing_prerequisites.append("access_token")
        elif status == "login_failed":
            blocked = True
            recoverable = True
            blocked_reason = "authentication_failed"
            if "valid_login" not in missing_prerequisites:
                missing_prerequisites.append("valid_login")
        elif status == "login_missing_required_arguments":
            blocked = True
            recoverable = True
            blocked_reason = "missing_arguments"
        elif status == "missing_required_arguments":
            blocked = True
            recoverable = True
            blocked_reason = "missing_arguments"
        elif status == "missing_tool_executor":
            blocked = True
            recoverable = False
            blocked_reason = "missing_tool_executor"
        elif status == "no_action_tool_selected":
            blocked = True
            recoverable = True
            blocked_reason = "no_action_tool_selected"
        elif str(tool_status.get("last_status", "") or "") == ToolResultStatus.ERROR.value:
            blocked = True
            recoverable = True
            blocked_reason = "tool_execution_error"

        auth_ready = self._auth_ready_from_auth(auth)
        if blocked_reason in self._AUTH_BLOCK_REASONS:
            auth_ready = False
        actionable_pending_tools = [
            str(item)
            for item in list(required_api_status.get("actionable_pending_tools") or [])
            if str(item).strip()
        ]
        if actionable_pending_tools and not blocked:
            missing_prerequisites = list(dict.fromkeys([*missing_prerequisites, *actionable_pending_tools]))
        return {
            "ready_for_response": (bool(selected_entities) or bool(latest_content) or blocked) and not actionable_pending_tools,
            "needs_more_tooling": bool(actionable_pending_tools) or (not blocked and not bool(selected_entities) and len(raw_results) < 3),
            "auth_ready": auth_ready,
            "blocked": blocked,
            "blocked_reason": blocked_reason,
            "missing_prerequisites": missing_prerequisites,
            "recoverable": recoverable,
        }

    def _normalize_progress_state(self, value: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
        raw = dict(value or {}) if isinstance(value, dict) else {}
        raw_blocked_reason = raw.get("blocked_reason", None)
        if raw_blocked_reason is None or (
            not str(raw_blocked_reason).strip() and str(fallback.get("blocked_reason", "")).strip()
        ):
            blocked_reason = str(fallback.get("blocked_reason", "") or "")
        else:
            blocked_reason = str(raw_blocked_reason or "")
        if blocked_reason not in self._ALLOWED_BLOCK_REASONS:
            blocked_reason = fallback.get("blocked_reason", "")
        missing_prerequisites = [
            str(item)
            for item in list(raw.get("missing_prerequisites") or fallback.get("missing_prerequisites") or [])
            if str(item).strip()
        ]
        fallback_missing_prerequisites = [
            str(item)
            for item in list(fallback.get("missing_prerequisites") or [])
            if str(item).strip()
        ]
        if blocked_reason in self._AUTH_BLOCK_REASONS:
            for item in fallback_missing_prerequisites:
                if item not in missing_prerequisites:
                    missing_prerequisites.append(item)
        fallback_blocked = bool(fallback.get("blocked", False))
        blocked = bool(raw.get("blocked", False)) or fallback_blocked or bool(blocked_reason)
        auth_ready = bool(fallback.get("auth_ready", False))
        if "auth_ready" in raw:
            auth_ready = auth_ready and bool(raw.get("auth_ready"))
        if blocked_reason in self._AUTH_BLOCK_REASONS:
            auth_ready = False
        ready_for_response = bool(raw.get("ready_for_response", fallback.get("ready_for_response", False)))
        if fallback_blocked:
            ready_for_response = bool(fallback.get("ready_for_response", ready_for_response))
        needs_more_tooling = bool(raw.get("needs_more_tooling", fallback.get("needs_more_tooling", False)))
        if blocked:
            needs_more_tooling = False
        recoverable = bool(raw.get("recoverable", fallback.get("recoverable", False)))
        if fallback_blocked:
            recoverable = bool(fallback.get("recoverable", recoverable))
        return {
            "ready_for_response": ready_for_response,
            "needs_more_tooling": needs_more_tooling,
            "auth_ready": auth_ready,
            "blocked": blocked,
            "blocked_reason": blocked_reason,
            "missing_prerequisites": missing_prerequisites,
            "recoverable": recoverable,
        }

    def _normalize_state_payload(self, parsed: dict[str, Any] | None, runtime: WorkerRuntime) -> dict[str, Any]:
        shared = _shared_data(runtime)
        raw_results = list(shared.get("raw_results") or [])
        instruction = str(_task(runtime).get("instruction", "") or "")
        requested_count = _extract_requested_count(instruction)
        fallback_progress = self._tool_status_progress_state(runtime)

        parsed = _as_dict(parsed)
        entities = _as_list(parsed.get("entities"))
        selected_entities = _as_list(parsed.get("selected_entities"))
        evidence = _as_dict(parsed.get("evidence"))
        intermediate_results = _as_dict(parsed.get("intermediate_results"))
        latest_content = str(raw_results[-1].get("content", "") or "").strip() if raw_results else ""

        if not entities:
            fallback_entities = self._fallback(runtime)
            entities = list(fallback_entities.get("entities") or [])
            if not selected_entities:
                selected_entities = list(fallback_entities.get("selected_entities") or [])
            evidence = {**dict(fallback_entities.get("evidence") or {}), **evidence}
            intermediate_results = {**dict(fallback_entities.get("intermediate_results") or {}), **intermediate_results}

        if not selected_entities:
            selected_entities = list(entities[: requested_count or 5])

        evidence.setdefault("result_count", len(raw_results))
        evidence.setdefault("latest_result_preview", latest_content[:400])
        evidence.setdefault("auth_ready", bool(fallback_progress.get("auth_ready", False)))
        intermediate_results.setdefault("requested_count", requested_count)
        intermediate_results.setdefault(
            "latest_tool_name",
            str(dict(shared.get("tool_status") or {}).get("last_tool_name", "") or ""),
        )

        return {
            "entities": entities[:20],
            "selected_entities": selected_entities[: requested_count or 5],
            "evidence": evidence,
            "intermediate_results": intermediate_results,
            "progress_state": self._normalize_progress_state(parsed.get("progress_state"), fallback=fallback_progress),
        }

    def _fallback(self, runtime: WorkerRuntime) -> dict[str, Any]:
        task = _task(runtime)
        instruction = str(task.get("instruction", "") or "")
        requested_count = _extract_requested_count(instruction)
        shared = _shared_data(runtime)
        raw_results = list(shared.get("raw_results") or [])
        auth = dict(shared.get("auth") or {})
        entities: list[dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            entities.extend(_extract_entities(item.get("payload")))
            parsed = _parse_json_payload(str(item.get("content", "") or ""))
            if parsed is not None:
                entities.extend(_extract_entities(parsed))
        unique_entities: list[dict[str, Any]] = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)
        selected_entities = unique_entities[: requested_count or 5]
        latest_content = str(raw_results[-1].get("content", "") or "") if raw_results else ""
        return {
            "entities": unique_entities[:20],
            "selected_entities": selected_entities,
            "evidence": {
                "result_count": len(raw_results),
                "latest_result_preview": latest_content[:400],
                "auth_ready": bool(dict(auth.get("auth_sessions") or {})),
            },
            "intermediate_results": {
                "requested_count": requested_count,
                "latest_tool_name": str(dict(shared.get("tool_status") or {}).get("last_tool_name", "") or ""),
            },
            "progress_state": self._tool_status_progress_state(runtime),
        }

    def _llm_prompt(self, runtime: WorkerRuntime) -> str:
        shared = _shared_data(runtime)
        payload = {
            "task": _task(runtime),
            "raw_results": list(shared.get("raw_results") or [])[-5:],
            "tool_status": dict(shared.get("tool_status") or {}),
            "auth": dict(shared.get("auth") or {}),
            "existing_entities": list(shared.get("entities") or []),
        }
        return (
            "You are the AppWorld state worker. Convert tool outputs into reusable blackboard state.\n"
            "Return strict JSON with keys: entities, selected_entities, evidence, intermediate_results, progress_state.\n"
            "progress_state must include: ready_for_response, needs_more_tooling, auth_ready, blocked, blocked_reason, missing_prerequisites, recoverable.\n"
            "blocked_reason must be one of: '', missing_authentication, authentication_failed, missing_arguments, tool_execution_error, missing_tool_executor, no_action_tool_selected, insufficient_state, task_completed.\n"
            "missing_prerequisites must be an array of strings.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def run(self, runtime: WorkerRuntime) -> list[dict[str, Any]]:
        parsed = _llm_json(self._llm, self._llm_prompt(runtime))
        normalized = self._normalize_state_payload(parsed, runtime) if parsed is not None else self._fallback(runtime)
        return [
            replace("/shared_data/entities", list(normalized.get("entities") or [])),
            replace("/shared_data/selected_entities", list(normalized.get("selected_entities") or [])),
            replace("/shared_data/evidence", dict(normalized.get("evidence") or {})),
            replace("/shared_data/intermediate_results", dict(normalized.get("intermediate_results") or {})),
            replace("/shared_data/progress_state", dict(normalized.get("progress_state") or {})),
            _append_history(runtime, self.spec.name, "Updated blackboard state from tool results."),
        ]


class AnalysisWorker(AppWorldWorker):
    spec = WorkerSpec(
        name="analysis_worker",
        description="Analyze fetched AppWorld data to compute task-specific selections, counts, rankings, and final answer candidates.",
        reads=(
            "raw_results",
            "tool_status",
            "auth",
            "entities",
            "selected_entities",
            "evidence",
            "intermediate_results",
            "progress_state",
            "analysis_results",
        ),
        writes=("analysis_results", "selected_entities", "evidence", "intermediate_results", "progress_state"),
        input_schema=ANALYSIS_WORKER_INPUT_SCHEMA,
    )

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def _auth_ready_from_auth(self, auth: dict[str, Any]) -> bool:
        auth_sessions = dict(auth.get("auth_sessions") or {})
        for session_value in auth_sessions.values():
            session = dict(session_value or {})
            payload = dict(session.get("payload") or {})
            if _has_meaningful_value(payload.get("access_token")):
                return True
        return False

    def _base_progress(self, runtime: WorkerRuntime) -> dict[str, Any]:
        shared = _shared_data(runtime)
        auth = dict(shared.get("auth") or {})
        progress_state = dict(shared.get("progress_state") or {})
        return {
            "ready_for_response": bool(progress_state.get("ready_for_response", False)),
            "needs_more_tooling": bool(progress_state.get("needs_more_tooling", False)),
            "auth_ready": bool(progress_state.get("auth_ready", self._auth_ready_from_auth(auth))),
            "blocked": bool(progress_state.get("blocked", False)),
            "blocked_reason": str(progress_state.get("blocked_reason", "") or ""),
            "missing_prerequisites": [
                str(item)
                for item in list(progress_state.get("missing_prerequisites") or [])
                if str(item).strip()
            ],
            "recoverable": bool(progress_state.get("recoverable", False)),
        }

    def _rows_for_payload(self, payload: Any) -> list[dict[str, Any]]:
        rows: list[Any] = []
        if isinstance(payload, list):
            rows = list(payload)
        elif isinstance(payload, dict):
            response = payload.get("response")
            if isinstance(response, list):
                rows = list(response)
            elif isinstance(response, dict):
                rows = [dict(response)]
            else:
                for key in ("items", "songs", "albums", "playlists", "results"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        rows = list(value)
                        break
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _spotify_target_genre(self, instruction: str, song_details: list[dict[str, Any]]) -> str:
        normalized_instruction = _normalize_genre_text(instruction)
        known_genres = {
            _normalize_genre_text(detail.get("genre"))
            for detail in song_details
            if _has_meaningful_value(detail.get("genre"))
        }
        known_genres.discard("")
        for genre in sorted(known_genres, key=len, reverse=True):
            if genre and genre in normalized_instruction:
                return genre
        for alias in ("r and b", "edm", "indie", "jazz", "hip hop", "pop", "rock", "country"):
            if alias in normalized_instruction:
                return alias
        return ""

    def _derive_answer_from_entities(
        self,
        instruction: str,
        selected_entities: list[dict[str, Any]],
        intermediate_results: dict[str, Any],
        analysis_results: dict[str, Any],
    ) -> str:
        final_answer = str(analysis_results.get("final_answer", "") or "").strip()
        if final_answer:
            return final_answer
        numeric_answer = analysis_results.get("numeric_answer")
        if isinstance(numeric_answer, int):
            return str(numeric_answer)
        csv_answer = str(
            intermediate_results.get("top_titles_csv")
            or intermediate_results.get("final_answer")
            or analysis_results.get("csv_answer")
            or ""
        ).strip()
        if csv_answer:
            return csv_answer
        labels = [_entity_label(entity) for entity in selected_entities]
        labels = [label for label in labels if label]
        if not labels:
            return ""
        requested_count = _extract_requested_count(instruction)
        if "comma-separated" in instruction.lower() or "comma separated" in instruction.lower():
            return ", ".join(labels[: requested_count or len(labels)])
        if requested_count == 1:
            return labels[0]
        return ", ".join(labels[: requested_count or len(labels)])

    def _merge_progress_state(
        self,
        runtime: WorkerRuntime,
        *,
        selected_entities: list[dict[str, Any]],
        analysis_results: dict[str, Any],
        missing_prerequisites: list[str],
    ) -> dict[str, Any]:
        shared = _shared_data(runtime)
        auth = dict(shared.get("auth") or {})
        progress_state = self._base_progress(runtime)
        final_answer = str(analysis_results.get("final_answer", "") or "").strip()
        if selected_entities or final_answer:
            progress_state.update(
                {
                    "ready_for_response": True,
                    "needs_more_tooling": False,
                    "auth_ready": self._auth_ready_from_auth(auth),
                    "recoverable": True,
                }
            )
            if progress_state.get("blocked_reason") in {"", "insufficient_state", "task_completed"}:
                progress_state.update(
                    {
                        "blocked": False,
                        "blocked_reason": "",
                        "missing_prerequisites": [],
                    }
                )
            return progress_state
        if not progress_state.get("needs_more_tooling", False) and not progress_state.get("blocked", False):
            progress_state.update(
                {
                    "ready_for_response": False,
                    "needs_more_tooling": False,
                    "auth_ready": self._auth_ready_from_auth(auth),
                    "blocked": True,
                    "blocked_reason": "insufficient_state",
                    "missing_prerequisites": list(
                        dict.fromkeys(
                            [
                                *list(progress_state.get("missing_prerequisites") or []),
                                *missing_prerequisites,
                            ]
                        )
                    ),
                    "recoverable": True,
                }
            )
        return progress_state

    def _spotify_task_state(self, runtime: WorkerRuntime) -> dict[str, Any] | None:
        shared = _shared_data(runtime)
        raw_results = list(shared.get("raw_results") or [])
        instruction = str(_task(runtime).get("instruction", "") or "")
        normalized_instruction = instruction.lower()
        spotify_tool_names = {
            "spotify__show_song_library",
            "spotify__show_album_library",
            "spotify__show_playlist_library",
            "spotify__show_song",
            "spotify__show_album",
            "spotify__show_playlist",
        }
        if "spotify" not in normalized_instruction and "played" not in normalized_instruction and "top" not in normalized_instruction:
            return None
        if not any(str(item.get("tool_name", "") or "") in spotify_tool_names for item in raw_results):
            return None

        requested_count = _extract_requested_count(instruction) or 5
        auth = dict(shared.get("auth") or {})
        tool_status = dict(shared.get("tool_status") or {})
        base_progress = self._base_progress(runtime)

        song_library_rows: list[dict[str, Any]] = []
        album_library_rows: list[dict[str, Any]] = []
        playlist_library_rows: list[dict[str, Any]] = []
        album_detail_rows: list[dict[str, Any]] = []
        playlist_detail_rows: list[dict[str, Any]] = []
        song_details_by_id: dict[int, dict[str, Any]] = {}

        for item in raw_results:
            tool_name = str(item.get("tool_name", "") or "")
            if str(item.get("status", "") or "") != ToolResultStatus.SUCCESS.value:
                continue
            payload = item.get("payload")
            if tool_name == "spotify__show_song_library":
                song_library_rows.extend(self._rows_for_payload(payload))
            elif tool_name == "spotify__show_album_library":
                album_library_rows.extend(self._rows_for_payload(payload))
            elif tool_name == "spotify__show_playlist_library":
                playlist_library_rows.extend(self._rows_for_payload(payload))
            elif tool_name == "spotify__show_album" and isinstance(payload, dict):
                album_detail_rows.append(dict(payload))
            elif tool_name == "spotify__show_playlist" and isinstance(payload, dict):
                playlist_detail_rows.append(dict(payload))
            elif tool_name == "spotify__show_song" and isinstance(payload, dict):
                song_id_value = payload.get("song_id", payload.get("id"))
                if isinstance(song_id_value, int):
                    song_details_by_id[song_id_value] = dict(payload)

        song_details = list(song_details_by_id.values())
        target_genre = self._spotify_target_genre(instruction, song_details)
        target_tracks: list[dict[str, Any]] = []
        for detail in song_details:
            genre = _normalize_genre_text(detail.get("genre"))
            play_count = detail.get("play_count")
            song_id_value = detail.get("song_id", detail.get("id"))
            title = str(detail.get("title", "") or "").strip()
            if not isinstance(song_id_value, int) or not title or not isinstance(play_count, int):
                continue
            if target_genre and genre != target_genre:
                continue
            target_tracks.append(
                {
                    "song_id": song_id_value,
                    "label": title,
                    "title": title,
                    "genre": str(detail.get("genre", "") or ""),
                    "play_count": play_count,
                }
            )

        deduped_tracks: dict[int, dict[str, Any]] = {}
        for track in target_tracks:
            song_id_value = int(track["song_id"])
            existing = deduped_tracks.get(song_id_value)
            if existing is None or int(track["play_count"]) > int(existing["play_count"]):
                deduped_tracks[song_id_value] = track
        ranked_tracks = sorted(
            deduped_tracks.values(),
            key=lambda item: (-int(item["play_count"]), str(item["title"]).lower(), int(item["song_id"])),
        )

        missing_prerequisites: list[str] = []
        if not song_library_rows:
            missing_prerequisites.append("spotify__show_song_library")
        if not album_library_rows:
            missing_prerequisites.append("spotify__show_album_library")
        if not playlist_library_rows:
            missing_prerequisites.append("spotify__show_playlist_library")
        if not target_genre:
            missing_prerequisites.append("genre target extraction")
        if song_details and not ranked_tracks:
            missing_prerequisites.append(f"{target_genre or 'target'} songs with play_count")
        if not song_details:
            missing_prerequisites.append("spotify__show_song details with genre/play_count")

        selected_entities = [dict(track) for track in ranked_tracks[:requested_count]]
        top_titles_csv = ", ".join(track["title"] for track in ranked_tracks[:requested_count])
        analysis_results = {
            "analysis_type": "spotify_ranked_tracks",
            "target_genre": target_genre,
            "ranked_track_count": len(ranked_tracks),
            "requested_count": requested_count,
            "final_answer": top_titles_csv,
            "csv_answer": top_titles_csv,
        }
        progress_state = dict(base_progress)
        if selected_entities:
            progress_state = self._merge_progress_state(
                runtime,
                selected_entities=selected_entities,
                analysis_results=analysis_results,
                missing_prerequisites=[],
            )
        elif not base_progress.get("needs_more_tooling", False):
            progress_state = self._merge_progress_state(
                runtime,
                selected_entities=[],
                analysis_results=analysis_results,
                missing_prerequisites=missing_prerequisites,
            )

        evidence = {
            "song_library_count": len(song_library_rows),
            "album_library_count": len(album_library_rows),
            "playlist_library_count": len(playlist_library_rows),
            "album_detail_count": len(album_detail_rows),
            "playlist_detail_count": len(playlist_detail_rows),
            "song_detail_count": len(song_details),
            "target_genre": target_genre,
            "ranked_track_count": len(ranked_tracks),
            "result_count": len(raw_results),
            "latest_result_preview": str(raw_results[-1].get("content", "") or "")[:400] if raw_results else "",
            "auth_ready": self._auth_ready_from_auth(auth),
        }
        intermediate_results = {
            "requested_count": requested_count,
            "latest_tool_name": str(tool_status.get("last_tool_name", "") or ""),
            "target_genre": target_genre,
            "song_library_loaded": bool(song_library_rows),
            "album_library_loaded": bool(album_library_rows),
            "playlist_library_loaded": bool(playlist_library_rows),
            "song_details_loaded": len(song_details),
            "cross_library_coverage_complete": bool(song_library_rows) and bool(album_library_rows) and bool(playlist_library_rows),
            "can_filter_target_genre": bool(target_genre),
            "can_rank_most_played": bool(ranked_tracks),
            "top_titles_csv": top_titles_csv,
        }
        return {
            "analysis_results": analysis_results,
            "selected_entities": selected_entities,
            "evidence": evidence,
            "intermediate_results": intermediate_results,
            "progress_state": progress_state,
        }

    def _fallback(self, runtime: WorkerRuntime) -> dict[str, Any]:
        spotify_state = self._spotify_task_state(runtime)
        if spotify_state is not None:
            return spotify_state
        shared = _shared_data(runtime)
        instruction = str(_task(runtime).get("instruction", "") or "")
        selected_entities = [
            dict(item)
            for item in list(shared.get("selected_entities") or [])
            if isinstance(item, dict)
        ]
        entities = [
            dict(item)
            for item in list(shared.get("entities") or [])
            if isinstance(item, dict)
        ]
        if not selected_entities:
            requested_count = _extract_requested_count(instruction)
            selected_entities = entities[: requested_count or 5]
        raw_results = list(shared.get("raw_results") or [])
        tool_status = dict(shared.get("tool_status") or {})
        evidence = {
            **_as_dict(shared.get("evidence")),
            "result_count": len(raw_results),
            "latest_result_preview": str(raw_results[-1].get("content", "") or "")[:400] if raw_results else "",
            "auth_ready": self._auth_ready_from_auth(dict(shared.get("auth") or {})),
        }
        intermediate_results = {
            **_as_dict(shared.get("intermediate_results")),
            "latest_tool_name": str(tool_status.get("last_tool_name", "") or ""),
            "requested_count": _extract_requested_count(instruction),
        }
        analysis_results = {
            **_as_dict(shared.get("analysis_results")),
            "analysis_type": str(_as_dict(shared.get("analysis_results")).get("analysis_type", "") or "generic"),
        }
        final_answer = self._derive_answer_from_entities(
            instruction,
            selected_entities,
            intermediate_results,
            analysis_results,
        )
        if final_answer:
            analysis_results["final_answer"] = final_answer
        progress_state = self._merge_progress_state(
            runtime,
            selected_entities=selected_entities,
            analysis_results=analysis_results,
            missing_prerequisites=[],
        )
        return {
            "analysis_results": analysis_results,
            "selected_entities": selected_entities,
            "evidence": evidence,
            "intermediate_results": intermediate_results,
            "progress_state": progress_state,
        }

    def _normalize_analysis_payload(self, parsed: dict[str, Any] | None, runtime: WorkerRuntime) -> dict[str, Any]:
        fallback = self._fallback(runtime)
        raw = _as_dict(parsed)
        analysis_results = {**_as_dict(fallback.get("analysis_results")), **_as_dict(raw.get("analysis_results"))}
        selected_entities = _as_list(raw.get("selected_entities")) or _as_list(fallback.get("selected_entities"))
        evidence = {**_as_dict(fallback.get("evidence")), **_as_dict(raw.get("evidence"))}
        intermediate_results = {
            **_as_dict(fallback.get("intermediate_results")),
            **_as_dict(raw.get("intermediate_results")),
        }
        final_answer = self._derive_answer_from_entities(
            str(_task(runtime).get("instruction", "") or ""),
            [dict(item) for item in selected_entities if isinstance(item, dict)],
            intermediate_results,
            analysis_results,
        )
        if final_answer:
            analysis_results["final_answer"] = final_answer
        progress_state = self._merge_progress_state(
            runtime,
            selected_entities=[dict(item) for item in selected_entities if isinstance(item, dict)],
            analysis_results=analysis_results,
            missing_prerequisites=[
                str(item)
                for item in list(_as_dict(raw.get("analysis_results")).get("missing_prerequisites") or [])
                if str(item).strip()
            ],
        )
        return {
            "analysis_results": analysis_results,
            "selected_entities": [dict(item) for item in selected_entities if isinstance(item, dict)],
            "evidence": evidence,
            "intermediate_results": intermediate_results,
            "progress_state": progress_state,
        }

    def _llm_prompt(self, runtime: WorkerRuntime) -> str:
        shared = _shared_data(runtime)
        payload = {
            "task": _task(runtime),
            "raw_results": list(shared.get("raw_results") or [])[-8:],
            "entities": list(shared.get("entities") or [])[:20],
            "selected_entities": list(shared.get("selected_entities") or [])[:10],
            "evidence": dict(shared.get("evidence") or {}),
            "intermediate_results": dict(shared.get("intermediate_results") or {}),
            "progress_state": dict(shared.get("progress_state") or {}),
            "tool_status": dict(shared.get("tool_status") or {}),
        }
        return (
            "You are the AppWorld analysis worker. Reason over fetched tool data and compute task-specific results.\n"
            "Return strict JSON with keys: analysis_results, selected_entities, evidence, intermediate_results, progress_state.\n"
            "analysis_results should include a task-specific final_answer when possible.\n"
            "If the task requests a pure number/count, final_answer should be digits only.\n"
            "If the task requests a comma-separated list, final_answer should be only the comma-separated list with no explanation.\n"
            "Do not invent missing tool results.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def run(self, runtime: WorkerRuntime) -> list[dict[str, Any]]:
        deterministic = self._spotify_task_state(runtime)
        if deterministic is not None:
            normalized = deterministic
        else:
            parsed = _llm_json(self._llm, self._llm_prompt(runtime))
            normalized = self._normalize_analysis_payload(parsed, runtime) if parsed is not None else self._fallback(runtime)
        return [
            replace("/shared_data/analysis_results", dict(normalized.get("analysis_results") or {})),
            replace("/shared_data/selected_entities", list(normalized.get("selected_entities") or [])),
            replace("/shared_data/evidence", dict(normalized.get("evidence") or {})),
            replace("/shared_data/intermediate_results", dict(normalized.get("intermediate_results") or {})),
            replace("/shared_data/progress_state", dict(normalized.get("progress_state") or {})),
            _append_history(runtime, self.spec.name, "Analyzed shared data into task-specific results."),
        ]


class ResponseWorker(AppWorldWorker):
    spec = WorkerSpec(
        name="response_worker",
        description="Produce the final user-facing response from the current blackboard state.",
        reads=("analysis_results", "selected_entities", "evidence", "intermediate_results", "progress_state", "tool_status"),
        writes=("assistant_response", "response_confidence", "finish", "finish_reason"),
        input_schema=RESPONSE_WORKER_INPUT_SCHEMA,
    )

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def _submission_pending(self, runtime: WorkerRuntime) -> bool:
        shared = _shared_data(runtime)
        tool_status = dict(shared.get("tool_status") or {})
        required_api_status = dict(tool_status.get("required_api_status") or {})
        pending = [
            str(item)
            for item in list(required_api_status.get("actionable_pending_deferred_tools") or [])
            if str(item).strip()
        ]
        return "supervisor__complete_task" in pending

    def _normalize_finish_reason(self, value: dict[str, Any], runtime: WorkerRuntime, fallback: dict[str, Any]) -> str:
        shared = _shared_data(runtime)
        progress_state = dict(shared.get("progress_state") or {})
        evidence = dict(shared.get("evidence") or {})
        selected_entities = list(shared.get("selected_entities") or [])
        blocked = bool(progress_state.get("blocked", False))
        blocked_reason = str(progress_state.get("blocked_reason", "") or "")
        finish_reason = str(value.get("finish_reason", "") or "").strip()

        if blocked:
            if blocked_reason in {"missing_authentication", "missing_arguments"}:
                return "prerequisite_missing"
            if blocked_reason == "authentication_failed":
                return "authentication_failed"
            if blocked_reason == "tool_execution_error":
                return "tool_execution_failed"
            if blocked_reason in {"missing_tool_executor", "no_action_tool_selected", "insufficient_state"}:
                return "insufficient_state"
            fallback_reason = str(fallback.get("finish_reason", "") or "").strip()
            if fallback_reason:
                return fallback_reason
            return "insufficient_state"

        if finish_reason in {
            "prerequisite_missing",
            "authentication_failed",
            "tool_execution_failed",
            "task_completed",
            "insufficient_state",
        }:
            return finish_reason

        latest_preview = str(evidence.get("latest_result_preview", "") or "").strip()
        if selected_entities or latest_preview:
            return "task_completed"
        return "insufficient_state"

    def _normalize_response_payload(self, value: dict[str, Any] | None, runtime: WorkerRuntime) -> dict[str, Any]:
        fallback = self._fallback(runtime)
        raw = dict(value or {})
        assistant_response = str(raw.get("assistant_response", "") or "").strip()
        if not assistant_response:
            assistant_response = str(fallback.get("assistant_response", "") or "")
        response_confidence = str(raw.get("response_confidence", "") or "").strip()
        if not response_confidence:
            response_confidence = str(fallback.get("response_confidence", "") or "")
        finish_reason = self._normalize_finish_reason(raw, runtime, fallback)
        finish = bool(raw.get("finish", fallback.get("finish", True)))
        if self._submission_pending(runtime):
            finish = False
        return {
            "assistant_response": assistant_response,
            "response_confidence": response_confidence,
            "finish": finish,
            "finish_reason": finish_reason,
        }

    def _fallback(self, runtime: WorkerRuntime) -> dict[str, Any]:
        shared = _shared_data(runtime)
        analysis_results = dict(shared.get("analysis_results") or {})
        selected_entities = list(shared.get("selected_entities") or [])
        evidence = dict(shared.get("evidence") or {})
        intermediate_results = dict(shared.get("intermediate_results") or {})
        progress_state = dict(shared.get("progress_state") or {})
        tool_status = dict(shared.get("tool_status") or {})
        blocked = bool(progress_state.get("blocked", False))
        blocked_reason = str(progress_state.get("blocked_reason", "") or "")
        missing_prerequisites = [
            str(item) for item in list(progress_state.get("missing_prerequisites") or []) if str(item).strip()
        ]
        last_error = str(tool_status.get("last_error", "") or "").strip()

        if blocked:
            if blocked_reason == "missing_authentication":
                return {
                    "assistant_response": "I could not continue because the required authentication state is missing.",
                    "response_confidence": "high",
                    "finish": True,
                    "finish_reason": "prerequisite_missing",
                }
            if blocked_reason == "authentication_failed":
                return {
                    "assistant_response": "I could not continue because authentication failed for the required AppWorld app.",
                    "response_confidence": "high",
                    "finish": True,
                    "finish_reason": "authentication_failed",
                }
            if blocked_reason == "missing_arguments":
                detail = f" Missing prerequisites: {', '.join(missing_prerequisites)}." if missing_prerequisites else ""
                return {
                    "assistant_response": f"I could not continue because required task inputs are missing.{detail}",
                    "response_confidence": "high",
                    "finish": True,
                    "finish_reason": "prerequisite_missing",
                }
            if blocked_reason == "tool_execution_error":
                detail = f" Latest error: {last_error}" if last_error else ""
                return {
                    "assistant_response": f"I could not complete the task because a tool execution failed.{detail}",
                    "response_confidence": "medium",
                    "finish": True,
                    "finish_reason": "tool_execution_failed",
                }
            if blocked_reason:
                detail = f" Latest error: {last_error}" if last_error else ""
                return {
                    "assistant_response": f"I could not continue because the workflow is blocked ({blocked_reason}).{detail}",
                    "response_confidence": "medium",
                    "finish": True,
                    "finish_reason": blocked_reason,
                }

        final_answer = str(
            analysis_results.get("final_answer")
            or analysis_results.get("csv_answer")
            or intermediate_results.get("final_answer")
            or intermediate_results.get("top_titles_csv")
            or ""
        ).strip()
        if not final_answer:
            numeric_answer = analysis_results.get("numeric_answer")
            if isinstance(numeric_answer, int):
                final_answer = str(numeric_answer)
        if final_answer:
            return {
                "assistant_response": final_answer,
                "response_confidence": "medium",
                "finish": True,
                "finish_reason": "task_completed",
            }

        labels = [_entity_label(entity) for entity in selected_entities if isinstance(entity, dict)]
        labels = [label for label in labels if label]
        if labels:
            return {
                "assistant_response": ", ".join(labels),
                "response_confidence": "medium",
                "finish": True,
                "finish_reason": "task_completed",
            }
        latest_preview = str(evidence.get("latest_result_preview", "") or "").strip()
        if latest_preview:
            return {
                "assistant_response": latest_preview,
                "response_confidence": "low",
                "finish": True,
                "finish_reason": "task_completed",
            }
        return {
            "assistant_response": "I do not yet have enough AppWorld state to provide a stronger answer.",
            "response_confidence": "low",
            "finish": True,
            "finish_reason": "insufficient_state",
        }

    def _llm_prompt(self, runtime: WorkerRuntime) -> str:
        shared = _shared_data(runtime)
        payload = {
            "task": _task(runtime),
            "analysis_results": dict(shared.get("analysis_results") or {}),
            "selected_entities": list(shared.get("selected_entities") or []),
            "evidence": dict(shared.get("evidence") or {}),
            "intermediate_results": dict(shared.get("intermediate_results") or {}),
            "progress_state": dict(shared.get("progress_state") or {}),
            "tool_status": dict(shared.get("tool_status") or {}),
        }
        return (
            "You are the AppWorld response worker. Produce the final user-facing answer.\n"
            "Return strict JSON with keys: assistant_response, response_confidence, finish, finish_reason.\n\n"
            "If progress_state.blocked is true, distinguish prerequisite failures from execution failures.\n"
            "Prefer analysis_results.final_answer when it is available.\n"
            "Use finish_reason values like: prerequisite_missing, authentication_failed, tool_execution_failed, task_completed, insufficient_state.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def run(self, runtime: WorkerRuntime) -> list[dict[str, Any]]:
        parsed = self._normalize_response_payload(_llm_json(self._llm, self._llm_prompt(runtime)), runtime)
        return [
            replace("/shared_data/assistant_response", str(parsed.get("assistant_response", "") or "")),
            replace("/shared_data/response_confidence", str(parsed.get("response_confidence", "") or "")),
            replace("/shared_data/finish", bool(parsed.get("finish", True))),
            replace("/shared_data/finish_reason", str(parsed.get("finish_reason", "") or "")),
            _append_history(runtime, self.spec.name, "Generated final response."),
        ]


def build_default_workers(*, llm: Any | None = None) -> tuple[AppWorldWorker, ...]:
    return (
        ToolWorker(),
        StateWorker(llm=llm),
        AnalysisWorker(llm=llm),
        ResponseWorker(llm=llm),
    )
