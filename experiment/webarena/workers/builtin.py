"""Builtin WebArena workers for the minimal blackboard flow."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

from experiment.common.neutral import Message, ToolCall
from experiment.webarena.core.target_urls import normalize_navigable_url
from experiment.webarena.workers.base import WebArenaWorker, WorkerResult, WorkerRuntime, WorkerSpec, replace


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_url(value: str) -> str:
    return normalize_navigable_url(value)


def _text_excerpt(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "over",
    "under",
    "all",
    "among",
    "any",
    "exactly",
}


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line and line.strip()]


def _extract_number_strings(text: str) -> list[str]:
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", str(text or ""))
    deduped: list[str] = []
    for item in matches:
        if item not in deduped:
            deduped.append(item)
    return deduped[:12]


def _extract_facts(text: str, *, tokens: list[str]) -> list[str]:
    facts: list[str] = []
    for line in _split_lines(text):
        lowered = line.lower()
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", lowered))
        has_token = any(token in lowered for token in tokens)
        if has_number or has_token:
            facts.append(_text_excerpt(line, limit=180))
        if len(facts) >= 10:
            break
    if facts:
        return facts
    excerpt = _text_excerpt(text, limit=180)
    return [excerpt] if excerpt else []


def _first_number_value(text: str) -> int | float | None:
    items = _extract_number_strings(text)
    if not items:
        return None
    token = items[0]
    if "." in token:
        try:
            return float(token)
        except ValueError:
            return None
    try:
        return int(token)
    except ValueError:
        return None


def _intent_tokens(goal: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(goal or "").lower())
    return [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]


def _extract_query_text(goal: str, *, tokens: list[str]) -> str:
    text = " ".join(str(goal or "").split())
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    for first, second in quoted:
        candidate = (first or second or "").strip()
        if candidate:
            return candidate[:120]
    for pattern in (
        r"search for (.+?)(?:[.?!]|$)",
        r"find (.+?)(?:[.?!]|$)",
        r"look up (.+?)(?:[.?!]|$)",
        r"get (.+?)(?:[.?!]|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = str(match.group(1) or "").strip(" ,.;:")
            if candidate:
                return candidate[:120]
    if tokens:
        return " ".join(tokens[:6])[:120]
    return ""


def _element_selector(element: dict[str, Any]) -> str:
    element_id = str(element.get("element_id", "") or "").strip()
    if element_id:
        return element_id
    text = " ".join(str(element.get("text", "") or "").split())
    if not text:
        return ""
    safe = text.replace('"', '\\"')
    return f"text={safe[:80]}"


def _element_score(element: dict[str, Any], *, tokens: list[str]) -> int:
    text = " ".join(
        [
            str(element.get("text", "") or ""),
            str(element.get("element_id", "") or ""),
            str(element.get("role", "") or ""),
            str(element.get("tag", "") or ""),
        ]
    ).lower()
    if not text:
        return 0
    return sum(1 for token in tokens if token in text)


def _is_click_like_element(element: dict[str, Any]) -> bool:
    tag = str(element.get("tag", "") or "").lower()
    role = str(element.get("role", "") or "").lower()
    return tag in {"a", "button"} or role in {"button", "link", "menuitem", "tab"}


def _pick_best_element(
    elements: list[dict[str, Any]],
    *,
    tokens: list[str],
    prefer_tags: set[str],
    exclude_selectors: set[str] | None = None,
) -> dict[str, Any] | None:
    excluded = set(exclude_selectors or set())
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        if not bool(element.get("enabled", True)):
            continue
        if not bool(element.get("visible", True)):
            continue
        tag = str(element.get("tag", "") or "").lower()
        role = str(element.get("role", "") or "").lower()
        if tag not in prefer_tags and role not in {"button", "link", "textbox", "searchbox", "combobox"}:
            continue
        selector = _element_selector(element)
        if not selector:
            continue
        if selector in excluded:
            continue
        score = _element_score(element, tokens=tokens)
        candidates.append((score, -index, element))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top_score, _, top = candidates[0]
    if top_score <= 0 and tokens:
        return None
    return top


def _coerce_schema_value(schema: dict[str, Any], *, text: str, facts: list[str]) -> Any:
    schema_type = str(schema.get("type", "") or "").strip().lower()
    default_text = _text_excerpt(text, limit=240)
    first_fact = facts[0] if facts else default_text
    if schema_type == "integer":
        value = _first_number_value(text)
        return int(value) if isinstance(value, (int, float)) else 0
    if schema_type == "number":
        value = _first_number_value(text)
        return float(value) if isinstance(value, (int, float)) else 0.0
    if schema_type == "boolean":
        return bool(first_fact)
    if schema_type == "array":
        item_schema = dict(schema.get("items") or {})
        if item_schema and str(item_schema.get("type", "") or "").strip().lower() in {"number", "integer"}:
            numbers = _extract_number_strings(text)
            if str(item_schema.get("type", "") or "").strip().lower() == "integer":
                return [int(float(item)) for item in numbers[:5]]
            return [float(item) for item in numbers[:5]]
        return facts[:5] if facts else ([default_text] if default_text else [])
    if schema_type == "object":
        properties = dict(schema.get("properties") or {})
        result: dict[str, Any] = {}
        for key, value in properties.items():
            key_name = str(key or "").strip()
            if not key_name:
                continue
            value_schema = value if isinstance(value, dict) else {"type": "string"}
            result[key_name] = _coerce_schema_value(value_schema, text=text, facts=facts)
        return result
    return str(first_fact or default_text or "")


def _build_structured_retrieval_response(
    *,
    schema: dict[str, Any],
    expected_template: Any,
    text: str,
    facts: list[str],
) -> str:
    candidate_schema = dict(schema or {})
    if not candidate_schema and isinstance(expected_template, dict):
        candidate_schema = {
            "type": "object",
            "properties": {str(k): {"type": "string"} for k in expected_template},
        }
    if not candidate_schema:
        if isinstance(expected_template, list):
            candidate_schema = {"type": "array", "items": {"type": "string"}}
        elif isinstance(expected_template, (int, float)):
            candidate_schema = {"type": "number"}
        elif isinstance(expected_template, bool):
            candidate_schema = {"type": "boolean"}
        else:
            candidate_schema = {"type": "string"}
    value = _coerce_schema_value(candidate_schema, text=text, facts=facts)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


_ALLOWED_AGENT_STATUSES = {
    "SUCCESS",
    "ACTION_NOT_ALLOWED_ERROR",
    "PERMISSION_DENIED_ERROR",
    "NOT_FOUND_ERROR",
    "DATA_VALIDATION_ERROR",
    "UNKNOWN_ERROR",
}


def _parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
    if code_block:
        text = str(code_block.group(1) or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _task_type_from_constraints(constraints: dict[str, Any]) -> str:
    category = str(constraints.get("task_category", "") or "").strip().lower()
    requires_final_response = bool(constraints.get("requires_final_response", False))
    if category == "navigation":
        return "NAVIGATE"
    if category in {"form", "interaction", "mutate"}:
        return "MUTATE"
    if category == "retrieval" or requires_final_response:
        return "RETRIEVE"
    return "MUTATE"


def _normalize_status(value: Any, *, default_success: bool) -> str:
    candidate = str(value or "").strip().upper()
    if candidate in _ALLOWED_AGENT_STATUSES:
        return candidate
    return "SUCCESS" if default_success else "UNKNOWN_ERROR"


def _coerce_retrieved_data_items(value: Any) -> list[Any]:
    parsed = _parse_json_like(value)
    if isinstance(parsed, (list, tuple)):
        items = list(parsed)
    elif parsed in (None, ""):
        items = []
    else:
        items = [parsed]
    normalized: list[Any] = []
    for item in items[:8]:
        if isinstance(item, (str, int, float, bool, dict)) or item is None:
            normalized.append(item)
        else:
            normalized.append(_text_excerpt(item, limit=240))
    return normalized


def _normalize_final_agent_response(
    *,
    payload: Any,
    constraints: dict[str, Any],
    verification: dict[str, Any],
    fallback_retrieved_data: list[Any],
) -> dict[str, Any]:
    candidate = dict(payload or {}) if isinstance(payload, dict) else {}
    default_task_type = _task_type_from_constraints(constraints)
    task_type = str(candidate.get("task_type") or candidate.get("performed_operation") or default_task_type).strip().upper()
    if task_type not in {"RETRIEVE", "MUTATE", "NAVIGATE"}:
        task_type = default_task_type
    default_success = bool(verification.get("goal_reached", False) or verification.get("answer_ready", False))
    status = _normalize_status(candidate.get("status"), default_success=default_success)
    if task_type == "RETRIEVE":
        retrieved_data = _coerce_retrieved_data_items(candidate.get("retrieved_data"))
        if not retrieved_data:
            retrieved_data = list(fallback_retrieved_data)
    else:
        retrieved_data = None
    error_details = candidate.get("error_details")
    if status == "SUCCESS":
        error_details = None
    elif not isinstance(error_details, str) or not error_details.strip():
        error_details = "Task could not be completed with available evidence."
    return {
        "task_type": task_type,
        "status": status,
        "retrieved_data": retrieved_data,
        "error_details": error_details,
    }


def _llm_json_safely(
    worker: WebArenaWorker,
    *,
    system_prompt: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    if not worker.llm_enabled:
        return None, {"worker_input_tokens": 0, "worker_output_tokens": 0}
    try:
        payload, usage = worker._invoke_json(system_prompt=system_prompt, context=context)
        return payload, usage
    except Exception:
        return None, {"worker_input_tokens": 0, "worker_output_tokens": 0}


def _heuristic_verification_from_state(
    *,
    constraints: dict[str, Any],
    current_page: dict[str, Any],
    page_evidence: dict[str, Any],
    previous_verification: dict[str, Any],
    action_history: list[Any],
) -> tuple[dict[str, Any], bool, str]:
    current_url = _normalize_url(current_page.get("url", ""))
    target_urls = [
        normalized
        for normalized in (_normalize_url(item) for item in list(constraints.get("target_urls", []) or []))
        if normalized
    ]
    requires_final_response = bool(constraints.get("requires_final_response", False))
    explicit_target_match = bool(current_url and current_url in target_urls)
    text_excerpt = str(page_evidence.get("text_excerpt") or current_page.get("visible_text_excerpt") or "")
    facts = list(page_evidence.get("facts", []) or [])
    number_mentions = list(page_evidence.get("number_mentions", []) or [])
    action_count = len(list(action_history or []))
    previous_action_count = int(previous_verification.get("action_count", 0) or 0)
    previous_facts_count = int(previous_verification.get("facts_count", 0) or 0)
    previous_url = _normalize_url(previous_verification.get("current_url", ""))
    intent_tokens = _intent_tokens(str(constraints.get("goal", "") or ""))
    token_hit_count = sum(1 for token in intent_tokens if token in text_excerpt.lower())
    has_target_urls = bool(target_urls)

    evidence_ready = explicit_target_match or bool(
        facts
        or number_mentions
        or token_hit_count >= 2
        or (len(text_excerpt) >= 120 and action_count > 0)
    )
    answer_ready = False
    if requires_final_response:
        answer_ready = explicit_target_match or bool(
            evidence_ready and (action_count >= 1 or not has_target_urls)
        )
        goal_reached = answer_ready
        needs_response = answer_ready
        finish = False
        finish_reason = ""
    else:
        semantic_hit_count = 0
        semantic_corpus = " ".join(
            [
                str(current_page.get("title", "") or ""),
                str(current_page.get("url", "") or ""),
                str(page_evidence.get("text_excerpt", "") or ""),
            ]
        ).lower()
        for token in intent_tokens:
            if token in semantic_corpus:
                semantic_hit_count += 1
        goal_reached = explicit_target_match or bool(
            (not has_target_urls)
            and (
                semantic_hit_count >= 2
                or (len(facts) >= 1 and action_count >= 1)
            )
        )
        needs_response = False
        finish = goal_reached
        finish_reason = "task_completed" if finish else ""

    progress_made = bool(
        (current_url and current_url != previous_url)
        or action_count > previous_action_count
        or len(facts) > previous_facts_count
    )
    if requires_final_response:
        if goal_reached or answer_ready:
            phase = "respond"
        elif has_target_urls and not explicit_target_match:
            phase = "navigate"
        else:
            phase = "evidence"
    else:
        phase = "complete" if goal_reached else "navigate"

    verification = {
        "goal_reached": goal_reached,
        "matched_target_url": current_url if goal_reached else "",
        "requires_final_response": requires_final_response,
        "needs_response": needs_response,
        "progress_made": progress_made or goal_reached,
        "evidence_ready": evidence_ready,
        "answer_ready": answer_ready,
        "action_count": action_count,
        "facts_count": len(facts),
        "number_mentions_count": len(number_mentions),
        "current_url": current_url,
        "previous_url": previous_url,
        "phase": phase,
        "task_category": str(constraints.get("task_category", "") or ""),
        "updated_at": _utc_now(),
    }
    return verification, finish, finish_reason


class PageStateWorker(WebArenaWorker):
    spec = WorkerSpec(
        name="page_state_worker",
        description="Summarize the latest browser observation into a normalized page state.",
        reads=("task_constraints", "last_observation", "current_page", "page_evidence", "verification", "action_history"),
        writes=("current_page", "page_evidence", "open_tabs", "verification", "finish", "finish_reason"),
        is_llm=True,
    )

    system_prompt = """You are the WebArena page_state_worker.

Return ONLY a JSON object with:
- current_page: object
- page_evidence: object
- open_tabs: array
- verification: object
- finish: boolean
- finish_reason: string

Rules:
- Do not choose browser actions
- Summarize only the latest observation into stable page state and state variables
- Keep current_page focused on url, title, visible_text_excerpt, interactive_elements, active_tab_index
- page_evidence must include whether the current URL matches a target URL
- verification must include: goal_reached, requires_final_response, needs_response, progress_made, evidence_ready, answer_ready, phase
- phase must be one of: navigate, evidence, respond, complete
- If output format is invalid, it will be rejected and you must retry with corrected JSON
"""

    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        shared = dict(runtime.state.get("shared_data") or {})
        constraints = dict(runtime.state.get("task_constraints") or {})
        observation = dict(shared.get("last_observation") or {})
        raw_elements = observation.get("elements")
        if raw_elements is None:
            raw_elements = observation.get("interactive_elements")
        raw_tabs = observation.get("tabs")
        if raw_tabs is None:
            raw_tabs = observation.get("open_tabs")
        current_page = {
            "url": str(observation.get("url", "") or ""),
            "title": str(observation.get("title") or observation.get("page_title") or ""),
            "visible_text_excerpt": _text_excerpt(observation.get("text") or observation.get("visible_text") or ""),
            "interactive_elements": list(raw_elements or [])[:50],
            "active_tab_index": int(observation.get("active_tab_index", 0) or 0),
        }
        target_urls = [
            normalized
            for normalized in (_normalize_url(item) for item in list(constraints.get("target_urls", []) or []))
            if normalized
        ]
        current_url = _normalize_url(current_page["url"])
        intent_tokens = _intent_tokens(str(constraints.get("goal", "") or ""))
        facts = _extract_facts(current_page["visible_text_excerpt"], tokens=intent_tokens)
        number_mentions = _extract_number_strings(current_page["visible_text_excerpt"])
        page_evidence = {
            "url": current_page["url"],
            "title": current_page["title"],
            "text_excerpt": current_page["visible_text_excerpt"],
            "target_url_match": current_url in target_urls if current_url else False,
            "element_count": len(list(raw_elements or [])),
            "facts": facts,
            "number_mentions": number_mentions,
            "updated_at": _utc_now(),
        }
        previous_verification = dict(shared.get("verification") or {})
        action_history = list(shared.get("action_history", []) or [])
        verification, finish, finish_reason = _heuristic_verification_from_state(
            constraints=constraints,
            current_page=current_page,
            page_evidence=page_evidence,
            previous_verification=previous_verification,
            action_history=action_history,
        )
        open_tabs = list(raw_tabs or [])
        usage = {"worker_input_tokens": 0, "worker_output_tokens": 0}
        if self.llm_enabled:
            llm_payload: dict[str, Any] | None = None
            max_attempts = 3
            retry_feedback = ""
            for _ in range(max_attempts):
                payload, attempt_usage = self._invoke_json(
                    system_prompt=self.system_prompt,
                    context={
                        "task_constraints": constraints,
                        "last_observation": observation,
                        "previous_state": {
                            "current_page": dict(shared.get("current_page") or {}),
                            "page_evidence": dict(shared.get("page_evidence") or {}),
                            "verification": previous_verification,
                            "finish": bool(shared.get("finish", False)),
                            "finish_reason": str(shared.get("finish_reason", "") or ""),
                        },
                        "action_history": action_history[-10:],
                        "heuristic_current_page": current_page,
                        "heuristic_page_evidence": page_evidence,
                        "retry_feedback": retry_feedback,
                    },
                )
                usage["worker_input_tokens"] += int(attempt_usage.get("worker_input_tokens", 0) or 0)
                usage["worker_output_tokens"] += int(attempt_usage.get("worker_output_tokens", 0) or 0)
                try:
                    llm_current_page = dict(payload.get("current_page") or {})
                    llm_page_evidence = dict(payload.get("page_evidence") or {})
                    llm_verification = dict(payload.get("verification") or {})
                    llm_tabs = payload.get("open_tabs")
                    llm_finish = payload.get("finish")
                    llm_finish_reason = payload.get("finish_reason")
                    if not llm_current_page:
                        raise ValueError("missing current_page")
                    if not llm_page_evidence:
                        raise ValueError("missing page_evidence")
                    if not llm_verification:
                        raise ValueError("missing verification")
                    required_verification = {
                        "goal_reached",
                        "requires_final_response",
                        "needs_response",
                        "progress_made",
                        "evidence_ready",
                        "answer_ready",
                        "phase",
                    }
                    missing = sorted(required_verification.difference(set(llm_verification)))
                    if missing:
                        raise ValueError(f"verification missing keys: {', '.join(missing)}")
                    phase = str(llm_verification.get("phase", "") or "").strip().lower()
                    if phase not in {"navigate", "evidence", "respond", "complete"}:
                        raise ValueError(f"invalid phase: {phase!r}")
                    llm_verification["phase"] = phase
                    merged_current_page = dict(current_page)
                    merged_current_page.update(llm_current_page)
                    if not list(merged_current_page.get("interactive_elements", []) or []):
                        merged_current_page["interactive_elements"] = list(current_page.get("interactive_elements", []) or [])
                    current_page = merged_current_page
                    merged_page_evidence = dict(page_evidence)
                    merged_page_evidence.update(llm_page_evidence)
                    page_evidence = merged_page_evidence
                    if isinstance(llm_tabs, list):
                        open_tabs = list(llm_tabs)
                    verification = llm_verification
                    finish = bool(llm_finish)
                    finish_reason = str(llm_finish_reason or "")
                    llm_payload = payload
                    break
                except Exception as exc:
                    retry_feedback = str(exc)
                    llm_payload = None
            if llm_payload is None:
                raise ValueError(f"page_state_worker failed to produce valid state payload after retries: {retry_feedback}")
        return WorkerResult(
            patch=(
                replace("/shared_data/current_page", current_page),
                replace("/shared_data/page_evidence", page_evidence),
                replace("/shared_data/open_tabs", open_tabs),
                replace("/shared_data/verification", verification),
                replace("/shared_data/finish", finish),
                replace("/shared_data/finish_reason", finish_reason),
            ),
            metadata=usage,
        )


class ArgumentGroundingWorker(WebArenaWorker):
    spec = WorkerSpec(
        name="argument_grounding_worker",
        description="Bind the next browser action from task constraints and current page state.",
        reads=("task_constraints", "current_page", "page_evidence", "verification", "action_history"),
        writes=("grounded_action", "action_arguments"),
        is_llm=False,
    )

    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        constraints = dict(runtime.state.get("task_constraints") or {})
        shared = dict(runtime.state.get("shared_data") or {})
        current_page = dict(shared.get("current_page") or {})
        verification = dict(shared.get("verification") or {})
        action_history = list(shared.get("action_history", []) or [])

        grounded_action: dict[str, Any] = {}
        current_url = _normalize_url(current_page.get("url", ""))
        enable_target_urls = bool(constraints.get("enable_target_urls", False))
        target_urls = []
        if enable_target_urls:
            target_urls = [
                normalized
                for normalized in (_normalize_url(item) for item in list(constraints.get("target_urls", []) or []))
                if normalized
            ]
        intent = str(constraints.get("goal", "") or "")
        intent_tokens = _intent_tokens(intent)
        elements = list(current_page.get("interactive_elements", []) or [])
        last_action = dict(action_history[-1] or {}) if action_history else {}
        last_tool = str(last_action.get("tool_name", "") or "").strip()
        explicit_target_match = bool(current_url and current_url in target_urls)
        default_phase = "navigate"
        if bool(constraints.get("requires_final_response", False)):
            if bool(verification.get("answer_ready", False) or verification.get("goal_reached", False)):
                default_phase = "respond"
            elif bool(target_urls) and not explicit_target_match:
                default_phase = "navigate"
            else:
                default_phase = "evidence"
        elif bool(verification.get("goal_reached", False)):
            default_phase = "complete"
        phase = str(verification.get("phase", "") or default_phase).strip().lower()
        avoid_selector = ""
        if not bool(verification.get("progress_made", False)) and last_tool == "browser__click":
            avoid_selector = str(dict(last_action.get("arguments") or {}).get("element_id", "") or "").strip()
        excluded_selectors = {avoid_selector} if avoid_selector else set()

        if phase in {"respond", "complete"}:
            grounded_action = {
                "tool_name": "",
                "arguments": {},
                "rationale": "No browser action is required in response/completion phase.",
            }

        if (
            not grounded_action
            and enable_target_urls
            and not bool(verification.get("goal_reached", False))
        ):
            for target_url in target_urls:
                if target_url and target_url != current_url:
                    grounded_action = {
                        "tool_name": "browser__goto",
                        "arguments": {"url": target_url},
                        "rationale": "Navigate directly to the expected target URL.",
                    }
                    break

        if not grounded_action and not bool(verification.get("goal_reached", False)):
            click_candidate = _pick_best_element(
                elements,
                tokens=intent_tokens,
                prefer_tags={"a", "button"},
                exclude_selectors=excluded_selectors,
            )
            if click_candidate is None:
                click_candidate = _pick_best_element(
                    elements,
                    tokens=[],
                    prefer_tags={"a", "button"},
                    exclude_selectors=excluded_selectors,
                )
            if click_candidate and _is_click_like_element(click_candidate):
                selector = _element_selector(click_candidate)
                if selector:
                    grounded_action = {
                        "tool_name": "browser__click",
                        "arguments": {"element_id": selector},
                        "rationale": "Click the most task-relevant actionable element on the current page.",
                    }

        if not grounded_action and not bool(verification.get("goal_reached", False)):
            input_candidate = _pick_best_element(
                elements,
                tokens=intent_tokens,
                prefer_tags={"input", "textarea"},
                exclude_selectors=excluded_selectors,
            )
            query = _extract_query_text(intent, tokens=intent_tokens)
            if input_candidate and query:
                selector = _element_selector(input_candidate)
                if selector:
                    typed_same_target = False
                    if last_tool == "browser__type":
                        last_arguments = dict(last_action.get("arguments") or {})
                        typed_same_target = str(last_arguments.get("element_id", "") or "").strip() == selector
                    if typed_same_target:
                        grounded_action = {
                            "tool_name": "browser__press",
                            "arguments": {"element_id": selector, "key": "Enter"},
                            "rationale": "Submit the previously typed task keyword query.",
                        }
                    else:
                        grounded_action = {
                            "tool_name": "browser__type",
                            "arguments": {"element_id": selector, "text": query},
                            "rationale": "Search with task keywords when no better clickable action is available.",
                        }

        if not grounded_action:
            scroll_y = 700.0
            if last_tool == "browser__scroll" and not bool(verification.get("progress_made", False)):
                last_y = float(dict(last_action.get("arguments") or {}).get("y", 0.0) or 0.0)
                scroll_y = -600.0 if last_y >= 0 else 700.0
            grounded_action = {
                "tool_name": "browser__scroll",
                "arguments": {"y": scroll_y},
                "rationale": "No relevant button is available on the current page; scroll to discover additional actionable elements.",
            }

        return WorkerResult(
            patch=(
                replace("/shared_data/grounded_action", grounded_action),
                replace("/shared_data/action_arguments", dict(grounded_action.get("arguments") or {})),
            )
        )


class BrowserActionWorker(WebArenaWorker):
    spec = WorkerSpec(
        name="browser_action_worker",
        description="Request one structured browser action from the grounded action slot.",
        reads=("grounded_action",),
        writes=("last_action", "action_history", "execution_error"),
        is_llm=False,
        can_use_tools=True,
    )

    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        execution = dict(runtime.state.get("execution") or {})
        shared = dict(runtime.state.get("shared_data") or {})
        grounded_action = dict(shared.get("grounded_action") or {})
        tool_name = str(grounded_action.get("tool_name", "") or "").strip()
        arguments = dict(grounded_action.get("arguments") or {})
        rationale = str(grounded_action.get("rationale", "") or "")

        if not tool_name:
            return WorkerResult(
                patch=(replace("/shared_data/execution_error", "No grounded browser action is available."),),
            )
        if tool_name not in runtime.tools_by_name:
            return WorkerResult(
                patch=(replace("/shared_data/execution_error", f"Unsupported browser tool: {tool_name}"),),
            )

        step_counter = int(execution.get("tool_call_counter", 0) or 0) + 1
        call_id = f"webarena-call-{step_counter}"
        last_action = {
            "tool_name": tool_name,
            "arguments": arguments,
            "rationale": rationale,
            "call_id": call_id,
            "requested_at": _utc_now(),
        }
        action_history = list(shared.get("action_history", []) or [])
        action_history.append(last_action)

        return WorkerResult(
            patch=(
                replace("/shared_data/last_action", last_action),
                replace("/shared_data/action_history", action_history[-20:]),
                replace("/shared_data/execution_error", ""),
                replace("/execution/tool_call_counter", step_counter),
            ),
            tool_call=ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                call_id=call_id,
                rationale=rationale,
                metadata={"worker": self.spec.name},
            ),
        )


class VerificationWorker(WebArenaWorker):
    spec = WorkerSpec(
        name="verification_worker",
        description="Judge whether the task has reached its goal or needs more browser work.",
        reads=("task_constraints", "current_page", "page_evidence", "last_observation", "action_history"),
        writes=("verification", "finish", "finish_reason"),
        is_llm=True,
    )

    system_prompt = """You are the WebArena verification_worker.

Return ONLY a JSON object with:
- verification: object
- finish: boolean
- finish_reason: string

Rules:
- Judge only whether the current page indicates progress or completion
- Do not choose actions or re-plan workers
- verification must include goal_reached, progress_made, matched_target_url, requires_final_response, needs_response, task_category, evidence_ready, answer_ready, phase
"""

    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        constraints = dict(runtime.state.get("task_constraints") or {})
        shared = dict(runtime.state.get("shared_data") or {})
        previous_verification = dict(shared.get("verification") or {})
        current_page = dict(shared.get("current_page") or {})
        page_evidence = dict(shared.get("page_evidence") or {})
        action_history = list(shared.get("action_history", []) or [])
        current_url = _normalize_url(current_page.get("url", ""))
        target_urls = [
            normalized
            for normalized in (_normalize_url(item) for item in list(constraints.get("target_urls", []) or []))
            if normalized
        ]
        requires_final_response = bool(constraints.get("requires_final_response", False))
        explicit_target_match = bool(current_url and current_url in target_urls)
        text_excerpt = str(
            page_evidence.get("text_excerpt")
            or current_page.get("visible_text_excerpt")
            or ""
        )
        facts = list(page_evidence.get("facts", []) or [])
        number_mentions = list(page_evidence.get("number_mentions", []) or [])
        action_count = len(action_history)
        previous_action_count = int(previous_verification.get("action_count", 0) or 0)
        previous_facts_count = int(previous_verification.get("facts_count", 0) or 0)
        previous_url = _normalize_url(previous_verification.get("current_url", ""))
        intent_tokens = _intent_tokens(str(constraints.get("goal", "") or ""))
        token_hit_count = sum(1 for token in intent_tokens if token in text_excerpt.lower())

        has_target_urls = bool(target_urls)
        evidence_ready = explicit_target_match or bool(
            facts
            or number_mentions
            or token_hit_count >= 2
            or (len(text_excerpt) >= 120 and action_count > 0)
        )
        answer_ready = False
        if requires_final_response:
            answer_ready = explicit_target_match or bool(
                evidence_ready and (action_count >= 1 or not has_target_urls)
            )
        if requires_final_response:
            goal_reached = answer_ready
            finish = False
            finish_reason = ""
            needs_response = answer_ready
        else:
            goal_reached = explicit_target_match
            finish = goal_reached
            finish_reason = "task_completed" if finish else ""
            needs_response = False

        progress_made = bool(
            (current_url and current_url != previous_url)
            or action_count > previous_action_count
            or len(facts) > previous_facts_count
        )
        if requires_final_response:
            if goal_reached or answer_ready:
                phase = "respond"
            elif has_target_urls and not explicit_target_match:
                phase = "navigate"
            else:
                phase = "evidence"
        else:
            phase = "complete" if goal_reached else "navigate"
        verification = {
            "goal_reached": goal_reached,
            "matched_target_url": current_url if goal_reached else "",
            "requires_final_response": requires_final_response,
            "needs_response": needs_response,
            "progress_made": progress_made or goal_reached,
            "evidence_ready": evidence_ready,
            "answer_ready": answer_ready,
            "action_count": action_count,
            "facts_count": len(facts),
            "number_mentions_count": len(number_mentions),
            "current_url": current_url,
            "previous_url": previous_url,
            "phase": phase,
            "task_category": str(constraints.get("task_category", "") or ""),
            "updated_at": _utc_now(),
        }
        llm_payload, usage = _llm_json_safely(
            self,
            system_prompt=self.system_prompt,
            context={
                "task_constraints": constraints,
                "current_page": current_page,
                "page_evidence": page_evidence,
                "last_observation": dict(shared.get("last_observation") or {}),
                "action_history": action_history,
                "heuristic_verification": verification,
                "heuristic_finish": finish,
                "heuristic_finish_reason": finish_reason,
            },
        )
        if isinstance(llm_payload, dict):
            verification = dict(llm_payload.get("verification") or verification)
            finish = bool(llm_payload.get("finish", finish))
            finish_reason = str(llm_payload.get("finish_reason", finish_reason) or "")
        if "phase" not in verification or not str(verification.get("phase", "") or "").strip():
            if requires_final_response:
                verification["phase"] = "respond" if bool(verification.get("answer_ready", False) or verification.get("goal_reached", False)) else "evidence"
            else:
                verification["phase"] = "complete" if bool(verification.get("goal_reached", False)) else "navigate"
        if requires_final_response:
            verification["requires_final_response"] = True
            verification["needs_response"] = bool(verification.get("goal_reached", False) or verification.get("answer_ready", False))
            verification["phase"] = str(
                verification.get("phase")
                or ("respond" if bool(verification.get("answer_ready", False)) else "evidence")
            )
            finish = False
            finish_reason = ""
        return WorkerResult(
            patch=(
                replace("/shared_data/verification", verification),
                replace("/shared_data/finish", finish),
                replace("/shared_data/finish_reason", finish_reason),
            ),
            metadata=usage,
        )


class ResponseWorker(WebArenaWorker):
    spec = WorkerSpec(
        name="response_worker",
        description="Compose the final textual answer from verified page evidence.",
        reads=("verification", "page_evidence", "current_page"),
        writes=("assistant_response",),
        is_llm=True,
    )

    system_prompt = """You are the WebArena response_worker.

Return ONLY a JSON object with:
- assistant_response: final response object with keys task_type, status, retrieved_data, error_details

Rules:
- task_type must be RETRIEVE, MUTATE, or NAVIGATE
- status must be one of SUCCESS, ACTION_NOT_ALLOWED_ERROR, PERMISSION_DENIED_ERROR, NOT_FOUND_ERROR, DATA_VALIDATION_ERROR, UNKNOWN_ERROR
- For RETRIEVE tasks, retrieved_data must be an array (even for a single value)
- For non-RETRIEVE tasks, retrieved_data must be null
- Use verified page evidence
- Return valid JSON only
"""

    def run(self, runtime: WorkerRuntime) -> WorkerResult:
        constraints = dict(runtime.state.get("task_constraints") or {})
        shared = dict(runtime.state.get("shared_data") or {})
        page_evidence = dict(shared.get("page_evidence") or {})
        current_page = dict(shared.get("current_page") or {})
        verification = dict(shared.get("verification") or {})
        text = str(
            page_evidence.get("text_excerpt")
            or current_page.get("visible_text_excerpt")
            or current_page.get("title")
            or current_page.get("url")
            or "Task completed."
        )
        facts = list(page_evidence.get("facts", []) or _extract_facts(text, tokens=_intent_tokens(str(constraints.get("goal", "") or ""))))
        response = _text_excerpt(text)
        if bool(constraints.get("requires_final_response", False)):
            retrieved_candidate = _build_structured_retrieval_response(
                schema=dict(constraints.get("response_schema") or {}),
                expected_template=constraints.get("expected_response_template"),
                text=text,
                facts=facts,
            )
            fallback_retrieved_data = _coerce_retrieved_data_items(retrieved_candidate)
            response_payload = _normalize_final_agent_response(
                payload={},
                constraints=constraints,
                verification=verification,
                fallback_retrieved_data=fallback_retrieved_data,
            )
        elif bool(verification.get("goal_reached", False)) and current_page.get("title"):
            response = f"{current_page.get('title')}: {response}"
        llm_payload, usage = _llm_json_safely(
            self,
            system_prompt=self.system_prompt,
            context={
                "task_constraints": constraints,
                "verification": verification,
                "page_evidence": page_evidence,
                "current_page": current_page,
                "heuristic_response": response,
            },
        )
        if bool(constraints.get("requires_final_response", False)):
            llm_candidate = None
            if isinstance(llm_payload, dict):
                llm_candidate = llm_payload.get("assistant_response", llm_payload)
            response_payload = _normalize_final_agent_response(
                payload=_parse_json_like(llm_candidate),
                constraints=constraints,
                verification=verification,
                fallback_retrieved_data=fallback_retrieved_data,
            )
            response = json.dumps(response_payload, ensure_ascii=False)
        elif isinstance(llm_payload, dict):
            response = str(llm_payload.get("assistant_response", response) or response)
        return WorkerResult(
            patch=(replace("/shared_data/assistant_response", response),),
            message=Message(role="assistant", content=response, metadata={"worker": self.spec.name}),
            finish_reason="task_completed",
            metadata=usage,
        )


def build_default_workers(*, llm: Any | None = None) -> tuple[WebArenaWorker, ...]:
    return (
        PageStateWorker(llm=llm),
        ArgumentGroundingWorker(),
        BrowserActionWorker(),
        ResponseWorker(llm=llm),
    )
