from __future__ import annotations

import json

from experiment.webarena.workers.builtin import (
    ArgumentGroundingWorker,
    PageStateWorker,
    ResponseWorker,
    VerificationWorker,
)
from experiment.webarena.workers.base import WorkerRuntime


class _FakeLLMResponse:
    def __init__(self, payload: dict, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.content = json.dumps(payload, ensure_ascii=False)
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, messages):
        self.messages = messages
        return _FakeLLMResponse(self.payload)


def test_page_state_worker_extracts_current_page_and_evidence():
    worker = PageStateWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"target_urls": ["http://localhost:8023/dashboard/todos"]},
            "shared_data": {
                "last_observation": {
                    "url": "http://localhost:8023/dashboard/todos",
                    "title": "Your To-Do List",
                    "text": "Todo item one. Todo item two.",
                    "elements": [{"element_id": "done", "tag": "button", "text": "Done", "enabled": True}],
                    "tabs": [{"index": 0, "url": "http://localhost:8023/dashboard/todos", "title": "Your To-Do List", "active": True}],
                    "active_tab_index": 0,
                }
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert payload["/shared_data/current_page"]["url"] == "http://localhost:8023/dashboard/todos"
    assert payload["/shared_data/page_evidence"]["target_url_match"] is True
    assert payload["/shared_data/open_tabs"][0]["active"] is True


def test_page_state_worker_uses_llm_override_when_available():
    worker = PageStateWorker(
        llm=_FakeLLM(
            {
                "current_page": {"url": "http://override", "title": "LLM Title", "visible_text_excerpt": "LLM summary", "interactive_elements": [], "active_tab_index": 0},
                "page_evidence": {"url": "http://override", "title": "LLM Title", "text_excerpt": "LLM summary", "target_url_match": False, "element_count": 0},
                "open_tabs": [{"index": 0, "url": "http://override", "title": "LLM Title", "active": True}],
                "verification": {
                    "goal_reached": False,
                    "requires_final_response": False,
                    "needs_response": False,
                    "progress_made": True,
                    "evidence_ready": False,
                    "answer_ready": False,
                    "phase": "navigate",
                },
                "finish": False,
                "finish_reason": "",
            }
        )
    )
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"target_urls": ["http://localhost:8023/dashboard/todos"]},
            "shared_data": {"last_observation": {"url": "http://localhost:8023/dashboard/todos", "title": "Your To-Do List", "text": "Todo"}},
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert payload["/shared_data/current_page"]["url"] == "http://override"
    assert result.metadata["worker_input_tokens"] == 11
    assert result.metadata["worker_output_tokens"] == 7


def test_verification_worker_marks_navigation_complete():
    worker = VerificationWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "target_urls": ["http://localhost:8023/dashboard/todos"],
                "requires_final_response": False,
                "task_category": "navigation",
            },
            "shared_data": {
                "current_page": {"url": "http://localhost:8023/dashboard/todos"},
                "page_evidence": {},
                "last_observation": {},
                "action_history": [],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert payload["/shared_data/verification"]["goal_reached"] is True
    assert payload["/shared_data/verification"]["phase"] == "complete"
    assert payload["/shared_data/finish"] is True
    assert payload["/shared_data/finish_reason"] == "task_completed"


def test_argument_grounding_worker_ignores_invalid_target_urls():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"target_urls": ["_", "^"]},
            "shared_data": {
                "current_page": {"url": "http://localhost:8023"},
                "page_evidence": {},
                "verification": {"goal_reached": False},
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == "browser__scroll"


def test_argument_grounding_worker_normalizes_regex_target_url():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"target_urls": ["^http://localhost:8023/repo/-/issues.*$"]},
            "shared_data": {
                "current_page": {"url": "http://localhost:8023"},
                "page_evidence": {},
                "verification": {"goal_reached": False},
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == "browser__scroll"


def test_argument_grounding_worker_clicks_relevant_element_without_target_url():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Open issues page", "target_urls": []},
            "shared_data": {
                "current_page": {
                    "url": "http://localhost:8023",
                    "interactive_elements": [
                        {"element_id": "", "tag": "a", "role": "link", "text": "Issues", "enabled": True},
                        {"element_id": "", "tag": "a", "role": "link", "text": "Projects", "enabled": True},
                    ],
                },
                "page_evidence": {},
                "verification": {"goal_reached": False},
                "action_history": [],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == "browser__click"
    assert payload["/shared_data/grounded_action"]["arguments"]["element_id"].startswith("text=Issues")


def test_argument_grounding_worker_ignores_hidden_candidates():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Open issues page", "target_urls": []},
            "shared_data": {
                "current_page": {
                    "url": "http://localhost:8023",
                    "interactive_elements": [
                        {"element_id": "", "tag": "a", "role": "link", "text": "Issues", "enabled": True, "visible": False},
                        {"element_id": "", "tag": "a", "role": "link", "text": "Issues", "enabled": True, "visible": True},
                    ],
                },
                "page_evidence": {},
                "verification": {"goal_reached": False},
                "action_history": [],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == "browser__click"
    assert payload["/shared_data/grounded_action"]["arguments"]["element_id"].startswith("text=Issues")


def test_argument_grounding_worker_types_then_submits_search_query():
    worker = ArgumentGroundingWorker()
    runtime_first = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Search for wireless mouse", "target_urls": []},
            "shared_data": {
                "current_page": {
                    "url": "http://localhost:7770",
                    "interactive_elements": [
                        {"element_id": "search", "tag": "input", "role": "searchbox", "text": "", "enabled": True}
                    ],
                },
                "page_evidence": {},
                "verification": {"goal_reached": False},
                "action_history": [],
            },
        },
        tools_by_name={},
    )
    first = worker.run(runtime_first)
    first_payload = {op["path"]: op["value"] for op in first.patch}
    assert first_payload["/shared_data/grounded_action"]["tool_name"] == "browser__type"
    assert "wireless mouse" in first_payload["/shared_data/grounded_action"]["arguments"]["text"]

    runtime_second = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Search for wireless mouse", "target_urls": []},
            "shared_data": {
                "current_page": {
                    "url": "http://localhost:7770",
                    "interactive_elements": [
                        {"element_id": "search", "tag": "input", "role": "searchbox", "text": "", "enabled": True}
                    ],
                },
                "page_evidence": {},
                "verification": {"goal_reached": False},
                "action_history": [
                    {
                        "tool_name": "browser__type",
                        "arguments": {"element_id": "search", "text": "wireless mouse"},
                    }
                ],
            },
        },
        tools_by_name={},
    )
    second = worker.run(runtime_second)
    second_payload = {op["path"]: op["value"] for op in second.patch}
    assert second_payload["/shared_data/grounded_action"]["tool_name"] == "browser__press"
    assert second_payload["/shared_data/grounded_action"]["arguments"]["key"] == "Enter"


def test_argument_grounding_worker_scrolls_when_no_actionable_elements():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Find order status", "target_urls": []},
            "shared_data": {
                "current_page": {"url": "http://localhost:7780/admin", "interactive_elements": []},
                "page_evidence": {},
                "verification": {"goal_reached": False, "progress_made": False},
                "action_history": [{"tool_name": "browser__scroll", "arguments": {"y": 700}}],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == "browser__scroll"
    assert payload["/shared_data/grounded_action"]["arguments"]["y"] < 0


def test_argument_grounding_worker_skips_actions_in_response_phase():
    worker = ArgumentGroundingWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {"goal": "Get top product", "target_urls": [], "requires_final_response": True},
            "shared_data": {
                "current_page": {
                    "url": "http://localhost:7780/admin",
                    "interactive_elements": [
                        {"element_id": "", "tag": "a", "role": "link", "text": "Dashboard", "enabled": True},
                    ],
                },
                "page_evidence": {"text_excerpt": "Quest Lumaflex Band 6"},
                "verification": {"goal_reached": True, "answer_ready": True, "phase": "respond"},
                "action_history": [],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/grounded_action"]["tool_name"] == ""


def test_verification_worker_uses_llm_override_when_available():
    worker = VerificationWorker(
        llm=_FakeLLM(
            {
                "verification": {
                    "goal_reached": False,
                    "matched_target_url": "",
                    "requires_final_response": False,
                    "needs_response": False,
                    "progress_made": False,
                    "task_category": "navigation",
                },
                "finish": False,
                "finish_reason": "",
            }
        )
    )
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "target_urls": ["http://localhost:8023/dashboard/todos"],
                "requires_final_response": False,
                "task_category": "navigation",
            },
            "shared_data": {
                "current_page": {"url": "http://localhost:8023/dashboard/todos"},
                "page_evidence": {},
                "last_observation": {},
                "action_history": [],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert payload["/shared_data/verification"]["goal_reached"] is False
    assert payload["/shared_data/finish"] is False
    assert result.metadata["worker_input_tokens"] == 11


def test_verification_worker_never_finishes_when_response_required():
    worker = VerificationWorker(
        llm=_FakeLLM(
            {
                "verification": {
                    "goal_reached": True,
                    "matched_target_url": "http://localhost:8023/dashboard/todos",
                    "requires_final_response": True,
                    "needs_response": True,
                    "progress_made": True,
                    "task_category": "retrieval",
                    "answer_ready": True,
                    "evidence_ready": True,
                },
                "finish": True,
                "finish_reason": "task_completed",
            }
        )
    )
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "target_urls": ["http://localhost:8023/dashboard/todos"],
                "requires_final_response": True,
                "task_category": "retrieval",
            },
            "shared_data": {
                "current_page": {"url": "http://localhost:8023/dashboard/todos"},
                "page_evidence": {"text_excerpt": "Todo item one."},
                "last_observation": {},
                "action_history": [{"tool_name": "browser__goto", "arguments": {"url": "http://localhost:8023/dashboard/todos"}}],
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    assert payload["/shared_data/verification"]["needs_response"] is True
    assert payload["/shared_data/finish"] is False
    assert payload["/shared_data/finish_reason"] == ""


def test_verification_worker_marks_retrieval_answer_ready_after_actions():
    worker = VerificationWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "target_urls": [],
                "requires_final_response": True,
                "task_category": "retrieval",
                "goal": "Get total number of pending reviews",
            },
            "shared_data": {
                "current_page": {"url": "http://localhost:7780/admin/reviews"},
                "page_evidence": {
                    "text_excerpt": "Pending Reviews 5 Approved Reviews 12",
                    "facts": ["Pending Reviews 5", "Approved Reviews 12"],
                    "number_mentions": ["5", "12"],
                },
                "last_observation": {},
                "action_history": [
                    {"tool_name": "browser__click", "arguments": {"element_id": "text=MARKETING"}},
                    {"tool_name": "browser__click", "arguments": {"element_id": "text=Pending Reviews"}},
                ],
                "verification": {"action_count": 1, "facts_count": 1, "current_url": "http://localhost:7780/admin"},
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    verification = payload["/shared_data/verification"]

    assert verification["requires_final_response"] is True
    assert verification["answer_ready"] is True
    assert verification["goal_reached"] is True
    assert payload["/shared_data/finish"] is False


def test_response_worker_builds_final_message_from_page_evidence():
    worker = ResponseWorker()
    runtime = WorkerRuntime(
        state={
            "shared_data": {
                "verification": {"goal_reached": True},
                "page_evidence": {"text_excerpt": "Todo item one. Todo item two."},
                "current_page": {"title": "Your To-Do List", "visible_text_excerpt": "Todo item one. Todo item two."},
            }
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert result.message is not None
    assert result.finish_reason == "task_completed"
    assert "Your To-Do List" in payload["/shared_data/assistant_response"]


def test_response_worker_formats_retrieval_object_response_from_schema():
    worker = ResponseWorker()
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "requires_final_response": True,
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "pending_count": {"type": "integer"},
                        "summary": {"type": "string"},
                    },
                },
                "goal": "Get total number of pending reviews",
            },
            "shared_data": {
                "verification": {"goal_reached": True},
                "page_evidence": {
                    "text_excerpt": "Pending Reviews 5 Approved Reviews 12",
                    "facts": ["Pending Reviews 5", "Approved Reviews 12"],
                    "number_mentions": ["5", "12"],
                },
                "current_page": {"title": "Reviews"},
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    response = json.loads(payload["/shared_data/assistant_response"])
    assert response["task_type"] == "RETRIEVE"
    assert response["status"] == "SUCCESS"
    assert isinstance(response["retrieved_data"], list)
    assert response["retrieved_data"][0]["pending_count"] == 5
    assert "Pending Reviews" in response["retrieved_data"][0]["summary"]
    assert response["error_details"] is None


def test_response_worker_uses_llm_override_when_available():
    worker = ResponseWorker(llm=_FakeLLM({"assistant_response": "LLM final answer"}))
    runtime = WorkerRuntime(
        state={
            "shared_data": {
                "verification": {"goal_reached": True},
                "page_evidence": {"text_excerpt": "Todo item one. Todo item two."},
                "current_page": {"title": "Your To-Do List", "visible_text_excerpt": "Todo item one. Todo item two."},
            }
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}

    assert payload["/shared_data/assistant_response"] == "LLM final answer"
    assert result.message is not None
    assert result.message.content == "LLM final answer"


def test_response_worker_repairs_invalid_retrieval_response_shape():
    worker = ResponseWorker(llm=_FakeLLM({"assistant_response": "Pending Reviews 5"}))
    runtime = WorkerRuntime(
        state={
            "task_constraints": {
                "requires_final_response": True,
                "response_schema": {"type": "array", "items": {"type": "string"}},
                "task_category": "retrieval",
            },
            "shared_data": {
                "verification": {"goal_reached": True, "answer_ready": True},
                "page_evidence": {"text_excerpt": "Pending Reviews 5"},
                "current_page": {"title": "Reviews"},
            },
        },
        tools_by_name={},
    )

    result = worker.run(runtime)
    payload = {op["path"]: op["value"] for op in result.patch}
    response = json.loads(payload["/shared_data/assistant_response"])
    assert response["task_type"] == "RETRIEVE"
    assert response["status"] == "SUCCESS"
    assert isinstance(response["retrieved_data"], list)
    assert response["retrieved_data"]
