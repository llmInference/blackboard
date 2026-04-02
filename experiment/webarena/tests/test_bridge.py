from __future__ import annotations

from experiment.common.neutral import ToolResultStatus
from experiment.webarena.bridge import (
    adapt_browser_result,
    browser_tools_to_specs,
    classify_task,
    requires_final_response,
    standardize_browser_observation,
    task_to_spec,
)


def test_task_to_spec_navigation_task():
    task = {
        "task_id": 44,
        "sites": ["gitlab"],
        "start_urls": ["__GITLAB__"],
        "intent": "Open my todos page",
        "intent_template": "Open my todos page",
        "eval": [
            {"evaluator": "AgentResponseEvaluator", "results_schema": {"type": "null"}, "expected": {"retrieved_data": None}},
            {"evaluator": "NetworkEventEvaluator", "expected": {"url": ["__GITLAB__/dashboard/todos"]}},
        ],
    }

    spec = task_to_spec(task, config_path="/tmp/task44.json")

    assert requires_final_response(task) is False
    assert classify_task(task) == "navigation"
    assert spec.task_id == "44"
    assert spec.domain == "webarena"
    assert spec.metadata["config_path"] == "/tmp/task44.json"
    assert spec.metadata["task_category"] == "navigation"
    assert spec.metadata["requires_final_response"] is False


def test_task_to_spec_retrieval_task():
    task = {
        "task_id": 47,
        "sites": ["shopping"],
        "start_urls": ["__SHOPPING__"],
        "intent": "Get the total amount spent over the past months. Return an object.",
        "eval": [
            {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {
                    "type": "object",
                    "properties": {
                        "order_count": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
                "expected": {"retrieved_data": {"order_count": 3, "amount": 10.99}},
            }
        ],
    }

    spec = task_to_spec(task)

    assert requires_final_response(task) is True
    assert classify_task(task) == "retrieval"
    assert spec.metadata["requires_final_response"] is True


def test_browser_tools_to_specs_exposes_minimal_surface():
    specs = {spec.name: spec for spec in browser_tools_to_specs()}

    assert "browser__goto" in specs
    assert "browser__click" in specs
    assert "browser__finish" in specs
    assert specs["browser__type"].parameters_json_schema["required"] == ["element_id", "text"]
    assert specs["browser__switch_tab"].metadata["mutates_state"] is False


def test_adapt_browser_result_normalizes_observation():
    raw_result = {
        "reward": 1.0,
        "done": False,
        "observation": {
            "url": "http://localhost:8023/dashboard/todos",
            "title": "Todos",
            "text": "My todos page",
            "tabs": [{"url": "http://localhost:8023/dashboard/todos", "title": "Todos", "active": True}],
            "elements": [{"id": "todo-link", "role": "link", "text": "To-Do List"}],
        },
    }

    tool_result, observation = adapt_browser_result("browser__click", raw_result, call_id="call-1")

    assert tool_result.status == ToolResultStatus.SUCCESS
    assert tool_result.call_id == "call-1"
    assert tool_result.payload["observation"]["page_title"] == "Todos"
    assert observation.payload["interactive_elements"][0]["element_id"] == "todo-link"


def test_standardize_browser_observation_handles_empty_payload():
    standardized = standardize_browser_observation({})

    assert standardized["url"] == ""
    assert standardized["interactive_elements"] == []
    assert standardized["open_tabs"] == []
