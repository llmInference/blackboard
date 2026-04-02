from __future__ import annotations

import json
from datetime import datetime, timezone

from experiment.appworld.core.state_bridge import turn_input_to_blackboard_state
from experiment.appworld.core.world_wrapper import AppWorldTaskRunner
from experiment.appworld.systems.blackboard_runner import run_blackboard_task
from experiment.appworld.systems.blackboard_system import AppWorldBlackboardSystem
from experiment.appworld.workers import build_default_workers
from experiment.appworld.workers.base import WorkerRuntime
from experiment.appworld.workers.builtin import AnalysisWorker, ResponseWorker, StateWorker, ToolWorker
from experiment.common.neutral import TaskSpec, ToolResult, ToolResultStatus, ToolSpec, TurnInput


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute_tool_call(self, tool_call):
        self.calls.append((tool_call.tool_name, dict(tool_call.arguments)))
        payload = {
            "items": [
                {"id": "song_1", "title": "Midnight Groove"},
                {"id": "song_2", "title": "Velvet Rain"},
            ]
        }
        if tool_call.tool_name == "supervisor__show_account_passwords":
            payload = {"spotify": "pw123", "gmail": "pw456"}
        return ToolResult(
            tool_name=tool_call.tool_name,
            status=ToolResultStatus.SUCCESS,
            content=str(payload),
            payload=payload,
        )


class _RoutingExecutor:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute_tool_call(self, tool_call):
        self.calls.append((tool_call.tool_name, dict(tool_call.arguments)))
        result = self.responses[tool_call.tool_name]
        if callable(result):
            result = result(tool_call)
        return result


def _tool_runtime(*, shared_data, tools, supervisor=None, executor=None, current_step=None):
    return WorkerRuntime(
        state={
            "task": {
                "metadata": {
                    "supervisor": supervisor
                    or {
                        "email": "ada@example.com",
                        "phone_number": "123",
                    }
                }
            },
            "shared_data": dict(shared_data),
            "execution": {"history": [], "current_step_id": "step_1", "current_step": dict(current_step or {})},
        },
        tools_by_name={tool.name: tool for tool in tools},
        tool_executor=executor,
    )


def _patch_value(patch, path):
    for op in patch:
        if op.get("path") == path:
            return op.get("value")
    raise AssertionError(f"Missing patch path {path}")


class _LLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeArchitectLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            """
{
  "workflow": {
    "workflow_id": "architect_custom",
    "goal": "custom",
    "steps": [
      {
        "id": "step_a",
        "worker": "tool_worker",
        "purpose": "Use tools first.",
        "reads": ["candidate_tools", "auth"],
        "writes": ["raw_results", "tool_status", "action_history", "auth"],
        "exit_conditions": ["tool work done"],
        "next": ["step_b"]
      },
      {
        "id": "step_b",
        "worker": "response_worker",
        "purpose": "Respond directly.",
        "reads": ["selected_entities", "evidence", "intermediate_results", "progress_state", "tool_status"],
        "writes": ["assistant_response", "response_confidence", "finish", "finish_reason"],
        "exit_conditions": ["response done"],
        "next": []
      }
    ]
  },
  "data_schema": {
    "raw_results": {
      "type": "array",
      "required": true,
      "producer": ["tool_worker"],
      "description": "Raw outputs"
    },
    "assistant_response": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Final answer"
    },
    "finish": {
      "type": "boolean",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done flag"
    },
    "finish_reason": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done reason"
    },
    "tool_status": {
      "type": "object",
      "required": true,
      "producer": ["tool_worker"],
      "description": "Tool status"
    }
  },
  "selected_workers": ["tool_worker", "response_worker"],
  "worker_instructions": {
    "tool_worker": "Use tools.",
    "response_worker": "Respond."
  }
}
            """.strip()
        )


class _NarrowWriteArchitectLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            """
{
  "workflow": {
    "workflow_id": "architect_narrow_writes",
    "goal": "custom",
    "steps": [
      {
        "id": "step_a",
        "worker": "tool_worker",
        "purpose": "Use tools first.",
        "reads": ["candidate_tools"],
        "writes": ["raw_results"],
        "exit_conditions": ["tool work done"],
        "next": ["step_b"]
      },
      {
        "id": "step_b",
        "worker": "response_worker",
        "purpose": "Respond directly.",
        "reads": ["selected_entities"],
        "writes": ["assistant_response", "finish", "finish_reason"],
        "exit_conditions": ["response done"],
        "next": []
      }
    ]
  },
  "data_schema": {
    "raw_results": {
      "type": "array",
      "required": true,
      "producer": ["tool_worker"],
      "description": "Raw outputs"
    },
    "assistant_response": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Final answer"
    },
    "finish": {
      "type": "boolean",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done flag"
    },
    "finish_reason": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done reason"
    }
  },
  "selected_workers": ["tool_worker", "response_worker"],
  "worker_instructions": {
    "tool_worker": "Use tools.",
    "response_worker": "Respond."
  }
}
            """.strip()
        )


class _InvalidNextArchitectLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            """
{
  "workflow": {
    "workflow_id": "architect_invalid_next",
    "goal": "custom",
    "steps": [
      {
        "id": "step_a",
        "worker": "tool_worker",
        "purpose": "Use tools first.",
        "reads": ["candidate_tools"],
        "writes": ["raw_results", "auth"],
        "exit_conditions": ["tool work done"],
        "next": ["success"]
      },
      {
        "id": "step_b",
        "worker": "response_worker",
        "purpose": "Respond directly.",
        "reads": ["selected_entities"],
        "writes": ["assistant_response", "finish", "finish_reason"],
        "exit_conditions": ["response done"],
        "next": []
      }
    ]
  },
  "data_schema": {
    "raw_results": {
      "type": "array",
      "required": true,
      "producer": ["tool_worker"],
      "description": "Raw outputs"
    },
    "assistant_response": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Final answer"
    },
    "finish": {
      "type": "boolean",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done flag"
    },
    "finish_reason": {
      "type": "string",
      "required": true,
      "producer": ["response_worker"],
      "description": "Done reason"
    }
  },
  "selected_workers": ["tool_worker", "response_worker"],
  "worker_instructions": {
    "tool_worker": "Use tools.",
    "response_worker": "Respond."
  }
}
            """.strip()
        )


class _MalformedArchitectLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse("not json at all")


class _RepairingArchitectLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        del prompt
        if self.calls == 1:
            return _LLMResponse('{"workflow":{"workflow_id":"broken"')
        return _LLMResponse(
            json.dumps(
                {
                    "workflow": {
                        "workflow_id": "architect_repaired",
                        "goal": "custom",
                        "steps": [
                            {
                                "id": "s1",
                                "worker": "tool_worker",
                                "purpose": "Fetch task-relevant data.",
                                "api_goal": "Call task-relevant APIs.",
                                "preferred_tools": ["spotify__search_tracks"],
                                "expected_outputs": ["raw_results"],
                                "reads": ["candidate_tools", "auth"],
                                "writes": ["raw_results", "tool_status", "action_history", "auth"],
                                "exit_conditions": ["tool work attempted"],
                                "next": ["s2"],
                            },
                            {
                                "id": "s2",
                                "worker": "response_worker",
                                "purpose": "Respond.",
                                "reads": ["selected_entities", "evidence", "intermediate_results", "progress_state", "tool_status"],
                                "writes": ["assistant_response", "response_confidence", "finish", "finish_reason"],
                                "exit_conditions": ["response done"],
                                "next": [],
                            },
                        ],
                    },
                    "data_schema": {
                        "raw_results": {
                            "type": "array",
                            "required": False,
                            "producer": ["tool_worker"],
                            "description": "Collected tool results.",
                        },
                        "assistant_response": {
                            "type": "string",
                            "required": True,
                            "producer": ["response_worker"],
                            "description": "Final answer.",
                        },
                    },
                    "selected_workers": ["tool_worker", "response_worker"],
                    "worker_instructions": {
                        "tool_worker": "Use tools.",
                        "response_worker": "Respond.",
                    },
                }
            )
        )


class _CoercibleArchitectLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            json.dumps(
                {
                    "workflow": {
                        "workflow_id": "architect_coercible",
                        "goal": "custom",
                        "steps": [
                            {
                                "id": "s1",
                                "worker": "tool_worker",
                                "purpose": "Login and fetch data.",
                                "api_goal": "Login to Spotify then fetch library data.",
                                "preferred_tools": "spotify__login",
                                "expected_outputs": "spotify_logged_in",
                                "reads": "auth",
                                "writes": "raw_results",
                                "exit_conditions": "tool work attempted",
                                "next": {"on_success": "s2", "on_failure": "s3"},
                            },
                            {
                                "id": "s2",
                                "worker": "state_worker",
                                "purpose": "Normalize results.",
                                "reads": ["raw_results"],
                                "writes": ["entities"],
                                "exit_conditions": ["state updated"],
                                "next": {"on_success": "s3"},
                            },
                            {
                                "id": "s3",
                                "worker": "response_worker",
                                "purpose": "Respond.",
                                "reads": ["selected_entities"],
                                "writes": ["assistant_response"],
                                "exit_conditions": ["finish=true"],
                                "next": {"on_success": None},
                            },
                        ],
                    },
                    "data_schema": {
                        "spotify_logged_in": {
                            "type": "boolean",
                            "required": True,
                            "producer": "tool_worker",
                            "description": "Whether Spotify authentication succeeded.",
                        },
                        "rnb_ranked_tracks": {
                            "type": "array",
                            "required": False,
                            "producer": "state_worker",
                            "description": "Ranked tracks.",
                        },
                        "top_4_rnb_titles_csv": {
                            "type": "string",
                            "required": False,
                            "producer": "response_worker",
                            "description": "Final list.",
                        },
                    },
                    "selected_workers": ["tool_worker", "state_worker", "response_worker"],
                    "worker_instructions": {
                        "tool_worker": ["Use tools only within role."],
                        "state_worker": ["Summarize results."],
                        "response_worker": ["Return concise answer."],
                    },
                }
            )
        )


class _FakeStateLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            """
{
  "entities": [{"label": "Song A", "title": "Song A"}],
  "selected_entities": [{"label": "Song A", "title": "Song A"}],
  "evidence": {"result_count": 1},
  "intermediate_results": {"requested_count": 1},
  "progress_state": {
    "ready_for_response": "yes",
    "needs_more_tooling": "",
    "auth_ready": "true",
    "blocked": 1,
    "blocked_reason": "not_a_valid_reason",
    "missing_prerequisites": ["access_token", 123],
    "recoverable": "sure"
  }
}
            """.strip()
        )


class _FakeBlockedResponseLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            json.dumps(
                {
                    "assistant_response": "Spotify authentication failed.",
                    "response_confidence": "high",
                    "finish": True,
                    "finish_reason": "authentication_failed",
                }
            )
        )


class _FakeMalformedStateLLM:
    def invoke(self, prompt: str):
        del prompt
        return _LLMResponse(
            """
{
  "entities": "not_a_list",
  "selected_entities": "also_not_a_list",
  "evidence": ["bad"],
  "intermediate_results": "bad",
  "progress_state": {
    "ready_for_response": false,
    "needs_more_tooling": true,
    "auth_ready": true,
    "blocked": false,
    "blocked_reason": "",
    "missing_prerequisites": [],
    "recoverable": true
  }
}
            """.strip()
        )


class _FakeSupervisor:
    first_name = "Ada"
    last_name = "Lovelace"
    email = "ada@example.com"
    phone_number = "123"


class _FakeApiDocs(dict):
    def keep_apps(self, allowed_apps):
        allowed = set(allowed_apps)
        return {
            app_name: dict(api_docs)
            for app_name, api_docs in self.items()
            if app_name in allowed
        }


class _FakeRequester:
    def request(self, app_name, api_name, **kwargs):
        if app_name == "supervisor" and api_name == "show_account_passwords":
            return {"spotify": "pw123"}
        if app_name == "spotify" and api_name == "login":
            return {"access_token": "token_1", "email": kwargs.get("email", "")}
        if app_name == "spotify" and api_name == "search_tracks":
            return {"items": [{"id": "song_1", "title": "Midnight Groove"}]}
        raise RuntimeError(f"unexpected tool call: {app_name}__{api_name}")


class _FakeWorld:
    save_calls = 0

    def __init__(self, **kwargs):
        del kwargs
        self.task = type(
            "FakeTask",
            (),
            {
                "id": "task_1",
                "instruction": "Give me the top 2 songs from Spotify as a comma-separated list.",
                "allowed_apps": ["spotify", "supervisor"],
                "app_descriptions": {"spotify": "Music app", "supervisor": "Supervisor app"},
                "supervisor": _FakeSupervisor(),
                "datetime": datetime(2026, 3, 23, tzinfo=timezone.utc),
                "db_version": "test",
                "api_docs": _FakeApiDocs(
                    {
                        "supervisor": {
                            "show_account_passwords": {
                                "description": "Show account passwords.",
                                "method": "GET",
                                "path": "/supervisor/passwords",
                                "parameters": [],
                                "response_schemas": {"success": {"spotify": "pw123"}},
                            }
                        },
                        "spotify": {
                            "login": {
                                "description": "Login to Spotify.",
                                "method": "POST",
                                "path": "/spotify/login",
                                "parameters": [
                                    {"name": "email", "type": "string", "required": True},
                                    {"name": "password", "type": "string", "required": True},
                                ],
                                "response_schemas": {"success": {"access_token": "token_1"}},
                            },
                            "search_tracks": {
                                "description": "Search Spotify tracks.",
                                "method": "GET",
                                "path": "/spotify/search",
                                "parameters": [
                                    {"name": "query", "type": "string", "required": True},
                                ],
                                "response_schemas": {"success": {"items": [{"id": "song_1", "title": "Midnight Groove"}]}},
                            },
                        },
                    }
                ),
            },
        )()
        self.requester = _FakeRequester()

    def evaluate(self, suppress_errors=True):
        del suppress_errors
        return {"success": False, "note": "fake"}

    def save(self):
        type(self).save_calls += 1
        return None

    def close(self):
        return None


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="task_1",
        title="task_1",
        domain="appworld",
        instruction="Give me the top 2 songs from Spotify as a comma-separated list.",
        metadata={
            "allowed_apps": ["spotify", "supervisor"],
            "supervisor": {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "email": "ada@example.com",
                "phone_number": "123",
            },
        },
    )


def _tools() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="supervisor__show_account_passwords",
            description="Show account passwords.",
            metadata={"app_name": "supervisor", "api_name": "show_account_passwords", "mutates_state": False},
        ),
        ToolSpec(
            name="spotify__search_tracks",
            description="Search Spotify tracks.",
            parameters_json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            metadata={"app_name": "spotify", "api_name": "search_tracks", "mutates_state": False},
        ),
        ToolSpec(
            name="spotify__login",
            description="Login to Spotify.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "password": {"type": "string"},
                },
                "required": ["email", "password"],
            },
            metadata={"app_name": "spotify", "api_name": "login", "mutates_state": True},
        ),
    )


def test_state_bridge_includes_worker_catalog() -> None:
    state = turn_input_to_blackboard_state(
        TurnInput(task=_task(), available_tools=_tools()),
        available_workers=tuple(worker.spec.to_dict() for worker in build_default_workers()),
    )
    assert state["execution"]["status"] == "uninitialized"
    assert state["environment"]["workers_catalog"]
    assert state["environment"]["workers_catalog"][0]["input_schema"]
    assert state["shared_data"]["analysis_results"] == {}
    assert state["shared_data"]["assistant_response"] == ""


def test_build_default_workers_includes_analysis_worker() -> None:
    worker_names = [worker.spec.name for worker in build_default_workers()]
    assert worker_names == ["tool_worker", "state_worker", "analysis_worker", "response_worker"]


def test_blackboard_session_runs_workers_and_calls_tools_internally() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    assert runtime.calls
    assert all(tool_name for tool_name, _ in runtime.calls)
    assert not output.requested_tool_calls
    assert output.emitted_messages


def test_blackboard_session_finishes_with_response() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    assert output.emitted_messages[0].content
    assert "kernel_status" in output.metadata


def test_blackboard_session_attempts_login_when_login_tool_exists() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    session.step(TurnInput(task=_task(), available_tools=_tools()))

    called_tools = [tool_name for tool_name, _ in runtime.calls]
    assert "supervisor__show_account_passwords" in called_tools
    assert "spotify__login" in called_tools


def test_blackboard_session_records_schema_filtered_worker_inputs() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    session.step(TurnInput(task=_task(), available_tools=_tools()))

    execution = dict(session.state.get("execution") or {})
    worker_inputs = list(execution.get("worker_input_history") or [])
    tool_input = next(item["input"] for item in worker_inputs if item.get("worker") == "tool_worker")
    analysis_input = next(item["input"] for item in worker_inputs if item.get("worker") == "analysis_worker")
    response_input = next(item["input"] for item in worker_inputs if item.get("worker") == "response_worker")
    assert dict(tool_input.get("execution") or {}).get("current_step_id")
    assert isinstance(dict(tool_input.get("execution") or {}).get("current_step") or {}, dict)
    assert "raw_results" in dict(analysis_input.get("shared_data") or {})
    assert "assistant_response" not in dict(analysis_input.get("shared_data") or {})
    assert "task" in response_input
    assert "shared_data" in response_input
    assert "raw_results" not in dict(response_input.get("shared_data") or {})
    assert "selected_entities" in dict(response_input.get("shared_data") or {})
    assert "analysis_results" in dict(response_input.get("shared_data") or {})


def test_blackboard_session_uses_llm_architect_when_available() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(_FakeArchitectLLM(), config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    session.step(TurnInput(task=_task(), available_tools=_tools()))

    workflow = dict(session.state.get("workflow") or {})
    execution = dict(session.state.get("execution") or {})
    steps = list(workflow.get("steps") or [])
    assert workflow.get("workflow_id") == "architect_custom"
    assert [step.get("worker") for step in steps] == ["tool_worker", "response_worker"]
    assert execution.get("used_fallback_architect") is False
    assert execution.get("fallback_reason") == ""
    assert not list(execution.get("warnings") or [])
    architect_debug = dict(execution.get("architect_debug") or {})
    assert architect_debug.get("raw_output")
    assert architect_debug.get("parse_error", "") == ""
    assert architect_debug.get("validation_errors") == []


def test_architect_step_writes_are_expanded_to_worker_scope() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(_NarrowWriteArchitectLLM(), config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    workflow = dict(session.state.get("workflow") or {})
    step_a = list(workflow.get("steps") or [])[0]
    assert "auth" in list(step_a.get("writes") or [])
    assert output.metadata.get("last_patch_error") == ""


def test_tool_worker_uses_supervisor_email_for_username_login_fields() -> None:
    worker = ToolWorker()
    runtime = type(
        "Runtime",
        (),
        {
            "state": {
                "task": {
                    "metadata": {
                        "supervisor": {
                            "email": "ada@example.com",
                        }
                    }
                },
                "shared_data": {
                    "auth": {
                        "supervisor": {"email": "ada@example.com"},
                        "account_passwords": {"spotify": "pw123"},
                        "auth_sessions": {},
                    }
                },
            },
            "tools_by_name": {
                "spotify__login": ToolSpec(
                    name="spotify__login",
                    description="Login to Spotify.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "password": {"type": "string"},
                        },
                        "required": ["username", "password"],
                    },
                    metadata={"app_name": "spotify", "api_name": "login"},
                )
            },
        },
    )()

    arguments = worker._build_arguments("spotify__login", runtime)
    assert arguments["username"] == "ada@example.com"
    assert arguments["password"] == "pw123"


def test_architect_invalid_next_is_normalized_to_sequential_step() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(_InvalidNextArchitectLLM(), config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    workflow = dict(session.state.get("workflow") or {})
    steps = list(workflow.get("steps") or [])
    assert steps[0].get("next") == ["step_b"]
    assert output.metadata.get("kernel_status") == "completed"


def test_architect_coerces_common_llm_shape_errors_without_fallback() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(_CoercibleArchitectLLM(), config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    session.step(TurnInput(task=_task(), available_tools=_tools()))

    execution = dict(session.state.get("execution") or {})
    workflow = dict(session.state.get("workflow") or {})
    data_schema = dict(session.state.get("data_schema") or {})
    steps = list(workflow.get("steps") or [])

    assert execution.get("used_fallback_architect") is False
    assert workflow.get("workflow_id") == "architect_coercible"
    assert steps[0].get("api_goal") == "Login to Spotify then fetch library data."
    assert steps[0].get("preferred_tools") == ["spotify__login"]
    assert steps[0].get("expected_outputs") == ["spotify_logged_in"]
    assert steps[0].get("next") == ["s2", "s3"]
    assert dict(data_schema.get("spotify_logged_in") or {}).get("producer") == ["tool_worker"]
    assert dict(data_schema.get("rnb_ranked_tracks") or {}).get("producer") == ["state_worker"]
    assert dict(execution.get("worker_instructions") or {}).get("tool_worker") == "Use tools only within role."


def test_architect_repairs_malformed_json_without_fallback() -> None:
    runtime = _FakeRuntime()
    llm = _RepairingArchitectLLM()
    system = AppWorldBlackboardSystem(llm, config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    session.step(TurnInput(task=_task(), available_tools=_tools()))

    execution = dict(session.state.get("execution") or {})
    workflow = dict(session.state.get("workflow") or {})
    architect_debug = dict(execution.get("architect_debug") or {})

    assert llm.calls >= 2
    assert execution.get("used_fallback_architect") is False
    assert workflow.get("workflow_id") == "architect_repaired"
    assert architect_debug.get("raw_output")
    assert architect_debug.get("repaired_output")
    assert architect_debug.get("repair_error", "") == ""


def test_tool_worker_stops_when_login_fails() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__login": ToolResult(
                tool_name="spotify__login",
                status=ToolResultStatus.ERROR,
                content="unauthorized",
                error_message="unauthorized",
                payload={},
            ),
            "spotify__show_downloaded_songs": ToolResult(
                tool_name="spotify__show_downloaded_songs",
                status=ToolResultStatus.SUCCESS,
                content="should_not_run",
                payload={},
            ),
        }
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__login", "spotify__show_downloaded_songs"],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "account_passwords": {"spotify": "pw123"},
            },
        },
        tools=_tools(),
        executor=executor,
    )

    patch = worker.run(runtime)
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert [name for name, _ in executor.calls] == ["spotify__login"]
    assert tool_status["status"] == "login_failed"
    assert tool_status["blocked_tool_name"] == "spotify__login"
    assert "unauthorized" in tool_status["last_error"]


def test_tool_worker_bootstraps_passwords_from_appworld_list_response() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "supervisor__show_account_passwords": ToolResult(
                tool_name="supervisor__show_account_passwords",
                status=ToolResultStatus.SUCCESS,
                content='{"response": [{"account_name": "spotify", "password": "pw123"}]}',
                payload={"response": [{"account_name": "spotify", "password": "pw123"}]},
            ),
            "spotify__login": ToolResult(
                tool_name="spotify__login",
                status=ToolResultStatus.SUCCESS,
                content='{"access_token": "token_1", "token_type": "Bearer"}',
                payload={"access_token": "token_1", "token_type": "Bearer"},
            ),
            "spotify__show_song_library": ToolResult(
                tool_name="spotify__show_song_library",
                status=ToolResultStatus.SUCCESS,
                content='{"items": [{"id": "song_1", "title": "Midnight Groove"}]}',
                payload={"items": [{"id": "song_1", "title": "Midnight Groove"}]},
            ),
        }
    )
    protected_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": [
                "supervisor__show_account_passwords",
                "spotify__login",
                "spotify__show_song_library",
            ],
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(
            _tools()[0],
            _tools()[2],
            protected_tool,
        ),
        executor=executor,
        current_step={
            "id": "step_1",
            "worker": "tool_worker",
            "purpose": "Login then fetch library.",
            "api_goal": "Get Spotify credentials, login, then read song library.",
            "preferred_tools": ["supervisor__show_account_passwords", "spotify__login", "spotify__show_song_library"],
            "expected_outputs": ["spotify_auth", "song_library"],
        },
    )

    patch = worker.run(runtime)
    auth = _patch_value(patch, "/shared_data/auth")
    raw_results = _patch_value(patch, "/shared_data/raw_results")
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert [name for name, _ in executor.calls] == [
        "supervisor__show_account_passwords",
        "spotify__login",
        "spotify__show_song_library",
    ]
    assert auth["account_passwords"] == {"spotify": "pw123"}
    assert auth["auth_sessions"]["spotify"]["payload"]["access_token"] == "token_1"
    assert [item["tool_name"] for item in raw_results] == [
        "supervisor__show_account_passwords",
        "spotify__login",
        "spotify__show_song_library",
    ]
    assert tool_status["last_tool_name"] == "spotify__show_song_library"
    assert tool_status["last_status"] == "success"


def test_tool_worker_limits_login_attempts_to_step_relevant_app() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__login": ToolResult(
                tool_name="spotify__login",
                status=ToolResultStatus.SUCCESS,
                content='{"access_token": "token_1", "token_type": "Bearer"}',
                payload={"access_token": "token_1", "token_type": "Bearer"},
            ),
        }
    )
    phone_login_tool = ToolSpec(
        name="phone__login",
        description="Login to phone.",
        parameters_json_schema={
            "type": "object",
            "properties": {"username": {"type": "string"}, "password": {"type": "string"}},
            "required": ["username", "password"],
        },
        metadata={"app_name": "phone", "api_name": "login", "mutates_state": True},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__login", "phone__login"],
            "auth": {
                "supervisor": {"email": "ada@example.com", "phone_number": "123"},
                "account_passwords": {"spotify": "pw123", "phone": "pw999"},
            },
        },
        tools=(
            _tools()[2],
            phone_login_tool,
        ),
        executor=executor,
        current_step={
            "id": "step_login_spotify",
            "worker": "tool_worker",
            "purpose": "Login to Spotify only.",
            "api_goal": "Authenticate to Spotify.",
            "preferred_tools": ["spotify__login"],
            "expected_outputs": ["spotify_auth"],
        },
    )

    patch = worker.run(runtime)
    auth = _patch_value(patch, "/shared_data/auth")

    assert executor.calls == [("spotify__login", {"email": "ada@example.com", "password": "pw123"})]
    assert auth["auth_sessions"]["spotify"]["payload"]["access_token"] == "token_1"
    assert "phone" not in auth.get("auth_sessions", {})


def test_tool_worker_blocks_protected_tool_without_access_token() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__show_downloaded_songs": ToolResult(
                tool_name="spotify__show_downloaded_songs",
                status=ToolResultStatus.SUCCESS,
                content="should_not_run",
                payload={},
            )
        }
    )
    protected_tool = ToolSpec(
        name="spotify__show_downloaded_songs",
        description="Show downloaded songs.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_downloaded_songs", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={"candidate_tools": ["spotify__show_downloaded_songs"], "auth": {"supervisor": {"email": "ada@example.com"}}},
        tools=(protected_tool,),
        executor=executor,
    )

    patch = worker.run(runtime)
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert executor.calls == []
    assert tool_status["status"] == "missing_access_token"
    assert tool_status["blocked_tool_name"] == "spotify__show_downloaded_songs"
    assert tool_status["missing_arguments"] == ["access_token"]


def test_tool_worker_blocks_when_required_arguments_are_missing() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__search_tracks": ToolResult(
                tool_name="spotify__search_tracks",
                status=ToolResultStatus.SUCCESS,
                content="should_not_run",
                payload={},
            )
        }
    )
    search_tool = ToolSpec(
        name="spotify__search_tracks",
        description="Search Spotify tracks.",
        parameters_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        metadata={"app_name": "spotify", "api_name": "search_tracks", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={"candidate_tools": ["spotify__search_tracks"], "auth": {"supervisor": {"email": "ada@example.com"}}},
        tools=(search_tool,),
        executor=executor,
    )

    patch = worker.run(runtime)
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert executor.calls == []
    assert tool_status["status"] == "missing_required_arguments"
    assert tool_status["blocked_tool_name"] == "spotify__search_tracks"
    assert tool_status["missing_arguments"] == ["query"]


def test_state_worker_fallback_maps_tool_status_to_structured_progress_state() -> None:
    worker = StateWorker()
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [],
            "tool_status": {
                "status": "missing_access_token",
                "blocked_tool_name": "spotify__show_downloaded_songs",
                "missing_arguments": ["access_token"],
                "last_error": "Missing access token",
            },
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    progress_state = _patch_value(patch, "/shared_data/progress_state")

    assert progress_state["blocked"] is True
    assert progress_state["blocked_reason"] == "missing_authentication"
    assert progress_state["missing_prerequisites"] == ["access_token"]
    assert progress_state["recoverable"] is True
    assert progress_state["ready_for_response"] is True
    assert progress_state["auth_ready"] is False
    assert progress_state["needs_more_tooling"] is False


def test_state_worker_requests_more_tooling_when_required_apis_are_pending() -> None:
    worker = StateWorker()
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [],
            "tool_status": {
                "status": "success",
                "required_api_status": {
                    "required_tools": ["spotify__show_song_library"],
                    "completed_tools": [],
                    "failed_tools": [],
                    "pending_tools": ["spotify__show_song_library"],
                    "deferred_tools": [],
                    "actionable_pending_tools": ["spotify__show_song_library"],
                    "all_actionable_completed": False,
                },
            },
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    progress_state = _patch_value(patch, "/shared_data/progress_state")

    assert progress_state["needs_more_tooling"] is True
    assert progress_state["ready_for_response"] is False
    assert "spotify__show_song_library" in progress_state["missing_prerequisites"]


def test_analysis_worker_deterministically_ranks_spotify_tracks_from_raw_results() -> None:
    worker = AnalysisWorker()
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [
                {
                    "tool_name": "spotify__show_song_library",
                    "status": "success",
                    "payload": {"response": [{"song_id": 1}, {"song_id": 2}]},
                    "content": "[]",
                },
                {
                    "tool_name": "spotify__show_album_library",
                    "status": "success",
                    "payload": {"response": [{"album_id": 10}]},
                    "content": "[]",
                },
                {
                    "tool_name": "spotify__show_playlist_library",
                    "status": "success",
                    "payload": {"response": [{"playlist_id": 20}]},
                    "content": "[]",
                },
                {
                    "tool_name": "spotify__show_song",
                    "status": "success",
                    "payload": {"song_id": 1, "title": "Track One", "genre": "EDM", "play_count": 100},
                    "content": "{}",
                },
                {
                    "tool_name": "spotify__show_song",
                    "status": "success",
                    "payload": {"song_id": 2, "title": "Track Two", "genre": "EDM", "play_count": 250},
                    "content": "{}",
                },
            ],
            "tool_status": {"last_tool_name": "spotify__show_song"},
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
            "progress_state": {
                "ready_for_response": False,
                "needs_more_tooling": False,
                "auth_ready": True,
                "blocked": False,
                "blocked_reason": "",
                "missing_prerequisites": [],
                "recoverable": True,
            },
        },
        tools=(),
        executor=None,
    )
    runtime.state["task"]["instruction"] = "Give me a comma-separated list of top 2 most played edm song titles."

    patch = worker.run(runtime)
    selected_entities = _patch_value(patch, "/shared_data/selected_entities")
    progress_state = _patch_value(patch, "/shared_data/progress_state")
    intermediate_results = _patch_value(patch, "/shared_data/intermediate_results")
    analysis_results = _patch_value(patch, "/shared_data/analysis_results")

    assert [item["title"] for item in selected_entities] == ["Track Two", "Track One"]
    assert intermediate_results["top_titles_csv"] == "Track Two, Track One"
    assert analysis_results["final_answer"] == "Track Two, Track One"
    assert progress_state["ready_for_response"] is True
    assert progress_state["blocked"] is False


def test_state_worker_normalizes_llm_progress_state_contract() -> None:
    worker = StateWorker(llm=_FakeStateLLM())
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [{"content": "{}", "payload": {}, "tool_name": "spotify__search_tracks", "status": "success"}],
            "tool_status": {"status": "missing_required_arguments", "missing_arguments": ["query"]},
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    progress_state = _patch_value(patch, "/shared_data/progress_state")

    assert progress_state["blocked"] is True
    assert progress_state["blocked_reason"] == "missing_arguments"
    assert progress_state["missing_prerequisites"] == ["access_token", "123"]
    assert progress_state["ready_for_response"] is True
    assert progress_state["recoverable"] is True
    assert progress_state["auth_ready"] is False


def test_state_worker_forces_auth_not_ready_when_access_token_is_missing() -> None:
    worker = StateWorker(llm=_FakeStateLLM())
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [],
            "tool_status": {
                "status": "missing_access_token",
                "blocked_tool_name": "spotify__show_song_library",
                "missing_arguments": ["access_token"],
                "last_error": "Missing access token",
            },
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    progress_state = _patch_value(patch, "/shared_data/progress_state")

    assert progress_state["blocked"] is True
    assert progress_state["blocked_reason"] == "missing_authentication"
    assert progress_state["auth_ready"] is False
    assert progress_state["needs_more_tooling"] is False
    assert "access_token" in progress_state["missing_prerequisites"]


def test_state_worker_preserves_fallback_authentication_failure_reason_when_llm_omits_it() -> None:
    worker = StateWorker(llm=_FakeStateLLM())
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [],
            "tool_status": {
                "status": "login_failed",
                "blocked_tool_name": "spotify__login",
                "missing_arguments": [],
                "last_error": "Invalid credentials",
            },
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    progress_state = _patch_value(patch, "/shared_data/progress_state")

    assert progress_state["blocked"] is True
    assert progress_state["blocked_reason"] == "authentication_failed"
    assert progress_state["auth_ready"] is False


def test_state_worker_tolerates_malformed_llm_container_types() -> None:
    worker = StateWorker(llm=_FakeMalformedStateLLM())
    runtime = _tool_runtime(
        shared_data={
            "raw_results": [
                {
                    "content": "{\"items\": [{\"id\": \"song_1\", \"title\": \"Midnight Groove\"}]}",
                    "payload": {"items": [{"id": "song_1", "title": "Midnight Groove"}]},
                    "tool_name": "spotify__show_song_library",
                    "status": "success",
                }
            ],
            "tool_status": {"last_tool_name": "spotify__show_song_library"},
            "auth": {"supervisor": {"email": "ada@example.com"}},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    entities = _patch_value(patch, "/shared_data/entities")
    selected_entities = _patch_value(patch, "/shared_data/selected_entities")
    evidence = _patch_value(patch, "/shared_data/evidence")
    intermediate_results = _patch_value(patch, "/shared_data/intermediate_results")

    assert entities
    assert selected_entities
    assert evidence["result_count"] == 1
    assert intermediate_results["latest_tool_name"] == "spotify__show_song_library"


def test_response_worker_distinguishes_prerequisite_missing() -> None:
    worker = ResponseWorker()
    runtime = _tool_runtime(
        shared_data={
            "selected_entities": [],
            "evidence": {},
            "progress_state": {
                "blocked": True,
                "blocked_reason": "missing_authentication",
                "missing_prerequisites": ["access_token"],
                "ready_for_response": True,
            },
            "tool_status": {"last_error": "Missing access token"},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    assistant_response = _patch_value(patch, "/shared_data/assistant_response")
    finish_reason = _patch_value(patch, "/shared_data/finish_reason")

    assert "authentication" in assistant_response.lower()
    assert finish_reason == "prerequisite_missing"


def test_response_worker_defers_finish_until_supervisor_complete_task_runs() -> None:
    worker = ResponseWorker()
    runtime = _tool_runtime(
        shared_data={
            "selected_entities": [{"label": "Song A", "title": "Song A"}],
            "evidence": {},
            "progress_state": {
                "blocked": False,
                "blocked_reason": "",
                "missing_prerequisites": [],
                "ready_for_response": True,
            },
            "tool_status": {
                "required_api_status": {
                    "actionable_pending_deferred_tools": ["supervisor__complete_task"],
                }
            },
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)

    assert _patch_value(patch, "/shared_data/assistant_response") == "Song A"
    assert _patch_value(patch, "/shared_data/finish_reason") == "task_completed"
    assert _patch_value(patch, "/shared_data/finish") is False


def test_response_worker_prefers_analysis_worker_final_answer() -> None:
    worker = ResponseWorker()
    runtime = _tool_runtime(
        shared_data={
            "analysis_results": {"final_answer": "81"},
            "selected_entities": [{"label": "Wrong Label"}],
            "evidence": {},
            "intermediate_results": {},
            "progress_state": {
                "blocked": False,
                "blocked_reason": "",
                "missing_prerequisites": [],
                "ready_for_response": True,
            },
            "tool_status": {},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)

    assert _patch_value(patch, "/shared_data/assistant_response") == "81"
    assert _patch_value(patch, "/shared_data/finish_reason") == "task_completed"


def test_blackboard_session_state_worker_can_jump_back_to_tool_worker_when_more_tooling_is_needed() -> None:
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=())
    session.state = {
        "workflow": {
            "steps": [
                {"id": "step_1", "worker": "tool_worker", "next": ["step_2"]},
                {"id": "step_2", "worker": "state_worker", "next": ["step_3"]},
                {"id": "step_3", "worker": "response_worker", "next": []},
            ]
        },
        "shared_data": {
            "progress_state": {
                "ready_for_response": False,
                "needs_more_tooling": True,
            }
        },
    }

    next_step = session._choose_next_step({"id": "step_2", "worker": "state_worker", "next": ["step_3"]})

    assert next_step == "step_1"


def test_blackboard_session_response_worker_can_jump_back_to_tool_worker_for_deferred_submission() -> None:
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=())
    session.state = {
        "workflow": {
            "steps": [
                {"id": "step_1", "worker": "tool_worker", "next": ["step_2"]},
                {"id": "step_2", "worker": "response_worker", "next": []},
            ]
        },
        "shared_data": {
            "assistant_response": "Song A",
            "tool_status": {
                "required_api_status": {
                    "actionable_pending_deferred_tools": ["supervisor__complete_task"],
                }
            },
        },
    }

    next_step = session._choose_next_step({"id": "step_2", "worker": "response_worker", "next": []})

    assert next_step == "step_1"


def test_blackboard_session_analysis_worker_can_jump_back_to_tool_worker_when_more_tooling_is_needed() -> None:
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=())
    session.state = {
        "workflow": {
            "steps": [
                {"id": "step_1", "worker": "tool_worker", "next": ["step_2"]},
                {"id": "step_2", "worker": "analysis_worker", "next": ["step_3"]},
                {"id": "step_3", "worker": "response_worker", "next": []},
            ]
        },
        "shared_data": {
            "progress_state": {
                "ready_for_response": False,
                "needs_more_tooling": True,
                "blocked": False,
            }
        },
    }

    next_step = session._choose_next_step({"id": "step_2", "worker": "analysis_worker", "next": ["step_3"]})

    assert next_step == "step_1"


def test_blackboard_session_analysis_worker_can_continue_to_response_when_ready() -> None:
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=())
    session.state = {
        "workflow": {
            "steps": [
                {"id": "step_1", "worker": "tool_worker", "next": ["step_2"]},
                {"id": "step_2", "worker": "analysis_worker", "next": ["step_3"]},
                {"id": "step_3", "worker": "response_worker", "next": []},
            ]
        },
        "shared_data": {
            "progress_state": {
                "ready_for_response": True,
                "needs_more_tooling": False,
                "blocked": False,
            }
        },
    }

    next_step = session._choose_next_step({"id": "step_2", "worker": "analysis_worker", "next": ["step_3"]})

    assert next_step == "step_3"


def test_response_worker_distinguishes_tool_execution_failure() -> None:
    worker = ResponseWorker()
    runtime = _tool_runtime(
        shared_data={
            "selected_entities": [],
            "evidence": {},
            "progress_state": {
                "blocked": True,
                "blocked_reason": "tool_execution_error",
                "missing_prerequisites": [],
                "ready_for_response": True,
            },
            "tool_status": {"last_error": "rate limited"},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    assistant_response = _patch_value(patch, "/shared_data/assistant_response")
    finish_reason = _patch_value(patch, "/shared_data/finish_reason")

    assert "tool execution failed" in assistant_response.lower()
    assert "rate limited" in assistant_response.lower()
    assert finish_reason == "tool_execution_failed"


def test_response_worker_normalizes_finish_reason_for_missing_authentication() -> None:
    worker = ResponseWorker(llm=_FakeBlockedResponseLLM())
    runtime = _tool_runtime(
        shared_data={
            "selected_entities": [],
            "evidence": {},
            "progress_state": {
                "blocked": True,
                "blocked_reason": "missing_authentication",
                "missing_prerequisites": ["access_token"],
                "ready_for_response": True,
            },
            "tool_status": {"last_error": "Missing access token"},
        },
        tools=(),
        executor=None,
    )

    patch = worker.run(runtime)
    assistant_response = _patch_value(patch, "/shared_data/assistant_response")
    finish_reason = _patch_value(patch, "/shared_data/finish_reason")

    assert assistant_response == "Spotify authentication failed."
    assert finish_reason == "prerequisite_missing"


def test_blackboard_session_surfaces_warning_when_architect_falls_back() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    execution = dict(session.state.get("execution") or {})
    warnings = list(execution.get("warnings") or [])
    history = list(execution.get("history") or [])
    data_schema = dict(session.state.get("data_schema") or {})
    assert execution.get("used_fallback_architect") is True
    assert warnings
    assert "fallback" in warnings[0].lower()
    assert execution.get("fallback_reason")
    assert "fallback" in str(execution.get("fallback_reason", "")).lower()
    assert "response_confidence" not in data_schema
    assert set(data_schema) == {
        "working_memory",
        "schema_extensions",
        "analysis_results",
        "assistant_response",
        "finish",
        "finish_reason",
    }
    assert history
    assert history[0].get("worker") == "architect"
    assert history[0].get("status") == "warning"
    assert output.metadata.get("used_fallback_architect") is True
    assert output.metadata.get("fallback_reason")
    architect_debug = dict(output.metadata.get("architect_debug") or {})
    assert architect_debug.get("invoke_error") == "No architect LLM configured."
    assert output.emitted_messages[0].metadata.get("used_fallback_architect") is True
    assert dict(output.emitted_messages[0].metadata.get("architect_debug") or {}).get("invoke_error") == "No architect LLM configured."


def test_blackboard_session_captures_architect_parse_failure_details() -> None:
    runtime = _FakeRuntime()
    system = AppWorldBlackboardSystem(_MalformedArchitectLLM(), config={"max_kernel_steps": 12, "max_no_progress": 2})
    session = system.create_session(_task(), _tools(), native_tools=(runtime,))
    output = session.step(TurnInput(task=_task(), available_tools=_tools()))

    execution = dict(session.state.get("execution") or {})
    architect_debug = dict(execution.get("architect_debug") or {})
    assert execution.get("used_fallback_architect") is True
    assert "json" in str(execution.get("fallback_reason", "") or "").lower()
    assert architect_debug.get("raw_output") == "not json at all"
    assert "not valid json" in str(architect_debug.get("parse_error", "") or "").lower()
    assert architect_debug.get("validation_errors") == []
    assert dict(output.metadata.get("architect_debug") or {}).get("raw_output") == "not json at all"


def test_run_session_result_carries_architect_fallback_metadata() -> None:
    _FakeWorld.save_calls = 0
    system = AppWorldBlackboardSystem(config={"max_kernel_steps": 12, "max_no_progress": 2})
    with AppWorldTaskRunner(
        task_id="task_1",
        experiment_name="test",
        world_factory=_FakeWorld,
    ) as runner:
        result = runner.run_session(system, max_steps=1)

    assert result.run_metadata.get("used_fallback_architect") is True
    assert result.run_metadata.get("fallback_reason")
    assert result.run_metadata.get("warnings")
    assert _FakeWorld.save_calls == 1


def test_task_runner_loads_oracle_required_apis_into_task_metadata() -> None:
    with AppWorldTaskRunner(
        task_id="50e1ac9_1",
        experiment_name="test",
        appworld_root="/home/syq/Documents/blackboard/appworld",
        world_factory=_FakeWorld,
    ) as runner:
        task_spec = runner.task_spec

    oracle_required_apis = list(task_spec.metadata.get("oracle_required_apis") or [])
    assert "spotify.login" in oracle_required_apis
    assert "spotify.show_song_library" in oracle_required_apis


def test_run_blackboard_task_uses_passed_llm() -> None:
    result = run_blackboard_task(
        task_id="task_1",
        experiment_name="test",
        llm=_FakeArchitectLLM(),
        world_factory=_FakeWorld,
        max_steps=1,
        system_config={"max_kernel_steps": 12, "max_no_progress": 2},
    )
    assert result.run_metadata.get("used_fallback_architect") is False
    assert result.run_metadata.get("warnings") == []


def test_tool_worker_prioritizes_oracle_required_apis() -> None:
    worker = ToolWorker()
    mutation_tool = ToolSpec(
        name="spotify__remove_song_from_playlist",
        description="Remove a song from a playlist.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "playlist_id": {"type": "string"}, "song_id": {"type": "string"}},
            "required": ["access_token", "playlist_id", "song_id"],
        },
        metadata={"app_name": "spotify", "api_name": "remove_song_from_playlist", "mutates_state": True},
    )
    oracle_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show your song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__remove_song_from_playlist", "spotify__show_song_library"],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(mutation_tool, oracle_tool),
        executor=None,
        supervisor={"email": "ada@example.com"},
    )
    runtime.state["task"]["instruction"] = "Give me the top 4 most played r&b song titles."
    runtime.state["task"]["metadata"]["oracle_required_apis"] = [
        "spotify.show_song_library",
        "spotify.login",
    ]

    selected_tool = worker._select_action_tool(runtime, worker._candidate_tools(runtime))
    assert selected_tool == "spotify__show_song_library"


def test_tool_worker_prioritizes_current_step_preferred_tools() -> None:
    worker = ToolWorker()
    mutation_tool = ToolSpec(
        name="spotify__remove_song_from_playlist",
        description="Remove a song from a playlist.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "playlist_id": {"type": "string"}, "song_id": {"type": "string"}},
            "required": ["access_token", "playlist_id", "song_id"],
        },
        metadata={"app_name": "spotify", "api_name": "remove_song_from_playlist", "mutates_state": True},
    )
    preferred_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show your song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__remove_song_from_playlist", "spotify__show_song_library"],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(mutation_tool, preferred_tool),
        executor=None,
        current_step={
            "id": "step_1",
            "worker": "tool_worker",
            "purpose": "Fetch library data only.",
            "api_goal": "Show the Spotify song library.",
            "preferred_tools": ["spotify__show_song_library"],
            "expected_outputs": ["song_library"],
        },
    )
    runtime.state["task"]["instruction"] = "Give me the top 4 most played r&b song titles."

    selected_tool = worker._select_action_tool(runtime, worker._candidate_tools(runtime))
    assert selected_tool == "spotify__show_song_library"


def test_tool_worker_preserves_prior_raw_results_across_steps() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__show_song_library": ToolResult(
                tool_name="spotify__show_song_library",
                status=ToolResultStatus.SUCCESS,
                content="{\"items\": [{\"id\": \"song_2\", \"title\": \"Velvet Rain\"}]}",
                payload={"items": [{"id": "song_2", "title": "Velvet Rain"}]},
            )
        }
    )
    preferred_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show your song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__show_song_library"],
            "raw_results": [
                {
                    "worker": "tool_worker",
                    "tool_name": "spotify__login",
                    "status": "success",
                    "payload": {"access_token": "token_1"},
                    "content": "{\"access_token\": \"token_1\"}",
                    "arguments": {"email": "ada@example.com"},
                }
            ],
            "action_history": [{"tool_name": "spotify__login", "status": "success"}],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(preferred_tool,),
        executor=executor,
        current_step={
            "id": "step_2",
            "worker": "tool_worker",
            "purpose": "Fetch song library.",
            "api_goal": "Show the Spotify song library.",
            "preferred_tools": ["spotify__show_song_library"],
            "expected_outputs": ["song_library"],
        },
    )

    patch = worker.run(runtime)
    raw_results = _patch_value(patch, "/shared_data/raw_results")

    assert len(raw_results) == 2
    assert raw_results[0]["tool_name"] == "spotify__login"
    assert raw_results[1]["tool_name"] == "spotify__show_song_library"


def test_tool_worker_executes_appworld_style_library_pagination_and_song_expansion() -> None:
    worker = ToolWorker()

    def library_result(tool_call):
        page_index = int(tool_call.arguments.get("page_index", 0))
        payload_by_page = {
            0: [{"song_id": 11, "title": "Track 11"}, {"song_id": 12, "title": "Track 12"}],
            1: [{"song_id": 13, "title": "Track 13"}],
            2: [],
        }
        payload = payload_by_page.get(page_index, [])
        return ToolResult(
            tool_name="spotify__show_song_library",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def song_result(tool_call):
        song_id = int(tool_call.arguments["song_id"])
        payload = {"id": song_id, "title": f"Track {song_id}", "genre": "R&B", "play_count": song_id}
        return ToolResult(
            tool_name="spotify__show_song",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    executor = _RoutingExecutor(
        {
            "spotify__show_song_library": library_result,
            "spotify__show_song": song_result,
        }
    )
    song_library_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "access_token": {"type": "string"},
                "page_index": {"type": "integer", "default": 0},
            },
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    song_tool = ToolSpec(
        name="spotify__show_song",
        description="Show song detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"song_id": {"type": "integer"}},
            "required": ["song_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__show_song_library", "spotify__show_song"],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(song_library_tool, song_tool),
        executor=executor,
        current_step={
            "id": "step_expand_song_library",
            "worker": "tool_worker",
            "purpose": "Fetch the song library and expand song details.",
            "api_goal": "List all saved songs and inspect each song.",
            "preferred_tools": ["spotify__show_song_library"],
            "expected_outputs": ["song_library", "expanded_tracks"],
        },
    )

    patch = worker.run(runtime)
    raw_results = _patch_value(patch, "/shared_data/raw_results")
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert executor.calls == [
        ("spotify__show_song_library", {"access_token": "token_1", "page_index": 0}),
        ("spotify__show_song_library", {"access_token": "token_1", "page_index": 1}),
        ("spotify__show_song_library", {"access_token": "token_1", "page_index": 2}),
        ("spotify__show_song", {"song_id": 11}),
        ("spotify__show_song", {"song_id": 12}),
        ("spotify__show_song", {"song_id": 13}),
    ]
    assert [item["tool_name"] for item in raw_results] == [
        "spotify__show_song_library",
        "spotify__show_song_library",
        "spotify__show_song_library",
        "spotify__show_song",
        "spotify__show_song",
        "spotify__show_song",
    ]
    assert tool_status["last_tool_name"] == "spotify__show_song"
    assert tool_status["last_status"] == "success"


def test_tool_worker_executes_album_and_playlist_expansion_without_duplicate_song_fetches() -> None:
    worker = ToolWorker()

    def album_library_result(tool_call):
        page_index = int(tool_call.arguments.get("page_index", 0))
        payload = [{"album_id": 201}] if page_index == 0 else []
        return ToolResult(
            tool_name="spotify__show_album_library",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def playlist_library_result(tool_call):
        page_index = int(tool_call.arguments.get("page_index", 0))
        payload = [{"playlist_id": 401}] if page_index == 0 else []
        return ToolResult(
            tool_name="spotify__show_playlist_library",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def album_result(tool_call):
        payload = {"album_id": int(tool_call.arguments["album_id"]), "songs": [{"id": 301}, {"id": 302}]}
        return ToolResult(
            tool_name="spotify__show_album",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    def playlist_result(tool_call):
        payload = {"playlist_id": int(tool_call.arguments["playlist_id"]), "songs": [{"id": 302}, {"id": 303}]}
        return ToolResult(
            tool_name="spotify__show_playlist",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    def song_result(tool_call):
        song_id = int(tool_call.arguments["song_id"])
        payload = {"id": song_id, "title": f"Track {song_id}", "genre": "R&B"}
        return ToolResult(
            tool_name="spotify__show_song",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    executor = _RoutingExecutor(
        {
            "spotify__show_album_library": album_library_result,
            "spotify__show_playlist_library": playlist_library_result,
            "spotify__show_album": album_result,
            "spotify__show_playlist": playlist_result,
            "spotify__show_song": song_result,
        }
    )
    album_library_tool = ToolSpec(
        name="spotify__show_album_library",
        description="Show album library.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "access_token": {"type": "string"},
                "page_index": {"type": "integer", "default": 0},
            },
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_album_library", "mutates_state": False},
    )
    playlist_library_tool = ToolSpec(
        name="spotify__show_playlist_library",
        description="Show playlist library.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "access_token": {"type": "string"},
                "page_index": {"type": "integer", "default": 0},
            },
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist_library", "mutates_state": False},
    )
    album_tool = ToolSpec(
        name="spotify__show_album",
        description="Show album detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"album_id": {"type": "integer"}},
            "required": ["album_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_album", "mutates_state": False},
    )
    playlist_tool = ToolSpec(
        name="spotify__show_playlist",
        description="Show playlist detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"playlist_id": {"type": "integer"}, "access_token": {"type": "string"}},
            "required": ["playlist_id", "access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist", "mutates_state": False},
    )
    song_tool = ToolSpec(
        name="spotify__show_song",
        description="Show song detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"song_id": {"type": "integer"}},
            "required": ["song_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": [
                "spotify__show_album_library",
                "spotify__show_playlist_library",
                "spotify__show_album",
                "spotify__show_playlist",
                "spotify__show_song",
            ],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(album_library_tool, playlist_library_tool, album_tool, playlist_tool, song_tool),
        executor=executor,
        current_step={
            "id": "step_expand_album_playlist",
            "worker": "tool_worker",
            "purpose": "Fetch album and playlist libraries, then expand them.",
            "api_goal": "List albums and playlists, then inspect contained songs.",
            "preferred_tools": ["spotify__show_album_library", "spotify__show_playlist_library"],
            "expected_outputs": ["album_library", "playlist_library", "expanded_tracks"],
        },
    )

    worker.run(runtime)

    assert executor.calls == [
        ("spotify__show_album_library", {"access_token": "token_1", "page_index": 0}),
        ("spotify__show_album_library", {"access_token": "token_1", "page_index": 1}),
        ("spotify__show_album", {"album_id": 201}),
        ("spotify__show_song", {"song_id": 301}),
        ("spotify__show_song", {"song_id": 302}),
        ("spotify__show_playlist_library", {"access_token": "token_1", "page_index": 0}),
        ("spotify__show_playlist_library", {"access_token": "token_1", "page_index": 1}),
        ("spotify__show_playlist", {"playlist_id": 401, "access_token": "token_1"}),
        ("spotify__show_song", {"song_id": 303}),
    ]


def test_tool_worker_passes_optional_access_token_when_available() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__show_playlist": ToolResult(
                tool_name="spotify__show_playlist",
                status=ToolResultStatus.SUCCESS,
                content='{"playlist_id": 401, "songs": []}',
                payload={"playlist_id": 401, "songs": []},
            )
        }
    )
    playlist_tool = ToolSpec(
        name="spotify__show_playlist",
        description="Show playlist detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "playlist_id": {"type": "integer"},
                "access_token": {"type": "string"},
            },
            "required": ["playlist_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__show_playlist"],
            "selected_entities": [{"playlist_id": 401}],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(playlist_tool,),
        executor=executor,
        current_step={
            "id": "step_playlist_detail",
            "worker": "tool_worker",
            "purpose": "Read one playlist.",
            "api_goal": "Inspect playlist details.",
            "preferred_tools": ["spotify__show_playlist"],
            "expected_outputs": ["playlist_detail"],
        },
    )

    worker.run(runtime)

    assert executor.calls == [
        ("spotify__show_playlist", {"playlist_id": 401, "access_token": "token_1"})
    ]


def test_tool_worker_follows_oracle_library_chain_even_if_current_step_names_only_song_library() -> None:
    worker = ToolWorker()

    def list_result(tool_call):
        tool_name = tool_call.tool_name
        page_index = int(tool_call.arguments.get("page_index", 0))
        if tool_name == "spotify__show_song_library":
            payload = [{"song_id": 11}] if page_index == 0 else []
        elif tool_name == "spotify__show_album_library":
            payload = [{"album_id": 201}] if page_index == 0 else []
        else:
            payload = [{"playlist_id": 401}] if page_index == 0 else []
        return ToolResult(
            tool_name=tool_name,
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def album_result(tool_call):
        return ToolResult(
            tool_name="spotify__show_album",
            status=ToolResultStatus.SUCCESS,
            content='{"album_id": 201, "songs": [{"id": 21}]}',
            payload={"album_id": 201, "songs": [{"id": 21}]},
        )

    def playlist_result(tool_call):
        return ToolResult(
            tool_name="spotify__show_playlist",
            status=ToolResultStatus.SUCCESS,
            content='{"playlist_id": 401, "songs": [{"id": 31}]}',
            payload={"playlist_id": 401, "songs": [{"id": 31}]},
        )

    def song_result(tool_call):
        song_id = int(tool_call.arguments["song_id"])
        return ToolResult(
            tool_name="spotify__show_song",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps({"id": song_id}),
            payload={"id": song_id},
        )

    executor = _RoutingExecutor(
        {
            "spotify__show_song_library": list_result,
            "spotify__show_album_library": list_result,
            "spotify__show_playlist_library": list_result,
            "spotify__show_album": album_result,
            "spotify__show_playlist": playlist_result,
            "spotify__show_song": song_result,
        }
    )
    song_library_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    album_library_tool = ToolSpec(
        name="spotify__show_album_library",
        description="Show album library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_album_library", "mutates_state": False},
    )
    playlist_library_tool = ToolSpec(
        name="spotify__show_playlist_library",
        description="Show playlist library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist_library", "mutates_state": False},
    )
    album_tool = ToolSpec(
        name="spotify__show_album",
        description="Show album detail.",
        parameters_json_schema={"type": "object", "properties": {"album_id": {"type": "integer"}}, "required": ["album_id"]},
        metadata={"app_name": "spotify", "api_name": "show_album", "mutates_state": False},
    )
    playlist_tool = ToolSpec(
        name="spotify__show_playlist",
        description="Show playlist detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"playlist_id": {"type": "integer"}, "access_token": {"type": "string"}},
            "required": ["playlist_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist", "mutates_state": False},
    )
    song_tool = ToolSpec(
        name="spotify__show_song",
        description="Show song detail.",
        parameters_json_schema={"type": "object", "properties": {"song_id": {"type": "integer"}}, "required": ["song_id"]},
        metadata={"app_name": "spotify", "api_name": "show_song", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": [
                "spotify__show_song_library",
                "spotify__show_album_library",
                "spotify__show_playlist_library",
                "spotify__show_album",
                "spotify__show_playlist",
                "spotify__show_song",
            ],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(song_library_tool, album_library_tool, playlist_library_tool, album_tool, playlist_tool, song_tool),
        executor=executor,
        current_step={
            "id": "step_library_chain",
            "worker": "tool_worker",
            "purpose": "Start from song library.",
            "api_goal": "Inspect the complete Spotify library footprint.",
            "preferred_tools": ["spotify__show_song_library"],
            "expected_outputs": ["song_library", "album_library", "playlist_library"],
        },
    )
    runtime.state["task"]["metadata"]["oracle_required_apis"] = [
        "spotify.show_song_library",
        "spotify.show_album_library",
        "spotify.show_playlist_library",
    ]

    worker.run(runtime)

    called_tool_names = [name for name, _ in executor.calls]
    assert "spotify__show_song_library" in called_tool_names
    assert "spotify__show_album_library" in called_tool_names
    assert "spotify__show_playlist_library" in called_tool_names


def test_tool_worker_records_required_api_status_and_executes_pending_oracle_tools() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "spotify__show_song_library": ToolResult(
                tool_name="spotify__show_song_library",
                status=ToolResultStatus.SUCCESS,
                content="[]",
                payload={"response": []},
            ),
            "spotify__show_album_library": ToolResult(
                tool_name="spotify__show_album_library",
                status=ToolResultStatus.SUCCESS,
                content="[]",
                payload={"response": []},
            ),
        }
    )
    song_library_tool = ToolSpec(
        name="spotify__show_song_library",
        description="Show song library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_song_library", "mutates_state": False},
    )
    album_library_tool = ToolSpec(
        name="spotify__show_album_library",
        description="Show album library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_album_library", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["spotify__show_song_library", "spotify__show_album_library"],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(song_library_tool, album_library_tool),
        executor=executor,
        current_step={"id": "step_1", "worker": "tool_worker", "purpose": "Collect required libraries."},
    )
    runtime.state["task"]["metadata"]["oracle_required_apis"] = [
        "spotify.login",
        "spotify.show_song_library",
        "spotify.show_album_library",
        "supervisor.show_profile",
        "supervisor.complete_task",
    ]

    patch = worker.run(runtime)
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert executor.calls == [
        ("spotify__show_album_library", {"access_token": "token_1", "page_index": 0}),
        ("spotify__show_song_library", {"access_token": "token_1", "page_index": 0}),
    ]
    required_api_status = dict(tool_status.get("required_api_status") or {})
    assert required_api_status["completed_tools"] == [
        "spotify__login",
        "spotify__show_album_library",
        "spotify__show_song_library",
        "supervisor__show_profile",
    ]
    assert required_api_status["pending_tools"] == []
    assert required_api_status["deferred_tools"] == ["supervisor__complete_task"]
    assert required_api_status["completed_deferred_tools"] == []
    assert required_api_status["pending_deferred_tools"] == ["supervisor__complete_task"]
    assert required_api_status["actionable_pending_deferred_tools"] == []
    assert required_api_status["all_actionable_completed"] is True


def test_tool_worker_executes_supervisor_complete_task_after_response_is_ready() -> None:
    worker = ToolWorker()
    executor = _RoutingExecutor(
        {
            "supervisor__complete_task": ToolResult(
                tool_name="supervisor__complete_task",
                status=ToolResultStatus.SUCCESS,
                content='{"message": "Marked the active task complete."}',
                payload={"message": "Marked the active task complete."},
            )
        }
    )
    complete_task_tool = ToolSpec(
        name="supervisor__complete_task",
        description="Mark the active task complete.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "answer": {"type": ["string", "null"], "default": None},
                "status": {"type": "string", "default": "success"},
            },
            "required": ["status"],
        },
        metadata={"app_name": "supervisor", "api_name": "complete_task", "mutates_state": True},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": ["supervisor__complete_task"],
            "assistant_response": "Song A, Song B",
            "finish_reason": "task_completed",
        },
        tools=(complete_task_tool,),
        executor=executor,
        current_step={"id": "step_submit", "worker": "tool_worker", "purpose": "Submit the final answer."},
    )
    runtime.state["task"]["instruction"] = "Return the top 2 song titles as a comma-separated list."
    runtime.state["task"]["metadata"]["oracle_required_apis"] = ["supervisor.complete_task"]

    patch = worker.run(runtime)
    raw_results = _patch_value(patch, "/shared_data/raw_results")
    tool_status = _patch_value(patch, "/shared_data/tool_status")

    assert executor.calls == [
        ("supervisor__complete_task", {"answer": "Song A, Song B", "status": "success"})
    ]
    assert raw_results[-1]["tool_name"] == "supervisor__complete_task"
    required_api_status = dict(tool_status.get("required_api_status") or {})
    assert required_api_status["pending_deferred_tools"] == []
    assert required_api_status["completed_deferred_tools"] == ["supervisor__complete_task"]
    assert required_api_status["actionable_pending_deferred_tools"] == []


def test_tool_worker_does_not_treat_nested_artist_or_song_ids_as_album_or_playlist_ids() -> None:
    worker = ToolWorker()

    def album_library_result(tool_call):
        page_index = int(tool_call.arguments.get("page_index", 0))
        payload = (
            [
                {
                    "album_id": 2,
                    "artists": [{"id": 32, "name": "Lucas Grey"}],
                    "song_ids": [8, 9, 10],
                }
            ]
            if page_index == 0
            else []
        )
        return ToolResult(
            tool_name="spotify__show_album_library",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def playlist_library_result(tool_call):
        page_index = int(tool_call.arguments.get("page_index", 0))
        payload = (
            [{"playlist_id": 501, "song_ids": [102, 104]}]
            if page_index == 0
            else []
        )
        return ToolResult(
            tool_name="spotify__show_playlist_library",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload={"response": payload},
        )

    def album_result(tool_call):
        payload = {"album_id": int(tool_call.arguments["album_id"]), "songs": [{"id": 8}]}
        return ToolResult(
            tool_name="spotify__show_album",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    def playlist_result(tool_call):
        payload = {"playlist_id": int(tool_call.arguments["playlist_id"]), "songs": [{"id": 102}]}
        return ToolResult(
            tool_name="spotify__show_playlist",
            status=ToolResultStatus.SUCCESS,
            content=json.dumps(payload),
            payload=payload,
        )

    executor = _RoutingExecutor(
        {
            "spotify__show_album_library": album_library_result,
            "spotify__show_playlist_library": playlist_library_result,
            "spotify__show_album": album_result,
            "spotify__show_playlist": playlist_result,
        }
    )
    album_library_tool = ToolSpec(
        name="spotify__show_album_library",
        description="Show album library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_album_library", "mutates_state": False},
    )
    playlist_library_tool = ToolSpec(
        name="spotify__show_playlist_library",
        description="Show playlist library.",
        parameters_json_schema={
            "type": "object",
            "properties": {"access_token": {"type": "string"}, "page_index": {"type": "integer", "default": 0}},
            "required": ["access_token"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist_library", "mutates_state": False},
    )
    album_tool = ToolSpec(
        name="spotify__show_album",
        description="Show album detail.",
        parameters_json_schema={"type": "object", "properties": {"album_id": {"type": "integer"}}, "required": ["album_id"]},
        metadata={"app_name": "spotify", "api_name": "show_album", "mutates_state": False},
    )
    playlist_tool = ToolSpec(
        name="spotify__show_playlist",
        description="Show playlist detail.",
        parameters_json_schema={
            "type": "object",
            "properties": {"playlist_id": {"type": "integer"}, "access_token": {"type": "string"}},
            "required": ["playlist_id"],
        },
        metadata={"app_name": "spotify", "api_name": "show_playlist", "mutates_state": False},
    )
    runtime = _tool_runtime(
        shared_data={
            "candidate_tools": [
                "spotify__show_album_library",
                "spotify__show_playlist_library",
                "spotify__show_album",
                "spotify__show_playlist",
            ],
            "auth": {
                "supervisor": {"email": "ada@example.com"},
                "auth_sessions": {"spotify": {"payload": {"access_token": "token_1"}}},
            },
        },
        tools=(album_library_tool, playlist_library_tool, album_tool, playlist_tool),
        executor=executor,
        current_step={
            "id": "step_library_chain",
            "worker": "tool_worker",
            "purpose": "Inspect libraries.",
            "api_goal": "Fetch album and playlist libraries then details.",
            "preferred_tools": ["spotify__show_album_library", "spotify__show_playlist_library"],
            "expected_outputs": ["album_library", "playlist_library"],
        },
    )

    worker.run(runtime)

    assert ("spotify__show_album", {"album_id": 2}) in executor.calls
    assert ("spotify__show_playlist", {"playlist_id": 501, "access_token": "token_1"}) in executor.calls
    assert ("spotify__show_album", {"album_id": 32}) not in executor.calls
    assert ("spotify__show_playlist", {"playlist_id": 102, "access_token": "token_1"}) not in executor.calls
