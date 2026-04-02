from __future__ import annotations

import json
from pathlib import Path

from experiment.common.neutral import Observation, TaskSpec, ToolResult, ToolResultStatus, TurnInput
from experiment.webarena.bridge import browser_tools_to_specs
from experiment.webarena.systems import WebArenaBlackboardSystem, run_blackboard_task


class _FakeLLMResponse:
    def __init__(self, payload: dict, *, prompt_tokens: int = 13, completion_tokens: int = 5) -> None:
        self.content = json.dumps(payload, ensure_ascii=False)
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class _SequenceLLM:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)

    def invoke(self, messages):
        self.messages = messages
        payload = self.payloads.pop(0)
        return _FakeLLMResponse(payload)


class _FakeRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.current_url = "http://localhost:8023"
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def current_observation(self):
        title = "Your To-Do List" if self.current_url.endswith("/dashboard/todos") else "Home"
        text = "Todo item one. Todo item two." if self.current_url.endswith("/dashboard/todos") else "Welcome home"
        return {
            "url": self.current_url,
            "title": title,
            "text": text,
            "tabs": [{"index": 0, "url": self.current_url, "title": title, "active": True}],
            "active_tab_index": 0,
            "elements": [],
        }

    def execute_tool_call(self, tool_call):
        self.current_url = str(tool_call.arguments.get("url", self.current_url))
        observation = {
            "url": self.current_url,
            "title": "Your To-Do List" if self.current_url.endswith("/dashboard/todos") else "Page",
            "text": "Todo item one. Todo item two.",
            "tabs": [{"index": 0, "url": self.current_url, "title": "Page", "active": True}],
            "active_tab_index": 0,
            "elements": [],
        }
        return (
            ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                status=ToolResultStatus.SUCCESS,
                content=f"navigated to {self.current_url}",
                payload={"observation": observation},
            ),
            Observation(
                source="webarena_browser",
                content=f"URL: {self.current_url}",
                payload=observation,
            ),
        )

    def evaluate(self, *, agent_response=None):
        return {"status": "success", "score": 1.0, "agent_response": agent_response}

    def close(self):
        self.closed = True


class _LoopingRuntime(_FakeRuntime):
    def __init__(self, *, fail_first: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.call_count = 0
        self.fail_first = fail_first

    def execute_tool_call(self, tool_call):
        self.call_count += 1
        success = self.call_count > 1
        current_url = str(tool_call.arguments.get("url", self.current_url)) if success else self.current_url
        error_message = "temporary browser error" if self.fail_first and self.call_count == 1 else ""
        self.current_url = current_url
        observation = {
            "url": self.current_url,
            "title": "Your To-Do List" if self.current_url.endswith("/dashboard/todos") else "Home",
            "text": "Todo item one. Todo item two." if self.current_url.endswith("/dashboard/todos") else "Welcome home",
            "tabs": [{"index": 0, "url": self.current_url, "title": "Page", "active": True}],
            "active_tab_index": 0,
            "elements": [],
        }
        return (
            ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                status=ToolResultStatus.ERROR if error_message else ToolResultStatus.SUCCESS,
                content=f"navigated to {self.current_url}",
                payload={"observation": observation},
                error_message=error_message,
            ),
            Observation(
                source="webarena_browser",
                content=f"URL: {self.current_url}",
                payload=observation,
            ),
        )


def _task_spec(*, requires_final_response: bool) -> TaskSpec:
    eval_payload = [
        {"evaluator": "NetworkEventEvaluator", "expected": {"url": ["http://localhost:8023/dashboard/todos"]}},
    ]
    if requires_final_response:
        eval_payload.insert(
            0,
            {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "string"},
                "expected": {"retrieved_data": {"summary": "Todo item one"}},
            },
        )
    else:
        eval_payload.insert(
            0,
            {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "null"},
                "expected": {"retrieved_data": None},
            },
        )
    return TaskSpec(
        task_id="44",
        instruction="Open my todos page",
        domain="webarena",
        metadata={
            "sites": ["gitlab"],
            "start_urls": ["http://localhost:8023"],
            "requires_final_response": requires_final_response,
            "task_category": "navigation" if not requires_final_response else "retrieval",
            "eval": eval_payload,
            "evaluators": ["AgentResponseEvaluator", "NetworkEventEvaluator"],
        },
    )


def test_webarena_blackboard_session_requests_goto_then_finishes():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True})
    runtime = _FakeRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))

    assert len(first.requested_tool_calls) == 1
    assert first.requested_tool_calls[0].tool_name == "browser__goto"
    assert first.requested_tool_calls[0].arguments["url"] == "http://localhost:8023/dashboard/todos"
    assert first.metadata["current_step_id"] == "inspect_current_page"
    assert first.metadata["communication_trace"][0]["event_type"] == "architect"
    assert first.metadata["architect_debug"]["stage_policy"]["initial_phase"] == "navigate"

    tool_result, observation = runtime.execute_tool_call(first.requested_tool_calls[0])
    second = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(tool_result,),
            new_observations=(observation,),
        )
    )

    assert second.finished is True
    assert not second.emitted_messages
    assert second.metadata["goal_condition_rate"] == 1.0
    assert len(second.metadata["trajectory"]) == 1
    assert second.metadata["trajectory"][0]["action"].startswith("browser__goto")
    assert second.metadata["trajectory"][0]["communication_trace"][0]["source"] == "architect"
    assert second.metadata["trajectory"][0]["observation_after"]["url"] == "http://localhost:8023/dashboard/todos"


def test_webarena_blackboard_session_composes_response_when_required():
    task = _task_spec(requires_final_response=True)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True})
    runtime = _FakeRuntime()
    runtime.current_url = "http://localhost:8023/dashboard/todos"
    session = system.create_session(task, tools, native_tools=(runtime,))

    output = session.step(TurnInput(task=task, available_tools=tools))

    assert output.finished is True
    assert output.emitted_messages
    assert "Todo" in output.emitted_messages[0].content or "todo" in output.emitted_messages[0].content.lower()


def test_webarena_blackboard_session_no_architect_records_fallback():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 2, "use_architect": False, "enable_target_urls": True})
    runtime = _FakeRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))

    assert first.metadata["used_fallback_architect"] is True
    assert first.metadata["fallback_reason"] == "architect_disabled"
    assert first.metadata["architect_debug"]["architect_disabled"] is True


def test_webarena_blackboard_session_architect_retries_before_fallback():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    llm = _SequenceLLM(
        [
            "invalid architect payload",
            {
                "workflow": {
                    "workflow_id": "webarena_navigation_retry",
                    "goal": "Open my todos page",
                    "steps": [
                        {
                            "id": "inspect_current_page",
                            "worker": "page_state_worker",
                            "purpose": "Extract page state.",
                            "reads": ["last_observation"],
                            "writes": ["current_page", "page_evidence", "open_tabs"],
                            "exit_conditions": ["page summarized"],
                            "next": [
                                {"condition": "goal_reached", "step": ""},
                                {"condition": "needs_action", "step": "bind_action_arguments"},
                            ],
                        },
                        {
                            "id": "bind_action_arguments",
                            "worker": "argument_grounding_worker",
                            "purpose": "Ground next action.",
                            "reads": ["task_constraints", "current_page", "page_evidence", "verification", "action_history"],
                            "writes": ["grounded_action", "action_arguments"],
                            "exit_conditions": ["action grounded"],
                            "next": [{"condition": "has_grounded_action", "step": "execute_browser_action"}],
                        },
                        {
                            "id": "execute_browser_action",
                            "worker": "browser_action_worker",
                            "purpose": "Execute action.",
                            "reads": ["grounded_action"],
                            "writes": ["last_action", "action_history", "execution_error"],
                            "exit_conditions": ["action executed"],
                            "next": [{"condition": "tool_call_available", "step": "inspect_current_page"}],
                        },
                    ],
                },
                "data_schema": {},
                "selected_workers": [
                    "page_state_worker",
                    "argument_grounding_worker",
                    "browser_action_worker",
                ],
                "worker_instructions": {},
                "warnings": [],
            },
            {
                "current_page": {
                    "url": "http://localhost:8023/",
                    "title": "Home",
                    "visible_text_excerpt": "Welcome home",
                    "interactive_elements": [],
                    "active_tab_index": 0,
                },
                "page_evidence": {
                    "url": "http://localhost:8023/",
                    "title": "Home",
                    "text_excerpt": "Welcome home",
                    "target_url_match": False,
                    "element_count": 0,
                },
                "open_tabs": [{"index": 0, "url": "http://localhost:8023/", "title": "Home", "active": True}],
                "verification": {
                    "goal_reached": False,
                    "matched_target_url": "",
                    "requires_final_response": False,
                    "needs_response": False,
                    "progress_made": False,
                    "evidence_ready": False,
                    "answer_ready": False,
                    "phase": "navigate",
                    "task_category": "navigation",
                },
                "finish": False,
                "finish_reason": "",
            },
        ]
    )
    system = WebArenaBlackboardSystem(llm=llm, config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True})
    runtime = _FakeRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))

    assert first.requested_tool_calls
    assert first.metadata["used_fallback_architect"] is False
    assert first.metadata["architect_debug"]["retry_attempt"] == 2


def test_webarena_blackboard_session_repeats_inspect_act_verify_until_goal():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 3, "enable_target_urls": True})
    runtime = _LoopingRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))
    first_result, first_observation = runtime.execute_tool_call(first.requested_tool_calls[0])
    second = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(first_result,),
            new_observations=(first_observation,),
        )
    )
    assert second.requested_tool_calls

    second_result, second_observation = runtime.execute_tool_call(second.requested_tool_calls[0])
    third = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(second_result,),
            new_observations=(second_observation,),
        )
    )

    assert third.finished is True
    assert runtime.call_count == 2
    assert len(third.metadata["trajectory"]) == 2
    assert third.metadata["goal_condition_rate"] == 1.0


def test_webarena_blackboard_session_recovers_from_browser_action_error():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 3, "enable_target_urls": True})
    runtime = _LoopingRuntime(fail_first=True)
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))
    first_result, first_observation = runtime.execute_tool_call(first.requested_tool_calls[0])
    second = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(first_result,),
            new_observations=(first_observation,),
        )
    )
    assert second.requested_tool_calls

    second_result, second_observation = runtime.execute_tool_call(second.requested_tool_calls[0])
    third = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(second_result,),
            new_observations=(second_observation,),
        )
    )

    assert third.finished is True
    assert third.metadata["circuit_breaker"]["tripped"] is False
    assert third.metadata["trajectory"][0]["tool_error"] == "temporary browser error"


def test_webarena_blackboard_session_enforces_bounded_write_scope():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    system = WebArenaBlackboardSystem(config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True})
    runtime = _FakeRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    class _BadWorker:
        spec = session.workers["page_state_worker"].spec

        def run(self, runtime):
            del runtime
            return type("BadResult", (), {"patch": ({"op": "replace", "path": "/shared_data/assistant_response", "value": "oops"},), "tool_call": None, "message": None, "finish_reason": "", "finished": False})()

    session.workers["page_state_worker"] = _BadWorker()
    output = session.step(TurnInput(task=task, available_tools=tools))

    assert output.finished is True
    assert output.metadata["finish_reason"] == "patch_error"
    assert output.metadata["circuit_breaker"]["reason"] == "patch_error"
    assert output.metadata["patch_error_count"] == 3
    assert output.metadata["worker_retry_count"] == 3


class _FakeTask:
    task_id = 44
    sites = ("gitlab",)
    start_urls = ("__GITLAB__",)

    def model_dump(self, mode: str = "json"):
        return {
            "task_id": 44,
            "sites": ["gitlab"],
            "start_urls": ["__GITLAB__"],
            "intent": "Open my todos page",
            "intent_template": "Open my todos page",
            "instantiation_dict": {},
            "eval": [
                {"evaluator": "AgentResponseEvaluator", "results_schema": {"type": "null"}, "expected": {"retrieved_data": None}},
                {"evaluator": "NetworkEventEvaluator", "expected": {"url": ["http://localhost:8023/dashboard/todos"]}},
            ],
            "revision": 2,
            "intent_template_id": 303,
        }


class _FakeConfig:
    def render_url(self, urls, sites):
        del sites
        return ["http://localhost:8023" if url == "__GITLAB__" else url for url in urls]


class _FakeWebArenaVerified:
    def __init__(self, *, config):
        self.config = _FakeConfig()
        self._config = config

    def get_task(self, task_id: int):
        assert task_id == 44
        return _FakeTask()


def test_run_blackboard_task_drives_runner_contract(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"environments": {"__GITLAB__": {"urls": ["http://localhost:8023"]}}}), encoding="utf-8")

    result = run_blackboard_task(
        task_id=44,
        config_path=config_path,
        output_dir=tmp_path / "run",
        webarena_factory=_FakeWebArenaVerified,
        runtime_factory=_FakeRuntime,
        system_config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True},
        max_steps=4,
    )

    assert result.finished is True
    assert result.steps == 2
    assert result.evaluation["score"] == 1.0
    assert result.tool_results[0].tool_name == "browser__goto"
    assert result.run_metadata["communication_trace"]
    assert result.run_metadata["goal_condition_rate"] == 1.0
    assert result.run_metadata["trajectory"][0]["action"].startswith("browser__goto")


def test_webarena_blackboard_session_accumulates_llm_worker_tokens():
    task = _task_spec(requires_final_response=False)
    tools = browser_tools_to_specs()
    llm = _SequenceLLM(
        [
            {
                "workflow": {
                    "workflow_id": "webarena_navigation_llm",
                    "goal": "Open my todos page",
                    "steps": [
                        {
                            "id": "inspect_current_page",
                            "worker": "page_state_worker",
                            "purpose": "Extract page state.",
                            "reads": ["last_observation"],
                            "writes": ["current_page", "page_evidence", "open_tabs"],
                            "exit_conditions": ["page summarized"],
                            "next": [
                                {"condition": "goal_reached", "step": ""},
                                {"condition": "needs_action", "step": "bind_action_arguments"},
                            ],
                        },
                        {
                            "id": "bind_action_arguments",
                            "worker": "argument_grounding_worker",
                            "purpose": "Ground next action.",
                            "reads": ["task_constraints", "current_page", "page_evidence", "verification"],
                            "writes": ["grounded_action", "action_arguments"],
                            "exit_conditions": ["action grounded"],
                            "next": [{"condition": "has_grounded_action", "step": "execute_browser_action"}],
                        },
                        {
                            "id": "execute_browser_action",
                            "worker": "browser_action_worker",
                            "purpose": "Execute action.",
                            "reads": ["grounded_action"],
                            "writes": ["last_action", "action_history", "execution_error"],
                            "exit_conditions": ["action executed"],
                            "next": [{"condition": "tool_call_available", "step": "inspect_current_page"}],
                        },
                    ],
                },
                "data_schema": {},
                "selected_workers": [
                    "page_state_worker",
                    "argument_grounding_worker",
                    "browser_action_worker",
                ],
                "worker_instructions": {},
                "warnings": [],
            },
            {
                "current_page": {
                    "url": "http://localhost:8023/",
                    "title": "Home",
                    "visible_text_excerpt": "Welcome home",
                    "interactive_elements": [],
                    "active_tab_index": 0,
                },
                "page_evidence": {
                    "url": "http://localhost:8023/",
                    "title": "Home",
                    "text_excerpt": "Welcome home",
                    "target_url_match": False,
                    "element_count": 0,
                },
                "open_tabs": [{"index": 0, "url": "http://localhost:8023/", "title": "Home", "active": True}],
                "verification": {
                    "goal_reached": False,
                    "matched_target_url": "",
                    "requires_final_response": False,
                    "needs_response": False,
                    "progress_made": False,
                    "evidence_ready": False,
                    "answer_ready": False,
                    "phase": "navigate",
                    "task_category": "navigation",
                },
                "finish": False,
                "finish_reason": "",
            },
            {
                "current_page": {
                    "url": "http://localhost:8023/dashboard/todos",
                    "title": "Your To-Do List",
                    "visible_text_excerpt": "Todo item one. Todo item two.",
                    "interactive_elements": [],
                    "active_tab_index": 0,
                },
                "page_evidence": {
                    "url": "http://localhost:8023/dashboard/todos",
                    "title": "Your To-Do List",
                    "text_excerpt": "Todo item one. Todo item two.",
                    "target_url_match": True,
                    "element_count": 0,
                },
                "open_tabs": [{"index": 0, "url": "http://localhost:8023/dashboard/todos", "title": "Your To-Do List", "active": True}],
                "verification": {
                    "goal_reached": True,
                    "matched_target_url": "http://localhost:8023/dashboard/todos",
                    "requires_final_response": False,
                    "needs_response": False,
                    "progress_made": True,
                    "evidence_ready": True,
                    "answer_ready": False,
                    "phase": "complete",
                    "task_category": "navigation",
                },
                "finish": True,
                "finish_reason": "task_completed",
            },
        ]
    )
    system = WebArenaBlackboardSystem(llm=llm, config={"max_kernel_steps": 8, "max_no_progress": 2, "enable_target_urls": True})
    runtime = _FakeRuntime()
    session = system.create_session(task, tools, native_tools=(runtime,))

    first = session.step(TurnInput(task=task, available_tools=tools))
    tool_result, observation = runtime.execute_tool_call(first.requested_tool_calls[0])
    second = session.step(
        TurnInput(
            task=task,
            available_tools=tools,
            new_tool_results=(tool_result,),
            new_observations=(observation,),
        )
    )

    assert second.finished is True
    assert second.metadata["worker_input_tokens"] == 26
    assert second.metadata["worker_output_tokens"] == 10
    assert second.metadata["architect_input_tokens"] == 13
    assert second.metadata["architect_output_tokens"] == 5
