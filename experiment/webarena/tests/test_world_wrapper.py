from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiment.common.neutral import Message, MultiAgentSession, MultiAgentSystem, TurnInput, TurnOutput
from experiment.webarena.core.world_wrapper import WebArenaTaskRunner


class FakeTask:
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
                {"evaluator": "NetworkEventEvaluator", "expected": {"url": ["__GITLAB__/dashboard/todos"]}},
            ],
            "revision": 2,
            "intent_template_id": 303,
        }


class FakeRegexTask:
    task_id = 102
    sites = ("gitlab",)
    start_urls = ("__GITLAB__",)

    def model_dump(self, mode: str = "json"):
        return {
            "task_id": 102,
            "sites": ["gitlab"],
            "start_urls": ["__GITLAB__"],
            "intent": "Navigate to issues with label filters",
            "intent_template": "Navigate to issues with label filters",
            "instantiation_dict": {},
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "null"},
                    "expected": {"retrieved_data": None},
                },
                {
                    "evaluator": "NetworkEventEvaluator",
                    "expected": {"url": "^__GITLAB__/byteblaze/a11y-syntax-highlighting/-/issues.*$"},
                },
                {
                    "evaluator": "NetworkEventEvaluator",
                    "expected": {"url": "__GITLAB__/api/graphql"},
                },
            ],
            "revision": 2,
            "intent_template_id": 999,
        }


class FakeConfig:
    def render_url(self, urls, sites):
        del sites
        rendered = []
        for url in urls:
            text = str(url)
            rendered.append(text.replace("__GITLAB__", "http://localhost:8023"))
        return rendered


class FakeWebArenaVerified:
    def __init__(self, *, config):
        self.config = FakeConfig()
        self._config_path = config

    def get_task(self, task_id: int):
        assert task_id == 44
        return FakeTask()


class FakeRegexWebArenaVerified(FakeWebArenaVerified):
    def get_task(self, task_id: int):
        assert task_id == 102
        return FakeRegexTask()


class FakeRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opened = False
        self.closed = False
        self.evaluated_with = None

    def open(self):
        self.opened = True
        return self

    def execute_tool_call(self, tool_call):
        from experiment.common.neutral import Observation, ToolResult

        return (
            ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                content="tool=browser__goto | status=success | url=http://localhost:8023/dashboard/todos",
                payload={"observation": {"url": "http://localhost:8023/dashboard/todos"}},
                metadata={},
            ),
            Observation(
                source="webarena_browser",
                content="URL: http://localhost:8023/dashboard/todos",
                payload={"url": "http://localhost:8023/dashboard/todos"},
                metadata={},
            ),
        )

    def evaluate(self, *, agent_response=None):
        self.evaluated_with = agent_response
        return {"status": "success", "score": 1.0}

    def close(self):
        self.closed = True


class FakeSession(MultiAgentSession):
    def __init__(self):
        self.calls = 0

    def step(self, turn_input: TurnInput) -> TurnOutput:
        self.calls += 1
        if self.calls == 1:
            return TurnOutput.request_tool("browser__goto", {"url": "http://localhost:8023/dashboard/todos"}, call_id="call-1")
        return TurnOutput(
            actions=(
                TurnOutput.emit_text("done").actions[0],
                TurnOutput.finish_turn("task_completed").actions[0],
            ),
            metadata={"finish_reason": "task_completed"},
        )


class FakeSystem(MultiAgentSystem):
    def create_session(self, task, tools=(), native_tools=()):
        self.created = {"task": task, "tools": tools, "native_tools": native_tools}
        return FakeSession()


class FakeNavigateOnlySession(MultiAgentSession):
    def __init__(self):
        self.calls = 0

    def step(self, turn_input: TurnInput) -> TurnOutput:
        self.calls += 1
        if self.calls == 1:
            return TurnOutput.request_tool("browser__goto", {"url": "http://localhost:8023/dashboard/todos"}, call_id="call-1")
        return TurnOutput.finish_turn("task_completed", metadata={"finish_reason": "task_completed"})


class FakeNavigateOnlySystem(MultiAgentSystem):
    def create_session(self, task, tools=(), native_tools=()):
        self.created = {"task": task, "tools": tools, "native_tools": native_tools}
        return FakeNavigateOnlySession()


def test_webarena_task_runner_builds_turn_inputs_and_runs_session(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"environments": {"__GITLAB__": {"urls": ["http://localhost:8023"]}}}), encoding="utf-8")

    runner = WebArenaTaskRunner(
        task_id=44,
        config_path=config_path,
        output_dir=tmp_path / "run",
        webarena_factory=FakeWebArenaVerified,
        runtime_factory=FakeRuntime,
    )
    system = FakeSystem()

    result = runner.run_session(system, max_steps=3)

    assert result.task.task_id == "44"
    assert result.finished is True
    assert result.steps == 2
    assert result.evaluation["score"] == 1.0
    assert result.tool_results[0].call_id == "call-1"
    assert result.observations[0].payload["url"] == "http://localhost:8023/dashboard/todos"
    assert result.messages[-1].content == "done"
    assert system.created["native_tools"][0].opened is True


def test_webarena_task_runner_close_closes_runtime(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"environments": {"__GITLAB__": {"urls": ["http://localhost:8023"]}}}), encoding="utf-8")

    runner = WebArenaTaskRunner(
        task_id=44,
        config_path=config_path,
        output_dir=tmp_path / "run",
        webarena_factory=FakeWebArenaVerified,
        runtime_factory=FakeRuntime,
    )
    runner.open()
    runtime = runner.runtime
    runner.close()

    assert runtime.closed is True


def test_webarena_task_runner_synthesizes_navigation_agent_response(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"environments": {"__GITLAB__": {"urls": ["http://localhost:8023"]}}}), encoding="utf-8")

    runner = WebArenaTaskRunner(
        task_id=44,
        config_path=config_path,
        output_dir=tmp_path / "run",
        webarena_factory=FakeWebArenaVerified,
        runtime_factory=FakeRuntime,
    )
    system = FakeNavigateOnlySystem()

    result = runner.run_session(system, max_steps=3)

    runtime = system.created["native_tools"][0]
    assert result.finished is True
    assert runtime.evaluated_with == {
        "task_type": "NAVIGATE",
        "status": "SUCCESS",
        "retrieved_data": None,
    }


def test_webarena_task_runner_extracts_rendered_target_urls_from_string_and_regex(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"environments": {"__GITLAB__": {"urls": ["http://localhost:8023"]}}}), encoding="utf-8")

    runner = WebArenaTaskRunner(
        task_id=102,
        config_path=config_path,
        output_dir=tmp_path / "run",
        webarena_factory=FakeRegexWebArenaVerified,
        runtime_factory=FakeRuntime,
    )
    runner.open()
    task_spec = runner.task_spec

    assert task_spec.metadata["rendered_target_urls"] == [
        "http://localhost:8023/byteblaze/a11y-syntax-highlighting/-/issues",
        "http://localhost:8023/api/graphql",
    ]
