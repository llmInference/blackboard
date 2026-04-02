from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from experiment.common.neutral import Message, TaskSpec, ToolResult
from experiment.webarena.examples import common as examples_common
from experiment.webarena.examples.common import episode_payload, run_blackboard_batch, summary_payload


@dataclass
class _FakeRunResult:
    task: TaskSpec
    evaluation: dict
    finished: bool
    steps: int
    messages: tuple[Message, ...]
    tool_results: tuple[ToolResult, ...]
    run_metadata: dict


def test_episode_payload_includes_common_analysis_fields():
    task = TaskSpec(
        task_id="44",
        instruction="Open my todos page",
        domain="webarena",
        metadata={"task_category": "navigation", "requires_final_response": False},
    )
    result = _FakeRunResult(
        task=task,
        evaluation={"success": True, "score": 1.0},
        finished=True,
        steps=2,
        messages=(Message(role="assistant", content="done"),),
        tool_results=(ToolResult(tool_name="browser__goto", call_id="c1"),),
        run_metadata={
            "finish_reason": "task_completed",
            "goal_condition_rate": 1.0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
            "total_tokens": 0,
            "fallback_action_count": 0,
            "patch_error_count": 0,
            "retrieval_precision": 1.0,
            "context_fragment_count": 3.0,
            "relevant_fragment_count": 3.0,
            "irrelevant_fragment_count": 0.0,
            "trajectory": [
                {
                    "step_id": 0,
                    "action": "browser__goto {\"url\": \"http://localhost:8023/dashboard/todos\"}",
                    "communication_trace": [
                        {"source": "architect", "channel": "structured", "content": {"intent": "Open my todos page"}},
                    ],
                }
            ],
            "communication_trace": [
                {"event_type": "architect", "selected_workers": ["page_state_worker", "verification_worker"]},
            ],
            "verification": {"goal_reached": True},
        },
    )

    payload = episode_payload(
        result,
        task_config={
            "sites": ["gitlab"],
            "intent": "Open my todos page",
            "start_urls": ["http://localhost:8023"],
        },
        run_id="demo_run",
        system_id="blackboard",
        system_family="blackboard",
    )

    assert payload["episode_id"] == "blackboard:44"
    assert payload["success"] is True
    assert payload["goal_condition_rate"] == 1.0
    assert payload["worker_input_tokens"] == 0
    assert payload["fallback_action_count"] == 0
    assert payload["trajectory"][0]["step_id"] == 0
    assert payload["communication_trace"][0]["event_type"] == "architect"
    assert payload["metadata"]["sites"] == ["gitlab"]


def test_summary_payload_exposes_mean_metrics(tmp_path: Path):
    episodes = [
        {
            "task_id": "44",
            "success": True,
            "finished": True,
            "steps": 2,
            "goal_condition_rate": 1.0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
            "total_tokens": 0,
            "tool_call_count": 1,
            "assistant_message_count": 0,
            "fallback_action_count": 0,
            "patch_error_count": 0,
            "retrieval_precision": 1.0,
            "context_fragment_count": 3.0,
            "relevant_fragment_count": 3.0,
            "irrelevant_fragment_count": 0.0,
            "finish_reason": "task_completed",
            "task_category": "navigation",
            "metadata": {"sites": ["gitlab"]},
        },
        {
            "task_id": "45",
            "success": False,
            "finished": True,
            "steps": 4,
            "goal_condition_rate": 0.0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
            "total_tokens": 0,
            "tool_call_count": 1,
            "assistant_message_count": 0,
            "fallback_action_count": 0,
            "patch_error_count": 1,
            "retrieval_precision": 0.0,
            "context_fragment_count": 4.0,
            "relevant_fragment_count": 4.0,
            "irrelevant_fragment_count": 0.0,
            "finish_reason": "patch_error",
            "task_category": "navigation",
            "metadata": {"sites": ["gitlab"]},
        },
    ]

    summary = summary_payload(
        episodes,
        task_set="debug",
        manifest_path=tmp_path / "manifest.json",
        config_path=tmp_path / "config.json",
    )

    assert summary["n_episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["mean_goal_condition_rate"] == 0.5
    assert summary["mean_steps"] == 3.0
    assert summary["mean_patch_error_rate"] == 0.125
    assert summary["site_breakdown"]["gitlab"] == 2
    assert summary["finish_reason_breakdown"]["task_completed"] == 1


def test_run_blackboard_batch_writes_fake_smoke_artifacts(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "task_id": 44,
                    "sites": ["gitlab"],
                    "intent": "Open my todos page",
                    "start_urls": ["http://localhost:8023"],
                }
            ]
        ),
        encoding="utf-8",
    )

    task = TaskSpec(
        task_id="44",
        instruction="Open my todos page",
        domain="webarena",
        metadata={"task_category": "navigation", "requires_final_response": False},
    )

    fake_result = _FakeRunResult(
        task=task,
        evaluation={"success": True, "score": 1.0},
        finished=True,
        steps=2,
        messages=(),
        tool_results=(ToolResult(tool_name="browser__goto", call_id="c1"),),
        run_metadata={
            "finish_reason": "task_completed",
            "goal_condition_rate": 1.0,
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
            "architect_input_tokens": 0,
            "architect_output_tokens": 0,
            "total_tokens": 0,
            "fallback_action_count": 0,
            "patch_error_count": 0,
            "retrieval_precision": 1.0,
            "context_fragment_count": 3.0,
            "relevant_fragment_count": 3.0,
            "irrelevant_fragment_count": 0.0,
            "trajectory": [],
            "communication_trace": [],
            "verification": {"goal_reached": True},
        },
    )

    monkeypatch.setattr(examples_common, "run_blackboard_task", lambda **kwargs: fake_result)

    output_dir = tmp_path / "smoke"
    episodes = run_blackboard_batch(
        dataset_path=dataset_path,
        config_path=tmp_path / "config.json",
        task_ids=[44],
        output_dir=output_dir,
        dataset_name="webarena-verified",
    )

    assert len(episodes) == 1
    assert (output_dir / "episode_results.jsonl").exists()
    assert (output_dir / "44" / "episode.json").exists()
    stored_episode = json.loads((output_dir / "44" / "episode.json").read_text(encoding="utf-8"))
    assert stored_episode["success"] is True
