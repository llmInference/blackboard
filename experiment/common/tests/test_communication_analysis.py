"""Tests for Experiment 1 communication analysis helpers."""
from __future__ import annotations

import json

from experiment.common.communication_analysis import (
    analyze_episode_groups,
    build_judge_records,
    build_trace_records,
    export_communication_artifacts,
    summarize_trace_records,
)


def test_build_trace_records_for_webarena_episode():
    episodes = [
        {
            "run_id": "run1",
            "system_id": "blackboard_full",
            "system_family": "blackboard",
            "episode_id": "ep1",
            "task_id": "1",
            "metadata": {"intent": "find item"},
            "trajectory": [
                {
                    "step_id": 0,
                    "action": "click [12]",
                    "decision_reason": "selected by planner",
                    "fallback_used": False,
                    "patch_error": "",
                    "expected_action": "click [12]",
                    "action_matched": True,
                    "communication_trace": [
                        {"source": "architect", "channel": "structured", "content": {"intent": "find item", "next_action": "click [12]"}},
                        {"source": "action_worker", "channel": "structured", "content": {"selected_action": "click [12]"}},
                    ],
                    "planner_state": {"intent": "find item", "next_action": "click [12]"},
                }
            ],
        }
    ]

    records = build_trace_records(episodes, env_name="webarena", group_type="system", group_id="blackboard")

    assert len(records) == 1
    assert records[0]["sender_source"] == "architect"
    assert records[0]["heuristic_drift"] is False
    assert records[0]["expected_action"] == "click [12]"


def test_build_trace_records_for_alfworld_episode_uses_planner_state():
    episodes = [
        {
            "run_id": "run1",
            "system_id": "blackboard_full",
            "system_family": "blackboard",
            "episode_id": "ep1",
            "gamefile": "/tmp/game.tw-pddl",
            "task_goal": "Put apple in fridge",
            "trajectory": [
                {
                    "step_id": 0,
                    "action": "look",
                    "decision_reason": "fallback",
                    "fallback_used": True,
                    "patch_error": "",
                    "planner_state": {
                        "subgoal": "acquire_goal_object",
                        "focus_object": "Apple",
                        "recommended_actions": ["take apple 1"],
                    },
                    "canonical_task": {"canonical_goal_object": "Apple"},
                }
            ],
        }
    ]

    records = build_trace_records(episodes, env_name="alfworld", group_type="mode", group_id="full")

    assert len(records) == 1
    assert records[0]["sender_source"] == "planner_state"
    assert records[0]["heuristic_drift"] is True
    assert "fallback_action" in records[0]["drift_reasons"]


def test_summarize_trace_records_and_judge_records():
    records = [
        {
            "sender_channel": "structured",
            "sender_source": "architect",
            "sender_message": "msg",
            "receiver_action": "click [1]",
            "heuristic_score": 1.0,
            "drift_reasons": [],
        },
        {
            "sender_channel": "structured",
            "sender_source": "architect",
            "sender_message": "msg2",
            "receiver_action": "look",
            "heuristic_score": 0.0,
            "heuristic_drift": True,
            "drift_reasons": ["fallback_action"],
        },
    ]

    summary = summarize_trace_records(records)
    judge_records = build_judge_records(
        [
            {
                **records[0],
                "env_name": "webarena",
                "group_type": "system",
                "group_id": "blackboard",
                "system_id": "blackboard_full",
                "system_family": "blackboard",
                "episode_id": "ep1",
                "task_id": "1",
                "step_id": 0,
                "sender_intent": "find item",
                "expected_action": "click [1]",
                "metadata": {},
            }
        ]
    )

    assert summary["semantic_drift_count"] == 1
    assert summary["semantic_drift_rate"] == 0.5
    assert summary["judge_ready_count"] == 2
    assert judge_records[0]["judge_id"] == "blackboard:ep1:0"


def test_export_communication_artifacts_writes_expected_files(tmp_path):
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "run_id": "run1",
                "system_id": "blackboard_full",
                "system_family": "blackboard",
                "episode_id": "ep1",
                "task_id": "1",
                "metadata": {"intent": "find item"},
                "trajectory": [
                    {
                        "step_id": 0,
                        "action": "click [12]",
                        "decision_reason": "selected by planner",
                        "fallback_used": False,
                        "patch_error": "",
                        "expected_action": "click [12]",
                        "action_matched": True,
                        "communication_trace": [
                            {"source": "architect", "channel": "structured", "content": {"intent": "find item", "next_action": "click [12]"}},
                            {"source": "action_worker", "channel": "structured", "content": {"selected_action": "click [12]"}},
                        ],
                        "planner_state": {"intent": "find item", "next_action": "click [12]"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = export_communication_artifacts(
        output_dir=tmp_path,
        env_name="webarena",
        group_specs=[{"group_type": "system", "group_id": "blackboard", "episodes_path": str(episodes_path)}],
    )

    assert (tmp_path / "comm_summary.json").exists()
    assert (tmp_path / "comm_trace.jsonl").exists()
    assert (tmp_path / "comm_judge.jsonl").exists()
    assert artifacts["summary_path"].endswith("comm_summary.json")


def test_analyze_episode_groups_aggregates_by_group(tmp_path):
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "run_id": "run1",
                "system_id": "blackboard_full",
                "system_family": "blackboard",
                "episode_id": "ep1",
                "task_id": "1",
                "metadata": {"intent": "find item"},
                "trajectory": [
                    {
                        "step_id": 0,
                        "action": "click [12]",
                        "decision_reason": "selected by planner",
                        "fallback_used": False,
                        "patch_error": "",
                        "expected_action": "click [12]",
                        "action_matched": True,
                        "communication_trace": [
                            {"source": "architect", "channel": "structured", "content": {"intent": "find item", "next_action": "click [12]"}},
                        ],
                        "planner_state": {"intent": "find item", "next_action": "click [12]"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary, trace_records, judge_records = analyze_episode_groups(
        [{"group_type": "system", "group_id": "blackboard", "episodes_path": str(episodes_path)}],
        env_name="webarena",
    )

    assert summary["overall"]["n_records"] == 1
    assert summary["by_group"]["blackboard"]["summary"]["semantic_drift_rate"] == 0.0
    assert len(trace_records) == 1
    assert len(judge_records) == 1
