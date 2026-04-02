"""Tests for cross-dataset aggregation helpers."""
from __future__ import annotations

import json

import pytest

from experiment.common.cross_dataset_analysis import (
    build_cross_dataset_ablation,
    build_cross_dataset_system_compare,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_cross_dataset_system_compare_aggregates_three_datasets(tmp_path):
    alf_episodes = tmp_path / "alf_episodes.jsonl"
    _write_jsonl(
        alf_episodes,
        [
            {
                "gamefile": "/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
                "success": True,
                "steps": 2,
                "total_tokens": 12,
                "goal_condition_rate": 1.0,
                "worker_input_tokens": 4,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "no_update_event_count": 0,
                "repeat_action_count": 0,
                "repeat_observation_count": 0,
                "stagnation_event_count": 0,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "environment_done",
                "final_status": "success",
                "workflow_final_status": "completed",
                "trajectory": [],
            }
        ],
    )
    alf_summary = tmp_path / "alf_system_compare_summary.json"
    alf_summary.write_text(
        json.dumps(
            {
                "systems": ["blackboard", "langgraph"],
                "n_gamefiles": 1,
                "workflow_mode": "planner_action",
                "system_summaries": {
                    "blackboard": {
                        "system_output_dir": str(tmp_path / "alf_blackboard"),
                        "result": {
                            "summary": {
                                "success_rate": 1.0,
                                "mean_goal_condition_rate": 1.0,
                                "mean_steps": 2.0,
                                "mean_total_tokens": 12.0,
                                "mean_worker_input_tokens": 4.0,
                            },
                            "episodes_path": str(alf_episodes),
                        },
                    },
                    "langgraph": {
                        "system_output_dir": str(tmp_path / "alf_langgraph"),
                        "result": {
                            "summary": {
                                "success_rate": 0.5,
                                "mean_goal_condition_rate": 0.5,
                                "mean_steps": 3.0,
                                "mean_total_tokens": 18.0,
                                "mean_worker_input_tokens": 6.0,
                            },
                            "episodes_path": str(alf_episodes),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    web_episodes = tmp_path / "web_episodes.jsonl"
    _write_jsonl(
        web_episodes,
        [
            {
                "task_id": "web-1",
                "success": True,
                "steps": 3,
                "total_tokens": 20,
                "goal_condition_rate": 0.8,
                "worker_input_tokens": 8,
                "worker_output_tokens": 4,
                "architect_input_tokens": 2,
                "architect_output_tokens": 2,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "stop_reason": "reference_completed",
                "final_status": "success",
                "workflow_final_status": "completed",
                "metadata": {"sites": ["reddit"]},
            }
        ],
    )
    web_summary = tmp_path / "web_system_compare_summary.json"
    web_summary.write_text(
        json.dumps(
            {
                "systems": ["blackboard", "langgraph"],
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "execution_backend": "reference_oracle",
                "system_summaries": {
                    "blackboard": {
                        "system_output_dir": str(tmp_path / "web_blackboard"),
                        "result": {
                            "summary": {
                                "success_rate": 0.75,
                                "mean_goal_condition_rate": 0.8,
                                "mean_steps": 3.0,
                                "mean_total_tokens": 20.0,
                                "mean_worker_input_tokens": 8.0,
                                "mean_retrieval_precision": 0.9,
                            },
                            "episodes_path": str(web_episodes),
                        },
                    },
                    "langgraph": {
                        "system_output_dir": str(tmp_path / "web_langgraph"),
                        "result": {
                            "summary": {
                                "success_rate": 0.25,
                                "mean_goal_condition_rate": 0.3,
                                "mean_steps": 4.0,
                                "mean_total_tokens": 28.0,
                                "mean_worker_input_tokens": 10.0,
                                "mean_retrieval_precision": 0.5,
                            },
                            "episodes_path": str(web_episodes),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    science_episodes = tmp_path / "science_episodes.jsonl"
    _write_jsonl(
        science_episodes,
        [
            {
                "task_id": "1-1",
                "success": True,
                "steps": 4,
                "total_tokens": 14,
                "goal_condition_rate": 0.7,
                "worker_input_tokens": 5,
                "worker_output_tokens": 4,
                "architect_input_tokens": 1,
                "architect_output_tokens": 2,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "stop_reason": "reference_completed",
                "final_status": "success",
                "workflow_final_status": "completed",
                "metadata": {"task_name": "1-1"},
            }
        ],
    )
    science_summary = tmp_path / "science_system_compare_summary.json"
    science_summary.write_text(
        json.dumps(
            {
                "systems": ["blackboard", "langgraph"],
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "system_summaries": {
                    "blackboard": {
                        "system_output_dir": str(tmp_path / "science_blackboard"),
                        "result": {
                            "summary": {
                                "success_rate": 0.5,
                                "mean_goal_condition_rate": 0.7,
                                "mean_steps": 4.0,
                                "mean_total_tokens": 14.0,
                                "mean_worker_input_tokens": 5.0,
                            },
                            "episodes_path": str(science_episodes),
                        },
                    },
                    "langgraph": {
                        "system_output_dir": str(tmp_path / "science_langgraph"),
                        "result": {
                            "summary": {
                                "success_rate": 0.25,
                                "mean_goal_condition_rate": 0.4,
                                "mean_steps": 5.0,
                                "mean_total_tokens": 18.0,
                                "mean_worker_input_tokens": 6.0,
                            },
                            "episodes_path": str(science_episodes),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_cross_dataset_system_compare(
        experiment_id="exp4",
        dataset_summaries={
            "alfworld": str(alf_summary),
            "webarena": str(web_summary),
            "scienceworld": str(science_summary),
        },
    )

    assert report["analysis_type"] == "cross_dataset_system_compare"
    assert report["dataset_count"] == 3
    assert report["aggregate_by_system"]["blackboard"]["dataset_count"] == 3
    assert report["aggregate_by_system"]["blackboard"]["metrics"]["success_rate"]["mean"] == pytest.approx(0.75)
    assert report["aggregate_by_system"]["langgraph"]["metrics"]["success_rate"]["mean"] == pytest.approx(1.0 / 3.0)
    assert report["system_order_by_mean_success_rate"][0] == "blackboard"


def test_build_cross_dataset_ablation_aggregates_mode_deltas(tmp_path):
    alf_full = tmp_path / "alf_full.jsonl"
    _write_jsonl(
        alf_full,
        [
            {
                "gamefile": "/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
                "success": True,
                "steps": 2,
                "total_tokens": 12,
                "goal_condition_rate": 1.0,
                "worker_input_tokens": 4,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "no_update_event_count": 0,
                "repeat_action_count": 0,
                "repeat_observation_count": 0,
                "stagnation_event_count": 0,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "environment_done",
                "final_status": "success",
                "workflow_final_status": "completed",
                "trajectory": [],
            }
        ],
    )
    alf_ablate = tmp_path / "alf_ablate.jsonl"
    _write_jsonl(
        alf_ablate,
        [
            {
                "gamefile": "/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
                "success": False,
                "steps": 3,
                "total_tokens": 18,
                "goal_condition_rate": 0.4,
                "worker_input_tokens": 6,
                "fallback_action_count": 1,
                "patch_error_count": 0,
                "no_update_event_count": 0,
                "repeat_action_count": 0,
                "repeat_observation_count": 0,
                "stagnation_event_count": 0,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "max_steps_reached",
                "final_status": "incomplete",
                "workflow_final_status": "completed",
                "trajectory": [],
            }
        ],
    )
    alf_summary = tmp_path / "alf_ablation_summary.json"
    alf_summary.write_text(
        json.dumps(
            {
                "modes": ["full", "ablate_c1"],
                "n_gamefiles": 1,
                "workflow_mode": "planner_action",
                "summaries": {
                    "full": {
                        "episodes_path": str(alf_full),
                        "success_rate": 1.0,
                        "mean_goal_condition_rate": 1.0,
                        "mean_steps": 2.0,
                        "mean_total_tokens": 12.0,
                        "mean_worker_input_tokens": 4.0,
                        "mean_fallback_rate": 0.0,
                        "mean_patch_error_rate": 0.0,
                    },
                    "ablate_c1": {
                        "episodes_path": str(alf_ablate),
                        "success_rate": 0.0,
                        "mean_goal_condition_rate": 0.4,
                        "mean_steps": 3.0,
                        "mean_total_tokens": 18.0,
                        "mean_worker_input_tokens": 6.0,
                        "mean_fallback_rate": 1.0,
                        "mean_patch_error_rate": 0.0,
                        "disabled_components": ["C1"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    web_full = tmp_path / "web_full.jsonl"
    _write_jsonl(
        web_full,
        [
            {
                "task_id": "web-1",
                "success": True,
                "steps": 3,
                "total_tokens": 20,
                "goal_condition_rate": 0.9,
                "worker_input_tokens": 8,
                "worker_output_tokens": 4,
                "architect_input_tokens": 2,
                "architect_output_tokens": 2,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "retrieval_precision": 0.9,
                "context_fragment_count": 5.0,
                "relevant_fragment_count": 4.0,
                "irrelevant_fragment_count": 1.0,
                "stop_reason": "reference_completed",
                "final_status": "success",
                "workflow_final_status": "completed",
                "metadata": {"sites": ["reddit"]},
            }
        ],
    )
    web_ablate = tmp_path / "web_ablate.jsonl"
    _write_jsonl(
        web_ablate,
        [
            {
                "task_id": "web-1",
                "success": False,
                "steps": 4,
                "total_tokens": 24,
                "goal_condition_rate": 0.5,
                "worker_input_tokens": 10,
                "worker_output_tokens": 4,
                "architect_input_tokens": 2,
                "architect_output_tokens": 2,
                "fallback_action_count": 1,
                "patch_error_count": 0,
                "retrieval_precision": 0.6,
                "context_fragment_count": 6.0,
                "relevant_fragment_count": 4.0,
                "irrelevant_fragment_count": 2.0,
                "stop_reason": "reference_completed",
                "final_status": "failure",
                "workflow_final_status": "completed",
                "metadata": {"sites": ["reddit"]},
            }
        ],
    )
    web_summary = tmp_path / "web_ablation_summary.json"
    web_summary.write_text(
        json.dumps(
            {
                "summary_type": "webarena_ablation",
                "modes": ["full", "ablate_c1"],
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "execution_backend": "reference_oracle",
                "summaries": {
                    "full": {
                        "mode_output_dir": str(tmp_path / "web_full_mode"),
                        "disabled_components": [],
                        "result": {
                            "summary": {
                                "success_rate": 0.8,
                                "mean_goal_condition_rate": 0.9,
                                "mean_steps": 3.0,
                                "mean_total_tokens": 20.0,
                                "mean_worker_input_tokens": 8.0,
                                "mean_fallback_rate": 0.0,
                                "mean_patch_error_rate": 0.0,
                                "mean_retrieval_precision": 0.9,
                                "mean_context_fragment_count": 5.0,
                                "mean_relevant_fragment_count": 4.0,
                                "mean_irrelevant_fragment_count": 1.0,
                            },
                            "episodes_path": str(web_full),
                        },
                    },
                    "ablate_c1": {
                        "mode_output_dir": str(tmp_path / "web_ablate_mode"),
                        "disabled_components": ["C1"],
                        "result": {
                            "summary": {
                                "success_rate": 0.2,
                                "mean_goal_condition_rate": 0.5,
                                "mean_steps": 4.0,
                                "mean_total_tokens": 24.0,
                                "mean_worker_input_tokens": 10.0,
                                "mean_fallback_rate": 1.0,
                                "mean_patch_error_rate": 0.0,
                                "mean_retrieval_precision": 0.6,
                                "mean_context_fragment_count": 6.0,
                                "mean_relevant_fragment_count": 4.0,
                                "mean_irrelevant_fragment_count": 2.0,
                            },
                            "episodes_path": str(web_ablate),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    science_full = tmp_path / "science_full.jsonl"
    _write_jsonl(
        science_full,
        [
            {
                "task_id": "1-1",
                "success": True,
                "steps": 4,
                "total_tokens": 14,
                "goal_condition_rate": 0.8,
                "worker_input_tokens": 5,
                "worker_output_tokens": 4,
                "architect_input_tokens": 1,
                "architect_output_tokens": 2,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "retrieval_precision": 0.7,
                "context_fragment_count": 4.0,
                "relevant_fragment_count": 3.0,
                "irrelevant_fragment_count": 1.0,
                "stop_reason": "reference_completed",
                "final_status": "success",
                "workflow_final_status": "completed",
                "metadata": {"task_name": "1-1"},
            }
        ],
    )
    science_ablate = tmp_path / "science_ablate.jsonl"
    _write_jsonl(
        science_ablate,
        [
            {
                "task_id": "1-1",
                "success": False,
                "steps": 5,
                "total_tokens": 19,
                "goal_condition_rate": 0.3,
                "worker_input_tokens": 7,
                "worker_output_tokens": 4,
                "architect_input_tokens": 1,
                "architect_output_tokens": 2,
                "fallback_action_count": 1,
                "patch_error_count": 0,
                "retrieval_precision": 0.4,
                "context_fragment_count": 5.0,
                "relevant_fragment_count": 3.0,
                "irrelevant_fragment_count": 2.0,
                "stop_reason": "reference_completed",
                "final_status": "failure",
                "workflow_final_status": "completed",
                "metadata": {"task_name": "1-1"},
            }
        ],
    )
    science_summary = tmp_path / "science_ablation_summary.json"
    science_summary.write_text(
        json.dumps(
            {
                "summary_type": "scienceworld_ablation",
                "modes": ["full", "ablate_c1"],
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "summaries": {
                    "full": {
                        "mode_output_dir": str(tmp_path / "science_full_mode"),
                        "disabled_components": [],
                        "result": {
                            "summary": {
                                "success_rate": 0.6,
                                "mean_goal_condition_rate": 0.8,
                                "mean_steps": 4.0,
                                "mean_total_tokens": 14.0,
                                "mean_worker_input_tokens": 5.0,
                                "mean_fallback_rate": 0.0,
                                "mean_patch_error_rate": 0.0,
                                "mean_retrieval_precision": 0.7,
                                "mean_context_fragment_count": 4.0,
                                "mean_relevant_fragment_count": 3.0,
                                "mean_irrelevant_fragment_count": 1.0,
                            },
                            "episodes_path": str(science_full),
                        },
                    },
                    "ablate_c1": {
                        "mode_output_dir": str(tmp_path / "science_ablate_mode"),
                        "disabled_components": ["C1"],
                        "result": {
                            "summary": {
                                "success_rate": 0.1,
                                "mean_goal_condition_rate": 0.3,
                                "mean_steps": 5.0,
                                "mean_total_tokens": 19.0,
                                "mean_worker_input_tokens": 7.0,
                                "mean_fallback_rate": 1.0,
                                "mean_patch_error_rate": 0.0,
                                "mean_retrieval_precision": 0.4,
                                "mean_context_fragment_count": 5.0,
                                "mean_relevant_fragment_count": 3.0,
                                "mean_irrelevant_fragment_count": 2.0,
                            },
                            "episodes_path": str(science_ablate),
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_cross_dataset_ablation(
        experiment_id="exp6",
        dataset_summaries={
            "alfworld": str(alf_summary),
            "webarena": str(web_summary),
            "scienceworld": str(science_summary),
        },
    )

    assert report["analysis_type"] == "cross_dataset_ablation"
    assert report["dataset_count"] == 3
    assert report["full_aggregate"]["metrics"]["success_rate"]["mean"] == pytest.approx(0.8)
    assert report["aggregate_by_mode"]["ablate_c1"]["dataset_count"] == 3
    assert report["aggregate_by_mode"]["ablate_c1"]["disabled_components"] == ["C1"]
    assert report["aggregate_by_mode"]["ablate_c1"]["full_minus_mode_metrics"]["success_rate"]["mean"] == pytest.approx(0.7)
    assert report["mode_order_by_mean_success_rate_delta"][0] == "ablate_c1"
