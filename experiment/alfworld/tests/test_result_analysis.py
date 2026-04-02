"""Tests for ALFWorld result analysis helpers."""
from __future__ import annotations

import json

from experiment.alfworld.utils.result_analysis import (
    analyze_ablation_summary,
    analyze_system_compare_summary,
    analyze_workflow_compare_summary,
    build_task_type_summaries,
    summarize_step_diagnostics,
)


def test_build_task_type_summaries_groups_by_gamefile_family():
    results = [
        {
            "gamefile": "/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
            "success": True,
            "steps": 2,
            "total_tokens": 10,
            "goal_condition_rate": 1.0,
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
        },
        {
            "gamefile": "/tmp/pick_heat_then_place-Apple-None-Fridge-1/trial/game.tw-pddl",
            "success": False,
            "steps": 3,
            "total_tokens": 8,
            "goal_condition_rate": 0.0,
            "fallback_action_count": 1,
            "patch_error_count": 0,
            "no_update_event_count": 0,
            "repeat_action_count": 0,
            "repeat_observation_count": 0,
            "stagnation_event_count": 1,
            "circuit_breaker_triggered": True,
            "waiting_for_user": False,
            "stop_reason": "stagnation_detected",
            "final_status": "stagnation",
            "workflow_final_status": "completed",
        },
    ]

    summaries = build_task_type_summaries(results)

    assert summaries["pick_and_place"]["n_episodes"] == 1
    assert summaries["pick_and_place"]["success_rate"] == 1.0
    assert summaries["pick_heat_then_place"]["success_rate"] == 0.0


def test_analyze_ablation_summary_adds_task_breakdowns(tmp_path):
    episodes_path = tmp_path / "full_episodes.jsonl"
    episodes_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "gamefile": "/tmp/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
                        "success": True,
                        "steps": 2,
                        "total_tokens": 12,
                        "goal_condition_rate": 1.0,
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
                        "trajectory": [
                            {
                                "action": "take apple 1",
                                "planner_state": {
                                    "recommended_actions": ["take apple 1", "go to fridge 1"],
                                    "failed_search_locations": ["drawer 1"],
                                },
                                "canonical_task": {
                                    "canonical_goal_object": "Apple",
                                },
                            }
                        ],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "workflow_mode": "planner_llm_action",
                "n_gamefiles": 1,
                "summaries": {
                    "full": {
                        "episodes_path": str(episodes_path),
                        "success_rate": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_ablation_summary(str(summary_path))

    assert analysis["analysis_type"] == "ablation"
    assert analysis["modes"]["full"]["task_type_summaries"]["pick_and_place"]["success_rate"] == 1.0
    assert analysis["modes"]["full"]["step_diagnostics"]["recommended_action_top1_follow_rate"] == 1.0
    assert analysis["modes"]["full"]["step_diagnostics"]["canonical_goal_mention_rate"] == 1.0


def test_analyze_workflow_compare_summary_adds_task_breakdowns(tmp_path):
    episodes_path = tmp_path / "planner_full_episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "gamefile": "/tmp/pick_two_obj_and_place-CD-None-Drawer-304/trial/game.tw-pddl",
                "success": False,
                "steps": 4,
                "total_tokens": 0,
                "goal_condition_rate": 0.0,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "no_update_event_count": 0,
                "repeat_action_count": 0,
                "repeat_observation_count": 0,
                "stagnation_event_count": 1,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "max_steps_reached",
                "final_status": "incomplete",
                "workflow_final_status": "completed",
                "trajectory": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "workflow_compare_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "workflow_modes": ["planner_action"],
                "n_gamefiles": 1,
                "workflow_summaries": {
                    "planner_action": {
                        "workflow_output_dir": str(tmp_path / "planner_action"),
                        "result": {
                            "summaries": {
                                "full": {
                                    "episodes_path": str(episodes_path),
                                    "success_rate": 0.0,
                                }
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_workflow_compare_summary(str(summary_path))

    assert analysis["analysis_type"] == "workflow_compare"
    assert "pick_two_obj" in analysis["workflow_summaries"]["planner_action"]["modes"]["full"]["task_type_summaries"]


def test_analyze_system_compare_summary_adds_task_breakdowns(tmp_path):
    episodes_path = tmp_path / "autogen_episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "gamefile": "/tmp/pick_two_obj_and_place-CD-None-Drawer-304/trial/game.tw-pddl",
                "success": False,
                "steps": 4,
                "total_tokens": 10,
                "goal_condition_rate": 0.0,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "no_update_event_count": 0,
                "repeat_action_count": 0,
                "repeat_observation_count": 0,
                "stagnation_event_count": 1,
                "circuit_breaker_triggered": False,
                "waiting_for_user": False,
                "stop_reason": "max_steps_reached",
                "final_status": "incomplete",
                "workflow_final_status": "completed",
                "trajectory": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "system_compare_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "systems": ["autogen"],
                "n_gamefiles": 1,
                "workflow_mode": "planner_action",
                "system_summaries": {
                    "autogen": {
                        "system_output_dir": str(tmp_path / "autogen"),
                        "result": {
                            "summary": {
                                "success_rate": 0.0,
                                "mean_steps": 4.0,
                            },
                            "episodes_path": str(episodes_path),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_system_compare_summary(str(summary_path))

    assert analysis["analysis_type"] == "system_compare"
    assert "pick_two_obj" in analysis["system_summaries"]["autogen"]["task_type_summaries"]


def test_summarize_step_diagnostics_computes_follow_rates():
    results = [
        {
            "trajectory": [
                {
                    "action": "take apple 1",
                    "planner_state": {
                        "recommended_actions": ["take apple 1", "go to fridge 1"],
                        "failed_search_locations": ["drawer 1"],
                    },
                    "canonical_task": {"canonical_goal_object": "Apple"},
                },
                {
                    "action": "go to countertop 1",
                    "planner_state": {
                        "recommended_actions": ["go to sinkbasin 1", "go to countertop 1"],
                        "failed_search_locations": ["drawer 1", "drawer 2"],
                    },
                    "canonical_task": {"canonical_goal_object": "Ladle"},
                },
            ]
        }
    ]

    summary = summarize_step_diagnostics(results)

    assert summary["total_steps"] == 2.0
    assert summary["steps_with_recommendations"] == 2.0
    assert summary["recommended_action_top1_follow_rate"] == 0.5
    assert summary["recommended_action_any_follow_rate"] == 1.0
    assert summary["failed_search_exact_repeat_avoid_rate"] == 1.0
    assert summary["canonical_goal_mention_rate"] == 0.5
