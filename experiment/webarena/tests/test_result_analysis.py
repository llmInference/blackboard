from __future__ import annotations

import json

from experiment.webarena.utils.result_analysis import (
    analyze_ablation_summary,
    analyze_system_compare_summary,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_ablation_summary_reads_mode_payloads(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes,
        [
            {
                "task_id": "web-1",
                "success": True,
                "steps": 3,
                "total_tokens": 20,
                "goal_condition_rate": 0.8,
                "worker_input_tokens": 8,
                "fallback_action_count": 0,
                "patch_error_count": 0,
                "retrieval_precision": 0.9,
                "context_fragment_count": 5.0,
                "relevant_fragment_count": 4.0,
                "irrelevant_fragment_count": 1.0,
                "metadata": {"sites": ["reddit"]},
            }
        ],
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary_type": "webarena_ablation",
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "execution_backend": "reference_oracle",
                "summaries": {
                    "full": {
                        "mode_output_dir": str(tmp_path / "full"),
                        "disabled_components": [],
                        "result": {
                            "summary": {
                                "success_rate": 0.8,
                                "mean_goal_condition_rate": 0.8,
                            },
                            "episodes_path": str(episodes),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_ablation_summary(str(summary_path))

    assert analysis["analysis_type"] == "ablation"
    assert analysis["execution_backend"] == "reference_oracle"
    assert analysis["mode_summaries"]["full"]["overall"]["success_rate"] == 0.8
    assert analysis["mode_summaries"]["full"]["site_summaries"]["reddit"]["success_rate"] == 1.0


def test_analyze_system_compare_summary_reads_system_payloads(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes,
        [
            {
                "task_id": "web-1",
                "success": False,
                "steps": 4,
                "total_tokens": 24,
                "goal_condition_rate": 0.5,
                "worker_input_tokens": 10,
                "fallback_action_count": 1,
                "patch_error_count": 0,
                "metadata": {"sites": ["gitlab"]},
            }
        ],
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "systems": ["blackboard"],
                "n_tasks": 1,
                "workflow_mode": "planner_action",
                "execution_backend": "reference_oracle",
                "system_summaries": {
                    "blackboard": {
                        "system_output_dir": str(tmp_path / "blackboard"),
                        "result": {
                            "summary": {
                                "success_rate": 0.25,
                                "mean_goal_condition_rate": 0.5,
                            },
                            "episodes_path": str(episodes),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_system_compare_summary(str(summary_path))

    assert analysis["analysis_type"] == "system_compare"
    assert analysis["system_summaries"]["blackboard"]["overall"]["success_rate"] == 0.25
    assert analysis["system_summaries"]["blackboard"]["site_summaries"]["gitlab"]["mean_steps"] == 4.0
