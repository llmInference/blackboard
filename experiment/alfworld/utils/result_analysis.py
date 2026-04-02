"""Helpers for post-processing ALFWorld experiment outputs."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .dataset_sampler import task_type_from_gamefile
from ..evaluators.metrics import compute_summary


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    results: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def build_task_type_summaries(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group episode results by ALFWorld task type and summarize each group."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        task_type = task_type_from_gamefile(str(result.get("gamefile", "")))
        grouped.setdefault(task_type, []).append(result)

    return {
        task_type: compute_summary(task_results)
        for task_type, task_results in grouped.items()
    }


def _normalize_text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def summarize_step_diagnostics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate optional per-step diagnostics embedded in episode trajectories."""
    total_steps = 0
    steps_with_recommendations = 0
    recommended_top1_follow = 0
    recommended_any_follow = 0
    steps_with_architect_decision = 0
    steps_with_failed_search_context = 0
    failed_search_exact_repeat_avoided = 0
    steps_with_canonical_goal = 0
    canonical_goal_mention_steps = 0

    for result in results:
        for step in result.get("trajectory", []):
            total_steps += 1
            planner_state = step.get("planner_state", {}) or {}
            canonical_task = step.get("canonical_task", {}) or {}
            action = str(step.get("action", "") or "")

            if str(planner_state.get("architect_decision", "") or "").strip():
                steps_with_architect_decision += 1

            recommended_actions = list(planner_state.get("recommended_actions", []) or [])
            if recommended_actions:
                steps_with_recommendations += 1
                if action == recommended_actions[0]:
                    recommended_top1_follow += 1
                if action in recommended_actions:
                    recommended_any_follow += 1

            failed_search_locations = set(planner_state.get("failed_search_locations", []) or [])
            if failed_search_locations:
                steps_with_failed_search_context += 1
                if action not in failed_search_locations:
                    failed_search_exact_repeat_avoided += 1

            canonical_goal = str(canonical_task.get("canonical_goal_object", "") or "")
            if canonical_goal:
                steps_with_canonical_goal += 1
                if _normalize_text_tokens(canonical_goal) & _normalize_text_tokens(action):
                    canonical_goal_mention_steps += 1

    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "total_steps": float(total_steps),
        "steps_with_recommendations": float(steps_with_recommendations),
        "steps_with_architect_decision": float(steps_with_architect_decision),
        "recommended_action_top1_follow_rate": _rate(recommended_top1_follow, steps_with_recommendations),
        "recommended_action_any_follow_rate": _rate(recommended_any_follow, steps_with_recommendations),
        "steps_with_failed_search_context": float(steps_with_failed_search_context),
        "failed_search_exact_repeat_avoid_rate": _rate(
            failed_search_exact_repeat_avoided, steps_with_failed_search_context
        ),
        "steps_with_canonical_goal": float(steps_with_canonical_goal),
        "canonical_goal_mention_rate": _rate(canonical_goal_mention_steps, steps_with_canonical_goal),
    }


def compute_robustness(task_type_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Compute cross-task-family variability metrics."""
    success_rates = [summary["success_rate"] for summary in task_type_summaries.values()]
    goal_rates = [summary["mean_goal_condition_rate"] for summary in task_type_summaries.values()]
    step_means = [summary["mean_steps"] for summary in task_type_summaries.values()]

    return {
        "task_type_count": float(len(task_type_summaries)),
        "success_rate_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
        "goal_condition_rate_std": statistics.stdev(goal_rates) if len(goal_rates) > 1 else 0.0,
        "mean_steps_std": statistics.stdev(step_means) if len(step_means) > 1 else 0.0,
    }


def analyze_ablation_summary(summary_path: str) -> Dict[str, Any]:
    """Attach per-task-type summaries and robustness metrics to an ablation run."""
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    analyzed_modes: Dict[str, Any] = {}
    for mode, mode_summary in summary.get("summaries", {}).items():
        episodes = load_jsonl(mode_summary["episodes_path"])
        task_type_summaries = build_task_type_summaries(episodes)
        analyzed_modes[mode] = {
            "overall": mode_summary,
            "task_type_summaries": task_type_summaries,
            "robustness": compute_robustness(task_type_summaries),
            "step_diagnostics": summarize_step_diagnostics(episodes),
        }

    return {
        "analysis_type": "ablation",
        "summary_path": str(path),
        "workflow_mode": summary.get("workflow_mode", ""),
        "architect_mode": summary.get("architect_mode", ""),
        "n_gamefiles": summary.get("n_gamefiles", 0),
        "modes": analyzed_modes,
    }


def analyze_workflow_compare_summary(summary_path: str) -> Dict[str, Any]:
    """Attach per-task-type summaries and robustness metrics to a workflow compare run."""
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    analyzed_workflows: Dict[str, Any] = {}
    for workflow_mode, workflow_summary in summary.get("workflow_summaries", {}).items():
        mode_analyses: Dict[str, Any] = {}
        result_payload = workflow_summary.get("result", {})
        for mode, mode_summary in result_payload.get("summaries", {}).items():
            episodes = load_jsonl(mode_summary["episodes_path"])
            task_type_summaries = build_task_type_summaries(episodes)
            mode_analyses[mode] = {
                "overall": mode_summary,
                "task_type_summaries": task_type_summaries,
                "robustness": compute_robustness(task_type_summaries),
                "step_diagnostics": summarize_step_diagnostics(episodes),
            }

        analyzed_workflows[workflow_mode] = {
            "workflow_output_dir": workflow_summary.get("workflow_output_dir", ""),
            "modes": mode_analyses,
        }

    return {
        "analysis_type": "workflow_compare",
        "summary_path": str(path),
        "workflow_modes": summary.get("workflow_modes", []),
        "architect_mode": summary.get("architect_mode", ""),
        "n_gamefiles": summary.get("n_gamefiles", 0),
        "workflow_summaries": analyzed_workflows,
    }


def analyze_system_compare_summary(summary_path: str) -> Dict[str, Any]:
    """Attach per-task-type summaries and robustness metrics to a system compare run."""
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    analyzed_systems: Dict[str, Any] = {}
    for system, system_summary in summary.get("system_summaries", {}).items():
        result_payload = system_summary.get("result", {})
        episodes_path = result_payload.get("episodes_path") or result_payload.get("standard_episodes_path", "")
        episodes = load_jsonl(episodes_path) if episodes_path else []
        task_type_summaries = build_task_type_summaries(episodes)
        overall = result_payload.get("summary", {})
        analyzed_systems[system] = {
            "system_output_dir": system_summary.get("system_output_dir", ""),
            "overall": overall,
            "task_type_summaries": task_type_summaries,
            "robustness": compute_robustness(task_type_summaries),
            "step_diagnostics": summarize_step_diagnostics(episodes),
        }

    return {
        "analysis_type": "system_compare",
        "summary_path": str(path),
        "systems": summary.get("systems", []),
        "workflow_mode": summary.get("workflow_mode", ""),
        "architect_mode": summary.get("architect_mode", ""),
        "n_gamefiles": summary.get("n_gamefiles", 0),
        "system_summaries": analyzed_systems,
    }
