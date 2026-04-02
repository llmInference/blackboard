"""Metrics computation for ALFWorld experiments."""
from __future__ import annotations

from typing import Any, Dict, List


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate episode results into summary metrics.

    Args:
        results: List of episode result dicts from run_episode()

    Returns:
        Dict with keys:
            success_rate, mean_steps, mean_total_tokens,
            mean_worker_input_tokens, mean_worker_output_tokens,
            mean_architect_input_tokens, mean_architect_output_tokens,
            mean_architect_total_tokens,
            mean_fallback_rate, mean_patch_error_rate,
            mean_goal_condition_rate, mean_no_update_rate,
            mean_repeat_action_rate, mean_repeat_observation_rate,
            mean_stagnation_rate,
            circuit_breaker_trigger_rate, waiting_for_user_rate,
            stop_reason_breakdown, final_status_breakdown,
            workflow_final_status_breakdown, n_episodes
    """
    n = len(results)
    if n == 0:
        return {
            "n_episodes": 0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "mean_total_tokens": 0.0,
            "mean_worker_input_tokens": 0.0,
            "mean_worker_output_tokens": 0.0,
            "mean_architect_input_tokens": 0.0,
            "mean_architect_output_tokens": 0.0,
            "mean_architect_total_tokens": 0.0,
            "mean_goal_condition_rate": 0.0,
            "mean_fallback_rate": 0.0,
            "mean_patch_error_rate": 0.0,
            "mean_no_update_rate": 0.0,
            "mean_repeat_action_rate": 0.0,
            "mean_repeat_observation_rate": 0.0,
            "mean_stagnation_rate": 0.0,
            "circuit_breaker_trigger_rate": 0.0,
            "waiting_for_user_rate": 0.0,
            "stop_reason_breakdown": {},
            "final_status_breakdown": {},
            "workflow_final_status_breakdown": {},
        }

    success_rate = sum(1 for r in results if r.get("success")) / n
    mean_steps = sum(r.get("steps", 0) for r in results) / n
    mean_total_tokens = sum(r.get("total_tokens", 0) for r in results) / n
    mean_worker_input_tokens = sum(r.get("worker_input_tokens", 0) for r in results) / n
    mean_worker_output_tokens = sum(r.get("worker_output_tokens", 0) for r in results) / n
    mean_architect_input_tokens = sum(r.get("architect_input_tokens", 0) for r in results) / n
    mean_architect_output_tokens = sum(r.get("architect_output_tokens", 0) for r in results) / n
    mean_goal_condition_rate = sum(r.get("goal_condition_rate", 0.0) for r in results) / n

    fallback_rates = [
        r.get("fallback_action_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]
    patch_error_rates = [
        r.get("patch_error_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]
    no_update_rates = [
        r.get("no_update_event_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]
    repeat_action_rates = [
        r.get("repeat_action_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]
    repeat_observation_rates = [
        r.get("repeat_observation_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]
    stagnation_rates = [
        r.get("stagnation_event_count", 0) / max(r.get("steps", 1), 1)
        for r in results
    ]

    circuit_breaker_trigger_rate = sum(1 for r in results if r.get("circuit_breaker_triggered")) / n
    waiting_for_user_rate = sum(1 for r in results if r.get("waiting_for_user")) / n

    stop_reason_breakdown: Dict[str, int] = {}
    final_status_breakdown: Dict[str, int] = {}
    workflow_final_status_breakdown: Dict[str, int] = {}
    for result in results:
        stop_reason = result.get("stop_reason", "")
        if stop_reason:
            stop_reason_breakdown[stop_reason] = stop_reason_breakdown.get(stop_reason, 0) + 1
        final_status = result.get("final_status", "")
        if final_status:
            final_status_breakdown[final_status] = final_status_breakdown.get(final_status, 0) + 1
        workflow_final_status = result.get("workflow_final_status", "")
        if workflow_final_status:
            workflow_final_status_breakdown[workflow_final_status] = (
                workflow_final_status_breakdown.get(workflow_final_status, 0) + 1
            )

    return {
        "n_episodes": n,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_total_tokens": mean_total_tokens,
        "mean_worker_input_tokens": mean_worker_input_tokens,
        "mean_worker_output_tokens": mean_worker_output_tokens,
        "mean_architect_input_tokens": mean_architect_input_tokens,
        "mean_architect_output_tokens": mean_architect_output_tokens,
        "mean_architect_total_tokens": mean_architect_input_tokens + mean_architect_output_tokens,
        "mean_goal_condition_rate": mean_goal_condition_rate,
        "mean_fallback_rate": sum(fallback_rates) / n,
        "mean_patch_error_rate": sum(patch_error_rates) / n,
        "mean_no_update_rate": sum(no_update_rates) / n,
        "mean_repeat_action_rate": sum(repeat_action_rates) / n,
        "mean_repeat_observation_rate": sum(repeat_observation_rates) / n,
        "mean_stagnation_rate": sum(stagnation_rates) / n,
        "circuit_breaker_trigger_rate": circuit_breaker_trigger_rate,
        "waiting_for_user_rate": waiting_for_user_rate,
        "stop_reason_breakdown": stop_reason_breakdown,
        "final_status_breakdown": final_status_breakdown,
        "workflow_final_status_breakdown": workflow_final_status_breakdown,
    }
