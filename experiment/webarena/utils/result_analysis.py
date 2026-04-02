"""Helpers for post-processing WebArena experiment outputs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _episode_site_group(result: Mapping[str, Any]) -> str:
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        sites = metadata.get("sites")
        if isinstance(sites, list) and sites:
            return "+".join(str(site) for site in sites)
    sites = result.get("sites")
    if isinstance(sites, list) and sites:
        return "+".join(str(site) for site in sites)
    return "unknown"


def summarize_episode_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute stable aggregate metrics over WebArena episodes."""

    def _values(field: str) -> List[float]:
        return [value for result in results if (value := _safe_float(result.get(field))) is not None]

    def _rate_values(field: str) -> List[float]:
        rates: List[float] = []
        for result in results:
            value = _safe_float(result.get(field))
            if value is None:
                continue
            steps = max(int(_safe_float(result.get("steps")) or 0), 1)
            rates.append(float(value / steps))
        return rates

    success_values = [
        1.0 if bool(result.get("success", False)) else 0.0
        for result in results
    ]
    return {
        "success_rate": _mean(success_values),
        "mean_goal_condition_rate": _mean(_values("goal_condition_rate")),
        "mean_steps": _mean(_values("steps")),
        "mean_total_tokens": _mean(_values("total_tokens")),
        "mean_worker_input_tokens": _mean(_values("worker_input_tokens")),
        "mean_fallback_rate": _mean(_rate_values("fallback_action_count")),
        "mean_patch_error_rate": _mean(_rate_values("patch_error_count")),
        "mean_retrieval_precision": _mean(_values("retrieval_precision")),
        "mean_context_fragment_count": _mean(_values("context_fragment_count")),
        "mean_relevant_fragment_count": _mean(_values("relevant_fragment_count")),
        "mean_irrelevant_fragment_count": _mean(_values("irrelevant_fragment_count")),
    }


def build_site_summaries(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group episode results by involved WebArena site(s)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(_episode_site_group(result), []).append(result)
    return {
        site_group: summarize_episode_results(site_results)
        for site_group, site_results in grouped.items()
    }


def compute_robustness(site_summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Compute cross-site variability metrics."""
    success_rates = [_safe_float(summary.get("success_rate")) or 0.0 for summary in site_summaries.values()]
    goal_rates = [_safe_float(summary.get("mean_goal_condition_rate")) or 0.0 for summary in site_summaries.values()]
    step_means = [_safe_float(summary.get("mean_steps")) or 0.0 for summary in site_summaries.values()]
    return {
        "site_group_count": float(len(site_summaries)),
        "success_rate_std": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
        "goal_condition_rate_std": statistics.stdev(goal_rates) if len(goal_rates) > 1 else 0.0,
        "mean_steps_std": statistics.stdev(step_means) if len(step_means) > 1 else 0.0,
    }


def _mode_result_payload(mode_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    result_payload = mode_summary.get("result")
    if isinstance(result_payload, dict):
        return result_payload
    return mode_summary


def _mode_overall_summary(mode_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mode_result_payload(mode_summary)
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def analyze_ablation_summary(summary_path: str) -> Dict[str, Any]:
    """Attach per-site summaries and robustness metrics to a WebArena ablation run."""
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    mode_summaries: Dict[str, Any] = {}
    for mode, raw_mode_summary in (summary.get("summaries") or {}).items():
        payload = _mode_result_payload(raw_mode_summary)
        episodes_path = str(payload.get("episodes_path", "") or "")
        episodes = load_jsonl(episodes_path) if episodes_path else []
        site_summaries = build_site_summaries(episodes)
        mode_summaries[mode] = {
            "overall": dict(_mode_overall_summary(raw_mode_summary)),
            "site_summaries": site_summaries,
            "robustness": compute_robustness(site_summaries),
            "disabled_components": list(raw_mode_summary.get("disabled_components") or []),
            "mode_output_dir": str(raw_mode_summary.get("mode_output_dir", "") or ""),
        }

    return {
        "analysis_type": "ablation",
        "summary_path": str(path),
        "summary_type": str(summary.get("summary_type", "") or ""),
        "workflow_mode": str(summary.get("workflow_mode", "") or ""),
        "execution_backend": str(summary.get("execution_backend", "") or ""),
        "n_tasks": int(summary.get("n_tasks", 0) or 0),
        "mode_summaries": mode_summaries,
        "modes": mode_summaries,
    }


def analyze_system_compare_summary(summary_path: str) -> Dict[str, Any]:
    """Attach per-site summaries and robustness metrics to a WebArena system-compare run."""
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    system_summaries: Dict[str, Any] = {}
    for system, raw_system_summary in (summary.get("system_summaries") or {}).items():
        payload = raw_system_summary.get("result") or {}
        episodes_path = str(payload.get("episodes_path") or payload.get("standard_episodes_path") or "")
        episodes = load_jsonl(episodes_path) if episodes_path else []
        site_summaries = build_site_summaries(episodes)
        system_summaries[system] = {
            "system_output_dir": str(raw_system_summary.get("system_output_dir", "") or ""),
            "overall": dict(payload.get("summary") or {}),
            "site_summaries": site_summaries,
            "robustness": compute_robustness(site_summaries),
        }

    return {
        "analysis_type": "system_compare",
        "summary_path": str(path),
        "summary_type": str(summary.get("summary_type", "") or ""),
        "systems": list(summary.get("systems") or []),
        "workflow_mode": str(summary.get("workflow_mode", "") or ""),
        "execution_backend": str(summary.get("execution_backend", "") or ""),
        "n_tasks": int(summary.get("n_tasks", 0) or 0),
        "system_summaries": system_summaries,
    }
