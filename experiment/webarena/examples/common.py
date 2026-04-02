"""Shared helpers for WebArena example runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment.common.task_registry import preview_task_set
from experiment.webarena.systems import run_blackboard_task
from experiment.webarena.utils import load_task_by_id, load_task_dataset
from experiment.webarena.utils.result_analysis import summarize_episode_results


def resolve_manifest(task_set_file: str | Path, *, config_override: str = "") -> dict[str, Any]:
    manifest_info = preview_task_set(task_set_file, limit=1)
    manifest_path = Path(manifest_info["manifest_path"]).resolve()
    task_paths = list(manifest_info.get("task_paths", []) or [])
    if not task_paths:
        raise ValueError("Task manifest did not resolve any task file.")
    dataset_path = Path(task_paths[0]).resolve()
    metadata = dict(manifest_info.get("metadata", {}) or {})

    if config_override:
        config_path = Path(config_override).resolve()
    else:
        template = str(metadata.get("config_template", "") or "").strip()
        if not template:
            raise ValueError("Manifest metadata.config_template is required when --config is omitted.")
        config_path = (manifest_path.parent / template).resolve()

    return {
        "manifest_info": manifest_info,
        "manifest_path": manifest_path,
        "dataset_path": dataset_path,
        "metadata": metadata,
        "config_path": config_path,
    }


def select_task_ids(
    *,
    dataset_path: Path,
    metadata: dict[str, Any],
    task_id_override: str = "",
    limit: int = 0,
) -> list[int]:
    if task_id_override:
        return [int(task_id_override)]
    configured = [int(item) for item in list(metadata.get("task_ids", []) or [])]
    if configured:
        return configured[:limit] if limit > 0 else configured
    dataset = load_task_dataset(dataset_path)
    selected = [int(task.get("task_id")) for task in dataset if str(task.get("task_id", "")).strip()]
    return selected[:limit] if limit > 0 else selected


def evaluation_success(evaluation: dict[str, Any]) -> bool:
    if "success" in evaluation:
        return bool(evaluation.get("success"))
    score = evaluation.get("score")
    try:
        return float(score or 0.0) > 0.0
    except Exception:
        return False


def _mean(episodes: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for episode in episodes:
        try:
            values.append(float(episode.get(field, 0.0) or 0.0))
        except Exception:
            continue
    return float(sum(values) / len(values)) if values else 0.0


def _rate_from_count_field(episodes: list[dict[str, Any]], field: str) -> float:
    if not episodes:
        return 0.0
    rates = [
        float(episode.get(field, 0) or 0) / max(int(episode.get("steps", 0) or 0), 1)
        for episode in episodes
    ]
    return float(sum(rates) / len(rates)) if rates else 0.0


def _breakdown(episodes: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        key = str(episode.get(field, "") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _site_breakdown(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        sites = list((episode.get("metadata", {}) or {}).get("sites", []) or [])
        key = "+".join(str(site) for site in sites) if sites else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def episode_payload(
    result: Any,
    *,
    task_config: dict[str, Any],
    run_id: str,
    system_id: str,
    system_family: str,
) -> dict[str, Any]:
    evaluation = dict(result.evaluation or {})
    success = evaluation_success(evaluation)
    run_metadata = dict(result.run_metadata or {})
    verification = dict(run_metadata.get("verification") or {})
    goal_condition_rate = float(run_metadata.get("goal_condition_rate", 1.0 if success else 0.0) or 0.0)
    worker_input_tokens = int(run_metadata.get("worker_input_tokens", 0) or 0)
    worker_output_tokens = int(run_metadata.get("worker_output_tokens", 0) or 0)
    architect_input_tokens = int(run_metadata.get("architect_input_tokens", 0) or 0)
    architect_output_tokens = int(run_metadata.get("architect_output_tokens", 0) or 0)
    total_tokens = int(run_metadata.get("total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = worker_input_tokens + worker_output_tokens + architect_input_tokens + architect_output_tokens
    trajectory = list(run_metadata.get("trajectory", []) or [])
    task_id = str(result.task.task_id)
    return {
        "run_id": run_id,
        "system_id": system_id,
        "system_family": system_family,
        "episode_id": f"{system_id}:{task_id}",
        "task_id": task_id,
        "success": success,
        "finished": bool(result.finished),
        "steps": int(result.steps),
        "goal_condition_rate": goal_condition_rate,
        "worker_input_tokens": worker_input_tokens,
        "worker_output_tokens": worker_output_tokens,
        "architect_input_tokens": architect_input_tokens,
        "architect_output_tokens": architect_output_tokens,
        "total_tokens": total_tokens,
        "fallback_action_count": int(run_metadata.get("fallback_action_count", 0) or 0),
        "patch_error_count": int(run_metadata.get("patch_error_count", 0) or 0),
        "retrieval_precision": float(run_metadata.get("retrieval_precision", goal_condition_rate) or 0.0),
        "context_fragment_count": float(run_metadata.get("context_fragment_count", 0.0) or 0.0),
        "relevant_fragment_count": float(run_metadata.get("relevant_fragment_count", 0.0) or 0.0),
        "irrelevant_fragment_count": float(run_metadata.get("irrelevant_fragment_count", 0.0) or 0.0),
        "task_category": str(result.task.metadata.get("task_category", "") or ""),
        "sites": list(task_config.get("sites", []) or []),
        "intent": str(task_config.get("intent", "") or ""),
        "tool_call_count": len(result.tool_results),
        "assistant_message_count": sum(1 for message in result.messages if message.role == "assistant"),
        "finish_reason": str(run_metadata.get("finish_reason", "") or ""),
        "termination_reason": str(run_metadata.get("finish_reason", "") or ""),
        "trajectory": trajectory,
        "evaluation": evaluation,
        "communication_trace": list(run_metadata.get("communication_trace", []) or []),
        "run_metadata": run_metadata,
        "metadata": {
            "sites": list(task_config.get("sites", []) or []),
            "intent": str(task_config.get("intent", "") or ""),
            "task_category": str(result.task.metadata.get("task_category", "") or ""),
            "start_urls": list(task_config.get("start_urls", []) or []),
            "requires_final_response": bool(result.task.metadata.get("requires_final_response", False)),
            "verification": verification,
        },
    }


def summary_payload(
    episodes: list[dict[str, Any]],
    *,
    task_set: str,
    manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    total = len(episodes)
    success_count = sum(1 for episode in episodes if bool(episode.get("success", False)))
    finished_count = sum(1 for episode in episodes if bool(episode.get("finished", False)))
    metrics = summarize_episode_results(episodes)
    return {
        "env_name": "webarena",
        "task_set": task_set,
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "n_episodes": total,
        "success_count": success_count,
        "finished_count": finished_count,
        "success_rate": float(metrics.get("success_rate", (success_count / total) if total else 0.0)),
        "mean_goal_condition_rate": float(metrics.get("mean_goal_condition_rate", 0.0) or 0.0),
        "mean_steps": float(metrics.get("mean_steps", 0.0) or 0.0),
        "mean_total_tokens": float(metrics.get("mean_total_tokens", 0.0) or 0.0),
        "mean_worker_input_tokens": float(metrics.get("mean_worker_input_tokens", 0.0) or 0.0),
        "mean_worker_output_tokens": _mean(episodes, "worker_output_tokens"),
        "mean_architect_input_tokens": _mean(episodes, "architect_input_tokens"),
        "mean_architect_output_tokens": _mean(episodes, "architect_output_tokens"),
        "mean_architect_total_tokens": _mean(episodes, "architect_input_tokens") + _mean(episodes, "architect_output_tokens"),
        "mean_tool_call_count": _mean(episodes, "tool_call_count"),
        "mean_assistant_message_count": _mean(episodes, "assistant_message_count"),
        "mean_fallback_rate": float(metrics.get("mean_fallback_rate", _rate_from_count_field(episodes, "fallback_action_count")) or 0.0),
        "mean_patch_error_rate": float(metrics.get("mean_patch_error_rate", _rate_from_count_field(episodes, "patch_error_count")) or 0.0),
        "mean_retrieval_precision": float(metrics.get("mean_retrieval_precision", 0.0) or 0.0),
        "mean_context_fragment_count": float(metrics.get("mean_context_fragment_count", 0.0) or 0.0),
        "mean_relevant_fragment_count": float(metrics.get("mean_relevant_fragment_count", 0.0) or 0.0),
        "mean_irrelevant_fragment_count": float(metrics.get("mean_irrelevant_fragment_count", 0.0) or 0.0),
        "finish_reason_breakdown": _breakdown(episodes, "finish_reason"),
        "task_category_breakdown": _breakdown(episodes, "task_category"),
        "site_breakdown": _site_breakdown(episodes),
        "episodes": episodes,
    }


def run_blackboard_batch(
    *,
    dataset_path: Path,
    config_path: Path,
    task_ids: list[int],
    output_dir: Path,
    dataset_name: str,
    system_config: dict[str, Any] | None = None,
    headless: bool = True,
    slow_mo_ms: int = 0,
    max_steps: int = 4,
    run_id: str = "",
    system_id: str = "blackboard",
    system_family: str = "blackboard",
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    resolved_run_id = run_id or output_dir.name
    for task_id in task_ids:
        task_config = load_task_by_id(dataset_path, task_id)
        task_output_dir = output_dir / str(task_id)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_blackboard_task(
            task_id=int(task_id),
            config_path=config_path,
            output_dir=task_output_dir,
            dataset_name=dataset_name,
            system_config=dict(system_config or {}),
            headless=bool(headless),
            slow_mo_ms=int(slow_mo_ms),
            max_steps=int(max_steps),
        )
        episode = episode_payload(
            result,
            task_config=task_config,
            run_id=resolved_run_id,
            system_id=system_id,
            system_family=system_family,
        )
        episodes.append(episode)
        (task_output_dir / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "episode_results.jsonl").write_text(
        "\n".join(json.dumps(episode, ensure_ascii=False) for episode in episodes) + ("\n" if episodes else ""),
        encoding="utf-8",
    )
    return episodes
