"""Helpers for loading WebArena task configs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class WebArenaTask:
    """Normalized view of one WebArena task."""

    config_path: str
    task_id: str
    sites: tuple[str, ...]
    intent: str
    start_urls: tuple[str, ...]
    n_evaluators: int
    requires_text_response: bool


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_sites(sites: Iterable[Any]) -> List[str]:
    normalized = [str(site) for site in sites if str(site).strip()]
    return sorted(dict.fromkeys(normalized))


def _dataset_summary(tasks: List[Dict[str, Any]], *, config_path: Path) -> Dict[str, Any]:
    all_sites: List[str] = []
    for task in tasks:
        all_sites.extend(str(site) for site in list(task.get("sites", []) or []))
    return {
        "config_path": str(config_path),
        "task_id": config_path.stem,
        "sites": _normalize_sites(all_sites),
        "n_tasks": len(tasks),
        "dataset": True,
    }


def load_task_dataset(config_path: str | Path) -> List[Dict[str, Any]]:
    """Load one or more raw WebArena task JSON objects from disk."""

    resolved_path = Path(config_path).resolve()
    payload = _load_json(resolved_path)
    if isinstance(payload, dict):
        return [dict(payload)]
    if isinstance(payload, list):
        tasks = [dict(item) for item in payload if isinstance(item, dict)]
        if not tasks:
            raise ValueError(f"No task records found in WebArena dataset: {resolved_path}")
        return tasks
    raise ValueError(f"Unsupported WebArena task config format: {resolved_path}")


def load_task_config(config_path: str | Path) -> Dict[str, Any]:
    """Load one raw WebArena task JSON object.

    If the file stores a dataset-level list of tasks, return a synthetic summary object
    with aggregated `sites` so callers such as task grouping can still operate.
    """

    resolved_path = Path(config_path).resolve()
    payload = _load_json(resolved_path)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            return dict(payload[0])
        tasks = [item for item in payload if isinstance(item, dict)]
        return _dataset_summary(tasks, config_path=resolved_path)
    raise ValueError(f"Unsupported WebArena task config format: {resolved_path}")


def load_task_by_id(config_path: str | Path, task_id: str | int) -> Dict[str, Any]:
    """Load exactly one task from a single-task file or dataset-level JSON file."""

    requested = str(task_id)
    for task in load_task_dataset(config_path):
        if str(task.get("task_id", "")) == requested:
            return task
    raise KeyError(f"Task id {requested!r} not found in {Path(config_path).resolve()}")


def parse_task(config_path: str | Path) -> WebArenaTask:
    """Parse one WebArena task config into a stable record."""

    resolved_path = Path(config_path).resolve()
    config = load_task_config(resolved_path)
    evaluators = list(config.get("eval", []) or [])
    expected_agent_response = any(
        str(item.get("evaluator", "") or "") == "AgentResponseEvaluator" for item in evaluators if isinstance(item, dict)
    )
    return WebArenaTask(
        config_path=str(resolved_path),
        task_id=str(config.get("task_id", resolved_path.stem)),
        sites=tuple(_normalize_sites(config.get("sites", []) or [])),
        intent=str(config.get("intent", "") or ""),
        start_urls=tuple(str(url) for url in list(config.get("start_urls", []) or [])),
        n_evaluators=len(evaluators),
        requires_text_response=expected_agent_response,
    )
