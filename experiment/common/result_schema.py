"""Shared result-schema helpers for experiment runners."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = "experiment_result.v1"

STANDARD_EPISODES_FILE = "episode_results.jsonl"
STANDARD_STATES_FILE = "captured_states.jsonl"
STANDARD_SUMMARY_FILE = "summary.json"
STANDARD_CONFIG_SNAPSHOT_FILE = "config_snapshot.json"


def _json_safe(value: Any) -> Any:
    """Convert common runtime objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_artifact_paths(output_dir: str | Path, run_id: str) -> Dict[str, str]:
    """Return both legacy and canonical artifact paths for one run."""
    output_path = Path(output_dir)
    return {
        "legacy_episodes_path": str(output_path / f"{run_id}_episodes.jsonl"),
        "legacy_states_path": str(output_path / f"{run_id}_states.jsonl"),
        "legacy_summary_path": str(output_path / f"{run_id}_summary.json"),
        "episodes_path": str(output_path / STANDARD_EPISODES_FILE),
        "states_path": str(output_path / STANDARD_STATES_FILE),
        "summary_path": str(output_path / STANDARD_SUMMARY_FILE),
        "config_snapshot_path": str(output_path / STANDARD_CONFIG_SNAPSHOT_FILE),
    }


def normalize_episode_result(
    result: Dict[str, Any],
    *,
    run_id: str,
    system_id: str,
    system_family: str,
    env_name: str,
) -> Dict[str, Any]:
    """Attach stable cross-system metadata to one episode result."""
    normalized = dict(result)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("record_type", "episode_result")
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("system_id", system_id)
    normalized.setdefault("system_family", system_family)
    normalized.setdefault("env_name", env_name)
    normalized.setdefault("task_id", str(normalized.get("gamefile", "")))
    normalized.setdefault("metadata", {})
    return _json_safe(normalized)


def normalize_captured_state(
    state: Dict[str, Any],
    *,
    run_id: str,
    system_id: str,
    system_family: str,
    env_name: str,
) -> Dict[str, Any]:
    """Attach stable cross-system metadata to one captured runtime state."""
    normalized = dict(state)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("record_type", "captured_state")
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("system_id", system_id)
    normalized.setdefault("system_family", system_family)
    normalized.setdefault("env_name", env_name)
    normalized.setdefault("task_id", str(normalized.get("gamefile", "")))
    return _json_safe(normalized)


def normalize_summary(
    summary: Dict[str, Any],
    *,
    run_id: str,
    system_id: str,
    system_family: str,
    env_name: str,
) -> Dict[str, Any]:
    """Attach stable cross-system metadata to one run summary."""
    normalized = dict(summary)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("record_type", "summary")
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("system_id", system_id)
    normalized.setdefault("system_family", system_family)
    normalized.setdefault("env_name", env_name)
    normalized.setdefault("metadata", {})
    return _json_safe(normalized)


def build_config_snapshot(
    *,
    run_id: str,
    system_id: str,
    system_family: str,
    env_name: str,
    config: Dict[str, Any],
    artifact_paths: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a canonical config snapshot payload for reproducibility."""
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "config_snapshot",
        "run_id": run_id,
        "system_id": system_id,
        "system_family": system_family,
        "env_name": env_name,
        "config": _json_safe(config),
        "artifacts": _json_safe(artifact_paths),
    }
