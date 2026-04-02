"""Generic result writer shared across experiment environments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from experiment.common.result_schema import (
    STANDARD_CONFIG_SNAPSHOT_FILE,
    STANDARD_EPISODES_FILE,
    STANDARD_STATES_FILE,
    STANDARD_SUMMARY_FILE,
    build_artifact_paths,
    build_config_snapshot,
    normalize_captured_state,
    normalize_episode_result,
    normalize_summary,
)


class ResultWriter:
    """Write episode, state, summary, and config artifacts with a shared schema."""

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        system_id: str | None = None,
        system_family: str = "blackboard",
        env_name: str = "unknown",
    ):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.system_id = system_id or run_id
        self.system_family = system_family
        self.env_name = env_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self.output_dir / f"{run_id}_episodes.jsonl"
        self._states_path = self.output_dir / f"{run_id}_states.jsonl"
        self._summary_path = self.output_dir / f"{run_id}_summary.json"
        self._standard_episodes_path = self.output_dir / STANDARD_EPISODES_FILE
        self._standard_states_path = self.output_dir / STANDARD_STATES_FILE
        self._standard_summary_path = self.output_dir / STANDARD_SUMMARY_FILE
        self._config_snapshot_path = self.output_dir / STANDARD_CONFIG_SNAPSHOT_FILE

    def write_episode(self, result: Dict[str, Any]) -> None:
        """Append one episode result as JSONL."""
        episode_record = dict(result)
        captured_states = episode_record.pop("captured_states", [])
        normalized_episode = normalize_episode_result(
            episode_record,
            run_id=self.run_id,
            system_id=self.system_id,
            system_family=self.system_family,
            env_name=self.env_name,
        )
        if captured_states:
            self.write_states(captured_states)
        for path in (self._episodes_path, self._standard_episodes_path):
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized_episode, ensure_ascii=False) + "\n")

    def write_states(self, captured_states: list[Dict[str, Any]]) -> None:
        """Append captured runtime states as JSONL."""
        normalized_states = [
            normalize_captured_state(
                state,
                run_id=self.run_id,
                system_id=self.system_id,
                system_family=self.system_family,
                env_name=self.env_name,
            )
            for state in captured_states
        ]
        for path in (self._states_path, self._standard_states_path):
            with open(path, "a", encoding="utf-8") as handle:
                for state in normalized_states:
                    handle.write(json.dumps(state, ensure_ascii=False) + "\n")

    def write_summary(self, metrics: Dict[str, Any]) -> None:
        """Write run summary JSON."""
        normalized_summary = normalize_summary(
            metrics,
            run_id=self.run_id,
            system_id=self.system_id,
            system_family=self.system_family,
            env_name=self.env_name,
        )
        for path in (self._summary_path, self._standard_summary_path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(normalized_summary, handle, ensure_ascii=False, indent=2)

    def write_config_snapshot(self, config: Dict[str, Any]) -> None:
        """Write canonical config snapshot."""
        snapshot = build_config_snapshot(
            run_id=self.run_id,
            system_id=self.system_id,
            system_family=self.system_family,
            env_name=self.env_name,
            config=config,
            artifact_paths=build_artifact_paths(self.output_dir, self.run_id),
        )
        with open(self._config_snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    @property
    def episodes_path(self) -> Path:
        return self._episodes_path

    @property
    def standard_episodes_path(self) -> Path:
        return self._standard_episodes_path

    @property
    def states_path(self) -> Path:
        return self._states_path

    @property
    def standard_states_path(self) -> Path:
        return self._standard_states_path

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    @property
    def standard_summary_path(self) -> Path:
        return self._standard_summary_path

    @property
    def config_snapshot_path(self) -> Path:
        return self._config_snapshot_path
