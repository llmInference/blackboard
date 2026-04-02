"""Backward-compatible ALFWorld wrapper for the shared ResultWriter."""
from __future__ import annotations

from experiment.common.result_writer import ResultWriter as _CommonResultWriter


class ResultWriter(_CommonResultWriter):
    """Keep the historic ALFWorld default env_name for existing callers/tests."""

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        system_id: str | None = None,
        system_family: str = "blackboard",
        env_name: str = "alfworld",
    ):
        super().__init__(
            output_dir,
            run_id,
            system_id=system_id,
            system_family=system_family,
            env_name=env_name,
        )


__all__ = ["ResultWriter"]
