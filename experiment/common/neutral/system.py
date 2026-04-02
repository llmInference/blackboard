"""System interfaces for the shared neutral protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from experiment.common.neutral.types import TaskSpec, ToolSpec, TurnInput, TurnOutput


class MultiAgentSession(ABC):
    """Stateful session created for one benchmark task."""

    @abstractmethod
    def step(self, turn_input: TurnInput) -> TurnOutput:
        """Advance the system by one neutral turn."""
        raise NotImplementedError

    def close(self) -> None:
        """Release session resources when a run completes."""
        return None


class MultiAgentSystem(ABC):
    """Factory for task-scoped multi-agent sessions."""

    @abstractmethod
    def create_session(
        self,
        task: TaskSpec,
        tools: tuple[ToolSpec, ...] = (),
        native_tools: tuple[Any, ...] = (),
    ) -> MultiAgentSession:
        """Create a new session for one task."""
        raise NotImplementedError

    def close_session(self, session: MultiAgentSession) -> None:
        """Close a previously created session."""
        session.close()
