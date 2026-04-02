"""Abstract base evaluator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
    """Base class for experiment evaluators."""

    @abstractmethod
    def run_episode(self, gamefile: str, **kwargs) -> Dict[str, Any]:
        """Run a single episode and return the result dict."""
        ...

    def run_batch(self, gamefiles: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run episodes for all gamefiles and return list of results."""
        return [self.run_episode(gf, **kwargs) for gf in gamefiles]

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate a list of episode results into summary metrics."""
        from .metrics import compute_summary
        return compute_summary(results)
