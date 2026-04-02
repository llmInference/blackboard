"""Dataset sampler for ALFWorld experiments."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List


# ALFWorld task type directory prefixes
_TASK_TYPES = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj",
    "pick_two_obj",
]


def task_type_from_gamefile(gamefile: str) -> str:
    """Return the canonical ALFWorld task type for a gamefile path."""
    for task_type in _TASK_TYPES:
        if task_type in gamefile:
            return task_type
    return "unknown"


class DatasetSampler:
    """
    Samples fixed, reproducible subsets of ALFWorld gamefiles.

    Scans data_root for game.tw-pddl files grouped by task type.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def _collect_by_type(self) -> dict[str, List[str]]:
        """Return {task_type: [gamefile_path, ...]} for all task types found."""
        by_type: dict[str, List[str]] = {t: [] for t in _TASK_TYPES}
        for gamefile in sorted(self.data_root.rglob("game.tw-pddl")):
            task_type = task_type_from_gamefile(str(gamefile))
            if task_type in by_type:
                by_type[task_type].append(str(gamefile))
        return by_type

    def sample(self, n_per_type: int, seed: int = 42) -> List[str]:
        """Return up to n_per_type gamefiles per task type, shuffled with seed."""
        rng = random.Random(seed)
        by_type = self._collect_by_type()
        result = []
        for task_type in _TASK_TYPES:
            files = by_type[task_type]
            rng.shuffle(files)
            result.extend(files[:n_per_type])
        return result

    def debug_split(self, seed: int = 42) -> List[str]:
        """12 tasks: 2 per task type."""
        return self.sample(n_per_type=2, seed=seed)

    def formal_split(self, seed: int = 42) -> List[str]:
        """60 tasks: 10 per task type."""
        return self.sample(n_per_type=10, seed=seed)
