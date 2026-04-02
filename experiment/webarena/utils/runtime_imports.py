"""Helpers for importing repo-local WebArena blackboard dependencies."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_webarena_runtime_imports() -> None:
    """Add repo-local dependency roots required by WebArena runtime LLM workers."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        repo_root / "blackboard" / "libs" / "kernel_system",
        repo_root / "langgraph" / "libs" / "langgraph",
        repo_root / "blackboard" / "libs" / "langgraph",
    ]

    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
