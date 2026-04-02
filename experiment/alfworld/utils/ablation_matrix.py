"""Mode definitions for ALFWorld ablation experiments."""
from __future__ import annotations

from typing import Iterable, List

from langgraph_kernel.ablation import AblationConfig

DEFAULT_RUNTIME_MODES = ["full", "ablate_c2", "ablate_c3"]
ALL_MODES = [
    "full",
    "ablate_c1",
    "ablate_c2",
    "ablate_c3",
    "ablate_c4",
    "ablate_c5",
    "ablate_c6",
]

_MODE_COMPONENTS = {
    "full": (),
    "ablate_c1": ("C1",),
    "ablate_c2": ("C2",),
    "ablate_c3": ("C3",),
    "ablate_c4": ("C4",),
    "ablate_c5": ("C5",),
    "ablate_c6": ("C1", "C2"),  # 自然语言通信 + 无 schema 验证
}


def resolve_ablation_modes(modes: Iterable[str] | None = None) -> List[str]:
    """Resolve a user-provided mode list while preserving the canonical order."""
    requested = list(modes or DEFAULT_RUNTIME_MODES)
    unknown = sorted(set(requested) - set(ALL_MODES))
    if unknown:
        raise ValueError(f"Unknown ablation modes: {unknown}. Valid modes: {ALL_MODES}")

    resolved = [mode for mode in ALL_MODES if mode in requested]
    return resolved


def build_ablation_config(mode: str) -> AblationConfig:
    """Return the AblationConfig for a named experiment mode."""
    if mode not in _MODE_COMPONENTS:
        raise ValueError(f"Unknown ablation mode: {mode!r}")

    components = _MODE_COMPONENTS[mode]
    return AblationConfig.full() if not components else AblationConfig.ablate(*components)


def get_mode_components(mode: str) -> tuple[str, ...]:
    """Return the disabled component IDs for the mode."""
    if mode not in _MODE_COMPONENTS:
        raise ValueError(f"Unknown ablation mode: {mode!r}")
    return _MODE_COMPONENTS[mode]
