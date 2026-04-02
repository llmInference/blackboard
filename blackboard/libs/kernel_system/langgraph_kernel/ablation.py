from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationConfig:
    """Control switches for kernel_system ablation experiments.

    Component semantics follow ``experiment.md``:
    - C1: JSON Patch communication + schema validation
    - C2: deterministic-kernel circuit breaker
    - C3: context slicing
    - C4: architect agent
    """

    use_json_patch: bool = True
    use_schema_validation: bool = True
    use_circuit_breaker: bool = True
    use_context_slicing: bool = True
    use_architect_agent: bool = True
    hard_max_steps: int = 50

    @classmethod
    def full(cls) -> "AblationConfig":
        """Return the full experiment configuration."""
        return cls()

    @classmethod
    def ablate(cls, *components: str) -> "AblationConfig":
        """Disable the requested components while leaving the others enabled."""
        config = cls()
        mapping = {
            "C1": ("use_json_patch", "use_schema_validation"),
            "C2": ("use_circuit_breaker",),
            "C3": ("use_context_slicing",),
            "C4": ("use_architect_agent",),
        }

        for component in components:
            attr_names = mapping.get(component, ())
            for attr_name in attr_names:
                setattr(config, attr_name, False)

        return config
