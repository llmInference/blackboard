"""Convenience runner for WebArena blackboard experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from experiment.webarena.core import WebArenaRunResult, WebArenaTaskRunner
from experiment.webarena.systems.blackboard_system import WebArenaBlackboardSystem
from experiment.webarena.utils import build_runtime_llm


def run_blackboard_task(
    *,
    task_id: int,
    config_path: str | Path,
    output_dir: str | Path,
    dataset_name: str = "webarena-verified",
    llm: Any | None = None,
    system_config: dict[str, Any] | None = None,
    headless: bool = True,
    slow_mo_ms: int = 0,
    webarena_factory: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    ui_login_func: Any | None = None,
    max_steps: int = 8,
) -> WebArenaRunResult:
    """Run one WebArena task through the WebArena blackboard system."""
    resolved_system_config = dict(system_config or {})
    if llm is None and str(resolved_system_config.get("llm_backend", "") or "").strip():
        llm = build_runtime_llm(
            llm_backend=str(resolved_system_config.get("llm_backend", "openai_compatible")),
            llm_model=str(resolved_system_config.get("llm_model", "") or ""),
            llm_model_env=str(resolved_system_config.get("llm_model_env", "") or ""),
            llm_base_url=str(resolved_system_config.get("llm_base_url", "") or ""),
            llm_base_url_env=str(resolved_system_config.get("llm_base_url_env", "") or ""),
            llm_api_key_env=str(resolved_system_config.get("llm_api_key_env", "") or ""),
            llm_temperature=float(resolved_system_config.get("llm_temperature", 0.0) or 0.0),
            llm_timeout=float(resolved_system_config.get("llm_timeout", 60.0) or 60.0),
        )
    system = WebArenaBlackboardSystem(llm=llm, config=resolved_system_config)
    with WebArenaTaskRunner(
        task_id=task_id,
        config_path=config_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        headless=headless,
        slow_mo_ms=slow_mo_ms,
        webarena_factory=webarena_factory,
        runtime_factory=runtime_factory,
        ui_login_func=ui_login_func,
    ) as runner:
        return runner.run_session(system, max_steps=max_steps)
