"""Convenience runner for AppWorld blackboard experiments."""
from __future__ import annotations

from typing import Any, Callable

from appworld import AppWorld
from experiment.appworld.core.world_wrapper import AppWorldRunResult, AppWorldTaskRunner
from experiment.appworld.systems.blackboard_system import AppWorldBlackboardSystem


def run_blackboard_task(
    *,
    task_id: str,
    experiment_name: str,
    dataset_name: str = "",
    llm: Any | None = None,
    remote_apis_url: str | None = None,
    appworld_root: str | None = None,
    load_ground_truth: bool = True,
    ground_truth_mode: str = "minimal",
    world_factory: Callable[..., AppWorld] | None = None,
    system_config: dict[str, Any] | None = None,
    max_steps: int = 1,
    world_kwargs: dict[str, Any] | None = None,
) -> AppWorldRunResult:
    """Run one AppWorld task through the AppWorld blackboard system."""
    system = AppWorldBlackboardSystem(llm=llm, config=system_config)
    with AppWorldTaskRunner(
        task_id=task_id,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        remote_apis_url=remote_apis_url,
        appworld_root=appworld_root,
        load_ground_truth=load_ground_truth,
        ground_truth_mode=ground_truth_mode,
        world_factory=world_factory,
        world_kwargs=world_kwargs,
    ) as runner:
        return runner.run_session(system, max_steps=max_steps)
