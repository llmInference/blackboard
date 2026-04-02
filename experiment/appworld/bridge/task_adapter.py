"""Task conversion helpers for the AppWorld neutral bridge."""
from __future__ import annotations

from typing import Any

from appworld.task import Task

from experiment.common.neutral import TaskSpec


def _supervisor_context(task: Task) -> str:
    supervisor = task.supervisor
    lines = [
        "Supervisor:",
        f"- Name: {supervisor.first_name} {supervisor.last_name}",
        f"- Email: {supervisor.email}",
        f"- Phone: {supervisor.phone_number}",
    ]
    return "\n".join(lines)


def task_to_spec(
    task: Task,
    *,
    dataset_name: str = "",
    task_metadata: dict[str, Any] | None = None,
) -> TaskSpec:
    """Convert one AppWorld task into the neutral task specification."""
    metadata = dict(task_metadata or {})
    metadata.update(
        {
            "dataset_name": str(dataset_name or ""),
            "allowed_apps": list(task.allowed_apps),
            "app_descriptions": dict(task.app_descriptions),
            "supervisor": {
                "first_name": task.supervisor.first_name,
                "last_name": task.supervisor.last_name,
                "email": task.supervisor.email,
                "phone_number": task.supervisor.phone_number,
            },
            "datetime": task.datetime.isoformat(),
            "db_version": task.db_version,
        }
    )

    context = [
        _supervisor_context(task),
        f"Current Datetime: {task.datetime.isoformat()}",
        f"Allowed Apps: {', '.join(task.allowed_apps)}",
    ]
    if task.app_descriptions:
        app_lines = ["App Descriptions:"]
        app_lines.extend(
            f"- {app_name}: {description}"
            for app_name, description in sorted(task.app_descriptions.items())
        )
        context.append("\n".join(app_lines))

    return TaskSpec(
        task_id=str(task.id),
        title=str(task.id),
        domain="appworld",
        instruction=str(task.instruction or "").strip(),
        context=tuple(context),
        metadata=metadata,
    )
