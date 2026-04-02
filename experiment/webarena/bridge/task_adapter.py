"""Task conversion helpers for the WebArena neutral bridge."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiment.common.neutral import TaskSpec


def _normalized_sites(task_config: Mapping[str, Any]) -> list[str]:
    return [str(site) for site in list(task_config.get("sites", []) or []) if str(site).strip()]


def _start_urls(task_config: Mapping[str, Any]) -> list[str]:
    return [str(url) for url in list(task_config.get("start_urls", []) or []) if str(url).strip()]


def _evaluator_names(task_config: Mapping[str, Any]) -> list[str]:
    evaluators: list[str] = []
    for item in list(task_config.get("eval", []) or []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("evaluator", "") or "").strip()
        if name:
            evaluators.append(name)
    return evaluators


def _agent_response_evaluator(task_config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for item in list(task_config.get("eval", []) or []):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("evaluator", "") or "") == "AgentResponseEvaluator":
            return item
    return None


def _results_schema_requires_text(results_schema: Any) -> bool:
    if not isinstance(results_schema, Mapping):
        return False
    schema_type = str(results_schema.get("type", "") or "").strip().lower()
    if schema_type and schema_type != "null":
        return True
    if results_schema.get("properties"):
        return True
    return False


def requires_final_response(task_config: Mapping[str, Any]) -> bool:
    """Return whether the task expects a final textual answer from the agent."""
    evaluator = _agent_response_evaluator(task_config)
    if evaluator is None:
        return False

    if _results_schema_requires_text(evaluator.get("results_schema")):
        return True

    expected = evaluator.get("expected")
    if isinstance(expected, Mapping):
        retrieved_data = expected.get("retrieved_data")
        if retrieved_data not in (None, "", [], {}):
            return True
    return False


def classify_task(task_config: Mapping[str, Any]) -> str:
    """Infer a coarse WebArena task category for routing and analysis."""
    sites = _normalized_sites(task_config)
    if len(sites) > 1:
        return "multi-site"

    intent = str(task_config.get("intent", "") or "").lower()
    evaluators = set(_evaluator_names(task_config))

    if "NetworkEventEvaluator" in evaluators and not requires_final_response(task_config):
        return "navigation"
    if requires_final_response(task_config):
        if any(token in intent for token in ("fill", "enter", "submit", "create", "edit", "update")):
            return "form"
        return "retrieval"
    if any(token in intent for token in ("open ", "navigate", "go to", "visit")):
        return "navigation"
    return "interaction"


def task_to_spec(
    task_config: Mapping[str, Any],
    *,
    config_path: str = "",
    dataset_name: str = "webarena-verified",
    config_template: str = "",
) -> TaskSpec:
    """Convert one WebArena task config into the neutral task specification."""
    sites = _normalized_sites(task_config)
    start_urls = _start_urls(task_config)
    evaluator_names = _evaluator_names(task_config)
    final_response_required = requires_final_response(task_config)
    category = classify_task(task_config)

    context: list[str] = []
    if sites:
        context.append(f"Sites: {', '.join(sites)}")
    if start_urls:
        context.append("Start URLs:\n" + "\n".join(f"- {url}" for url in start_urls))
    if evaluator_names:
        context.append("Evaluators: " + ", ".join(evaluator_names))

    metadata = {
        "dataset_name": str(dataset_name or "webarena-verified"),
        "config_path": str(config_path or ""),
        "config_template": str(config_template or ""),
        "sites": sites,
        "start_urls": start_urls,
        "intent_template": str(task_config.get("intent_template", "") or ""),
        "instantiation_dict": dict(task_config.get("instantiation_dict") or {}),
        "evaluators": evaluator_names,
        "eval": list(task_config.get("eval", []) or []),
        "requires_final_response": final_response_required,
        "task_category": category,
        "revision": task_config.get("revision"),
        "intent_template_id": task_config.get("intent_template_id"),
    }

    return TaskSpec(
        task_id=str(task_config.get("task_id", "")),
        title=f"WebArena Task {task_config.get('task_id', '')}".strip(),
        domain="webarena",
        instruction=str(task_config.get("intent", "") or "").strip(),
        context=tuple(context),
        metadata=metadata,
    )
