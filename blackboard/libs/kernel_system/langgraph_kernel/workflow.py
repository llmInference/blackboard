from __future__ import annotations

import copy
from typing import Any


TERMINAL_WORKER_NAMES = {"", "END"}


def _resolve_field_value(domain_state: dict[str, Any], field: str | None) -> Any:
    if not field or not isinstance(field, str):
        return None

    tokens = field.replace(".", "/").lstrip("/").split("/")
    current: Any = domain_state

    for token in tokens:
        if token == "":
            continue
        if isinstance(current, dict):
            current = current.get(token)
            continue
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (TypeError, ValueError, IndexError):
                return None
            continue
        return None

    return current


def _matching_state_key(rules: dict[str, Any], current_value: Any) -> str | None:
    if current_value in rules:
        return current_value
    if current_value is None:
        return None

    current_str = str(current_value)
    if current_str in rules:
        return current_str
    return None


def extract_worker_name(entry: Any) -> str | None:
    if entry is None:
        return None

    if isinstance(entry, str):
        worker = entry.strip()
        if worker in TERMINAL_WORKER_NAMES:
            return None
        return worker or None

    if isinstance(entry, dict):
        for key in ("worker", "worker_name", "route_to"):
            if key not in entry:
                continue
            worker = entry.get(key)
            if worker is None:
                return None
            worker_name = str(worker).strip()
            if worker_name in TERMINAL_WORKER_NAMES:
                return None
            return worker_name or None

    return None


def get_matching_workflow_entry(
    domain_state: dict[str, Any],
    workflow_rules: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None, Any]:
    for field_name, rules in (workflow_rules or {}).items():
        if not isinstance(rules, dict):
            continue
        state_key = _matching_state_key(rules, domain_state.get(field_name))
        if state_key is None:
            continue
        return field_name, state_key, rules[state_key]
    return None, None, None


def evaluate_condition(condition: Any, domain_state: dict[str, Any]) -> bool:
    if condition in (None, "", {}):
        return True

    if isinstance(condition, bool):
        return condition

    if isinstance(condition, list):
        return all(evaluate_condition(item, domain_state) for item in condition)

    if not isinstance(condition, dict):
        return bool(condition)

    if "all" in condition:
        items = condition.get("all") or []
        return all(evaluate_condition(item, domain_state) for item in items)

    if "any" in condition:
        items = condition.get("any") or []
        return any(evaluate_condition(item, domain_state) for item in items)

    if "not" in condition:
        return not evaluate_condition(condition.get("not"), domain_state)

    field = condition.get("field")
    value = _resolve_field_value(domain_state, field)

    if "exists" in condition:
        return (value is not None) == bool(condition.get("exists"))

    if condition.get("truthy") is True:
        return bool(value)
    if condition.get("truthy") is False:
        return not bool(value)
    if condition.get("falsy") is True:
        return not bool(value)

    if "equals" in condition:
        return value == condition.get("equals")
    if "eq" in condition:
        return value == condition.get("eq")
    if "not_equals" in condition:
        return value != condition.get("not_equals")
    if "ne" in condition:
        return value != condition.get("ne")

    if "in" in condition:
        candidates = condition.get("in") or []
        return value in candidates
    if "not_in" in condition:
        candidates = condition.get("not_in") or []
        return value not in candidates

    if "contains" in condition:
        needle = condition.get("contains")
        if isinstance(value, str):
            return str(needle) in value
        if isinstance(value, dict):
            return needle in value
        if isinstance(value, (list, tuple, set)):
            return needle in value
        return False

    for op_name, comparator in (
        ("gt", lambda lhs, rhs: lhs > rhs),
        ("gte", lambda lhs, rhs: lhs >= rhs),
        ("lt", lambda lhs, rhs: lhs < rhs),
        ("lte", lambda lhs, rhs: lhs <= rhs),
    ):
        if op_name not in condition:
            continue
        try:
            return comparator(value, condition.get(op_name))
        except TypeError:
            return False

    if field is not None:
        return bool(value)

    return False


def resolve_transition_target(transition: Any, domain_state: dict[str, Any]) -> str | None:
    if transition is None:
        return None

    if isinstance(transition, str):
        target = transition.strip()
        return target or None

    if isinstance(transition, dict):
        if "cases" in transition and isinstance(transition["cases"], list):
            for case in transition["cases"]:
                if not isinstance(case, dict):
                    continue
                when = case.get("when", case.get("condition"))
                if evaluate_condition(when, domain_state):
                    return resolve_transition_target(case.get("target", case.get("next")), domain_state)
            return resolve_transition_target(transition.get("default"), domain_state)

        if "when" in transition or "condition" in transition:
            when = transition.get("when", transition.get("condition"))
            if evaluate_condition(when, domain_state):
                target = transition.get("then", transition.get("target", transition.get("next")))
                return resolve_transition_target(target, domain_state)
            return resolve_transition_target(transition.get("else", transition.get("default")), domain_state)

        for key in ("target", "next", "default"):
            if key in transition:
                return resolve_transition_target(transition.get(key), domain_state)

    return None


def determine_next_status(
    current_state: dict[str, Any],
    workflow_rules: dict[str, dict[str, Any]],
    current_worker: str,
) -> str | None:
    if not workflow_rules or not current_worker:
        return None

    status_field, current_status, entry = get_matching_workflow_entry(current_state, workflow_rules)
    if status_field and current_status is not None and extract_worker_name(entry) == current_worker:
        if isinstance(entry, dict):
            for key in ("next", "next_state", "transition"):
                if key not in entry:
                    continue
                target = resolve_transition_target(entry.get(key), current_state)
                if target:
                    return target

        status_list = list(workflow_rules[status_field].keys())
        try:
            current_index = status_list.index(current_status)
        except ValueError:
            return None
        if current_index + 1 < len(status_list):
            return status_list[current_index + 1]
        return None

    for rules in workflow_rules.values():
        if not isinstance(rules, dict):
            continue
        for state_value, candidate in rules.items():
            if extract_worker_name(candidate) != current_worker:
                continue
            status_list = list(rules.keys())
            try:
                current_index = status_list.index(state_value)
            except ValueError:
                return None
            if current_index + 1 < len(status_list):
                return status_list[current_index + 1]
            return None

    return None


def get_status_field(workflow_rules: dict[str, dict[str, Any]]) -> str | None:
    for field_name, rules in (workflow_rules or {}).items():
        if isinstance(rules, dict) and rules:
            return field_name
    return None


def derive_selected_workers(
    task_flow: list[dict[str, Any]] | None,
    workflow_rules: dict[str, dict[str, Any]] | None,
) -> list[str]:
    ordered_workers: list[str] = []

    for task in task_flow or []:
        if not isinstance(task, dict):
            continue
        worker_name = extract_worker_name(task.get("worker"))
        if worker_name and worker_name not in ordered_workers:
            ordered_workers.append(worker_name)

    for rules in (workflow_rules or {}).values():
        if not isinstance(rules, dict):
            continue
        for entry in rules.values():
            worker_name = extract_worker_name(entry)
            if worker_name and worker_name not in ordered_workers:
                ordered_workers.append(worker_name)

    return ordered_workers


def sync_workflow_state_enums(
    data_schema: dict[str, Any],
    workflow_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(data_schema, dict):
        return {}

    schema = copy.deepcopy(data_schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema

    for field_name, rules in (workflow_rules or {}).items():
        if not isinstance(rules, dict) or field_name not in properties:
            continue
        prop = properties.get(field_name)
        if not isinstance(prop, dict):
            continue
        properties[field_name] = {
            **prop,
            "type": prop.get("type", "string"),
            "enum": list(rules.keys()),
        }

    return schema
