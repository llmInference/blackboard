"""Deterministic planner worker for ALFWorld multi-worker experiments."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langgraph_kernel.worker.base import RuleWorkerAgent

from .action_worker import (
    _META_ACTIONS,
    _count_goal_deliveries,
    _expand_entity_variants,
    _extract_object_phrase,
    _extract_task_metadata,
    _extract_target_phrase,
    _goal_transform_done,
    _infer_held_item_variants,
    _light_activated,
    _matches_variants,
    _task_requires_transform,
)

_SURFACE_LOCATION_TYPES = {
    "armchair",
    "bed",
    "cabinettop",
    "coffee_table",
    "coffeetable",
    "countertop",
    "desk",
    "diningtable",
    "sidetable",
    "sofa",
    "table",
}
_CONTAINER_LOCATION_TYPES = {
    "cabinet",
    "drawer",
    "fridge",
    "microwave",
    "safe",
    "sinkbasin",
}
_UTENSIL_OBJECTS = {"butterknife", "fork", "knife", "ladle", "spatula", "spoon"}
_DISHWARE_OBJECTS = {"bowl", "cup", "glass", "kettle", "mug", "pan", "plate", "pot"}
_FOOD_OBJECTS = {
    "apple",
    "banana",
    "bread",
    "egg",
    "lettuce",
    "orange",
    "pepper",
    "potato",
    "tomato",
}
_SMALL_OBJECTS = {
    "book",
    "cd",
    "cellphone",
    "creditcard",
    "keychain",
    "newspaper",
    "remotecontrol",
    "watch",
}


def _normalize_location_type(label: str) -> str:
    normalized = re.sub(r"\s+\d+$", "", label.lower()).strip()
    return normalized.split()[0] if normalized else ""


def _extract_search_location(action: str) -> str:
    lowered = action.lower()
    if lowered.startswith(("go to ", "open ", "close ", "examine ")):
        return _extract_object_phrase(action)
    if lowered.startswith("take "):
        return _extract_target_phrase(action)
    return ""


def _observation_shows_empty(observation: str) -> bool:
    return "you see nothing" in observation.lower()


def _goal_object_preferences(goal_object: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", "", goal_object.lower())
    if normalized in _UTENSIL_OBJECTS:
        return ["countertop", "sinkbasin", "cabinet", "drawer"]
    if normalized in _DISHWARE_OBJECTS:
        return ["cabinet", "countertop", "sinkbasin", "drawer"]
    if normalized in _FOOD_OBJECTS:
        return ["countertop", "fridge", "diningtable", "sidetable", "microwave"]
    if normalized in _SMALL_OBJECTS:
        return ["drawer", "sidetable", "desk", "countertop", "armchair", "sofa"]
    return ["countertop", "sidetable", "cabinet", "drawer", "sinkbasin", "fridge"]


def _build_goal_object_guidance(task_goal: str, goal_object: str) -> str:
    canonical = goal_object.strip()
    if not canonical:
        return "Follow planner focus_object as the canonical required object."

    canonical_variants = _expand_entity_variants(canonical)
    task_goal_tokens = _expand_entity_variants(task_goal)
    if canonical_variants and not (canonical_variants & task_goal_tokens):
        return (
            f"Canonical goal object: {canonical}. "
            "If the natural-language task wording suggests a nearby substitute, still follow focus_object exactly."
        )

    return f"Canonical goal object: {canonical}. Prefer actions that target this object exactly."


def _build_search_guidance(context: dict[str, Any], task_meta: dict[str, Any]) -> dict[str, Any]:
    available_actions = list(context.get("available_actions", []))
    action_history = list(context.get("action_history", []))
    observation_history = list(context.get("observation_history", []))
    recent_history = action_history[-4:]
    goal_variants = set(task_meta["goal_object_variants"])
    preferred_location_types = _goal_object_preferences(str(task_meta["goal_object"]))

    searched_locations: list[str] = []
    failed_search_locations: list[str] = []
    failed_type_counts: Counter[str] = Counter()

    for action, observation in zip(action_history, observation_history[1:]):
        location = _extract_search_location(action)
        if location and location not in searched_locations:
            searched_locations.append(location)

        if location and _observation_shows_empty(observation):
            if location not in failed_search_locations:
                failed_search_locations.append(location)
            failed_type_counts[_normalize_location_type(location)] += 1

    repeated_failed_type = ""
    repeated_failed_count = 0
    if failed_type_counts:
        repeated_failed_type, repeated_failed_count = max(
            failed_type_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
        if repeated_failed_count < 2:
            repeated_failed_type = ""
            repeated_failed_count = 0

    ranked_actions: list[tuple[float, str]] = []
    for index, action in enumerate(available_actions):
        action_lower = action.lower()
        location = _extract_search_location(action)
        location_type = _normalize_location_type(location)
        object_variants = _expand_entity_variants(_extract_object_phrase(action))

        score = float(-index * 0.01)

        if action_lower in _META_ACTIONS:
            score -= 50.0

        if action in recent_history:
            score -= 12.0

        if action_lower.startswith("take ") and _matches_variants(object_variants, goal_variants):
            score += 200.0
        elif _matches_variants(object_variants, goal_variants):
            score += 60.0

        if action_lower.startswith(("go to ", "open ")):
            score += 8.0

        if location:
            if location in searched_locations:
                score -= 4.0
            else:
                score += 10.0

            if location in failed_search_locations:
                score -= 30.0

            if location_type in preferred_location_types:
                score += max(0.0, 12.0 - preferred_location_types.index(location_type) * 2.0)

            if repeated_failed_type:
                if location_type == repeated_failed_type:
                    score -= 24.0
                elif location_type:
                    score += 12.0

            if location_type in _SURFACE_LOCATION_TYPES:
                score += 3.0
            if location_type in _CONTAINER_LOCATION_TYPES and action_lower.startswith("open "):
                score += 2.0

        ranked_actions.append((score, action))

    ranked_actions.sort(key=lambda item: (-item[0], item[1]))
    recommended_actions = [action for _, action in ranked_actions[:3]]

    if recommended_actions and any(
        action.lower().startswith("take ") and _matches_variants(_expand_entity_variants(_extract_object_phrase(action)), goal_variants)
        for action in recommended_actions
    ):
        search_guidance = "The goal object is directly available now. Prefer taking it immediately."
    elif repeated_failed_type and recommended_actions:
        preferred = ", ".join(recommended_actions[:2])
        search_guidance = (
            f"Already checked {repeated_failed_count} empty {repeated_failed_type} locations. "
            f"Pivot to a different location type instead of opening another {repeated_failed_type}. "
            f"Prefer: {preferred}."
        )
    elif recommended_actions:
        preferred = ", ".join(recommended_actions[:2])
        search_guidance = f"Search unexplored likely locations for the goal object. Prefer: {preferred}."
    else:
        search_guidance = "Search unexplored locations that are most likely to contain the goal object."

    return {
        "searched_locations": searched_locations,
        "failed_search_locations": failed_search_locations,
        "recommended_actions": recommended_actions,
        "search_guidance": search_guidance,
    }


class PlannerWorker(RuleWorkerAgent):
    """Produce a compact structured subgoal for the downstream action worker."""

    def __init__(self, llm: Any = None, instruction: str | None = None) -> None:
        self._instruction = instruction or ""
        self._llm = llm

    def _think(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        task_meta = _extract_task_metadata(context)
        task_type = str(task_meta["task_type"])
        action_history = list(context.get("action_history", []))
        held_items = _infer_held_item_variants(action_history)
        holding_goal_object = any(task_meta["goal_object_variants"] & held for held in held_items)

        desired_goal_count = 2 if "pick_two_obj" in task_type else 1
        delivered_goal_count = _count_goal_deliveries(
            action_history,
            task_meta["goal_object_variants"],
            task_meta["target_receptacle_variants"],
        )
        needs_transform, transform_verb, _, transform_appliance = _task_requires_transform(task_type)
        transform_completed = _goal_transform_done(action_history, transform_verb, task_meta["goal_object_variants"])
        light_activated = _light_activated(action_history, task_meta["target_receptacle_variants"])

        subgoal = "search_goal_object"
        focus_object = task_meta["goal_object"]
        focus_receptacle = task_meta["target_receptacle"]
        planner_summary = "Search for the goal object."

        if "look_at_obj_in_light" in task_type:
            if not holding_goal_object:
                subgoal = "acquire_goal_object"
                planner_summary = "Find and pick up the inspection object."
            elif not light_activated:
                subgoal = "activate_light"
                planner_summary = "Go to the light source and turn it on."
            else:
                subgoal = "inspect_goal_object"
                planner_summary = "Examine the held object under the active light."
        elif needs_transform:
            if not holding_goal_object:
                subgoal = "acquire_goal_object"
                planner_summary = "Acquire the goal object before applying the required transform."
            elif not transform_completed:
                subgoal = f"{transform_verb}_goal_object"
                focus_receptacle = transform_appliance
                planner_summary = f"Bring the goal object to the {transform_appliance} and apply {transform_verb}."
            elif delivered_goal_count < desired_goal_count:
                subgoal = "place_goal_object"
                planner_summary = "Move the transformed goal object to the final receptacle."
            else:
                subgoal = "task_completed"
                planner_summary = "The required transformed objects have already been delivered."
        else:
            if not holding_goal_object:
                subgoal = "acquire_goal_object"
                planner_summary = "Acquire the goal object."
            elif delivered_goal_count < desired_goal_count:
                subgoal = "place_goal_object"
                planner_summary = "Deliver the held goal object to the target receptacle."
            else:
                subgoal = "task_completed"
                planner_summary = "The required objects have already been delivered."

        search_guidance = _build_search_guidance(context, task_meta)
        fields = {
            "planner_summary": planner_summary,
            "subgoal": subgoal,
            "focus_object": focus_object,
            "focus_receptacle": focus_receptacle,
            "required_transform": transform_verb,
            "transform_completed": transform_completed,
            "light_activated": light_activated,
            "delivered_goal_count": delivered_goal_count,
            "goal_delivery_target_count": desired_goal_count,
            "goal_object_guidance": _build_goal_object_guidance(context.get("task_goal", ""), focus_object),
            "searched_locations": search_guidance["searched_locations"],
            "failed_search_locations": search_guidance["failed_search_locations"],
            "recommended_actions": search_guidance["recommended_actions"],
            "search_guidance": search_guidance["search_guidance"],
        }

        patches: list[dict[str, Any]] = []
        for field, value in fields.items():
            patches.append(
                {
                    "op": "replace" if field in context else "add",
                    "path": f"/{field}",
                    "value": value,
                }
            )
        return patches
