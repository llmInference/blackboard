"""Deterministic ALFWorld action selection worker."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langgraph_kernel.worker.base import RuleWorkerAgent


_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "with",
    "your",
}
_META_ACTIONS = {"help", "inventory", "look"}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and token not in _STOPWORDS
    }


def _expand_entity_variants(label: str) -> set[str]:
    if not label or label.lower() == "none":
        return set()
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    base_tokens = _tokenize(spaced)
    compact = re.sub(r"[^a-z0-9]+", "", spaced.lower())
    if compact:
        base_tokens.add(compact)
    return base_tokens


def _extract_task_metadata(context: dict[str, Any]) -> dict[str, set[str] | str]:
    canonical_task_type = str(context.get("canonical_task_type", "") or "")
    canonical_goal_object = str(context.get("canonical_goal_object", "") or "")
    canonical_target_receptacle = str(context.get("canonical_target_receptacle", "") or "")
    gamefile = context.get("current_gamefile", "")
    task_type = canonical_task_type
    goal_object = canonical_goal_object
    helper_object = ""
    target_receptacle = canonical_target_receptacle

    if gamefile:
        task_dir = Path(gamefile).parent.parent.name
        parts = task_dir.split("-")
        if len(parts) >= 5:
            parsed_task_type, parsed_goal_object, parsed_helper_object, parsed_target_receptacle = parts[:4]
            task_type = task_type or parsed_task_type
            goal_object = goal_object or parsed_goal_object
            helper_object = helper_object or parsed_helper_object
            target_receptacle = target_receptacle or parsed_target_receptacle

    return {
        "task_type": task_type,
        "goal_object": goal_object,
        "helper_object": helper_object,
        "target_receptacle": target_receptacle,
        "goal_object_variants": _expand_entity_variants(goal_object),
        "helper_object_variants": _expand_entity_variants(helper_object),
        "target_receptacle_variants": _expand_entity_variants(target_receptacle),
    }


def _extract_object_phrase(action: str) -> str:
    patterns = [
        (r"^take (.+?) from ", 1),
        (r"^move (.+?) to ", 1),
        (r"^(?:clean|heat|cool|slice) (.+?) with ", 1),
        (r"^(?:toggle|use|open|close|examine|go to) (.+)$", 1),
    ]
    for pattern, group in patterns:
        match = re.match(pattern, action.lower())
        if match:
            return match.group(group)
    return ""


def _extract_target_phrase(action: str) -> str:
    patterns = [
        (r"^take .+? from (.+)$", 1),
        (r"^move .+? to (.+)$", 1),
        (r"^(?:clean|heat|cool|slice) .+? with (.+)$", 1),
    ]
    for pattern, group in patterns:
        match = re.match(pattern, action.lower())
        if match:
            return match.group(group)
    return ""


def _infer_held_item_variants(action_history: list[str]) -> list[set[str]]:
    held: list[set[str]] = []
    for action in action_history:
        obj_phrase = _extract_object_phrase(action)
        if not obj_phrase:
            continue
        variants = _expand_entity_variants(obj_phrase)
        if action.startswith("take "):
            held.append(variants)
        elif action.startswith("move "):
            held = [item for item in held if item != variants]
    return held


def _matches_variants(action_tokens: set[str], variants: set[str]) -> bool:
    return bool(variants and action_tokens & variants)


def _target_matches_variants(target_phrase: str, variants: set[str]) -> bool:
    if not target_phrase or not variants:
        return False
    target_variants = _expand_entity_variants(target_phrase)
    return bool(target_variants & variants)


def _count_goal_deliveries(action_history: list[str], goal_variants: set[str], target_variants: set[str]) -> int:
    delivered = 0
    for action in action_history:
        if not action.startswith("move "):
            continue
        obj_variants = _expand_entity_variants(_extract_object_phrase(action))
        target = _expand_entity_variants(_extract_target_phrase(action))
        if obj_variants == goal_variants and target_variants and target & target_variants:
            delivered += 1
    return delivered


def _task_requires_transform(task_type: str) -> tuple[bool, str, set[str], str]:
    if "pick_clean_then_place" in task_type:
        return True, "clean", {"sinkbasin", "sink"}, "sinkbasin"
    if "pick_heat_then_place" in task_type:
        return True, "heat", {"microwave"}, "microwave"
    if "pick_cool_then_place" in task_type:
        return True, "cool", {"fridge", "refrigerator"}, "fridge"
    return False, "", set(), ""


def _goal_transform_done(action_history: list[str], transform_verb: str, goal_variants: set[str]) -> bool:
    if not transform_verb or not goal_variants:
        return False
    prefix = f"{transform_verb} "
    for action in action_history:
        if not action.startswith(prefix):
            continue
        obj_variants = _expand_entity_variants(_extract_object_phrase(action))
        if obj_variants & goal_variants:
            return True
    return False


def _goal_on_target(action_history: list[str], goal_variants: set[str], target_variants: set[str]) -> bool:
    if not goal_variants or not target_variants:
        return False
    for action in reversed(action_history):
        if action.startswith("move "):
            obj_variants = _expand_entity_variants(_extract_object_phrase(action))
            target = _expand_entity_variants(_extract_target_phrase(action))
            if obj_variants == goal_variants:
                return bool(target & target_variants)
        if action.startswith("take "):
            obj_variants = _expand_entity_variants(_extract_object_phrase(action))
            if obj_variants == goal_variants:
                return False
    return False


def _light_activated(action_history: list[str], target_variants: set[str]) -> bool:
    if not target_variants:
        return False
    for action in action_history:
        if action.startswith("toggle ") or action.startswith("use "):
            obj_variants = _expand_entity_variants(_extract_object_phrase(action))
            if obj_variants & target_variants:
                return True
    return False


class ActionWorker(RuleWorkerAgent):
    """Choose a valid ALFWorld action using lightweight goal matching."""

    def __init__(self, llm: Any = None, instruction: str | None = None) -> None:
        self._instruction = instruction or ""
        self._llm = llm

    def _think(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        available_actions = list(context.get("available_actions", []))
        selected_action, reason = self._select_action(context, available_actions)
        patches: list[dict[str, Any]] = []

        patches.append(
            {
                "op": "replace" if "selected_action" in context else "add",
                "path": "/selected_action",
                "value": selected_action,
            }
        )
        patches.append(
            {
                "op": "replace" if "decision_reason" in context else "add",
                "path": "/decision_reason",
                "value": reason,
            }
        )
        return patches

    def _select_action(
        self,
        context: dict[str, Any],
        available_actions: list[str],
    ) -> tuple[str, str]:
        if not available_actions:
            return "", "No admissible actions were provided by ALFWorld."

        goal_text = context.get("task_goal", "")
        observation_text = context.get("current_observation", "")
        action_history = list(context.get("action_history", []))
        recent_history = action_history[-4:]
        goal_tokens = _tokenize(goal_text)
        observation_tokens = _tokenize(observation_text)
        task_meta = _extract_task_metadata(context)
        task_type = str(task_meta["task_type"])
        desired_goal_count = 2 if "pick_two_obj" in task_type else 1
        held_item_variants = _infer_held_item_variants(action_history)
        holding_goal_object = any(task_meta["goal_object_variants"] & held for held in held_item_variants)
        holding_helper_object = any(task_meta["helper_object_variants"] & held for held in held_item_variants)
        delivered_goal_count = _count_goal_deliveries(
            action_history,
            task_meta["goal_object_variants"],
            task_meta["target_receptacle_variants"],
        )
        needs_transform, transform_verb, transform_appliance_variants, transform_appliance_label = _task_requires_transform(task_type)
        goal_transform_done = _goal_transform_done(action_history, transform_verb, task_meta["goal_object_variants"])
        transform_pending = needs_transform and not goal_transform_done
        looking_for_more_goal_objects = delivered_goal_count < desired_goal_count
        goal_already_on_target = _goal_on_target(
            action_history,
            task_meta["goal_object_variants"],
            task_meta["target_receptacle_variants"],
        )
        light_done = _light_activated(action_history, task_meta["target_receptacle_variants"])

        best_action = available_actions[0]
        best_score = float("-inf")
        best_reason = "Fallback to the first admissible action."

        if goal_transform_done and holding_goal_object:
            for action in available_actions:
                if not action.startswith("move "):
                    continue
                obj_variants = _expand_entity_variants(_extract_object_phrase(action))
                target_phrase = _extract_target_phrase(action)
                if _matches_variants(obj_variants, task_meta["goal_object_variants"]) and _target_matches_variants(
                    target_phrase, task_meta["target_receptacle_variants"]
                ):
                    return action, "post-transform hard rule: move goal object to final receptacle"

        for index, action in enumerate(available_actions):
            action_tokens = _tokenize(action)
            action_lower = action.lower()
            obj_phrase = _extract_object_phrase(action)
            obj_variants = _expand_entity_variants(obj_phrase)
            target_phrase = _extract_target_phrase(action)
            target_variants = _expand_entity_variants(target_phrase)
            score = float(-index * 0.01)
            reasons: list[str] = []

            repeat_count = action_history.count(action)
            if repeat_count:
                score -= 2.0 * repeat_count
                reasons.append(f"penalized repeated action x{repeat_count + 1}")
            if action in recent_history:
                score -= 4.0
                reasons.append("penalized recent repeat")

            if action in _META_ACTIONS:
                score -= 4.0
                reasons.append("meta action fallback")
            else:
                score += 1.0
                reasons.append("non-meta action")

            goal_overlap = sorted(action_tokens & goal_tokens)
            if goal_overlap:
                score += 4.0 * len(goal_overlap)
                reasons.append(f"matches goal tokens {', '.join(goal_overlap)}")

            observation_overlap = sorted(action_tokens & observation_tokens)
            if observation_overlap:
                score += 1.5 * len(observation_overlap)
                reasons.append(f"matches observation tokens {', '.join(observation_overlap)}")

            goal_lower = goal_text.lower()
            if action_lower.startswith("take ") and any(
                keyword in goal_lower for keyword in ("take", "pick", "grab", "clean", "heat", "cool", "put", "place")
            ):
                score += 2.0
                reasons.append("supports acquiring task object")
            if action_lower.startswith("put ") and any(keyword in goal_lower for keyword in ("put", "place")):
                score += 3.0
                reasons.append("supports placement goal")
            if action_lower.startswith("open "):
                score += 0.5
                reasons.append("may reveal new objects")
            if action_lower.startswith("toggle ") and any(keyword in goal_lower for keyword in ("light", "lamp")):
                score += 2.0
                reasons.append("supports lighting goal")

            if _matches_variants(action_tokens, task_meta["goal_object_variants"]):
                score += 7.0
                reasons.append("references goal object")

            if _matches_variants(action_tokens, task_meta["target_receptacle_variants"]):
                score += 5.0
                reasons.append("references target receptacle")

            if _matches_variants(action_tokens, task_meta["helper_object_variants"]):
                score += 3.0
                reasons.append("references helper object")

            if action_lower.startswith("take ") and _matches_variants(obj_variants, task_meta["goal_object_variants"]):
                score += 10.0
                reasons.append("acquires goal object")
                if goal_already_on_target and not looking_for_more_goal_objects:
                    score -= 10.0
                    reasons.append("penalized taking goal object already placed")
            elif action_lower.startswith("take "):
                score -= 12.0
                reasons.append("penalized taking unrelated object")

            if action_lower.startswith("move "):
                if _matches_variants(obj_variants, task_meta["goal_object_variants"]) and _target_matches_variants(
                    target_phrase, task_meta["target_receptacle_variants"]
                ):
                    move_score = 12.0
                    if transform_pending:
                        move_score -= 10.0
                        reasons.append("penalized moving goal object to target before required transform")
                    score += move_score
                    reasons.append("moves goal object toward target receptacle")
                elif not _matches_variants(obj_variants, task_meta["goal_object_variants"]):
                    score -= 8.0
                    reasons.append("penalized unrelated move")

            if action_lower.startswith("go to ") and _matches_variants(obj_variants, task_meta["target_receptacle_variants"]):
                score += 8.0 if holding_goal_object else 4.0
                reasons.append("navigates toward target receptacle")
                if "look_at_obj_in_light" in task_type and light_done:
                    score -= 8.0
                    reasons.append("penalized revisiting light source after activation")

            if action_lower.startswith("open ") and _matches_variants(obj_variants, task_meta["target_receptacle_variants"]):
                score += 6.0 if holding_goal_object else 3.0
                reasons.append("opens target receptacle")
                if goal_already_on_target:
                    score -= 4.0
                    reasons.append("penalized reopening target after placement")

            if action_lower.startswith("examine "):
                score -= 1.5
                reasons.append("slight penalty for examine")

            if looking_for_more_goal_objects and action_lower.startswith("go to ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score += 5.0
                reasons.append("navigates toward goal object mention")

            if needs_transform:
                if transform_pending and holding_goal_object:
                    if action_lower.startswith("go to ") and _matches_variants(obj_variants, transform_appliance_variants):
                        score += 14.0
                        reasons.append(f"navigates toward required {transform_appliance_label}")
                    if action_lower.startswith("open ") and _matches_variants(obj_variants, transform_appliance_variants):
                        score += 10.0
                        reasons.append(f"opens required {transform_appliance_label}")
                    if action_lower.startswith(f"{transform_verb} ") and _matches_variants(
                        obj_variants, task_meta["goal_object_variants"]
                    ):
                        score += 18.0
                        reasons.append(f"applies required {transform_verb} step")
                    if action_lower.startswith("move ") and _matches_variants(
                        obj_variants, task_meta["goal_object_variants"]
                    ) and _matches_variants(target_variants, transform_appliance_variants):
                        score += 8.0
                        reasons.append(f"moves goal object toward {transform_appliance_label}")
                elif not holding_goal_object and transform_pending and action_lower.startswith("take "):
                    if not _matches_variants(obj_variants, task_meta["goal_object_variants"]):
                        score -= 6.0
                        reasons.append("penalized taking non-goal object before transform step")
                elif goal_transform_done:
                    if action_lower.startswith(f"{transform_verb} "):
                        score -= 40.0
                        reasons.append(f"penalized repeated {transform_verb} after completion")
                    if action_lower.startswith("go to ") and _matches_variants(
                        obj_variants, task_meta["target_receptacle_variants"]
                    ):
                        score += 9.0
                        reasons.append("navigates to final receptacle after transform")
                    if action_lower.startswith("open ") and _matches_variants(
                        obj_variants, task_meta["target_receptacle_variants"]
                    ):
                        score += 7.0
                        reasons.append("opens final receptacle after transform")
                    if action_lower.startswith("move ") and _matches_variants(
                        obj_variants, task_meta["goal_object_variants"]
                    ) and _target_matches_variants(target_phrase, task_meta["target_receptacle_variants"]):
                        score += 24.0
                        reasons.append("moves transformed goal object to final receptacle")

            if "pick_heat_then_place" in task_type and action_lower.startswith("heat ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score += 12.0
                reasons.append("applies required heating step")
            if "pick_cool_then_place" in task_type and action_lower.startswith("cool ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score += 12.0
                reasons.append("applies required cooling step")
            if "pick_clean_then_place" in task_type and action_lower.startswith("clean ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score += 12.0
                reasons.append("applies required cleaning step")
            if "look_at_obj_in_light" in task_type:
                if (action_lower.startswith("toggle ") or action_lower.startswith("use ")) and _matches_variants(
                    obj_variants, task_meta["target_receptacle_variants"]
                ):
                    score += 12.0
                    reasons.append("supports lighting interaction")
                    if holding_goal_object:
                        score += 6.0
                        reasons.append("activates light while holding goal object")
                if action_lower.startswith("go to ") and _matches_variants(obj_variants, task_meta["target_receptacle_variants"]):
                    score += 5.0
                    reasons.append("navigates toward light source")
                if action_lower.startswith("take ") and _matches_variants(obj_variants, task_meta["goal_object_variants"]):
                    score += 8.0
                    reasons.append("acquires object to inspect under light")
                if light_done and holding_goal_object and action_lower.startswith("examine ") and _matches_variants(
                    obj_variants, task_meta["goal_object_variants"]
                ):
                    score += 16.0
                    reasons.append("examines goal object after light activation")
                if light_done and (action_lower.startswith("toggle ") or action_lower.startswith("use ")):
                    score -= 12.0
                    reasons.append("penalized repeated light activation")

            if holding_goal_object and action_lower.startswith("move ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score += 4.0
                reasons.append("acts while already holding goal object")

            if holding_goal_object and action_lower.startswith("take ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score -= 10.0
                reasons.append("penalized re-taking goal object already held")

            if holding_helper_object and action_lower.startswith("move ") and _matches_variants(
                obj_variants, task_meta["helper_object_variants"]
            ):
                score -= 4.0
                reasons.append("penalized moving helper object without goal progress")

            if len(recent_history) >= 2 and action == recent_history[-2]:
                score -= 4.0
                reasons.append("penalized alternating loop pattern")

            if recent_history:
                last_action = recent_history[-1].lower()
                if last_action.startswith("take ") and action_lower.startswith("move "):
                    last_object = _expand_entity_variants(_extract_object_phrase(last_action))
                    if last_object and last_object == obj_variants and _extract_target_phrase(last_action) == target_phrase:
                        score -= 6.0
                        reasons.append("penalized immediate take/move reversal")
                if last_action.startswith("move ") and action_lower.startswith("take "):
                    last_object = _expand_entity_variants(_extract_object_phrase(last_action))
                    if last_object and last_object == obj_variants:
                        score -= 5.0
                        reasons.append("penalized immediate move/take reversal")

            if delivered_goal_count >= desired_goal_count and action_lower.startswith("move "):
                score -= 20.0
                reasons.append("penalized extra move after enough deliveries")
            if goal_already_on_target and action_lower.startswith("move ") and _matches_variants(
                obj_variants, task_meta["goal_object_variants"]
            ):
                score -= 12.0
                reasons.append("penalized moving goal object away from target")

            if score > best_score:
                best_action = action
                best_score = score
                best_reason = "; ".join(reasons)

        return best_action, best_reason
