"""
State Bridge - Utilities for converting between ALFWorld and Kernel System states

This module provides functions to:
1. Convert ALFWorld observations to Kernel State
2. Update Kernel State with new observations
3. Extract actions from Kernel State for ALFWorld execution
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_TASK_DESC = "Complete the current ALFWorld task."


@lru_cache(maxsize=1024)
def _load_task_metadata_from_gamefile(gamefile_path: str) -> Dict[str, str]:
    gamefile = Path(gamefile_path)
    traj_path = gamefile.with_name("traj_data.json")
    task_dir = gamefile.parent.parent.name if gamefile.parent.parent else gamefile.stem
    metadata = {
        "task_desc": "",
        "canonical_task_type": task_dir.split("-")[0] if task_dir else "",
        "canonical_goal_object": "",
        "canonical_target_receptacle": "",
    }

    if traj_path.exists():
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("turk_annotations", {}).get("anns", [])
        if anns:
            task_desc = anns[0].get("task_desc", "").strip()
            if task_desc:
                metadata["task_desc"] = task_desc

        pddl_params = data.get("pddl_params", {}) or {}
        metadata["canonical_goal_object"] = str(pddl_params.get("object_target") or "").strip()
        metadata["canonical_target_receptacle"] = str(
            pddl_params.get("parent_target") or pddl_params.get("toggle_target") or ""
        ).strip()

    if not metadata["task_desc"]:
        metadata["task_desc"] = task_dir.replace("_", " ").replace("-", " ").strip()

    return metadata


def _load_task_desc_from_gamefile(gamefile_path: str) -> str:
    return _load_task_metadata_from_gamefile(gamefile_path)["task_desc"]


def extract_task_desc(infos: Dict[str, Any]) -> str:
    """Extract the task description from ALFWorld infos."""
    if infos.get("extra.goal"):
        return infos["extra.goal"][0]

    gamefile = _extract_gamefile(infos)
    if gamefile:
        return _load_task_desc_from_gamefile(gamefile)

    return _DEFAULT_TASK_DESC


def extract_task_metadata(infos: Dict[str, Any]) -> Dict[str, str]:
    gamefile = _extract_gamefile(infos)
    if not gamefile:
        return {
            "canonical_task_type": "",
            "canonical_goal_object": "",
            "canonical_target_receptacle": "",
        }
    metadata = _load_task_metadata_from_gamefile(gamefile)
    return {
        "canonical_task_type": metadata.get("canonical_task_type", ""),
        "canonical_goal_object": metadata.get("canonical_goal_object", ""),
        "canonical_target_receptacle": metadata.get("canonical_target_receptacle", ""),
    }


def _extract_gamefile(infos: Dict[str, Any]) -> str:
    gamefiles = infos.get("extra.gamefile") or []
    for gamefile in gamefiles:
        if isinstance(gamefile, str) and gamefile:
            return gamefile
    return ""


def _build_user_prompt(
    task_desc: str,
    obs: str,
    admissible_commands: List[str],
    action_history: List[str] | None = None,
) -> str:
    previous_actions = action_history or []
    previous_actions_text = ", ".join(previous_actions[-5:]) if previous_actions else "None"
    return f"""Task: {task_desc}

Current Observation:
{obs}

Available Actions:
{', '.join(admissible_commands)}

Previous Actions:
{previous_actions_text}
"""


def obs_to_kernel_state(obs: str, infos: Dict) -> Dict[str, Any]:
    """
    Convert ALFWorld observation to Kernel State.

    This function creates the initial Kernel State from ALFWorld's observation.
    The text observation is passed directly as user_prompt to leverage LLM's
    natural language understanding.

    Args:
        obs: Text observation from ALFWorld
        infos: Info dictionary from ALFWorld containing:
            - extra.goal: Task description
            - admissible_commands: List of valid actions

    Returns:
        Kernel State dictionary with:
            - user_prompt: Complete prompt including task, observation, and actions
            - domain_state: Business state with task info and current observation
            - Other Kernel System fields initialized to empty
    """
    task_desc = extract_task_desc(infos)
    admissible_commands = infos["admissible_commands"][0]
    gamefile = _extract_gamefile(infos)
    task_metadata = extract_task_metadata(infos)
    user_prompt = _build_user_prompt(task_desc, obs, admissible_commands)

    return {
        "user_prompt": user_prompt,
        "domain_state": {
            "user_prompt": user_prompt,
            "task_goal": task_desc,
            "current_observation": obs,  # Store text observation directly
            "available_actions": admissible_commands,
            "selected_action": "",
            "decision_reason": "",
            "action_history": [],
            "observation_history": [obs],
            "current_gamefile": gamefile,
            "architect_decision": "",
            "canonical_task_type": task_metadata["canonical_task_type"],
            "canonical_goal_object": task_metadata["canonical_goal_object"],
            "canonical_target_receptacle": task_metadata["canonical_target_receptacle"],
            "workflow_status": "deciding",
            "episode_id": "",
            "step_id": 0,
            "fallback_used": False,
        },
        "task_flow": [],
        "data_schema": {},
        "workflow_rules": {},
        "worker_instructions": {},
        "selected_workers": [],
        "pending_patch": [],
        "patch_error": "",
        "step_count": 0,
        "turn_worker_input_tokens": 0,
        "turn_worker_output_tokens": 0,
        "turn_architect_input_tokens": 0,
        "turn_architect_output_tokens": 0,
        "retry_count": 0,
        "error_feedback": "",
        "no_update_count": 0,
        "status_history": [],
        "conversation_history": [],
        "pending_user_question": "",
        "user_response": "",
        "waiting_for_user": False,
    }


def update_kernel_state(prev_result: Dict, new_obs: str, infos: Dict) -> Dict[str, Any]:
    """
    Update Kernel State with new observation from ALFWorld.

    This function is called after each ALFWorld step to inject the new observation
    into the Kernel State, preparing it for the next Kernel System invocation.

    Args:
        prev_result: Previous Kernel State result
        new_obs: New observation text from ALFWorld
        infos: Info dictionary from ALFWorld

    Returns:
        Updated Kernel State with new observation and cleared selected_action
    """
    # Get previous domain_state
    domain_state = prev_result.get("domain_state", {}).copy()

    # Update current observation
    domain_state["current_observation"] = new_obs

    # Update available actions
    domain_state["available_actions"] = infos["admissible_commands"][0]
    domain_state["current_gamefile"] = _extract_gamefile(infos) or domain_state.get("current_gamefile", "")
    new_task_desc = extract_task_desc(infos)
    if new_task_desc and new_task_desc != _DEFAULT_TASK_DESC:
        domain_state["task_goal"] = new_task_desc
    task_metadata = extract_task_metadata(infos)
    for field, value in task_metadata.items():
        if value:
            domain_state[field] = value
        else:
            domain_state.setdefault(field, "")

    # Add to history
    domain_state.setdefault("observation_history", []).append(new_obs)
    if domain_state.get("selected_action"):
        domain_state.setdefault("action_history", []).append(domain_state["selected_action"])

    # Clear selected_action to avoid re-execution
    domain_state["selected_action"] = ""
    domain_state["decision_reason"] = ""
    domain_state["workflow_status"] = domain_state.get("workflow_entry_status", "deciding")

    # Rebuild user_prompt with new observation and context
    task_desc = domain_state["task_goal"]
    admissible_commands = domain_state["available_actions"]
    user_prompt = _build_user_prompt(
        task_desc,
        new_obs,
        admissible_commands,
        domain_state.get("action_history", []),
    )
    domain_state["user_prompt"] = user_prompt

    # Preserve other fields
    return {
        **prev_result,
        "user_prompt": user_prompt,
        "domain_state": domain_state,
        "pending_patch": [],  # Clear pending_patch
        "patch_error": "",
        "step_count": 0,
        "turn_worker_input_tokens": 0,
        "turn_worker_output_tokens": 0,
        "turn_architect_input_tokens": 0,
        "turn_architect_output_tokens": 0,
        "retry_count": 0,
        "error_feedback": "",
        "no_update_count": 0,
        "status_history": [],
    }


def extract_action_with_meta(result: Dict, admissible_commands: List[str]) -> tuple:
    """
    Extract ALFWorld action from Kernel System result, with fallback metadata.

    Returns:
        (action, fallback_used): action string and whether fallback was triggered
    """
    domain_state = result.get("domain_state", {})
    selected_action = domain_state.get("selected_action", "")

    if selected_action and selected_action in admissible_commands:
        return selected_action, False

    fallback = "look" if "look" in admissible_commands else (admissible_commands[0] if admissible_commands else "")
    print(f"Warning: Invalid or empty action '{selected_action}', using '{fallback}' instead")
    return fallback, True


def extract_action(result: Dict, admissible_commands: List[str]) -> str:
    """
    Extract ALFWorld action from Kernel System result.

    This function reads the selected_action field from domain_state and validates
    it against ALFWorld's admissible_commands.

    Args:
        result: Kernel State result after graph invocation
        admissible_commands: List of valid actions from ALFWorld

    Returns:
        Valid action string to execute in ALFWorld
        Falls back to "look" if action is invalid or empty
    """
    action, _ = extract_action_with_meta(result, admissible_commands)
    return action
