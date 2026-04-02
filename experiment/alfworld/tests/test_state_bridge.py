"""
Unit tests for state_bridge module.

Tests the conversion functions between ALFWorld and Kernel System states.
"""
import json
import pytest
from core.state_bridge import extract_action, obs_to_kernel_state, update_kernel_state


def test_obs_to_kernel_state():
    """Test converting ALFWorld observation to Kernel State."""
    obs = "You are in the kitchen. You see a apple 1 on the table 1."
    infos = {
        "extra.goal": ["Put the apple in the fridge"],
        "admissible_commands": [["go to table 1", "go to fridge 1", "look"]]
    }

    state = obs_to_kernel_state(obs, infos)

    # Check user_prompt contains all necessary information
    assert "user_prompt" in state
    assert "Task: Put the apple in the fridge" in state["user_prompt"]
    assert obs in state["user_prompt"]
    assert "go to table 1" in state["user_prompt"]

    # Check domain_state structure
    assert state["domain_state"]["current_observation"] == obs
    assert state["domain_state"]["task_goal"] == "Put the apple in the fridge"
    assert state["domain_state"]["available_actions"] == ["go to table 1", "go to fridge 1", "look"]
    assert state["domain_state"]["selected_action"] == ""
    assert state["domain_state"]["user_prompt"] == state["user_prompt"]
    assert state["domain_state"]["action_history"] == []
    assert state["domain_state"]["observation_history"] == [obs]
    assert state["domain_state"]["workflow_status"] == "deciding"

    # Check other Kernel System fields
    assert state["task_flow"] == []
    assert state["data_schema"] == {}
    assert state["workflow_rules"] == {}
    assert state["worker_instructions"] == {}
    assert state["selected_workers"] == []
    assert state["pending_patch"] == []
    assert state["step_count"] == 0


def test_update_kernel_state():
    """Test updating Kernel State with new observation."""
    # Create initial state
    prev_result = {
        "domain_state": {
            "task_goal": "Put the apple in the fridge",
            "current_observation": "You are in the kitchen.",
            "available_actions": ["go to table 1", "look"],
            "selected_action": "go to table 1",
            "decision_reason": "heading to table",
            "action_history": [],
            "observation_history": ["You are in the kitchen."],
            "workflow_status": "completed",
        },
        "step_count": 0
    }

    new_obs = "You are now at table 1. You see a apple 1 on the table 1."
    infos = {
        "admissible_commands": [["take apple 1", "examine apple 1", "look"]]
    }

    updated_state = update_kernel_state(prev_result, new_obs, infos)

    # Check observation updated
    assert updated_state["domain_state"]["current_observation"] == new_obs

    # Check available actions updated
    assert updated_state["domain_state"]["available_actions"] == ["take apple 1", "examine apple 1", "look"]

    # Check history updated
    assert "go to table 1" in updated_state["domain_state"]["action_history"]
    assert new_obs in updated_state["domain_state"]["observation_history"]

    # Check selected_action cleared
    assert updated_state["domain_state"]["selected_action"] == ""
    assert updated_state["domain_state"]["decision_reason"] == ""
    assert updated_state["domain_state"]["workflow_status"] == "deciding"

    # Check runtime counters reset for the next graph invocation
    assert updated_state["step_count"] == 0

    # Check user_prompt updated
    assert new_obs in updated_state["user_prompt"]
    assert "go to table 1" in updated_state["user_prompt"]


def test_obs_to_kernel_state_reads_task_from_gamefile(tmp_path):
    gamefile = tmp_path / "trial_1" / "game.tw-pddl"
    gamefile.parent.mkdir(parents=True)
    gamefile.write_text("{}", encoding="utf-8")
    traj_data = {
        "turk_annotations": {
            "anns": [{"task_desc": "Examine the apple under the lamp."}]
        },
        "pddl_params": {
            "object_target": "Apple",
            "parent_target": "DeskLamp",
        },
    }
    (gamefile.parent / "traj_data.json").write_text(json.dumps(traj_data), encoding="utf-8")

    state = obs_to_kernel_state(
        "You are in a room.",
        {
            "extra.gamefile": [str(gamefile)],
            "admissible_commands": [["look"]],
        },
    )

    assert state["domain_state"]["task_goal"] == "Examine the apple under the lamp."
    assert state["domain_state"]["current_gamefile"] == str(gamefile)
    assert state["domain_state"]["canonical_goal_object"] == "Apple"
    assert state["domain_state"]["canonical_target_receptacle"] == "DeskLamp"


def test_update_kernel_state_tolerates_missing_gamefile():
    updated = update_kernel_state(
        {
            "domain_state": {
                "task_goal": "Put the apple in the fridge",
                "current_observation": "You are in the kitchen.",
                "available_actions": ["look"],
                "selected_action": "look",
                "decision_reason": "fallback",
                "action_history": [],
                "observation_history": ["You are in the kitchen."],
                "current_gamefile": "/tmp/game.tw-pddl",
                "canonical_goal_object": "Apple",
                "canonical_target_receptacle": "Fridge",
                "workflow_status": "completed",
            }
        },
        "You are still in the kitchen.",
        {
            "extra.gamefile": [None],
            "admissible_commands": [["look"]],
        },
    )

    assert updated["domain_state"]["current_gamefile"] == "/tmp/game.tw-pddl"
    assert updated["domain_state"]["task_goal"] == "Put the apple in the fridge"
    assert updated["domain_state"]["canonical_goal_object"] == "Apple"
    assert updated["domain_state"]["canonical_target_receptacle"] == "Fridge"


def test_update_kernel_state_uses_workflow_entry_status():
    updated = update_kernel_state(
        {
            "domain_state": {
                "task_goal": "Put the apple in the fridge",
                "current_observation": "You are in the kitchen.",
                "available_actions": ["look"],
                "selected_action": "look",
                "decision_reason": "planning",
                "action_history": [],
                "observation_history": ["You are in the kitchen."],
                "workflow_status": "completed",
                "workflow_entry_status": "planning",
            }
        },
        "You are still in the kitchen.",
        {
            "admissible_commands": [["look"]],
        },
    )

    assert updated["domain_state"]["workflow_status"] == "planning"


def test_extract_action_valid():
    """Test extracting valid action from Kernel State."""
    result = {
        "domain_state": {
            "selected_action": "go to table 1"
        }
    }
    admissible = ["go to table 1", "go to fridge 1", "look"]

    action = extract_action(result, admissible)
    assert action == "go to table 1"


def test_extract_action_invalid():
    """Test extracting invalid action falls back to 'look'."""
    result = {
        "domain_state": {
            "selected_action": "invalid action"
        }
    }
    admissible = ["go to table 1", "look"]

    action = extract_action(result, admissible)
    assert action == "look"


def test_extract_action_empty():
    """Test extracting empty action falls back to 'look'."""
    result = {
        "domain_state": {
            "selected_action": ""
        }
    }
    admissible = ["go to table 1", "look"]

    action = extract_action(result, admissible)
    assert action == "look"


def test_extract_action_missing_field():
    """Test extracting action when field is missing."""
    result = {
        "domain_state": {}
    }
    admissible = ["go to table 1", "look"]

    action = extract_action(result, admissible)
    assert action == "look"
