"""Tests for ALFWorld-specific architect and worker nodes."""
from workers.action_worker import ActionWorker
from workers.alfworld_architect import ALFWorldArchitect
from workers.llm_alfworld_architect import LLMALFWorldArchitect
from workers.planner_worker import PlannerWorker
from utils.mock_llm import ALFWorldMockActionLLM


def test_action_worker_selects_admissible_action():
    worker = ActionWorker()
    context = {
        "task_goal": "Put the apple in the fridge",
        "current_observation": "You are in the kitchen. You see an apple on the table.",
        "available_actions": ["look", "take apple 1", "go to fridge 1"],
        "selected_action": "",
        "action_history": [],
    }

    patch = worker._think(context)

    assert any(op["path"] == "/selected_action" for op in patch)
    selected = next(op["value"] for op in patch if op["path"] == "/selected_action")
    assert selected in context["available_actions"]


def test_alfworld_architect_preserves_domain_state():
    architect = ALFWorldArchitect()
    state = {
        "user_prompt": "Task: Put the apple in the fridge",
        "domain_state": {
            "user_prompt": "Task: Put the apple in the fridge",
            "task_goal": "Put the apple in the fridge",
            "current_observation": "You are in the kitchen.",
            "available_actions": ["look"],
            "selected_action": "",
            "action_history": [],
            "observation_history": ["You are in the kitchen."],
            "current_gamefile": "/tmp/game.tw-pddl",
        },
    }

    result = architect(state)

    assert result["selected_workers"] == ["action_worker"]
    assert result["workflow_rules"]["workflow_status"]["deciding"] == "action_worker"
    assert result["domain_state"]["task_goal"] == "Put the apple in the fridge"
    assert result["domain_state"]["workflow_status"] == "deciding"


def test_action_worker_prefers_goal_object_over_unrelated_take():
    worker = ActionWorker()
    context = {
        "task_goal": "place a knife on the microwave oven table",
        "current_observation": "You see a mug on the sidetable and a knife on the countertop.",
        "available_actions": [
            "take mug 1 from sidetable 1",
            "take knife 1 from countertop 1",
        ],
        "selected_action": "",
        "action_history": [],
        "current_gamefile": "/tmp/pick_and_place_simple-Knife-None-SideTable-3/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    selected = next(op["value"] for op in patch if op["path"] == "/selected_action")
    assert selected == "take knife 1 from countertop 1"


def test_action_worker_prefers_transform_path_when_goal_object_held():
    worker = ActionWorker()
    context = {
        "task_goal": "Place a tomato back in the refrigerator when done heating in the microwave.",
        "current_observation": "You are in front of the fridge.",
        "available_actions": [
            "move tomato 1 to fridge 1",
            "go to microwave 1",
            "open microwave 1",
        ],
        "selected_action": "",
        "action_history": ["take tomato 1 from fridge 1"],
        "current_gamefile": "/tmp/pick_heat_then_place_in_recep-Tomato-None-Fridge-23/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    selected = next(op["value"] for op in patch if op["path"] == "/selected_action")
    assert selected == "go to microwave 1"


def test_action_worker_moves_to_target_after_transform_done():
    worker = ActionWorker()
    context = {
        "task_goal": "Place a cooled mug on a coffee machine.",
        "current_observation": "You are near the fridge holding a mug.",
        "available_actions": [
            "cool mug 1 with fridge 1",
            "go to coffeemachine 1",
            "move mug 1 to coffeemachine 1",
        ],
        "selected_action": "",
        "action_history": [
            "take mug 1 from coffeemachine 1",
            "go to fridge 1",
            "open fridge 1",
            "cool mug 1 with fridge 1",
        ],
        "current_gamefile": "/tmp/pick_cool_then_place_in_recep-Mug-None-CoffeeMachine-16/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    selected = next(op["value"] for op in patch if op["path"] == "/selected_action")
    assert selected == "move mug 1 to coffeemachine 1"


def test_action_worker_examines_goal_after_light_activation():
    worker = ActionWorker()
    context = {
        "task_goal": "Grab the disc from the shelf, turn on the lamp on the cabinet",
        "current_observation": "You are by the cabinet lamp holding the disc.",
        "available_actions": [
            "use desklamp 1",
            "examine cd 1",
            "go to shelf 3",
        ],
        "selected_action": "",
        "action_history": [
            "take cd 1 from shelf 3",
            "go to desklamp 1",
            "use desklamp 1",
        ],
        "current_gamefile": "/tmp/look_at_obj_in_light-CD-None-DeskLamp-302/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    selected = next(op["value"] for op in patch if op["path"] == "/selected_action")
    assert selected == "examine cd 1"


def test_planner_worker_sets_structured_subgoal_for_transform_task():
    worker = PlannerWorker()
    context = {
        "task_goal": "Place a cooled mug on a coffee machine.",
        "current_observation": "You are holding a mug.",
        "available_actions": ["go to fridge 1", "cool mug 1 with fridge 1"],
        "action_history": ["take mug 1 from coffeemachine 1"],
        "current_gamefile": "/tmp/pick_cool_then_place_in_recep-Mug-None-CoffeeMachine-16/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    values = {op["path"]: op["value"] for op in patch}
    assert values["/subgoal"] == "cool_goal_object"
    assert values["/focus_object"] == "Mug"
    assert values["/focus_receptacle"] == "fridge"
    assert values["/required_transform"] == "cool"


def test_planner_worker_recommends_pivot_after_repeated_empty_drawers():
    worker = PlannerWorker()
    context = {
        "task_goal": "place a knife on the microwave oven table",
        "current_observation": "You arrive at drawer 3. The drawer 3 is closed.",
        "available_actions": [
            "go to cabinet 1",
            "go to countertop 1",
            "go to drawer 4",
            "go to sinkbasin 1",
            "open drawer 3",
            "look",
        ],
        "action_history": [
            "go to drawer 1",
            "open drawer 1",
            "go to drawer 2",
            "open drawer 2",
            "go to drawer 3",
        ],
        "observation_history": [
            "You are in the middle of a room.",
            "You arrive at drawer 1. The drawer 1 is closed.",
            "You open the drawer 1. The drawer 1 is open. In it, you see nothing.",
            "You arrive at drawer 2. The drawer 2 is closed.",
            "You open the drawer 2. The drawer 2 is open. In it, you see nothing.",
            "You arrive at drawer 3. The drawer 3 is closed.",
        ],
        "current_gamefile": "/tmp/pick_and_place_simple-Knife-None-SideTable-3/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    values = {op["path"]: op["value"] for op in patch}

    assert values["/searched_locations"] == ["drawer 1", "drawer 2", "drawer 3"]
    assert values["/failed_search_locations"] == ["drawer 1", "drawer 2"]
    assert values["/recommended_actions"][0] == "go to countertop 1"
    assert "pivot" in values["/search_guidance"].lower()
    assert "drawer" in values["/search_guidance"].lower()


def test_planner_worker_marks_focus_object_as_canonical_when_task_text_uses_alias():
    worker = PlannerWorker()
    context = {
        "task_goal": "put clean spoon on the table",
        "current_observation": (
            "You arrive at countertop 2. On the countertop 2, you see a ladle 2 and a spoon 1."
        ),
        "available_actions": [
            "take ladle 2 from countertop 2",
            "take spoon 1 from countertop 2",
            "go to sinkbasin 1",
        ],
        "action_history": [
            "go to countertop 1",
            "go to sinkbasin 1",
            "go to countertop 2",
        ],
        "observation_history": [
            "You are in the middle of a room.",
            "You arrive at countertop 1. On the countertop 1, you see nothing.",
            "You arrive at sinkbasin 1. On the sinkbasin 1, you see a cup 1.",
            "You arrive at countertop 2. On the countertop 2, you see a ladle 2 and a spoon 1.",
        ],
        "current_gamefile": "/tmp/pick_clean_then_place_in_recep-Ladle-None-DiningTable-27/trial_1/game.tw-pddl",
    }

    patch = worker._think(context)
    values = {op["path"]: op["value"] for op in patch}

    assert values["/focus_object"] == "Ladle"
    assert values["/recommended_actions"][0] == "take ladle 2 from countertop 2"
    assert "canonical goal object: Ladle".lower() in values["/goal_object_guidance"].lower()


def test_alfworld_architect_planner_action_mode_uses_two_workers():
    architect = ALFWorldArchitect(workflow_mode="planner_action")
    result = architect(
        {
            "user_prompt": "Task: Put the apple in the fridge",
            "domain_state": {
                "task_goal": "Put the apple in the fridge",
                "current_observation": "You are in the kitchen.",
                "available_actions": ["look"],
                "action_history": [],
                "observation_history": ["You are in the kitchen."],
            },
        }
    )

    assert result["selected_workers"] == ["planner_worker", "action_worker"]
    assert result["workflow_rules"]["workflow_status"]["planning"] == "planner_worker"
    assert result["workflow_rules"]["workflow_status"]["acting"] == "action_worker"
    assert result["domain_state"]["workflow_status"] == "planning"
    assert result["domain_state"]["workflow_entry_status"] == "planning"


def test_alfworld_architect_planner_llm_action_mode_uses_llm_worker():
    architect = ALFWorldArchitect(workflow_mode="planner_llm_action")
    result = architect(
        {
            "user_prompt": "Task: Put the apple in the fridge",
            "domain_state": {
                "task_goal": "Put the apple in the fridge",
                "current_observation": "You are in the kitchen.",
                "available_actions": ["look"],
                "action_history": [],
                "observation_history": ["You are in the kitchen."],
                "canonical_goal_object": "Apple",
                "canonical_target_receptacle": "Fridge",
            },
        }
    )

    assert result["selected_workers"] == ["planner_worker", "llm_action_worker"]
    assert result["workflow_rules"]["workflow_status"]["acting"] == "llm_action_worker"
    assert "/selected_action" in result["worker_instructions"]["llm_action_worker"]
    assert "available_actions" in result["worker_instructions"]["llm_action_worker"]
    assert "recommended_actions" in result["worker_instructions"]["llm_action_worker"]
    assert "goal_object_guidance" in result["worker_instructions"]["llm_action_worker"]
    assert "recommended_actions" in result["data_schema"]["properties"]
    assert "goal_object_guidance" in result["data_schema"]["properties"]
    assert result["domain_state"]["canonical_goal_object"] == "Apple"
    assert result["domain_state"]["canonical_target_receptacle"] == "Fridge"
    assert result["domain_state"]["recommended_actions"] == []


def test_llm_alfworld_architect_specializes_worker_instructions():
    architect = LLMALFWorldArchitect(ALFWorldMockActionLLM(), workflow_mode="planner_llm_action")
    result = architect(
        {
            "user_prompt": "Task: Put the apple in the fridge",
            "domain_state": {
                "task_goal": "Put the apple in the fridge",
                "current_observation": "You are in the kitchen.",
                "available_actions": ["look", "take apple 1", "go to fridge 1"],
                "action_history": [],
                "observation_history": ["You are in the kitchen."],
                "canonical_goal_object": "Apple",
                "canonical_target_receptacle": "Fridge",
            },
        }
    )

    assert result["selected_workers"] == ["planner_worker", "llm_action_worker"]
    assert "Apple" in result["domain_state"]["architect_decision"]
    assert "Apple" in result["worker_instructions"]["llm_action_worker"]

    planning_rule = result["workflow_rules"]["workflow_status"]["planning"]
    assert planning_rule["worker"] == "planner_worker"
    assert planning_rule["next"]["default"] == "acting"
    assert result["workflow_rules"]["workflow_status"]["replanning"]["worker"] == "planner_worker"
    assert any(task.get("state") == "replanning" for task in result["task_flow"])
