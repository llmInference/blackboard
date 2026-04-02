"""Test LLMActionWorker._think() with use_json_patch=False."""
import sys
sys.path.insert(0, "/home/syq/Documents/blackboard/blackboard/libs/kernel_system")
sys.path.insert(0, "/home/syq/Documents/blackboard/experiment/alfworld")

from utils.mock_llm import ALFWorldMockActionLLM
from workers.llm_action_worker import LLMActionWorker

CONTEXT = {
    "task_goal": "put a hot apple in fridge",
    "current_observation": "You are in the kitchen. You see a microwave, a fridge, and a counter.",
    "available_actions": [
        "go to microwave 1",
        "go to fridge 1",
        "go to counter 1",
        "open microwave 1",
        "look",
    ],
    "subgoal": "heat_goal_object",
    "focus_object": "apple",
    "focus_receptacle": "microwave",
    "required_transform": "heat",
    "recommended_actions": ["go to microwave 1"],
}

llm = ALFWorldMockActionLLM()
worker = LLMActionWorker(llm=llm, instruction="test")

print("=== use_json_patch=True ===")
result = worker._think(CONTEXT, use_json_patch=True)
print(f"type: {type(result)}")
print(f"result: {result}")

print("\n=== use_json_patch=False ===")
result = worker._think(CONTEXT, use_json_patch=False)
print(f"type: {type(result)}")
print(f"result: {result}")

# 验证 selected_action 不为空
if isinstance(result, list):
    actions = {op["path"]: op["value"] for op in result if "path" in op and "value" in op}
    selected = actions.get("/selected_action", "")
    print(f"\nselected_action: '{selected}'")
    assert selected, "FAIL: selected_action is empty!"
    assert selected in CONTEXT["available_actions"], f"FAIL: '{selected}' not in available_actions"
    print("PASS: selected_action is valid")
else:
    print(f"FAIL: result is not a list, got: {type(result)}: {result!r}")
