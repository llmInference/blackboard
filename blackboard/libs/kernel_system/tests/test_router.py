from langgraph.graph import END

from langgraph_kernel.kernel.router import WorkflowRouter
from langgraph_kernel.workflow import determine_next_status

WORKERS = ["planner_worker", "executor_worker"]


def make_state(domain_state, patch_error="", step_count=0, workflow_rules=None):
    return {
        "domain_state": domain_state,
        "patch_error": patch_error,
        "step_count": step_count,
        "workflow_rules": workflow_rules
        or {
            "status": {
                "planning": "planner_worker",
                "executing": "executor_worker",
            }
        },
    }


def test_routes_to_correct_worker():
    router = WorkflowRouter(WORKERS)
    state = make_state({"status": "planning"})
    assert router.route(state) == "planner_worker"


def test_routes_structured_rule_to_worker():
    router = WorkflowRouter(WORKERS)
    state = make_state(
        {"workflow_status": "planning"},
        workflow_rules={
            "workflow_status": {
                "planning": {"worker": "planner_worker", "next": "executing"},
                "executing": {"worker": "executor_worker", "next": "completed"},
                "completed": None,
            }
        },
    )
    assert router.route(state) == "planner_worker"


def test_routes_to_end_on_structured_terminal_rule():
    router = WorkflowRouter(WORKERS)
    state = make_state(
        {"workflow_status": "completed"},
        workflow_rules={
            "workflow_status": {
                "planning": {"worker": "planner_worker", "next": "completed"},
                "completed": None,
            }
        },
    )
    assert router.route(state) == END


def test_routes_to_end_on_patch_error():
    router = WorkflowRouter(WORKERS)
    state = make_state({"status": "planning"}, patch_error="some error")
    assert router.route(state) == END


def test_routes_to_end_on_max_steps():
    router = WorkflowRouter(WORKERS, max_steps=5)
    state = make_state({"status": "planning"}, step_count=5)
    assert router.route(state) == END


def test_routes_to_end_when_no_match():
    router = WorkflowRouter(WORKERS)
    state = make_state({"status": "done"})
    assert router.route(state) == END


def test_determine_next_status_uses_conditional_branch():
    workflow_rules = {
        "workflow_status": {
            "planning": {
                "worker": "planner_worker",
                "next": {
                    "cases": [
                        {
                            "when": {"field": "failed_search_locations", "truthy": True},
                            "target": "replanning",
                        }
                    ],
                    "default": "executing",
                },
            },
            "replanning": {"worker": "planner_worker", "next": "executing"},
            "executing": {"worker": "executor_worker", "next": "completed"},
            "completed": None,
        }
    }

    assert determine_next_status(
        {"workflow_status": "planning", "failed_search_locations": ["drawer 1"]},
        workflow_rules,
        "planner_worker",
    ) == "replanning"
    assert determine_next_status(
        {"workflow_status": "planning", "failed_search_locations": []},
        workflow_rules,
        "planner_worker",
    ) == "executing"
