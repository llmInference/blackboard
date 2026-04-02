from __future__ import annotations

from langgraph_kernel.types import KernelState


class FixedPipelineArchitect:
    """Deterministic architect used when the LLM-based architect is ablated."""

    FIXED_WORKFLOW = {
        "status": {
            "analyzing": "analyzer",
            "planning": "planner",
            "executing": "executor",
            "reviewing": "reviewer",
            "done": None,
        }
    }

    FIXED_SCHEMA = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "user_prompt": {"type": "string"},
            "result": {"type": "string"},
            "plan": {"type": "array", "items": {"type": "string"}},
            "analysis": {"type": "string"},
            "review": {"type": "string"},
        },
        "required": ["status", "user_prompt"],
    }

    def __call__(self, state: KernelState) -> dict:
        domain_state = state.get("domain_state") or {}
        user_prompt = domain_state.get("user_prompt", "")

        return {
            "task_flow": [
                {"subtask": "analyze", "worker": "analyzer"},
                {"subtask": "plan", "worker": "planner"},
                {"subtask": "execute", "worker": "executor"},
                {"subtask": "review", "worker": "reviewer"},
            ],
            "data_schema": self.FIXED_SCHEMA,
            "workflow_rules": self.FIXED_WORKFLOW,
            "worker_instructions": {},
            "selected_workers": ["analyzer", "planner", "executor", "reviewer"],
            "domain_state": {"user_prompt": user_prompt, "status": "analyzing"},
            "pending_patch": [],
            "patch_error": "",
            "step_count": 0,
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
            "conversation_history": [],
            "pending_user_question": "",
            "user_response": "",
            "waiting_for_user": False,
        }
