import pytest
from langgraph_kernel.kernel.validator import PatchValidator

SCHEMA = {
    "type": "object",
    "required": ["status"],
    "properties": {
        "status": {"type": "string", "enum": ["init", "planning", "done"]},
        "plan": {"type": "array", "items": {"type": "string"}},
    },
}

validator = PatchValidator()


def test_valid_patch():
    state = {"status": "init"}
    patch = [{"op": "replace", "path": "/status", "value": "planning"}]
    new_state, error = validator.validate(state, patch, SCHEMA)
    assert error is None
    assert new_state["status"] == "planning"


def test_invalid_patch_operation():
    state = {"status": "init"}
    patch = [{"op": "remove", "path": "/nonexistent"}]
    new_state, error = validator.validate(state, patch, SCHEMA)
    assert error is not None
    assert "patch apply error" in error
    assert new_state == state  # 原始状态不变


def test_schema_violation():
    state = {"status": "init"}
    patch = [{"op": "replace", "path": "/status", "value": "invalid_value"}]
    new_state, error = validator.validate(state, patch, SCHEMA)
    assert error is not None
    assert "schema validation error" in error
    assert new_state == state


def test_add_optional_field():
    state = {"status": "init"}
    patch = [{"op": "add", "path": "/plan", "value": ["step1", "step2"]}]
    new_state, error = validator.validate(state, patch, SCHEMA)
    assert error is None
    assert new_state["plan"] == ["step1", "step2"]
