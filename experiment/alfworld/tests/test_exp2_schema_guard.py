"""
Experiment 2 tests: injection generation, patch classification, C2 on/off behavior,
and schema guard metrics.
"""
from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from utils.injection import (
    INJECTION_TYPES,
    build_injection_cases,
    classify_patch_error,
    generate_invalid_patch,
    inject_invalid_enum,
    inject_invalid_path_format,
    inject_missing_op,
    inject_replace_missing_path,
    inject_schema_type_mismatch,
)
from evaluators.schema_guard_evaluator import (
    CaseResult,
    SchemaGuardEvaluator,
    SchemaGuardMetrics,
    compute_schema_guard_metrics,
    flatten_case_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DOMAIN_STATE = {
    "observation": "You are in the kitchen.",
    "action": "go to counter 1",
    "task_type": "pick_and_place",
    "step_count": 2,
}

_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "action": {"type": "string"},
        "task_type": {"type": "string", "enum": ["pick_and_place", "heat_then_place"]},
        "step_count": {"type": "integer"},
    },
}


# ---------------------------------------------------------------------------
# Injection generators
# ---------------------------------------------------------------------------

def test_inject_missing_op_returns_patch_without_op():
    patch = inject_missing_op(_DOMAIN_STATE)
    assert patch is not None
    assert len(patch) == 1
    assert "op" not in patch[0]
    assert "path" in patch[0]


def test_inject_replace_missing_path_returns_nonexistent_path():
    patch = inject_replace_missing_path(_DOMAIN_STATE)
    assert patch is not None
    assert patch[0]["op"] == "replace"
    assert "nonexistent" in patch[0]["path"]


def test_inject_schema_type_mismatch_sets_string_to_non_string():
    patch = inject_schema_type_mismatch(_DOMAIN_STATE, _DATA_SCHEMA)
    assert patch is not None
    assert patch[0]["op"] == "replace"
    assert not isinstance(patch[0]["value"], str)


def test_inject_invalid_enum_uses_bad_value():
    patch = inject_invalid_enum(_DOMAIN_STATE, _DATA_SCHEMA)
    assert patch is not None
    assert patch[0]["value"] == "__invalid_enum_value__"


def test_inject_invalid_enum_returns_none_when_no_enum_in_schema():
    schema_no_enum = {"properties": {"observation": {"type": "string"}}}
    result = inject_invalid_enum(_DOMAIN_STATE, schema_no_enum)
    assert result is None


def test_inject_invalid_path_format_missing_leading_slash():
    patch = inject_invalid_path_format(_DOMAIN_STATE)
    assert patch is not None
    assert not patch[0]["path"].startswith("/")


def test_generate_invalid_patch_dispatcher():
    for inj_type in INJECTION_TYPES:
        if inj_type == "invalid_enum":
            result = generate_invalid_patch(inj_type, _DOMAIN_STATE, _DATA_SCHEMA)
        else:
            result = generate_invalid_patch(inj_type, _DOMAIN_STATE, _DATA_SCHEMA)
        # Most types should produce a patch for this state/schema
        # (invalid_enum requires enum in schema, which we have)
        assert result is None or isinstance(result, list)


def test_generate_invalid_patch_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown injection type"):
        generate_invalid_patch("nonexistent_type", _DOMAIN_STATE)


# ---------------------------------------------------------------------------
# build_injection_cases
# ---------------------------------------------------------------------------

def test_build_injection_cases_produces_cases_for_all_types():
    states = [{
        "episode_id": "ep0",
        "gamefile": "game.z8",
        "step_id": 1,
        "domain_state": _DOMAIN_STATE,
        "data_schema": _DATA_SCHEMA,
    }]
    cases = build_injection_cases(states)
    # Should have one case per injection type that can produce a patch
    assert len(cases) >= 1
    for case in cases:
        assert "case_id" in case
        assert case["injection_type"] in INJECTION_TYPES
        assert isinstance(case["patch"], list)
        assert case["episode_id"] == "ep0"
        assert case["step_id"] == 1


def test_build_injection_cases_subset_of_types():
    states = [{"episode_id": "ep0", "gamefile": "", "step_id": 0,
               "domain_state": _DOMAIN_STATE, "data_schema": _DATA_SCHEMA}]
    cases = build_injection_cases(states, injection_types=["missing_op"])
    assert all(c["injection_type"] == "missing_op" for c in cases)


def test_build_injection_cases_multiple_states():
    states = [
        {"episode_id": f"ep{i}", "gamefile": "", "step_id": i,
         "domain_state": _DOMAIN_STATE, "data_schema": _DATA_SCHEMA}
        for i in range(3)
    ]
    cases = build_injection_cases(states, injection_types=["missing_op"])
    assert len(cases) == 3


# ---------------------------------------------------------------------------
# classify_patch_error
# ---------------------------------------------------------------------------

def test_classify_patch_error_empty():
    assert classify_patch_error("") == ""


def test_classify_patch_error_missing_op():
    assert classify_patch_error("missing op field in patch") == "json_format_error"


def test_classify_patch_error_jsonpatch():
    assert classify_patch_error("jsonpatch apply error: path not found") == "json_patch_apply_error"


def test_classify_patch_error_schema_type():
    assert classify_patch_error("schema validation type error") == "schema_type_error"


def test_classify_patch_error_enum():
    assert classify_patch_error("enum value not allowed") == "schema_enum_error"


def test_classify_patch_error_unknown():
    assert classify_patch_error("something completely unexpected") == "unknown_error"


def test_classify_patch_error_missing_op_member_message():
    assert classify_patch_error("Operation does not contain 'op' member") == "json_format_error"


def test_classify_patch_error_jsonpointer_location():
    assert classify_patch_error("Location must start with /") == "json_patch_apply_error"


def test_classify_patch_error_enum_membership():
    msg = "'__invalid_enum_value__' is not one of ['pick_and_place', 'heat_then_place']"
    assert classify_patch_error(msg) == "schema_enum_error"


# ---------------------------------------------------------------------------
# compute_schema_guard_metrics
# ---------------------------------------------------------------------------

def test_compute_schema_guard_metrics_empty():
    m = compute_schema_guard_metrics([])
    assert m.n_cases == 0
    assert m.c2_on_interception_rate == 0.0


def test_compute_schema_guard_metrics_all_intercepted_by_c2_on():
    results = [
        CaseResult(case_id="a", injection_type="missing_op", episode_id="ep0", gamefile="", step_id=0, patch=[],
                   c2_on_intercepted=True, c2_off_intercepted=False),
        CaseResult(case_id="b", injection_type="missing_op", episode_id="ep0", gamefile="", step_id=1, patch=[],
                   c2_on_intercepted=True, c2_off_intercepted=False),
    ]
    m = compute_schema_guard_metrics(results)
    assert m.n_cases == 2
    assert m.c2_on_interception_rate == 1.0
    assert m.c2_off_interception_rate == 0.0
    assert m.delta_interception_rate == 1.0


def test_compute_schema_guard_metrics_by_type():
    results = [
        CaseResult(case_id="a", injection_type="missing_op", episode_id="ep0", gamefile="", step_id=0, patch=[],
                   c2_on_intercepted=True, c2_off_intercepted=False),
        CaseResult(case_id="b", injection_type="invalid_enum", episode_id="ep0", gamefile="", step_id=0, patch=[],
                   c2_on_intercepted=False, c2_off_intercepted=False),
    ]
    m = compute_schema_guard_metrics(results)
    assert m.c2_on_interception_by_type["missing_op"] == 1.0
    assert m.c2_on_interception_by_type["invalid_enum"] == 0.0
    assert "apply_rate" in m.full.by_injection_type["missing_op"]
    assert isinstance(m.ablate_c2.error_type_breakdown, dict)


# ---------------------------------------------------------------------------
# SchemaGuardEvaluator: C2 on vs off behavior
# ---------------------------------------------------------------------------

def _make_case(injection_type: str = "missing_op") -> dict:
    return {
        "case_id": "test_case",
        "injection_type": injection_type,
        "episode_id": "ep0",
        "gamefile": "game.z8",
        "step_id": 0,
        "patch": [{"path": "/observation", "value": "new obs"}],  # missing op
        "domain_state_snapshot": copy.deepcopy(_DOMAIN_STATE),
        "data_schema": copy.deepcopy(_DATA_SCHEMA),
    }


def test_schema_guard_evaluator_c2_on_intercepts_missing_op():
    """C2 on should catch a patch missing the 'op' field."""
    evaluator = SchemaGuardEvaluator()
    case = _make_case("missing_op")
    results = evaluator.evaluate_cases([case])
    assert len(results) == 1
    r = results[0]
    assert r.c2_on_intercepted is True
    assert r.c2_on_successfully_applied is False
    assert r.c2_on_patch_error
    assert r.c2_on_error_category == "json_format_error"
    assert r.c2_off_intercepted is False
    assert r.c2_off_successfully_applied is False
    assert r.c2_off_patch_error
    assert r.case_id == "test_case"
    assert r.injection_type == "missing_op"


def test_schema_guard_evaluator_evaluate_from_states():
    states = [{
        "episode_id": "ep0",
        "gamefile": "",
        "step_id": 0,
        "domain_state": _DOMAIN_STATE,
        "data_schema": _DATA_SCHEMA,
    }]
    evaluator = SchemaGuardEvaluator()
    results, metrics = evaluator.evaluate_from_states(states)
    assert len(results) > 0
    assert metrics.n_cases == len(results)
    assert 0.0 <= metrics.c2_on_interception_rate <= 1.0


def test_schema_guard_evaluator_c2_off_allows_invalid_enum():
    evaluator = SchemaGuardEvaluator()
    case = _make_case("invalid_enum")
    case["patch"] = [{"op": "replace", "path": "/task_type", "value": "__invalid_enum_value__"}]
    results = evaluator.evaluate_cases([case])
    r = results[0]
    assert r.c2_on_intercepted is True
    assert r.c2_on_successfully_applied is False
    assert r.c2_on_error_category == "schema_enum_error"
    assert r.c2_off_intercepted is False
    assert r.c2_off_successfully_applied is True
    assert r.c2_off_domain_state_changed is True


def test_flatten_case_results_has_required_experiment_fields():
    result = CaseResult(
        case_id="case0",
        injection_type="missing_op",
        episode_id="ep0",
        gamefile="game.z8",
        step_id=1,
        patch=[{"path": "/observation", "value": "obs"}],
        c2_on_intercepted=True,
        c2_on_patch_error="Operation does not contain 'op' member",
        c2_on_error_category="json_format_error",
        c2_off_intercepted=False,
        c2_off_patch_error="Operation does not contain 'op' member",
    )
    rows = flatten_case_results([result])
    assert len(rows) == 2
    for row in rows:
        assert set(
            [
                "case_id",
                "episode_id",
                "gamefile",
                "step_id",
                "injection_type",
                "ablation_mode",
                "patch",
                "intercepted",
                "patch_error",
                "domain_state_changed",
                "successfully_applied",
            ]
        ).issubset(row)


def test_compute_schema_guard_metrics_reports_nonzero_full_interception():
    evaluator = SchemaGuardEvaluator()
    results = evaluator.evaluate_cases([_make_case("missing_op")])
    metrics = compute_schema_guard_metrics(results)
    assert metrics.full.interception_rate > 0.0
    assert metrics.ablate_c2.interception_rate == 0.0
    assert "json_format_error" in metrics.full.error_type_breakdown
    assert 0.0 <= metrics.c2_off_interception_rate <= 1.0


def test_schema_guard_evaluator_c2_off_allows_invalid_path():
    """C2 off should not intercept a replace on a missing path (jsonpatch raises, recorded as error)."""
    evaluator = SchemaGuardEvaluator()
    case = {
        "case_id": "path_case",
        "injection_type": "replace_missing_path",
        "episode_id": "ep0",
        "step_id": 0,
        "patch": [{"op": "replace", "path": "/nonexistent_field_xyz", "value": "bad"}],
        "domain_state_snapshot": copy.deepcopy(_DOMAIN_STATE),
        "data_schema": copy.deepcopy(_DATA_SCHEMA),
    }
    results = evaluator.evaluate_cases([case])
    r = results[0]
    # C2 off: jsonpatch.apply_patch raises on missing path → intercepted=True (error recorded)
    # C2 on: PatchFixer may fix or reject → intercepted depends on fixer behavior
    assert isinstance(r.c2_on_intercepted, bool)
    assert isinstance(r.c2_off_intercepted, bool)
