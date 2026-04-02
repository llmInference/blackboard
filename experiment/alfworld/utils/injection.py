"""Invalid patch injection utilities for Experiment 2 (schema guard evaluation)."""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Injection strategy names
# ---------------------------------------------------------------------------
INJECTION_TYPES = [
    "missing_op",
    "replace_missing_path",
    "schema_type_mismatch",
    "invalid_enum",
    "invalid_path_format",
]


def _make_case_id() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Individual injection generators
# ---------------------------------------------------------------------------

def inject_missing_op(domain_state: dict, **_) -> Optional[List[dict]]:
    """Patch item missing the 'op' field."""
    keys = [k for k in domain_state if not k.startswith("_")]
    if not keys:
        return None
    field = keys[0]
    return [{"path": f"/{field}", "value": domain_state[field]}]  # no "op"


def inject_replace_missing_path(domain_state: dict, **_) -> Optional[List[dict]]:
    """replace op on a path that does not exist in domain_state."""
    return [{"op": "replace", "path": "/nonexistent_field_xyz", "value": "injected"}]


def inject_schema_type_mismatch(domain_state: dict, data_schema: dict, **_) -> Optional[List[dict]]:
    """Set a string field to a list (type mismatch)."""
    props = data_schema.get("properties", {}) if data_schema else {}
    for field, spec in props.items():
        if spec.get("type") == "string" and field in domain_state:
            return [{"op": "replace", "path": f"/{field}", "value": [1, 2, 3]}]
    # Fallback: pick first string-valued field from domain_state
    for field, val in domain_state.items():
        if isinstance(val, str) and not field.startswith("_"):
            return [{"op": "replace", "path": f"/{field}", "value": {"injected": True}}]
    return None


def inject_invalid_enum(domain_state: dict, data_schema: dict, **_) -> Optional[List[dict]]:
    """Set an enum-constrained field to a value not in the enum."""
    props = data_schema.get("properties", {}) if data_schema else {}
    for field, spec in props.items():
        if "enum" in spec and field in domain_state:
            bad_value = "__invalid_enum_value__"
            return [{"op": "replace", "path": f"/{field}", "value": bad_value}]
    return None


def inject_invalid_path_format(domain_state: dict, **_) -> Optional[List[dict]]:
    """Malformed JSON Pointer path (missing leading slash)."""
    keys = [k for k in domain_state if not k.startswith("_")]
    if not keys:
        return None
    field = keys[0]
    return [{"op": "replace", "path": f"no_slash_{field}", "value": "bad"}]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_GENERATORS = {
    "missing_op": inject_missing_op,
    "replace_missing_path": inject_replace_missing_path,
    "schema_type_mismatch": inject_schema_type_mismatch,
    "invalid_enum": inject_invalid_enum,
    "invalid_path_format": inject_invalid_path_format,
}


def generate_invalid_patch(
    injection_type: str,
    domain_state: dict,
    data_schema: Optional[dict] = None,
) -> Optional[List[dict]]:
    """
    Generate an invalid patch of the given injection_type.

    Returns a list of patch operations, or None if the strategy cannot
    produce a patch for the given state/schema.
    """
    gen = _GENERATORS.get(injection_type)
    if gen is None:
        raise ValueError(f"Unknown injection type: {injection_type!r}. "
                         f"Valid types: {INJECTION_TYPES}")
    return gen(domain_state=domain_state, data_schema=data_schema or {})


# ---------------------------------------------------------------------------
# Patch error classifier
# ---------------------------------------------------------------------------

def classify_patch_error(patch_error: str) -> str:
    """
    Map a raw patch_error string to a canonical error category.

    Categories:
        json_format_error       – patch item is not valid JSON structure
        json_patch_apply_error  – jsonpatch could not apply the operation
        schema_type_error       – value type violates schema
        schema_enum_error       – value not in allowed enum
        invalid_action_error    – action not in admissible set
        unknown_error           – anything else
    """
    if not patch_error:
        return ""
    err = patch_error.lower()
    if ("missing" in err and "op" in err) or ("does not contain 'op'" in err) or ("缺少" in err and "op" in err):
        return "json_format_error"
    if (
        "patch apply error" in err
        or "jsonpatch" in err
        or "jsonpointer" in err
        or "can't replace" in err
        or "location must start with /" in err
        or "路径不存在" in err
    ):
        return "json_patch_apply_error"
    if ("type" in err and ("schema" in err or "validation" in err)) or "not of type" in err:
        return "schema_type_error"
    if "enum" in err or "is not one of" in err:
        return "schema_enum_error"
    if "action" in err or "admissible" in err:
        return "invalid_action_error"
    return "unknown_error"


# ---------------------------------------------------------------------------
# Injection case builder
# ---------------------------------------------------------------------------

def build_injection_cases(
    captured_states: List[Dict[str, Any]],
    injection_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    For each captured state and each injection type, produce an injection case dict.

    Args:
        captured_states: List of dicts with keys:
            episode_id, gamefile, step_id, domain_state, data_schema
        injection_types: Subset of INJECTION_TYPES to use (default: all)

    Returns:
        List of injection case dicts with keys:
            case_id, episode_id, gamefile, step_id,
            injection_type, patch, domain_state_snapshot, data_schema
    """
    types = injection_types or INJECTION_TYPES
    cases = []
    for state_record in captured_states:
        domain_state = state_record.get("domain_state", {})
        data_schema = state_record.get("data_schema", {})
        for inj_type in types:
            patch = generate_invalid_patch(inj_type, domain_state, data_schema)
            if patch is None:
                continue
            cases.append({
                "case_id": _make_case_id(),
                "episode_id": state_record.get("episode_id", ""),
                "gamefile": state_record.get("gamefile", ""),
                "step_id": state_record.get("step_id", 0),
                "injection_type": inj_type,
                "patch": patch,
                "domain_state_snapshot": copy.deepcopy(domain_state),
                "data_schema": copy.deepcopy(data_schema),
            })
    return cases
