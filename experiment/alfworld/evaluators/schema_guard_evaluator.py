"""Schema guard evaluator for Experiment 2 (C2 on vs off comparison)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jsonpatch
from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.kernel.node import kernel_node

from experiment.alfworld.utils.injection import (
    INJECTION_TYPES,
    build_injection_cases,
    classify_patch_error,
)

FULL_MODE = "full"
ABLATE_C2_MODE = "ablate_c2"


@dataclass
class CaseResult:
    case_id: str
    injection_type: str
    episode_id: str
    gamefile: str
    step_id: int
    patch: List[Dict[str, Any]]

    # C2 on
    c2_on_intercepted: bool = False
    c2_on_patch_error: str = ""
    c2_on_domain_state_changed: bool = False
    c2_on_successfully_applied: bool = False
    c2_on_error_category: str = ""

    # C2 off
    c2_off_intercepted: bool = False
    c2_off_patch_error: str = ""
    c2_off_domain_state_changed: bool = False
    c2_off_successfully_applied: bool = False
    c2_off_error_category: str = ""


@dataclass
class ModeMetrics:
    n_cases: int = 0
    interception_rate: float = 0.0
    apply_rate: float = 0.0
    error_type_breakdown: Dict[str, int] = field(default_factory=dict)
    by_injection_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SchemaGuardMetrics:
    n_cases: int = 0
    n_by_type: Dict[str, int] = field(default_factory=dict)

    # Legacy fields kept for compatibility with existing tests/callers.
    c2_on_interception_rate: float = 0.0
    c2_on_interception_by_type: Dict[str, float] = field(default_factory=dict)
    c2_off_interception_rate: float = 0.0
    c2_off_interception_by_type: Dict[str, float] = field(default_factory=dict)
    delta_interception_rate: float = 0.0

    # Experiment 2 reporting fields.
    full: ModeMetrics = field(default_factory=ModeMetrics)
    ablate_c2: ModeMetrics = field(default_factory=ModeMetrics)


def _make_kernel_state(domain_state: dict, data_schema: dict, patch: list, ablation: AblationConfig) -> dict:
    """Build a minimal KernelState dict for kernel_node."""
    return {
        "domain_state": copy.deepcopy(domain_state),
        "data_schema": data_schema or {},
        "pending_patch": copy.deepcopy(patch),
        "ablation_config": ablation,
        "workflow_rules": {},
        "worker_instructions": {},
        "selected_workers": [],
        "task_flow": [],
        "patch_error": "",
        "step_count": 0,
        "current_worker": "test_worker",
        "retry_count": 0,
        "error_feedback": "",
        "no_update_count": 0,
        "status_history": [],
        "state_tree": None,
        "current_node_id": None,
        "circuit_breaker": None,
        "circuit_breaker_triggered": False,
        "circuit_breaker_reason": "",
        "circuit_breaker_details": {},
        "conversation_history": [],
        "pending_user_question": "",
        "user_response": "",
        "waiting_for_user": False,
    }


def _replay_raw_patch(domain_state: dict, patch: list) -> str:
    """Reproduce raw jsonpatch errors for C2-off reporting."""
    try:
        jsonpatch.apply_patch(copy.deepcopy(domain_state), copy.deepcopy(patch))
    except Exception as exc:  # pragma: no cover - branch depends on patch type
        return str(exc)
    return ""


def _extract_patch_error(
    result: Dict[str, Any],
    ablation: AblationConfig,
    original_state: dict,
    patch: list,
) -> str:
    """Normalize kernel output into a single patch_error field for reporting."""
    patch_error = result.get("patch_error", "")
    if patch_error:
        return patch_error

    retry_count = result.get("retry_count", 0)
    error_feedback = result.get("error_feedback", "")
    if retry_count > 0 and error_feedback:
        return error_feedback

    new_domain_state = result.get("domain_state", original_state)
    if not ablation.use_schema_validation and new_domain_state == original_state:
        return _replay_raw_patch(original_state, patch)

    return ""


def _run_case_with_ablation(
    case: Dict[str, Any],
    ablation: AblationConfig,
) -> tuple[bool, str, bool, bool]:
    """
    Run kernel_node for one injection case under the given ablation config.

    Returns:
        (intercepted, patch_error, domain_state_changed, successfully_applied)
    """
    state = _make_kernel_state(
        case["domain_state_snapshot"],
        case["data_schema"],
        case["patch"],
        ablation,
    )
    original_state = copy.deepcopy(case["domain_state_snapshot"])
    result = kernel_node(state)

    new_domain_state = result.get("domain_state", original_state)
    domain_state_changed = new_domain_state != original_state
    patch_error = _extract_patch_error(result, ablation, original_state, case["patch"])
    intercepted = bool(result.get("patch_error")) or bool(
        result.get("retry_count", 0) > 0 and result.get("error_feedback", "")
    )
    successfully_applied = domain_state_changed and not intercepted and not patch_error

    return intercepted, patch_error, domain_state_changed, successfully_applied


def flatten_case_results(results: List[CaseResult]) -> List[Dict[str, Any]]:
    """Expand paired C2-on/C2-off results into per-mode records."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "case_id": result.case_id,
                "episode_id": result.episode_id,
                "gamefile": result.gamefile,
                "step_id": result.step_id,
                "injection_type": result.injection_type,
                "ablation_mode": FULL_MODE,
                "patch": copy.deepcopy(result.patch),
                "intercepted": result.c2_on_intercepted,
                "patch_error": result.c2_on_patch_error,
                "error_category": result.c2_on_error_category,
                "domain_state_changed": result.c2_on_domain_state_changed,
                "successfully_applied": result.c2_on_successfully_applied,
            }
        )
        rows.append(
            {
                "case_id": result.case_id,
                "episode_id": result.episode_id,
                "gamefile": result.gamefile,
                "step_id": result.step_id,
                "injection_type": result.injection_type,
                "ablation_mode": ABLATE_C2_MODE,
                "patch": copy.deepcopy(result.patch),
                "intercepted": result.c2_off_intercepted,
                "patch_error": result.c2_off_patch_error,
                "error_category": result.c2_off_error_category,
                "domain_state_changed": result.c2_off_domain_state_changed,
                "successfully_applied": result.c2_off_successfully_applied,
            }
        )
    return rows


def _compute_mode_metrics(
    results: List[CaseResult],
    *,
    intercepted_attr: str,
    applied_attr: str,
    error_attr: str,
) -> ModeMetrics:
    mode_metrics = ModeMetrics(n_cases=len(results))
    if not results:
        return mode_metrics

    type_totals: Dict[str, int] = {}
    type_intercepts: Dict[str, int] = {}
    type_applies: Dict[str, int] = {}

    intercept_total = 0
    apply_total = 0

    for result in results:
        injection_type = result.injection_type
        type_totals[injection_type] = type_totals.get(injection_type, 0) + 1

        intercepted = bool(getattr(result, intercepted_attr))
        if intercepted:
            intercept_total += 1
            type_intercepts[injection_type] = type_intercepts.get(injection_type, 0) + 1

        applied = bool(getattr(result, applied_attr))
        if applied:
            apply_total += 1
            type_applies[injection_type] = type_applies.get(injection_type, 0) + 1

        error_category = getattr(result, error_attr)
        if error_category:
            mode_metrics.error_type_breakdown[error_category] = (
                mode_metrics.error_type_breakdown.get(error_category, 0) + 1
            )

    mode_metrics.interception_rate = intercept_total / len(results)
    mode_metrics.apply_rate = apply_total / len(results)

    for injection_type, count in type_totals.items():
        mode_metrics.by_injection_type[injection_type] = {
            "n_cases": count,
            "interception_rate": type_intercepts.get(injection_type, 0) / count,
            "apply_rate": type_applies.get(injection_type, 0) / count,
        }

    return mode_metrics


def compute_schema_guard_metrics(results: List[CaseResult]) -> SchemaGuardMetrics:
    if not results:
        return SchemaGuardMetrics()

    type_counts: Dict[str, int] = {}
    for result in results:
        type_counts[result.injection_type] = type_counts.get(result.injection_type, 0) + 1

    full_metrics = _compute_mode_metrics(
        results,
        intercepted_attr="c2_on_intercepted",
        applied_attr="c2_on_successfully_applied",
        error_attr="c2_on_error_category",
    )
    ablate_metrics = _compute_mode_metrics(
        results,
        intercepted_attr="c2_off_intercepted",
        applied_attr="c2_off_successfully_applied",
        error_attr="c2_off_error_category",
    )

    metrics = SchemaGuardMetrics(
        n_cases=len(results),
        n_by_type=type_counts,
        c2_on_interception_rate=full_metrics.interception_rate,
        c2_off_interception_rate=ablate_metrics.interception_rate,
        delta_interception_rate=full_metrics.interception_rate - ablate_metrics.interception_rate,
        full=full_metrics,
        ablate_c2=ablate_metrics,
    )

    for injection_type, values in full_metrics.by_injection_type.items():
        metrics.c2_on_interception_by_type[injection_type] = values["interception_rate"]
    for injection_type, values in ablate_metrics.by_injection_type.items():
        metrics.c2_off_interception_by_type[injection_type] = values["interception_rate"]

    return metrics


class SchemaGuardEvaluator:
    """
    Evaluates C2 (schema validation) effectiveness by injecting invalid patches
    and comparing interception rates with C2 on vs off.
    """

    def __init__(
        self,
        injection_types: Optional[List[str]] = None,
    ) -> None:
        self.injection_types = injection_types or INJECTION_TYPES
        self._c2_on = AblationConfig.full()
        self._c2_off = AblationConfig.ablate("C2")

    def evaluate_cases(self, cases: List[Dict[str, Any]]) -> List[CaseResult]:
        """Run all injection cases and return per-case results."""
        results = []
        for case in cases:
            result = CaseResult(
                case_id=case["case_id"],
                injection_type=case["injection_type"],
                episode_id=case.get("episode_id", ""),
                gamefile=case.get("gamefile", ""),
                step_id=case.get("step_id", 0),
                patch=copy.deepcopy(case["patch"]),
            )

            (
                result.c2_on_intercepted,
                result.c2_on_patch_error,
                result.c2_on_domain_state_changed,
                result.c2_on_successfully_applied,
            ) = _run_case_with_ablation(case, self._c2_on)
            result.c2_on_error_category = classify_patch_error(result.c2_on_patch_error)

            (
                result.c2_off_intercepted,
                result.c2_off_patch_error,
                result.c2_off_domain_state_changed,
                result.c2_off_successfully_applied,
            ) = _run_case_with_ablation(case, self._c2_off)
            result.c2_off_error_category = classify_patch_error(result.c2_off_patch_error)

            results.append(result)
        return results

    def evaluate_from_states(
        self,
        captured_states: List[Dict[str, Any]],
    ) -> tuple[List[CaseResult], SchemaGuardMetrics]:
        """Build injection cases from captured states, evaluate, and return metrics."""
        cases = build_injection_cases(captured_states, self.injection_types)
        results = self.evaluate_cases(cases)
        metrics = compute_schema_guard_metrics(results)
        return results, metrics
