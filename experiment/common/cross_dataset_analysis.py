"""Cross-dataset aggregation helpers for experiments 4/5/6."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from experiment.alfworld.utils.result_analysis import (
    analyze_ablation_summary as analyze_alfworld_ablation_summary,
    analyze_system_compare_summary as analyze_alfworld_system_compare_summary,
)
from experiment.scienceworld.utils.result_analysis import (
    analyze_ablation_summary as analyze_scienceworld_ablation_summary,
    analyze_system_compare_summary as analyze_scienceworld_system_compare_summary,
)
from experiment.webarena.utils.result_analysis import (
    analyze_ablation_summary as analyze_webarena_ablation_summary,
    analyze_system_compare_summary as analyze_webarena_system_compare_summary,
)

DATASET_ORDER = ("alfworld", "webarena", "scienceworld")
SYSTEM_COMPARE_METRICS = (
    "success_rate",
    "mean_goal_condition_rate",
    "mean_steps",
    "mean_total_tokens",
    "mean_worker_input_tokens",
    "mean_fallback_rate",
    "mean_patch_error_rate",
    "mean_retrieval_precision",
)
ABLATION_METRICS = (
    "success_rate",
    "mean_goal_condition_rate",
    "mean_steps",
    "mean_total_tokens",
    "mean_worker_input_tokens",
    "mean_fallback_rate",
    "mean_patch_error_rate",
    "mean_retrieval_precision",
    "mean_context_fragment_count",
    "mean_relevant_fragment_count",
    "mean_irrelevant_fragment_count",
)

_SYSTEM_COMPARE_ANALYZERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "alfworld": analyze_alfworld_system_compare_summary,
    "webarena": analyze_webarena_system_compare_summary,
    "scienceworld": analyze_scienceworld_system_compare_summary,
}
_ABLATION_ANALYZERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "alfworld": analyze_alfworld_ablation_summary,
    "webarena": analyze_webarena_ablation_summary,
    "scienceworld": analyze_scienceworld_ablation_summary,
}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _summarize_values(values: list[float]) -> Dict[str, float]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _ordered_datasets(dataset_summaries: Mapping[str, str]) -> list[str]:
    known = [dataset for dataset in DATASET_ORDER if dataset_summaries.get(dataset)]
    extras = sorted(dataset for dataset in dataset_summaries if dataset not in DATASET_ORDER and dataset_summaries.get(dataset))
    return known + extras


def _dataset_n_items(analysis: Mapping[str, Any]) -> int:
    return int(analysis.get("n_gamefiles") or analysis.get("n_tasks") or 0)


def _normalize_metrics(payload: Mapping[str, Any], metric_keys: tuple[str, ...]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key in metric_keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            normalized[key] = value
    return normalized


def _extract_compare_systems(analysis: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    systems: Dict[str, Dict[str, Any]] = {}
    for system, payload in (analysis.get("system_summaries") or {}).items():
        overall = payload.get("overall") or {}
        systems[system] = {
            "metrics": _normalize_metrics(overall, SYSTEM_COMPARE_METRICS),
            "robustness": payload.get("robustness") or {},
            "system_output_dir": payload.get("system_output_dir", ""),
        }
    return systems


def _extract_ablation_modes(analysis: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    source = analysis.get("mode_summaries")
    if not isinstance(source, dict):
        source = analysis.get("modes") if isinstance(analysis.get("modes"), dict) else {}

    modes: Dict[str, Dict[str, Any]] = {}
    for mode, payload in source.items():
        overall = payload.get("overall") or {}
        modes[mode] = {
            "metrics": _normalize_metrics(overall, ABLATION_METRICS),
            "robustness": payload.get("robustness") or {},
            "disabled_components": list(payload.get("disabled_components") or []),
            "mode_output_dir": payload.get("mode_output_dir", ""),
        }
    return modes


def _mode_delta(full_metrics: Mapping[str, float], mode_metrics: Mapping[str, float]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key in ABLATION_METRICS:
        full_value = full_metrics.get(key)
        mode_value = mode_metrics.get(key)
        if full_value is None or mode_value is None:
            continue
        delta[key] = float(full_value - mode_value)
    return delta


def build_cross_dataset_system_compare(
    *,
    experiment_id: str,
    dataset_summaries: Mapping[str, str],
) -> Dict[str, Any]:
    """Aggregate Experiment 4/5 system-compare results across datasets."""
    datasets: Dict[str, Dict[str, Any]] = {}
    metric_values_by_system: Dict[str, Dict[str, list[float]]] = {}
    per_dataset_by_system: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset in _ordered_datasets(dataset_summaries):
        summary_path = dataset_summaries[dataset]
        analysis = _SYSTEM_COMPARE_ANALYZERS[dataset](summary_path)
        systems = _extract_compare_systems(analysis)
        datasets[dataset] = {
            "dataset": dataset,
            "summary_path": str(Path(summary_path)),
            "analysis_type": analysis.get("analysis_type", ""),
            "workflow_mode": analysis.get("workflow_mode", ""),
            "execution_backend": analysis.get("execution_backend", ""),
            "n_items": _dataset_n_items(analysis),
            "systems": systems,
        }

        for system, payload in systems.items():
            metric_values = metric_values_by_system.setdefault(system, {})
            per_dataset = per_dataset_by_system.setdefault(system, {})
            per_dataset[dataset] = dict(payload.get("metrics") or {})
            for metric, value in (payload.get("metrics") or {}).items():
                metric_values.setdefault(metric, []).append(value)

    aggregate_by_system: Dict[str, Dict[str, Any]] = {}
    for system in sorted(metric_values_by_system):
        metrics = {
            metric: _summarize_values(values)
            for metric, values in metric_values_by_system[system].items()
            if values
        }
        success_values = metric_values_by_system[system].get("success_rate", [])
        goal_values = metric_values_by_system[system].get("mean_goal_condition_rate", [])
        aggregate_by_system[system] = {
            "dataset_count": len(per_dataset_by_system.get(system, {})),
            "datasets": sorted(per_dataset_by_system.get(system, {})),
            "by_dataset": per_dataset_by_system.get(system, {}),
            "metrics": metrics,
            "cross_dataset_robustness": {
                "success_rate_std": _std(success_values),
                "mean_goal_condition_rate_std": _std(goal_values),
            },
        }

    system_order = sorted(
        aggregate_by_system,
        key=lambda system: (
            -aggregate_by_system[system].get("metrics", {}).get("success_rate", {}).get("mean", 0.0),
            system,
        ),
    )

    return {
        "analysis_type": "cross_dataset_system_compare",
        "experiment_id": experiment_id,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "aggregate_by_system": aggregate_by_system,
        "system_order_by_mean_success_rate": system_order,
    }


def build_cross_dataset_ablation(
    *,
    experiment_id: str,
    dataset_summaries: Mapping[str, str],
) -> Dict[str, Any]:
    """Aggregate Experiment 6 ablation results across datasets."""
    datasets: Dict[str, Dict[str, Any]] = {}
    full_metric_values: Dict[str, list[float]] = {}
    full_by_dataset: Dict[str, Dict[str, float]] = {}
    delta_values_by_mode: Dict[str, Dict[str, list[float]]] = {}
    delta_by_dataset_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}
    disabled_components_by_mode: Dict[str, set[str]] = {}

    for dataset in _ordered_datasets(dataset_summaries):
        summary_path = dataset_summaries[dataset]
        analysis = _ABLATION_ANALYZERS[dataset](summary_path)
        modes = _extract_ablation_modes(analysis)
        full_metrics = dict(modes.get("full", {}).get("metrics") or {})
        if full_metrics:
            full_by_dataset[dataset] = full_metrics
            for metric, value in full_metrics.items():
                full_metric_values.setdefault(metric, []).append(value)

        dataset_modes: Dict[str, Dict[str, Any]] = {}
        for mode, payload in modes.items():
            mode_metrics = dict(payload.get("metrics") or {})
            mode_entry = {
                "metrics": mode_metrics,
                "robustness": payload.get("robustness") or {},
                "disabled_components": list(payload.get("disabled_components") or []),
                "mode_output_dir": payload.get("mode_output_dir", ""),
            }
            if mode != "full":
                delta = _mode_delta(full_metrics, mode_metrics)
                mode_entry["full_minus_mode"] = delta
                delta_by_dataset_by_mode.setdefault(mode, {})[dataset] = delta
                disabled_components_by_mode.setdefault(mode, set()).update(mode_entry["disabled_components"])
                for metric, value in delta.items():
                    delta_values_by_mode.setdefault(mode, {}).setdefault(metric, []).append(value)
            dataset_modes[mode] = mode_entry

        datasets[dataset] = {
            "dataset": dataset,
            "summary_path": str(Path(summary_path)),
            "analysis_type": analysis.get("analysis_type", ""),
            "workflow_mode": analysis.get("workflow_mode", ""),
            "execution_backend": analysis.get("execution_backend", ""),
            "n_items": _dataset_n_items(analysis),
            "modes": dataset_modes,
        }

    full_aggregate = {
        "dataset_count": len(full_by_dataset),
        "datasets": sorted(full_by_dataset),
        "by_dataset": full_by_dataset,
        "metrics": {
            metric: _summarize_values(values)
            for metric, values in full_metric_values.items()
            if values
        },
    }

    aggregate_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in sorted(delta_values_by_mode):
        aggregate_by_mode[mode] = {
            "dataset_count": len(delta_by_dataset_by_mode.get(mode, {})),
            "datasets": sorted(delta_by_dataset_by_mode.get(mode, {})),
            "disabled_components": sorted(disabled_components_by_mode.get(mode, set())),
            "by_dataset": delta_by_dataset_by_mode.get(mode, {}),
            "full_minus_mode_metrics": {
                metric: _summarize_values(values)
                for metric, values in delta_values_by_mode[mode].items()
                if values
            },
        }

    mode_order = sorted(
        aggregate_by_mode,
        key=lambda mode: (
            -aggregate_by_mode[mode].get("full_minus_mode_metrics", {}).get("success_rate", {}).get("mean", 0.0),
            mode,
        ),
    )

    return {
        "analysis_type": "cross_dataset_ablation",
        "experiment_id": experiment_id,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "full_aggregate": full_aggregate,
        "aggregate_by_mode": aggregate_by_mode,
        "mode_order_by_mean_success_rate_delta": mode_order,
    }


def _fmt(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.3f}"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown report for human inspection."""
    lines: list[str] = []
    title = str(report.get("experiment_id", "cross_dataset")).upper()
    lines.append(f"# {title} Cross-Dataset Report")
    lines.append("")
    lines.append(f"- analysis_type: `{report.get('analysis_type', '')}`")
    lines.append(f"- dataset_count: `{report.get('dataset_count', 0)}`")
    lines.append("")

    if report.get("analysis_type") == "cross_dataset_system_compare":
        lines.append("## Dataset Breakdown")
        lines.append("")
        lines.append("| Dataset | System | Success | Goal | Steps | Total Tokens | Retrieval |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for dataset in _ordered_datasets(report.get("datasets", {})):
            systems = report.get("datasets", {}).get(dataset, {}).get("systems", {})
            for system in sorted(systems):
                metrics = systems[system].get("metrics", {})
                lines.append(
                    f"| {dataset} | {system} | {_fmt(metrics.get('success_rate'))} | "
                    f"{_fmt(metrics.get('mean_goal_condition_rate'))} | {_fmt(metrics.get('mean_steps'))} | "
                    f"{_fmt(metrics.get('mean_total_tokens'))} | {_fmt(metrics.get('mean_retrieval_precision'))} |"
                )
        lines.append("")
        lines.append("## Cross-Dataset Aggregate")
        lines.append("")
        lines.append("| System | Datasets | Mean Success | Success Std | Mean Goal | Mean Tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for system in report.get("system_order_by_mean_success_rate", []):
            aggregate = report.get("aggregate_by_system", {}).get(system, {})
            metrics = aggregate.get("metrics", {})
            lines.append(
                f"| {system} | {aggregate.get('dataset_count', 0)} | "
                f"{_fmt(metrics.get('success_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('success_rate', {}).get('std'))} | "
                f"{_fmt(metrics.get('mean_goal_condition_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('mean_total_tokens', {}).get('mean'))} |"
            )
    else:
        lines.append("## Dataset Breakdown")
        lines.append("")
        lines.append("| Dataset | Mode | Success | Goal | Total Tokens | Delta Success | Delta Retrieval |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for dataset in _ordered_datasets(report.get("datasets", {})):
            modes = report.get("datasets", {}).get(dataset, {}).get("modes", {})
            for mode in sorted(modes):
                metrics = modes[mode].get("metrics", {})
                delta = modes[mode].get("full_minus_mode", {})
                lines.append(
                    f"| {dataset} | {mode} | {_fmt(metrics.get('success_rate'))} | "
                    f"{_fmt(metrics.get('mean_goal_condition_rate'))} | {_fmt(metrics.get('mean_total_tokens'))} | "
                    f"{_fmt(delta.get('success_rate'))} | {_fmt(delta.get('mean_retrieval_precision'))} |"
                )
        lines.append("")
        lines.append("## Cross-Dataset Aggregate")
        lines.append("")
        lines.append("| Mode | Datasets | Mean Delta Success | Delta Success Std | Mean Delta Goal | Mean Delta Tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for mode in report.get("mode_order_by_mean_success_rate_delta", []):
            aggregate = report.get("aggregate_by_mode", {}).get(mode, {})
            metrics = aggregate.get("full_minus_mode_metrics", {})
            lines.append(
                f"| {mode} | {aggregate.get('dataset_count', 0)} | "
                f"{_fmt(metrics.get('success_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('success_rate', {}).get('std'))} | "
                f"{_fmt(metrics.get('mean_goal_condition_rate', {}).get('mean'))} | "
                f"{_fmt(metrics.get('mean_total_tokens', {}).get('mean'))} |"
            )

    lines.append("")
    return "\n".join(lines)
