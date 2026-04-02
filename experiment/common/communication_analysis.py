"""Communication-trace analysis helpers for Experiment 1."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick_sender_trace(step: Dict[str, Any]) -> Dict[str, Any]:
    traces = list(step.get("communication_trace", []) or [])
    if not traces:
        return {}
    receiver_like_sources = {"action_worker", "executor", "action_node", "prompt"}
    for trace in traces:
        source = str(trace.get("source", "") or "")
        if source not in receiver_like_sources:
            return trace
    return traces[0]


def _extract_sender_message(step: Dict[str, Any]) -> Tuple[str, str, str]:
    sender_trace = _pick_sender_trace(step)
    if sender_trace:
        return (
            str(sender_trace.get("source", "") or "derived_sender"),
            str(sender_trace.get("channel", "") or "derived"),
            _json_text(sender_trace.get("content", "")),
        )

    planner_state = step.get("planner_state", {}) or {}
    if planner_state:
        return ("planner_state", "structured", _json_text(planner_state))

    decision_reason = str(step.get("decision_reason", "") or "")
    return ("decision_reason", "derived", decision_reason)


def _extract_sender_intent(step: Dict[str, Any], episode: Dict[str, Any]) -> str:
    planner_state = step.get("planner_state", {}) or {}
    if str(planner_state.get("intent", "") or "").strip():
        return str(planner_state.get("intent", "") or "")
    if str(planner_state.get("subgoal", "") or "").strip():
        parts = [str(planner_state.get("subgoal", "") or "").strip()]
        focus_object = str(planner_state.get("focus_object", "") or "").strip()
        focus_receptacle = str(planner_state.get("focus_receptacle", "") or "").strip()
        if focus_object:
            parts.append(f"object={focus_object}")
        if focus_receptacle:
            parts.append(f"receptacle={focus_receptacle}")
        return " ".join(parts)
    canonical_task = step.get("canonical_task", {}) or {}
    canonical_goal = str(canonical_task.get("canonical_goal_object", "") or "").strip()
    canonical_receptacle = str(canonical_task.get("canonical_target_receptacle", "") or "").strip()
    if canonical_goal or canonical_receptacle:
        return " ".join(part for part in (canonical_goal, canonical_receptacle) if part)
    metadata = episode.get("metadata", {}) or {}
    return (
        str(metadata.get("intent", "") or "").strip()
        or str(episode.get("task_goal", "") or "").strip()
        or str(episode.get("task_id", "") or "").strip()
    )


def _extract_expected_action(step: Dict[str, Any]) -> str:
    expected_action = str(step.get("expected_action", "") or "").strip()
    if expected_action:
        return expected_action
    planner_state = step.get("planner_state", {}) or {}
    next_action = str(planner_state.get("next_action", "") or "").strip()
    if next_action:
        return next_action
    recommended_actions = list(planner_state.get("recommended_actions", []) or [])
    return str(recommended_actions[0] if recommended_actions else "").strip()


def _drift_reasons(step: Dict[str, Any], expected_action: str) -> List[str]:
    reasons: List[str] = []
    if str(step.get("patch_error", "") or "").strip():
        reasons.append("patch_error")
    if bool(step.get("fallback_used", False)):
        reasons.append("fallback_action")
    if "action_matched" in step and not bool(step.get("action_matched", False)):
        reasons.append("expected_action_mismatch")
    elif expected_action and str(step.get("action", "") or "").strip() and expected_action != str(step.get("action", "") or "").strip():
        recommended_actions = list((step.get("planner_state", {}) or {}).get("recommended_actions", []) or [])
        if recommended_actions:
            reasons.append("recommended_action_mismatch")

    recommended_actions = list((step.get("planner_state", {}) or {}).get("recommended_actions", []) or [])
    action = str(step.get("action", "") or "").strip()
    if recommended_actions and action and action not in recommended_actions and "recommended_action_mismatch" not in reasons:
        reasons.append("recommended_action_mismatch")
    return reasons


def build_trace_records(
    episodes: Iterable[Dict[str, Any]],
    *,
    env_name: str,
    group_type: str,
    group_id: str,
) -> List[Dict[str, Any]]:
    """Normalize per-step communication records from episode results."""
    records: List[Dict[str, Any]] = []
    for episode in episodes:
        for step in list(episode.get("trajectory", []) or []):
            sender_source, sender_channel, sender_message = _extract_sender_message(step)
            sender_intent = _extract_sender_intent(step, episode)
            receiver_action = str(step.get("action", "") or "").strip()
            expected_action = _extract_expected_action(step)
            drift_reasons = _drift_reasons(step, expected_action)
            heuristic_drift = bool(drift_reasons)
            heuristic_score = 0.0 if heuristic_drift else 1.0
            record = {
                "run_id": str(episode.get("run_id", "") or ""),
                "system_id": str(episode.get("system_id", "") or ""),
                "system_family": str(episode.get("system_family", "") or ""),
                "env_name": env_name,
                "group_type": group_type,
                "group_id": group_id,
                "episode_id": str(episode.get("episode_id", "") or ""),
                "task_id": str(episode.get("task_id", episode.get("gamefile", "")) or ""),
                "step_id": int(step.get("step_id", 0) or 0),
                "sender_source": sender_source,
                "sender_channel": sender_channel,
                "sender_intent": sender_intent,
                "sender_message": sender_message,
                "receiver_action": receiver_action,
                "expected_action": expected_action,
                "receiver_reason": str(step.get("decision_reason", "") or ""),
                "fallback_used": bool(step.get("fallback_used", False)),
                "patch_error": str(step.get("patch_error", "") or ""),
                "action_matched": bool(step.get("action_matched", False)) if "action_matched" in step else None,
                "heuristic_drift": heuristic_drift,
                "heuristic_score": heuristic_score,
                "drift_reasons": drift_reasons,
                "metadata": {
                    "context_mode": str(step.get("context_mode", episode.get("metadata", {}).get("context_mode", "")) or ""),
                    "task_goal": str(episode.get("task_goal", "") or ""),
                },
            }
            if sender_message or sender_intent or receiver_action:
                records.append(record)
    return records


def summarize_trace_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate communication records into Experiment 1 metrics."""
    rows = list(records)
    if not rows:
        return {
            "n_records": 0,
            "semantic_drift_count": 0,
            "semantic_drift_rate": 0.0,
            "heuristic_alignment_rate": 0.0,
            "fallback_drift_count": 0,
            "patch_error_drift_count": 0,
            "expected_action_mismatch_count": 0,
            "recommended_action_mismatch_count": 0,
            "judge_ready_count": 0,
            "channel_breakdown": {},
            "source_breakdown": {},
        }

    channel_breakdown: Dict[str, int] = {}
    source_breakdown: Dict[str, int] = {}
    semantic_drift_count = 0
    fallback_drift_count = 0
    patch_error_drift_count = 0
    expected_action_mismatch_count = 0
    recommended_action_mismatch_count = 0
    judge_ready_count = 0

    for row in rows:
        sender_channel = str(row.get("sender_channel", "") or "unknown")
        sender_source = str(row.get("sender_source", "") or "unknown")
        channel_breakdown[sender_channel] = channel_breakdown.get(sender_channel, 0) + 1
        source_breakdown[sender_source] = source_breakdown.get(sender_source, 0) + 1

        reasons = list(row.get("drift_reasons", []) or [])
        if reasons:
            semantic_drift_count += 1
        if "fallback_action" in reasons:
            fallback_drift_count += 1
        if "patch_error" in reasons:
            patch_error_drift_count += 1
        if "expected_action_mismatch" in reasons:
            expected_action_mismatch_count += 1
        if "recommended_action_mismatch" in reasons:
            recommended_action_mismatch_count += 1
        if str(row.get("sender_message", "") or "").strip() and str(row.get("receiver_action", "") or "").strip():
            judge_ready_count += 1

    heuristic_alignment_rate = float(sum(float(row.get("heuristic_score", 0.0)) for row in rows) / len(rows))
    return {
        "n_records": len(rows),
        "semantic_drift_count": semantic_drift_count,
        "semantic_drift_rate": float(semantic_drift_count / len(rows)),
        "heuristic_alignment_rate": heuristic_alignment_rate,
        "fallback_drift_count": fallback_drift_count,
        "patch_error_drift_count": patch_error_drift_count,
        "expected_action_mismatch_count": expected_action_mismatch_count,
        "recommended_action_mismatch_count": recommended_action_mismatch_count,
        "judge_ready_count": judge_ready_count,
        "channel_breakdown": channel_breakdown,
        "source_breakdown": source_breakdown,
    }


def build_judge_records(trace_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract compact LLM-judge inputs from normalized trace records."""
    judge_records: List[Dict[str, Any]] = []
    for row in trace_records:
        if not str(row.get("sender_message", "") or "").strip():
            continue
        if not str(row.get("receiver_action", "") or "").strip():
            continue
        judge_records.append(
            {
                "judge_id": (
                    f"{row.get('group_id', '')}:"
                    f"{row.get('episode_id', '')}:"
                    f"{row.get('step_id', 0)}"
                ),
                "env_name": row.get("env_name", ""),
                "group_type": row.get("group_type", ""),
                "group_id": row.get("group_id", ""),
                "system_id": row.get("system_id", ""),
                "system_family": row.get("system_family", ""),
                "episode_id": row.get("episode_id", ""),
                "task_id": row.get("task_id", ""),
                "step_id": row.get("step_id", 0),
                "sender_intent": row.get("sender_intent", ""),
                "sender_message": row.get("sender_message", ""),
                "receiver_action": row.get("receiver_action", ""),
                "expected_action": row.get("expected_action", ""),
                "heuristic_score": row.get("heuristic_score", 0.0),
                "heuristic_drift": row.get("heuristic_drift", False),
                "drift_reasons": row.get("drift_reasons", []),
                "metadata": row.get("metadata", {}),
            }
        )
    return judge_records


def analyze_episode_groups(
    group_specs: Iterable[Dict[str, str]],
    *,
    env_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Analyze multiple episode groups and return summary + trace/judge rows."""
    all_trace_records: List[Dict[str, Any]] = []
    by_group: Dict[str, Dict[str, Any]] = {}
    for spec in group_specs:
        episodes_path = str(spec.get("episodes_path", "") or "")
        if not episodes_path:
            continue
        path = Path(episodes_path)
        if not path.is_file():
            continue
        group_id = str(spec.get("group_id", path.stem) or path.stem)
        group_type = str(spec.get("group_type", "group") or "group")
        records = build_trace_records(load_jsonl(path), env_name=env_name, group_type=group_type, group_id=group_id)
        all_trace_records.extend(records)
        by_group[group_id] = {
            "group_type": group_type,
            "episodes_path": str(path),
            "summary": summarize_trace_records(records),
        }

    judge_records = build_judge_records(all_trace_records)
    communication_summary = {
        "env_name": env_name,
        "overall": summarize_trace_records(all_trace_records),
        "by_group": by_group,
    }
    return communication_summary, all_trace_records, judge_records


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_communication_artifacts(
    *,
    output_dir: str | Path,
    env_name: str,
    group_specs: Iterable[Dict[str, str]],
) -> Dict[str, str]:
    """Write communication analysis artifacts next to a summary analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    communication_summary, trace_records, judge_records = analyze_episode_groups(group_specs, env_name=env_name)

    summary_path = output_path / "comm_summary.json"
    trace_path = output_path / "comm_trace.jsonl"
    judge_path = output_path / "comm_judge.jsonl"
    summary_path.write_text(json.dumps(communication_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(trace_path, trace_records)
    write_jsonl(judge_path, judge_records)
    return {
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "judge_path": str(judge_path),
    }
