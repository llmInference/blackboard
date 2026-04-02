"""Analyze ALFWorld experiment summaries into per-task-type reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.common.communication_analysis import export_communication_artifacts
from experiment.alfworld.utils.result_analysis import (
    analyze_ablation_summary,
    analyze_system_compare_summary,
    analyze_workflow_compare_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ALFWorld summary outputs")
    parser.add_argument("--summary-file", required=True, help="Path to ablation_summary.json or workflow_compare_summary.json")
    parser.add_argument("--output-file", default="", help="Optional output path for analysis JSON")
    return parser.parse_args()


def _collect_group_specs(payload: dict) -> list[dict[str, str]]:
    if "system_summaries" in payload:
        return [
            {
                "group_type": "system",
                "group_id": system,
                "episodes_path": (
                    system_summary.get("result", {}).get("episodes_path")
                    or system_summary.get("result", {}).get("standard_episodes_path", "")
                ),
            }
            for system, system_summary in payload.get("system_summaries", {}).items()
        ]
    if "workflow_summaries" in payload:
        specs: list[dict[str, str]] = []
        for workflow_mode, workflow_summary in payload.get("workflow_summaries", {}).items():
            for mode, mode_summary in workflow_summary.get("result", {}).get("summaries", {}).items():
                specs.append(
                    {
                        "group_type": "workflow_mode",
                        "group_id": f"{workflow_mode}:{mode}",
                        "episodes_path": str(mode_summary.get("episodes_path", "")),
                    }
                )
        return specs
    if "summaries" in payload:
        return [
            {
                "group_type": "mode",
                "group_id": mode,
                "episodes_path": str(mode_summary.get("episodes_path", "")),
            }
            for mode, mode_summary in payload.get("summaries", {}).items()
        ]
    return []


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_file)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    if "system_summaries" in payload:
        analysis = analyze_system_compare_summary(str(summary_path))
    elif "workflow_summaries" in payload:
        analysis = analyze_workflow_compare_summary(str(summary_path))
    elif "summaries" in payload:
        analysis = analyze_ablation_summary(str(summary_path))
    else:
        raise ValueError(f"Unsupported summary file shape: {summary_path}")

    output_path = Path(args.output_file) if args.output_file else summary_path.with_name(f"{summary_path.stem}_analysis.json")
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis written to: {output_path}")

    artifacts = export_communication_artifacts(
        output_dir=summary_path.parent,
        env_name="alfworld",
        group_specs=_collect_group_specs(payload),
    )
    print(f"Communication summary written to: {artifacts['summary_path']}")
    print(f"Communication trace written to: {artifacts['trace_path']}")
    print(f"Communication judge records written to: {artifacts['judge_path']}")


if __name__ == "__main__":
    main()
