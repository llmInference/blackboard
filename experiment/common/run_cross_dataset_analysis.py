"""CLI for cross-dataset aggregation of experiments 4/5/6."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.common.cross_dataset_analysis import (
    build_cross_dataset_ablation,
    build_cross_dataset_system_compare,
    render_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment summaries across ALFWorld, WebArena, and ScienceWorld")
    parser.add_argument("--experiment", choices=["exp4", "exp5", "exp6"], required=True, help="Experiment identifier")
    parser.add_argument("--alfworld-summary", default="", help="Path to ALFWorld summary JSON")
    parser.add_argument("--webarena-summary", default="", help="Path to WebArena summary JSON")
    parser.add_argument("--scienceworld-summary", default="", help="Path to ScienceWorld summary JSON")
    parser.add_argument("--output-path", default="", help="Optional output JSON path")
    parser.add_argument("--markdown-path", default="", help="Optional output Markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_summaries = {
        "alfworld": args.alfworld_summary,
        "webarena": args.webarena_summary,
        "scienceworld": args.scienceworld_summary,
    }
    dataset_summaries = {dataset: path for dataset, path in dataset_summaries.items() if path}
    if not dataset_summaries:
        raise ValueError("At least one dataset summary path is required.")

    if args.experiment in {"exp4", "exp5"}:
        report = build_cross_dataset_system_compare(
            experiment_id=args.experiment,
            dataset_summaries=dataset_summaries,
        )
    else:
        report = build_cross_dataset_ablation(
            experiment_id=args.experiment,
            dataset_summaries=dataset_summaries,
        )

    output_path = Path(args.output_path) if args.output_path else Path.cwd() / f"{args.experiment}_cross_dataset_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_path) if args.markdown_path else output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Cross-dataset summary written to: {output_path}")
    print(f"Cross-dataset markdown written to: {markdown_path}")


if __name__ == "__main__":
    main()
