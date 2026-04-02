"""Run a minimal WebArena Blackboard smoke slice from a task manifest."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.common.communication_analysis import export_communication_artifacts
from experiment.webarena.examples.common import resolve_manifest, run_blackboard_batch, select_task_ids, summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a WebArena Blackboard smoke task")
    parser.add_argument(
        "--task-set-file",
        default=str(REPO_ROOT / "experiment/common/task_sets/webarena_script_browser_smoke.json"),
        help="WebArena task manifest.",
    )
    parser.add_argument("--config", default="", help="Local WebArena config JSON. Defaults to manifest metadata.config_template.")
    parser.add_argument("--task-id", default="", help="Optional single task id override.")
    parser.add_argument("--limit", type=int, default=1, help="Max number of tasks to run.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiment/webarena/outputs" / f"blackboard_smoke_{time.strftime('%Y%m%d_%H%M%S')}"),
        help="Output directory for smoke artifacts.",
    )
    parser.add_argument("--dataset-name", default="webarena-verified", help="Dataset name recorded in outputs.")
    parser.add_argument("--max-steps", type=int, default=4, help="Max neutral turns.")
    parser.add_argument("--max-kernel-steps", type=int, default=8, help="Max internal kernel steps per turn.")
    parser.add_argument("--max-no-progress", type=int, default=2, help="Circuit-break threshold.")
    parser.add_argument("--llm-backend", default="", help="Optional LLM backend for LLM-capable workers.")
    parser.add_argument("--llm-model", default="", help="Optional explicit LLM model name.")
    parser.add_argument("--llm-model-env", default="", help="Optional env var for the LLM model.")
    parser.add_argument("--llm-base-url", default="", help="Optional explicit OpenAI-compatible base URL.")
    parser.add_argument("--llm-base-url-env", default="", help="Optional env var for the base URL.")
    parser.add_argument("--llm-api-key-env", default="", help="Optional env var for the API key.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Optional LLM temperature.")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="Optional LLM timeout seconds.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run Playwright headless.")
    parser.add_argument("--slow-mo-ms", type=int, default=0, help="Optional Playwright slow motion.")
    parser.add_argument("--summary-only", action="store_true", help="Print only the summary JSON.")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    resolved = resolve_manifest(args.task_set_file, config_override=args.config)
    manifest_info = resolved["manifest_info"]
    manifest_path = resolved["manifest_path"]
    dataset_path = resolved["dataset_path"]
    metadata = resolved["metadata"]
    config_path = resolved["config_path"]
    output_dir = Path(args.output_dir).resolve()
    selected_task_ids = select_task_ids(
        dataset_path=dataset_path,
        metadata=metadata,
        task_id_override=args.task_id,
        limit=args.limit,
    )
    episodes = run_blackboard_batch(
        dataset_path=dataset_path,
        config_path=config_path,
        task_ids=selected_task_ids,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        run_id=output_dir.name,
        system_id="blackboard",
        system_family="blackboard",
        system_config={
            "max_kernel_steps": args.max_kernel_steps,
            "max_no_progress": args.max_no_progress,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "llm_model_env": args.llm_model_env,
            "llm_base_url": args.llm_base_url,
            "llm_base_url_env": args.llm_base_url_env,
            "llm_api_key_env": args.llm_api_key_env,
            "llm_temperature": args.llm_temperature,
            "llm_timeout": args.llm_timeout,
        },
        headless=bool(args.headless),
        slow_mo_ms=int(args.slow_mo_ms),
        max_steps=int(args.max_steps),
    )
    summary = summary_payload(
        episodes,
        task_set=str(manifest_info.get("task_set", "")),
        manifest_path=manifest_path,
        config_path=config_path,
    )
    communication = export_communication_artifacts(
        output_dir=output_dir,
        env_name="webarena",
        group_specs=[
            {
                "group_type": "system",
                "group_id": "blackboard",
                "episodes_path": str(output_dir / "episode_results.jsonl"),
            }
        ],
    )
    summary["communication_analysis"] = communication
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.summary_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        payload = {
            "task_manifest": manifest_info,
            "selected_task_ids": selected_task_ids,
            "summary": summary,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
