"""
Capture real ALFWorld runtime states for Experiment 2 schema-guard evaluation.

Usage:
    python run_capture_states.py --data-root /path/to/alfworld/data --output-dir outputs/exp1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.alfworld.core.adapter import ALFWorldKernelAdapter
from experiment.alfworld.environments.env_wrapper import ALFWorldEnvWrapper
from experiment.alfworld.evaluators.alfworld_evaluator import ALFWorldEvaluator
from experiment.alfworld.utils.llm_factory import build_runtime_llm
from experiment.alfworld.utils.result_writer import ResultWriter
from experiment.alfworld.utils.task_selection import select_gamefiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture real ALFWorld runtime states")
    parser.add_argument("--data-root", default="", help="ALFWorld dataset root for sampler-based gamefile selection")
    parser.add_argument("--output-dir", default="outputs/exp1", help="Directory for episode/state logs")
    parser.add_argument("--run-id", default="capture_states", help="Run identifier used in output filenames")
    parser.add_argument("--gamefiles-file", default="", help="Optional JSON file containing an explicit gamefile list")
    parser.add_argument("--task-set-file", default="", help="Optional task-set manifest JSON for gamefile selection")
    parser.add_argument("--split", choices=["debug", "formal"], default="debug", help="Dataset split size")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sampled gamefiles to run")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum ALFWorld steps per episode")
    parser.add_argument("--model-name", default="alfworld_capture_states", help="Model label written to logs")
    parser.add_argument("--env-config", default=None, help="Optional ALFWorld YAML config path")
    parser.add_argument(
        "--workflow-mode",
        choices=["single_action", "planner_action", "planner_llm_action"],
        default="planner_llm_action",
        help="Workflow mode used while capturing runtime states",
    )
    parser.add_argument(
        "--architect-mode",
        choices=["deterministic", "llm"],
        default="llm",
        help="Architect mode used while capturing runtime states",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["none", "mock", "openai_compatible"],
        default="none",
        help="LLM backend for capture runs",
    )
    parser.add_argument("--llm-model", default="", help="Model name for openai_compatible backend")
    parser.add_argument("--llm-model-env", default="", help="Optional env var name containing the model name")
    parser.add_argument("--llm-base-url", default="", help="Base URL for openai_compatible backend")
    parser.add_argument("--llm-base-url-env", default="", help="Optional env var name containing the base URL")
    parser.add_argument("--llm-api-key-env", default="", help="Optional env var name containing the API key")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="Timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gamefiles, selection_meta = select_gamefiles(
        data_root=args.data_root,
        split=args.split,
        seed=args.seed,
        limit=args.limit,
        gamefiles_file=args.gamefiles_file,
        task_set_file=args.task_set_file,
    )
    selection_source = selection_meta.get("selection_source", "unknown")
    selection_target = (
        selection_meta.get("task_set_file")
        or selection_meta.get("gamefiles_file")
        or selection_meta.get("data_root")
        or args.data_root
    )
    print(f"Selected {len(gamefiles)} gamefiles via {selection_source}: {selection_target}")

    env = ALFWorldEnvWrapper(config_path=args.env_config)
    llm = build_runtime_llm(
        workflow_mode=args.workflow_mode,
        architect_mode=args.architect_mode,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_model_env=args.llm_model_env,
        llm_base_url=args.llm_base_url,
        llm_base_url_env=args.llm_base_url_env,
        llm_api_key_env=args.llm_api_key_env,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
    )
    adapter = ALFWorldKernelAdapter(
        env,
        llm=llm,
        config={
            "workflow_mode": args.workflow_mode,
            "architect_mode": args.architect_mode,
        },
    )
    writer = ResultWriter(
        args.output_dir,
        run_id=args.run_id,
        system_id="blackboard_capture_states",
        system_family="blackboard",
        env_name="alfworld",
    )
    writer.write_config_snapshot(
        {
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "run_id": args.run_id,
            "gamefiles_file": args.gamefiles_file,
            "task_set_file": args.task_set_file,
            "split": args.split,
            "seed": args.seed,
            "limit": args.limit,
            "selection_meta": selection_meta,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "env_config": args.env_config,
            "capture_states": True,
            "workflow_mode": args.workflow_mode,
            "architect_mode": args.architect_mode,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "llm_model_env": args.llm_model_env,
            "llm_base_url": args.llm_base_url,
            "llm_base_url_env": args.llm_base_url_env,
            "llm_api_key_env": args.llm_api_key_env,
            "llm_temperature": args.llm_temperature,
            "llm_timeout": args.llm_timeout,
        }
    )
    evaluator = ALFWorldEvaluator(adapter=adapter, result_writer=writer)

    try:
        evaluator.run_batch(
            gamefiles,
            run_id=args.run_id,
            experiment_id="capture_states",
            model_name=args.model_name,
            max_steps=args.max_steps,
            capture_states=True,
        )
        summary = evaluator.finalize()
    finally:
        env.close()

    print(f"Episodes written to: {writer.episodes_path}")
    print(f"Canonical episodes written to: {writer.standard_episodes_path}")
    print(f"Captured states written to: {writer.states_path}")
    print(f"Canonical states written to: {writer.standard_states_path}")
    print(f"Summary written to: {writer.summary_path}")
    print(f"Canonical summary written to: {writer.standard_summary_path}")
    print(f"Config snapshot written to: {writer.config_snapshot_path}")
    print(f"Summary metrics: {summary}")


if __name__ == "__main__":
    main()
