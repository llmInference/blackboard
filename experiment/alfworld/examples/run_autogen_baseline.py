"""Run the AutoGen ALFWorld baseline on a sampled task set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.alfworld.environments.env_wrapper import ALFWorldEnvWrapper
from experiment.alfworld.evaluators.alfworld_evaluator import ALFWorldEvaluator
from experiment.alfworld.systems.autogen_runner import ALFWorldAutoGenRunner
from experiment.alfworld.utils.autogen_factory import build_autogen_model_client_factory
from experiment.alfworld.utils.result_writer import ResultWriter
from experiment.alfworld.utils.task_selection import select_gamefiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoGen baseline on ALFWorld")
    parser.add_argument("--data-root", default="", help="ALFWorld dataset root for sampler-based gamefile selection")
    parser.add_argument("--output-dir", default="outputs/autogen_baseline", help="Directory for output artifacts")
    parser.add_argument("--gamefiles-file", default="", help="Optional JSON file with explicit gamefiles")
    parser.add_argument("--task-set-file", default="", help="Optional task-set manifest JSON for gamefile selection")
    parser.add_argument("--run-id", default="autogen_baseline", help="Run identifier")
    parser.add_argument("--split", choices=["debug", "formal"], default="debug", help="Dataset split size")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sampled gamefiles to run")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum ALFWorld steps per episode")
    parser.add_argument("--model-name", default="autogen_plain", help="Model label used in logs")
    parser.add_argument("--env-config", default=None, help="Optional ALFWorld YAML config path")
    parser.add_argument(
        "--workflow-mode",
        choices=["single_action", "planner_action", "planner_llm_action"],
        default="planner_action",
        help="AutoGen workflow shape to run",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai_compatible"],
        default="mock",
        help="LLM backend for AutoGen baseline",
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

    model_client_factory = build_autogen_model_client_factory(
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_model_env=args.llm_model_env,
        llm_base_url=args.llm_base_url,
        llm_base_url_env=args.llm_base_url_env,
        llm_api_key_env=args.llm_api_key_env,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
    )
    env = ALFWorldEnvWrapper(config_path=args.env_config)
    runner = ALFWorldAutoGenRunner(
        env,
        model_client_factory,
        config={"workflow_mode": args.workflow_mode},
    )
    writer = ResultWriter(
        args.output_dir,
        run_id=args.run_id,
        system_id=f"autogen_{args.workflow_mode}",
        system_family="autogen",
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
            "limit": args.limit,
            "seed": args.seed,
            "selection_meta": selection_meta,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "env_config": args.env_config,
            "workflow_mode": args.workflow_mode,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
            "llm_model_env": args.llm_model_env,
            "llm_base_url": args.llm_base_url,
            "llm_base_url_env": args.llm_base_url_env,
            "llm_api_key_env": args.llm_api_key_env,
            "llm_temperature": args.llm_temperature,
            "llm_timeout": args.llm_timeout,
            "n_gamefiles": len(gamefiles),
        }
    )
    evaluator = ALFWorldEvaluator(adapter=runner, result_writer=writer)

    try:
        evaluator.run_batch(
            gamefiles,
            run_id=args.run_id,
            experiment_id="autogen_baseline",
            model_name=args.model_name,
            max_steps=args.max_steps,
        )
        summary = evaluator.finalize()
    finally:
        env.close()

    selected_gamefiles_path = Path(args.output_dir) / "selected_gamefiles.json"
    selected_gamefiles_path.write_text(json.dumps(gamefiles, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_target = (
        selection_meta.get("task_set_file")
        or selection_meta.get("gamefiles_file")
        or selection_meta.get("data_root")
        or args.data_root
    )
    print(f"Selected {len(gamefiles)} gamefiles via {selection_meta.get('selection_source', 'unknown')}: {selection_target}")
    print(f"Selected gamefiles written to: {selected_gamefiles_path}")
    print(f"Episodes written to: {writer.episodes_path}")
    print(f"Canonical episodes written to: {writer.standard_episodes_path}")
    print(f"Summary written to: {writer.summary_path}")
    print(f"Canonical summary written to: {writer.standard_summary_path}")
    print(f"Config snapshot written to: {writer.config_snapshot_path}")
    print(f"Summary metrics: {summary}")


if __name__ == "__main__":
    main()
