"""
Run ALFWorld ablation experiments on a fixed sampled task set.

Usage:
    python run_ablation.py --data-root /path/to/alfworld/data --output-dir outputs/ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.alfworld.core.adapter import ALFWorldKernelAdapter
from experiment.alfworld.environments.env_wrapper import ALFWorldEnvWrapper
from experiment.alfworld.evaluators.alfworld_evaluator import ALFWorldEvaluator
from experiment.alfworld.utils.ablation_matrix import (
    build_ablation_config,
    get_mode_components,
    resolve_ablation_modes,
)
from experiment.alfworld.utils.llm_factory import build_runtime_llm
from experiment.alfworld.utils.result_writer import ResultWriter
from experiment.alfworld.utils.task_selection import select_gamefiles
from experiment.common.result_schema import build_config_snapshot


def run_ablation_suite(
    *,
    gamefiles: List[str],
    modes: List[str],
    output_dir: str,
    evaluator_factory: Callable[[str, str], ALFWorldEvaluator],
    max_steps: int,
    model_name: str,
    workflow_mode: str = "single_action",
    architect_mode: str = "deterministic",
) -> Dict[str, object]:
    """Run all requested modes on the same gamefiles and write a consolidated summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_gamefiles_path = output_path / "selected_gamefiles.json"
    selected_gamefiles_path.write_text(
        json.dumps(gamefiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    suite_config = {
        "gamefiles": gamefiles,
        "modes": modes,
        "max_steps": max_steps,
        "model_name": model_name,
        "workflow_mode": workflow_mode,
        "architect_mode": architect_mode,
    }
    suite_snapshot = build_config_snapshot(
        run_id="ablation_suite",
        system_id="ablation_suite",
        system_family="experiment_suite",
        env_name="alfworld",
        config=suite_config,
        artifact_paths={
            "selected_gamefiles_path": str(selected_gamefiles_path),
            "ablation_summary_path": str(output_path / "ablation_summary.json"),
        },
    )
    (output_path / "config_snapshot.json").write_text(
        json.dumps(suite_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_mode: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        mode_output_dir = output_path / mode
        evaluator = evaluator_factory(mode, str(mode_output_dir))
        try:
            evaluator.run_batch(
                gamefiles,
                run_id=mode,
                experiment_id=mode,
                model_name=model_name,
                max_steps=max_steps,
            )
            summary = dict(evaluator.finalize())
        finally:
            env = getattr(getattr(evaluator, "adapter", None), "env", None)
            if env is not None and hasattr(env, "close"):
                env.close()

        summary["mode"] = mode
        summary["disabled_components"] = list(get_mode_components(mode))
        summary["n_gamefiles"] = len(gamefiles)
        summary["episodes_path"] = str(mode_output_dir / f"{mode}_episodes.jsonl")
        summary["summary_path"] = str(mode_output_dir / f"{mode}_summary.json")
        summary["standard_episodes_path"] = str(mode_output_dir / "episode_results.jsonl")
        summary["standard_summary_path"] = str(mode_output_dir / "summary.json")
        summary["config_snapshot_path"] = str(mode_output_dir / "config_snapshot.json")
        per_mode[mode] = summary

    consolidated = {
        "modes": modes,
        "n_gamefiles": len(gamefiles),
        "workflow_mode": workflow_mode,
        "architect_mode": architect_mode,
        "gamefiles_path": str(selected_gamefiles_path),
        "config_snapshot_path": str(output_path / "config_snapshot.json"),
        "summaries": per_mode,
    }

    summary_path = output_path / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return consolidated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALFWorld ablation experiments")
    parser.add_argument("--data-root", default="", help="ALFWorld dataset root for sampler-based gamefile selection")
    parser.add_argument("--output-dir", default="outputs/ablation", help="Directory for ablation outputs")
    parser.add_argument("--gamefiles-file", default="", help="Optional JSON file containing an explicit gamefile list")
    parser.add_argument("--task-set-file", default="", help="Optional task-set manifest JSON for gamefile selection")
    parser.add_argument(
        "--modes",
        default="full,ablate_c2,ablate_c3",
        help="Comma-separated ablation modes (e.g. full,ablate_c2,ablate_c3)",
    )
    parser.add_argument("--split", choices=["debug", "formal"], default="debug", help="Dataset split size")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sampled gamefiles to run")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum ALFWorld steps per episode")
    parser.add_argument("--model-name", default="deterministic_alfworld", help="Model label used in logs")
    parser.add_argument("--env-config", default=None, help="Optional ALFWorld YAML config path")
    parser.add_argument(
        "--workflow-mode",
        choices=["single_action", "planner_action", "planner_llm_action"],
        default="single_action",
        help="ALFWorld workflow mode used by the architect",
    )
    parser.add_argument(
        "--architect-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Architect implementation used for the ALFWorld workflow",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["none", "mock", "openai_compatible"],
        default="none",
        help="LLM backend for llm workflow modes",
    )
    parser.add_argument("--llm-model", default="", help="Model name for openai_compatible backend")
    parser.add_argument(
        "--llm-model-env",
        default="",
        help="Optional env var name containing the model name for openai_compatible backend",
    )
    parser.add_argument("--llm-base-url", default="", help="Base URL for openai_compatible backend")
    parser.add_argument(
        "--llm-base-url-env",
        default="",
        help="Optional env var name containing the base URL for openai_compatible backend",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="",
        help="Optional env var name containing the API key for openai_compatible backend",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for openai_compatible backend",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for openai_compatible backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = resolve_ablation_modes([m.strip() for m in args.modes.split(",") if m.strip()])
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
    print(f"Running modes: {', '.join(modes)}")
    print(f"Workflow mode: {args.workflow_mode}")
    print(f"Architect mode: {args.architect_mode}")
    print(f"LLM backend: {args.llm_backend}")

    def evaluator_factory(mode: str, mode_output_dir: str) -> ALFWorldEvaluator:
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
                "ablation": build_ablation_config(mode),
                "kernel": {"max_steps": args.max_steps},
                "workflow_mode": args.workflow_mode,
                "architect_mode": args.architect_mode,
            },
        )
        writer = ResultWriter(
            mode_output_dir,
            run_id=mode,
            system_id=f"blackboard_{mode}",
            system_family="blackboard",
            env_name="alfworld",
        )
        writer.write_config_snapshot(
            {
                "mode": mode,
                "data_root": args.data_root,
                "output_dir": mode_output_dir,
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
        return ALFWorldEvaluator(adapter=adapter, result_writer=writer)

    consolidated = run_ablation_suite(
        gamefiles=gamefiles,
        modes=modes,
        output_dir=args.output_dir,
        evaluator_factory=evaluator_factory,
        max_steps=args.max_steps,
        model_name=args.model_name,
        workflow_mode=args.workflow_mode,
        architect_mode=args.architect_mode,
    )

    print(f"Selected gamefiles written to: {consolidated['gamefiles_path']}")
    print(f"Ablation summary written to: {Path(args.output_dir) / 'ablation_summary.json'}")
    for mode in modes:
        mode_summary = consolidated["summaries"][mode]
        print(
            f"{mode}: success_rate={mode_summary.get('success_rate', 0.0):.3f}, "
            f"mean_steps={mode_summary.get('mean_steps', 0.0):.3f}, "
            f"circuit_breaker_trigger_rate={mode_summary.get('circuit_breaker_trigger_rate', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
