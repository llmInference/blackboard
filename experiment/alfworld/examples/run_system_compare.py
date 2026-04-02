"""Run fixed-gamefile ALFWorld comparisons across system families."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.alfworld.core.adapter import ALFWorldKernelAdapter
from experiment.alfworld.environments.env_wrapper import ALFWorldEnvWrapper
from experiment.alfworld.evaluators.alfworld_evaluator import ALFWorldEvaluator
from experiment.alfworld.systems.autogen_runner import ALFWorldAutoGenRunner
from experiment.alfworld.systems.langgraph_runner import ALFWorldLangGraphRunner
from experiment.alfworld.utils.ablation_matrix import build_ablation_config
from experiment.alfworld.utils.autogen_factory import build_autogen_model_client_factory
from experiment.alfworld.utils.llm_factory import build_runtime_llm
from experiment.alfworld.utils.result_writer import ResultWriter
from experiment.alfworld.utils.task_selection import select_gamefiles
from experiment.common.result_schema import build_config_snapshot


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from disk."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedupe_episode_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one record per gamefile/task, preferring the latest payload."""
    deduped: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []
    for record in records:
        key = str(record.get("gamefile") or record.get("task_id") or record.get("episode_id") or "")
        if key not in deduped:
            ordered_keys.append(key)
        deduped[key] = record
    return [deduped[key] for key in ordered_keys]


def rewrite_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Rewrite one JSONL file with deduped records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_resume_gamefiles(*, gamefiles: List[str], writer: ResultWriter) -> Dict[str, Any]:
    """Compute remaining gamefiles for resume and normalize existing JSONL artifacts."""
    existing_records = load_jsonl(writer.episodes_path)
    if not existing_records and writer.standard_episodes_path.exists():
        existing_records = load_jsonl(writer.standard_episodes_path)

    deduped_records = dedupe_episode_records(existing_records)
    if existing_records and len(deduped_records) != len(existing_records):
        rewrite_jsonl(writer.episodes_path, deduped_records)
        rewrite_jsonl(writer.standard_episodes_path, deduped_records)

    completed_gamefiles = {
        str(record.get("gamefile") or record.get("task_id") or "")
        for record in deduped_records
        if str(record.get("gamefile") or record.get("task_id") or "").strip()
    }
    remaining_gamefiles = [gamefile for gamefile in gamefiles if str(gamefile) not in completed_gamefiles]
    return {
        "completed_count": len(completed_gamefiles),
        "remaining_count": len(remaining_gamefiles),
        "remaining_gamefiles": remaining_gamefiles,
        "deduped_existing_count": len(deduped_records),
    }


def run_system_compare(
    *,
    gamefiles: List[str],
    systems: List[str],
    output_dir: str,
    run_system_suite,
    workflow_mode: str,
    architect_mode: str,
) -> Dict[str, object]:
    """Run the same sampled gamefiles across multiple system families."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_gamefiles_path = output_path / "selected_gamefiles.json"
    selected_gamefiles_path.write_text(
        json.dumps(gamefiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    suite_snapshot = build_config_snapshot(
        run_id="system_compare_suite",
        system_id="system_compare_suite",
        system_family="experiment_suite",
        env_name="alfworld",
        config={
            "gamefiles": gamefiles,
            "systems": systems,
            "workflow_mode": workflow_mode,
            "architect_mode": architect_mode,
        },
        artifact_paths={
            "selected_gamefiles_path": str(selected_gamefiles_path),
            "system_compare_summary_path": str(output_path / "system_compare_summary.json"),
        },
    )
    (output_path / "config_snapshot.json").write_text(
        json.dumps(suite_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    system_summaries: Dict[str, Dict[str, object]] = {}
    for system in systems:
        system_output_dir = output_path / system
        system_result = run_system_suite(system, str(system_output_dir))
        system_summaries[system] = {
            "system": system,
            "system_output_dir": str(system_output_dir),
            "result": system_result,
        }

    consolidated = {
        "systems": systems,
        "n_gamefiles": len(gamefiles),
        "workflow_mode": workflow_mode,
        "architect_mode": architect_mode,
        "gamefiles_path": str(selected_gamefiles_path),
        "config_snapshot_path": str(output_path / "config_snapshot.json"),
        "system_summaries": system_summaries,
    }

    summary_path = output_path / "system_compare_summary.json"
    summary_path.write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return consolidated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ALFWorld systems on fixed sampled gamefiles")
    parser.add_argument("--data-root", default="", help="ALFWorld dataset root for sampler-based gamefile selection")
    parser.add_argument("--output-dir", default="outputs/system_compare", help="Directory for compare outputs")
    parser.add_argument("--gamefiles-file", default="", help="Optional JSON file containing an explicit gamefile list")
    parser.add_argument("--task-set-file", default="", help="Optional task-set manifest JSON for gamefile selection")
    parser.add_argument(
        "--systems",
        default="blackboard,langgraph,autogen",
        help="Comma-separated systems to compare",
    )
    parser.add_argument("--split", choices=["debug", "formal"], default="debug", help="Dataset split size")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sampled gamefiles to run")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum ALFWorld steps per episode")
    parser.add_argument("--model-name", default="alfworld_system_compare", help="Model label used in logs")
    parser.add_argument("--env-config", default=None, help="Optional ALFWorld YAML config path")
    parser.add_argument(
        "--workflow-mode",
        choices=["single_action", "planner_action", "planner_llm_action"],
        default="single_action",
        help="Workflow shape used across systems",
    )
    parser.add_argument(
        "--blackboard-mode",
        default="full",
        help="Ablation mode used for the blackboard system",
    )
    parser.add_argument(
        "--architect-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Architect implementation used for blackboard/langgraph",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["none", "mock", "openai_compatible"],
        default="mock",
        help="LLM backend for systems that require an LLM",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-system episode files in the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = [system.strip() for system in args.systems.split(",") if system.strip()]
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
    print(f"Systems: {', '.join(systems)}")
    print(f"Workflow mode: {args.workflow_mode}")
    print(f"Architect mode: {args.architect_mode}")
    print(f"LLM backend: {args.llm_backend}")

    def run_system_suite(system: str, system_output_dir: str) -> Dict[str, object]:
        env = ALFWorldEnvWrapper(config_path=args.env_config)
        try:
            if system == "blackboard":
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
                        "ablation": build_ablation_config(args.blackboard_mode),
                        "kernel": {"max_steps": args.max_steps},
                        "workflow_mode": args.workflow_mode,
                        "architect_mode": args.architect_mode,
                    },
                )
                run_id = f"blackboard_{args.blackboard_mode}"
                experiment_id = f"blackboard:{args.blackboard_mode}"
                writer = ResultWriter(
                    system_output_dir,
                    run_id=run_id,
                    system_id=f"blackboard_{args.workflow_mode}_{args.blackboard_mode}",
                    system_family="blackboard",
                    env_name="alfworld",
                )
                writer.write_config_snapshot(
                    {
                        "system": "blackboard",
                        "mode": args.blackboard_mode,
                        "workflow_mode": args.workflow_mode,
                        "architect_mode": args.architect_mode,
                        "data_root": args.data_root,
                        "output_dir": system_output_dir,
                        "gamefiles_file": args.gamefiles_file,
                        "task_set_file": args.task_set_file,
                        "split": args.split,
                        "limit": args.limit,
                        "seed": args.seed,
                        "selection_meta": selection_meta,
                        "max_steps": args.max_steps,
                        "model_name": args.model_name,
                        "env_config": args.env_config,
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
            elif system == "langgraph":
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
                adapter = ALFWorldLangGraphRunner(
                    env,
                    llm,
                    config={
                        "workflow_mode": args.workflow_mode,
                        "architect_mode": args.architect_mode,
                    },
                )
                run_id = "langgraph"
                experiment_id = "langgraph_baseline"
                writer = ResultWriter(
                    system_output_dir,
                    run_id=run_id,
                    system_id=f"langgraph_{args.workflow_mode}",
                    system_family="langgraph",
                    env_name="alfworld",
                )
                writer.write_config_snapshot(
                    {
                        "system": "langgraph",
                        "workflow_mode": args.workflow_mode,
                        "architect_mode": args.architect_mode,
                        "data_root": args.data_root,
                        "output_dir": system_output_dir,
                        "gamefiles_file": args.gamefiles_file,
                        "task_set_file": args.task_set_file,
                        "split": args.split,
                        "limit": args.limit,
                        "seed": args.seed,
                        "selection_meta": selection_meta,
                        "max_steps": args.max_steps,
                        "model_name": args.model_name,
                        "env_config": args.env_config,
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
            elif system == "autogen":
                model_client_factory = build_autogen_model_client_factory(
                    llm_backend="mock" if args.llm_backend == "none" else args.llm_backend,
                    llm_model=args.llm_model,
                    llm_model_env=args.llm_model_env,
                    llm_base_url=args.llm_base_url,
                    llm_base_url_env=args.llm_base_url_env,
                    llm_api_key_env=args.llm_api_key_env,
                    llm_temperature=args.llm_temperature,
                    llm_timeout=args.llm_timeout,
                )
                adapter = ALFWorldAutoGenRunner(
                    env,
                    model_client_factory,
                    config={"workflow_mode": args.workflow_mode},
                )
                run_id = "autogen"
                experiment_id = "autogen_baseline"
                writer = ResultWriter(
                    system_output_dir,
                    run_id=run_id,
                    system_id=f"autogen_{args.workflow_mode}",
                    system_family="autogen",
                    env_name="alfworld",
                )
                writer.write_config_snapshot(
                    {
                        "system": "autogen",
                        "workflow_mode": args.workflow_mode,
                        "data_root": args.data_root,
                        "output_dir": system_output_dir,
                        "gamefiles_file": args.gamefiles_file,
                        "task_set_file": args.task_set_file,
                        "split": args.split,
                        "limit": args.limit,
                        "seed": args.seed,
                        "selection_meta": selection_meta,
                        "max_steps": args.max_steps,
                        "model_name": args.model_name,
                        "env_config": args.env_config,
                        "llm_backend": "mock" if args.llm_backend == "none" else args.llm_backend,
                        "llm_model": args.llm_model,
                        "llm_model_env": args.llm_model_env,
                        "llm_base_url": args.llm_base_url,
                        "llm_base_url_env": args.llm_base_url_env,
                        "llm_api_key_env": args.llm_api_key_env,
                        "llm_temperature": args.llm_temperature,
                        "llm_timeout": args.llm_timeout,
                    }
                )
            else:
                raise ValueError(f"Unsupported system: {system}")

            evaluator = ALFWorldEvaluator(adapter=adapter, result_writer=writer)
            gamefiles_to_run = list(gamefiles)
            if args.resume:
                resume_state = prepare_resume_gamefiles(gamefiles=gamefiles, writer=writer)
                gamefiles_to_run = list(resume_state["remaining_gamefiles"])
                print(
                    f"[resume:{system}] completed={resume_state['completed_count']} "
                    f"remaining={resume_state['remaining_count']}",
                    flush=True,
                )
            if gamefiles_to_run:
                evaluator.run_batch(
                    gamefiles_to_run,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    model_name=args.model_name,
                    max_steps=args.max_steps,
                )
            else:
                print(f"[resume:{system}] all requested gamefiles already completed", flush=True)
            summary = evaluator.finalize()
            return {
                "system": system,
                "system_id": writer.system_id,
                "system_family": writer.system_family,
                "workflow_mode": args.workflow_mode,
                "architect_mode": args.architect_mode if system in {"blackboard", "langgraph"} else "",
                "summary": summary,
                "episodes_path": str(writer.episodes_path),
                "summary_path": str(writer.summary_path),
                "standard_episodes_path": str(writer.standard_episodes_path),
                "standard_summary_path": str(writer.standard_summary_path),
                "config_snapshot_path": str(writer.config_snapshot_path),
            }
        finally:
            env.close()

    consolidated = run_system_compare(
        gamefiles=gamefiles,
        systems=systems,
        output_dir=args.output_dir,
        run_system_suite=run_system_suite,
        workflow_mode=args.workflow_mode,
        architect_mode=args.architect_mode,
    )

    print(f"Selected gamefiles written to: {consolidated['gamefiles_path']}")
    print(f"System compare summary written to: {Path(args.output_dir) / 'system_compare_summary.json'}")
    for system in systems:
        summary = consolidated["system_summaries"][system]["result"]["summary"]
        print(
            f"{system}: "
            f"success_rate={summary.get('success_rate', 0.0):.3f}, "
            f"mean_steps={summary.get('mean_steps', 0.0):.3f}, "
            f"mean_total_tokens={summary.get('mean_total_tokens', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
