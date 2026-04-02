"""Run one real AppWorld blackboard smoke task with an OpenAI-compatible LLM."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appworld import load_task_ids

from experiment.appworld.systems import run_blackboard_task
from experiment.common.openai_compatible_llm import chat_text


class OpenAICompatibleInvoker:
    """Minimal invoke() adapter expected by architect/state/response workers."""

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> Any:
        response = chat_text(
            system_prompt=(
                "You are a precise JSON-only planning and state synthesis model. "
                "Return exactly one JSON object and no surrounding prose."
            ),
            user_prompt=prompt,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            force_json_mode=True,
        )
        return SimpleNamespace(content=str(response["raw_text"] or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AppWorld blackboard smoke task")
    parser.add_argument("--task-id", default="", help="AppWorld task id. Defaults to the first task in the selected dataset.")
    parser.add_argument("--dataset-name", default="dev", help="AppWorld split used when task id is omitted.")
    parser.add_argument("--experiment-name", default="appworld_blackboard_smoke", help="Experiment name for AppWorld.")
    parser.add_argument("--remote-apis-url", required=True, help="Remote AppWorld API server URL.")
    parser.add_argument("--appworld-root", default="", help="Optional local AppWorld repo root.")
    parser.add_argument("--llm-model", default="", help="OpenAI-compatible model name.")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-api-key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--timeout", type=float, default=60.0, help="OpenAI-compatible timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max completion tokens.")
    parser.add_argument("--max-steps", type=int, default=3, help="Max neutral turns for the smoke run.")
    parser.add_argument("--max-kernel-steps", type=int, default=12, help="Max kernel steps per turn.")
    parser.add_argument("--max-no-progress", type=int, default=2, help="Circuit-break threshold.")
    parser.add_argument("--wall-clock-timeout", type=int, default=0, help="Optional whole-run timeout in seconds. 0 disables it.")
    parser.add_argument("--summary-only", action="store_true", help="Print only a concise debug summary instead of the full JSON payload.")
    parser.add_argument(
        "--load-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load AppWorld ground truth so official evaluation can run.",
    )
    parser.add_argument(
        "--ground-truth-mode",
        default="minimal",
        help="AppWorld ground truth mode passed during task load.",
    )
    return parser.parse_args()


def _timeout_handler(signum, frame):
    del signum, frame
    raise TimeoutError("Smoke run exceeded wall-clock timeout.")


def _build_payload(result: Any) -> dict[str, Any]:
    return {
        "task_id": result.task.task_id,
        "finished": result.finished,
        "steps": result.steps,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "name": message.name,
                "metadata": dict(message.metadata),
            }
            for message in result.messages
        ],
        "tool_results": [
            {
                "tool_name": tool_result.tool_name,
                "status": tool_result.status.value,
                "content": tool_result.content,
                "metadata": dict(tool_result.metadata),
            }
            for tool_result in result.tool_results
        ],
        "run_metadata": dict(result.run_metadata),
        "evaluation": dict(result.evaluation),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    run_metadata = dict(payload.get("run_metadata") or {})
    tool_status = dict(run_metadata.get("tool_status") or {})
    progress_state = dict(run_metadata.get("progress_state") or {})
    evaluation = dict(payload.get("evaluation") or {})
    summary = {
        "task_id": payload.get("task_id"),
        "finished": payload.get("finished"),
        "steps": payload.get("steps"),
        "kernel_status": run_metadata.get("kernel_status"),
        "used_fallback_architect": run_metadata.get("used_fallback_architect"),
        "fallback_reason": run_metadata.get("fallback_reason"),
        "architect_debug": dict(run_metadata.get("architect_debug") or {}),
        "active_worker": run_metadata.get("active_worker"),
        "last_worker": run_metadata.get("last_worker"),
        "finish_reason": run_metadata.get("finish_reason"),
        "tool_status": tool_status,
        "progress_state": progress_state,
        "evaluation": {
            "success": evaluation.get("success"),
            "task_goal_completion": evaluation.get("task_goal_completion"),
            "scenario_goal_completion": evaluation.get("scenario_goal_completion"),
            "evaluation_error": evaluation.get("evaluation_error"),
        },
        "history_tail": list(run_metadata.get("history_tail") or []),
        "last_message": payload.get("messages", [])[-1] if payload.get("messages") else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    task_id = args.task_id or load_task_ids(args.dataset_name)[0]
    llm = OpenAICompatibleInvoker(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    started_at = time.time()
    if args.wall_clock_timeout > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(args.wall_clock_timeout))
    try:
        result = run_blackboard_task(
            task_id=task_id,
            experiment_name=args.experiment_name,
            dataset_name=args.dataset_name,
            llm=llm,
            remote_apis_url=args.remote_apis_url,
            appworld_root=args.appworld_root or None,
            load_ground_truth=bool(args.load_ground_truth),
            ground_truth_mode=str(args.ground_truth_mode or "minimal"),
            system_config={
                "max_kernel_steps": args.max_kernel_steps,
                "max_no_progress": args.max_no_progress,
            },
            max_steps=args.max_steps,
        )
        payload = _build_payload(result)
        payload["wall_clock_seconds"] = round(time.time() - started_at, 2)
        if args.summary_only:
            _print_summary(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    except TimeoutError as exc:
        payload = {
            "task_id": task_id,
            "timed_out": True,
            "error": str(exc),
            "wall_clock_seconds": round(time.time() - started_at, 2),
            "hint": "Lower --max-steps/--max-kernel-steps or inspect worker prompts and remote API latency.",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(124) from exc
    finally:
        if args.wall_clock_timeout > 0:
            signal.alarm(0)


if __name__ == "__main__":
    main()
