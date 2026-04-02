"""Plain LangGraph baseline runner for ALFWorld."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[3]
_LANGGRAPH_LIB_ROOT = _ROOT / "langgraph" / "libs" / "langgraph"
if str(_LANGGRAPH_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANGGRAPH_LIB_ROOT))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from experiment.alfworld.core.adapter import ALFWorldKernelAdapter
from experiment.alfworld.core.state_bridge import (
    extract_action_with_meta,
    obs_to_kernel_state,
    update_kernel_state,
)
from experiment.alfworld.workers.action_worker import ActionWorker
from experiment.alfworld.workers.alfworld_architect import ALFWorldArchitect
from experiment.alfworld.workers.llm_action_worker import LLMActionWorker
from experiment.alfworld.workers.llm_alfworld_architect import LLMALFWorldArchitect
from experiment.alfworld.workers.planner_worker import PlannerWorker


class PlainALFWorldState(TypedDict, total=False):
    """Minimal state used by the plain LangGraph baseline."""

    user_prompt: str
    domain_state: Dict[str, Any]
    data_schema: Dict[str, Any]
    task_flow: list[Dict[str, Any]]
    worker_instructions: Dict[str, str]
    selected_workers: list[str]
    patch_error: str
    no_update_count: int
    retry_count: int
    error_feedback: str
    turn_worker_input_tokens: int
    turn_worker_output_tokens: int
    turn_architect_input_tokens: int
    turn_architect_output_tokens: int


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _parse_pointer(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"unsupported patch path: {path!r}")
    if path == "/":
        return []
    return [_decode_pointer_token(token) for token in path.lstrip("/").split("/")]


def _resolve_parent(container: Any, tokens: list[str]) -> tuple[Any, str]:
    if not tokens:
        raise ValueError("root replacement is not supported")

    current = container
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"missing path segment: {token}")
            current = current[token]
            continue
        if isinstance(current, list):
            index = int(token)
            current = current[index]
            continue
        raise TypeError(f"unsupported container type at {token}: {type(current).__name__}")
    return current, tokens[-1]


def apply_plain_json_patch(domain_state: Dict[str, Any], patch: Any) -> tuple[Dict[str, Any], str, int]:
    """Apply a restricted JSON Patch list directly to domain_state."""
    if not isinstance(patch, list):
        return domain_state, "plain_langgraph_invalid_patch_payload", 0

    updated = copy.deepcopy(domain_state)
    applied_ops = 0

    try:
        for op in patch:
            if not isinstance(op, dict):
                raise ValueError(f"invalid patch operation: {op!r}")
            operation = str(op.get("op", "")).strip()
            path = str(op.get("path", "")).strip()
            tokens = _parse_pointer(path)
            parent, key = _resolve_parent(updated, tokens)

            if isinstance(parent, dict):
                exists = key in parent
                if operation == "add":
                    parent[key] = op.get("value")
                elif operation == "replace":
                    if not exists:
                        raise KeyError(f"replace target missing: {path}")
                    parent[key] = op.get("value")
                elif operation == "remove":
                    if not exists:
                        raise KeyError(f"remove target missing: {path}")
                    parent.pop(key)
                else:
                    raise ValueError(f"unsupported patch op: {operation}")
            elif isinstance(parent, list):
                if key == "-":
                    if operation != "add":
                        raise ValueError(f"unsupported list operation for '-': {operation}")
                    parent.append(op.get("value"))
                else:
                    index = int(key)
                    if operation == "add":
                        parent.insert(index, op.get("value"))
                    elif operation == "replace":
                        parent[index] = op.get("value")
                    elif operation == "remove":
                        parent.pop(index)
                    else:
                        raise ValueError(f"unsupported patch op: {operation}")
            else:
                raise TypeError(f"unsupported patch target: {type(parent).__name__}")
            applied_ops += 1
    except Exception as exc:
        return domain_state, f"plain_langgraph_patch_error:{exc}", applied_ops

    return updated, "", applied_ops


class ALFWorldLangGraphRunner:
    """Baseline ALFWorld runner built with plain LangGraph, without kernel_system."""

    def __init__(self, env, llm: Optional[BaseChatModel], config: Optional[Dict[str, Any]] = None):
        self.env = env
        self.llm = llm
        self.config = config or {}
        self.turn_graph = None

    def _build_architect(self):
        workflow_mode = self.config.get("workflow_mode", "single_action")
        architect_mode = self.config.get("architect_mode", "deterministic")
        if architect_mode == "llm":
            if self.llm is None:
                raise ValueError("architect_mode='llm' requires a configured llm")
            return LLMALFWorldArchitect(self.llm, workflow_mode=workflow_mode)
        return ALFWorldArchitect(self.llm, workflow_mode=workflow_mode)

    def _make_worker(self, worker_name: str, instruction: str | None):
        if worker_name == "planner_worker":
            return PlannerWorker(self.llm, instruction)
        if worker_name == "llm_action_worker":
            if self.llm is None:
                raise ValueError("workflow_mode='planner_llm_action' requires a configured llm")
            return LLMActionWorker(self.llm, instruction)
        if worker_name == "action_worker":
            return ActionWorker(self.llm, instruction)
        raise ValueError(f"unsupported ALFWorld worker: {worker_name}")

    def _run_worker_node(
        self,
        state: PlainALFWorldState,
        *,
        worker_name: str,
        next_status: str,
    ) -> Dict[str, Any]:
        instructions = state.get("worker_instructions", {}) or {}
        worker = self._make_worker(worker_name, instructions.get(worker_name))
        worker_state = {
            "domain_state": copy.deepcopy(state.get("domain_state", {})),
            "data_schema": copy.deepcopy(state.get("data_schema", {})),
            "retry_count": int(state.get("retry_count", 0)),
            "error_feedback": state.get("error_feedback", ""),
        }
        worker_result = worker(worker_state)
        pending_patch = worker_result.get("pending_patch", [])
        updated_domain_state, patch_error, applied_ops = apply_plain_json_patch(
            state.get("domain_state", {}),
            pending_patch,
        )
        updated_domain_state["workflow_status"] = next_status

        no_update_count = 0
        if patch_error:
            no_update_count = 1
        elif applied_ops == 0:
            no_update_count = 1

        return {
            "domain_state": updated_domain_state,
            "patch_error": patch_error,
            "no_update_count": no_update_count,
            "turn_worker_input_tokens": int(state.get("turn_worker_input_tokens", 0))
            + int(worker_result.get("worker_input_tokens", 0)),
            "turn_worker_output_tokens": int(state.get("turn_worker_output_tokens", 0))
            + int(worker_result.get("worker_output_tokens", 0)),
        }

    def build_turn_graph(self):
        """Build one-turn plain LangGraph workflow."""
        workflow_mode = self.config.get("workflow_mode", "single_action")
        architect = self._build_architect()

        def architect_node(state: PlainALFWorldState) -> Dict[str, Any]:
            result = architect(state)
            return {
                "domain_state": result.get("domain_state", {}),
                "data_schema": result.get("data_schema", {}),
                "task_flow": result.get("task_flow", []),
                "worker_instructions": result.get("worker_instructions", {}),
                "selected_workers": result.get("selected_workers", []),
                "patch_error": "",
                "no_update_count": 0,
                "retry_count": 0,
                "error_feedback": "",
                "turn_architect_input_tokens": int(result.get("turn_architect_input_tokens", 0)),
                "turn_architect_output_tokens": int(result.get("turn_architect_output_tokens", 0)),
                "turn_worker_input_tokens": 0,
                "turn_worker_output_tokens": 0,
            }

        builder = StateGraph(PlainALFWorldState)
        builder.add_node("architect", architect_node)
        if workflow_mode == "single_action":
            builder.add_node(
                "action",
                lambda state: self._run_worker_node(state, worker_name="action_worker", next_status="completed"),
            )
            builder.add_edge(START, "architect")
            builder.add_edge("architect", "action")
            builder.add_edge("action", END)
        elif workflow_mode == "planner_action":
            builder.add_node(
                "planner",
                lambda state: self._run_worker_node(state, worker_name="planner_worker", next_status="acting"),
            )
            builder.add_node(
                "action",
                lambda state: self._run_worker_node(state, worker_name="action_worker", next_status="completed"),
            )
            builder.add_edge(START, "architect")
            builder.add_edge("architect", "planner")
            builder.add_edge("planner", "action")
            builder.add_edge("action", END)
        elif workflow_mode == "planner_llm_action":
            builder.add_node(
                "planner",
                lambda state: self._run_worker_node(state, worker_name="planner_worker", next_status="acting"),
            )
            builder.add_node(
                "action",
                lambda state: self._run_worker_node(
                    state,
                    worker_name="llm_action_worker",
                    next_status="completed",
                ),
            )
            builder.add_edge(START, "architect")
            builder.add_edge("architect", "planner")
            builder.add_edge("planner", "action")
            builder.add_edge("action", END)
        else:
            raise ValueError(f"Unsupported workflow_mode: {workflow_mode}")

        return builder.compile()

    def run_episode(
        self,
        max_steps: int = 50,
        episode_id: str = "",
        run_id: str = "",
        experiment_id: str = "",
        model_name: str = "",
        token_callback=None,
        gamefile: Optional[str] = None,
        capture_states: bool = False,
    ) -> Dict[str, Any]:
        """Run one ALFWorld episode with a plain LangGraph baseline."""
        if self.env.env is None and gamefile is None:
            self.env.init(batch_size=1)
        obs, infos = self.env.reset(gamefile=gamefile)
        kernel_like_state = obs_to_kernel_state(obs[0], infos)
        task_desc = kernel_like_state["domain_state"]["task_goal"]
        gamefile = kernel_like_state["domain_state"].get("current_gamefile", "")

        if self.turn_graph is None:
            self.turn_graph = self.build_turn_graph()

        trajectory = []
        fallback_action_count = 0
        patch_error_count = 0
        no_update_event_count = 0
        repeat_action_count = 0
        repeat_observation_count = 0
        stagnation_event_count = 0
        worker_input_tokens = 0
        worker_output_tokens = 0
        architect_input_tokens = 0
        architect_output_tokens = 0
        captured_states = []
        circuit_breaker_triggered = False
        circuit_breaker_reason = ""
        waiting_for_user = False
        workflow_final_status = kernel_like_state["domain_state"].get("workflow_status", "deciding")
        stop_reason = "max_steps_reached"
        latest_infos = infos
        action_hits: Dict[str, int] = {}
        observation_hits: Dict[str, int] = {}
        pair_hits: Dict[tuple[str, str], int] = {}
        previous_action = ""
        previous_observation = obs[0]

        breaker_cfg = self.config.get("episode_breaker", {})
        repeat_pair_threshold = max(int(breaker_cfg.get("repeat_pair_threshold", 3)), 2)
        repeat_observation_threshold = max(int(breaker_cfg.get("repeat_observation_threshold", 3)), 2)
        repeat_action_threshold = max(int(breaker_cfg.get("repeat_action_threshold", 3)), 2)
        use_episode_breaker = bool(self.config.get("use_episode_breaker", True))

        for step in range(max_steps):
            result = self.turn_graph.invoke(kernel_like_state)

            token_info = {
                "worker_input_tokens": int(result.get("turn_worker_input_tokens", 0)),
                "worker_output_tokens": int(result.get("turn_worker_output_tokens", 0)),
                "architect_input_tokens": int(result.get("turn_architect_input_tokens", 0)),
                "architect_output_tokens": int(result.get("turn_architect_output_tokens", 0)),
            }
            if token_callback is not None:
                token_info = token_callback(result)
            worker_input_tokens += token_info.get("worker_input_tokens", 0)
            worker_output_tokens += token_info.get("worker_output_tokens", 0)
            architect_input_tokens += token_info.get("architect_input_tokens", 0)
            architect_output_tokens += token_info.get("architect_output_tokens", 0)

            patch_err = result.get("patch_error", "")
            if patch_err:
                patch_error_count += 1
            if result.get("no_update_count", 0):
                no_update_event_count += 1

            domain_state = result.get("domain_state", {})
            workflow_final_status = domain_state.get("workflow_status", workflow_final_status)

            action, fallback_used = extract_action_with_meta(result, infos["admissible_commands"][0])
            if fallback_used:
                fallback_action_count += 1

            step_diagnostics = ALFWorldKernelAdapter._build_step_diagnostics(
                domain_state,
                observation_before=kernel_like_state.get("domain_state", {}).get("current_observation", obs[0]),
                available_actions_before=infos["admissible_commands"][0],
            )

            if capture_states:
                selected_action = result.get("domain_state", {}).get("selected_action", "")
                fallback_reason = ""
                fallback_source = ""
                if fallback_used:
                    fallback_reason = f"invalid_or_empty_action:{selected_action or '<empty>'}"
                    fallback_source = "look" if "look" in infos["admissible_commands"][0] else "first_admissible"
                captured_states.append(
                    ALFWorldKernelAdapter._capture_runtime_state(
                        result,
                        episode_id=episode_id,
                        gamefile=gamefile,
                        step_id=step,
                        fallback_used=fallback_used,
                        fallback_action=action,
                        fallback_reason=fallback_reason,
                        fallback_source=fallback_source,
                        default_data_schema=kernel_like_state.get("data_schema", {}),
                    )
                )

            obs, reward, done, infos = self.env.step([action])
            latest_infos = infos

            trajectory.append(
                {
                    "step_id": step,
                    "action": action,
                    "fallback_used": fallback_used,
                    "observation": obs[0],
                    "reward": reward[0],
                    "done": done[0],
                    "decision_reason": domain_state.get("decision_reason", ""),
                    "patch_error": patch_err,
                    "retry_count": result.get("retry_count", 0),
                    **step_diagnostics,
                }
            )

            action_hits[action] = action_hits.get(action, 0) + 1
            observation_hits[obs[0]] = observation_hits.get(obs[0], 0) + 1
            pair = (action, obs[0])
            pair_hits[pair] = pair_hits.get(pair, 0) + 1

            if previous_action and action == previous_action:
                repeat_action_count += 1
            if previous_observation == obs[0]:
                repeat_observation_count += 1
            if pair_hits[pair] > 1 or observation_hits[obs[0]] > 1:
                stagnation_event_count += 1

            previous_action = action
            previous_observation = obs[0]

            if done[0]:
                stop_reason = "environment_done"
                break

            if use_episode_breaker:
                pair_repeat = pair_hits[pair] >= repeat_pair_threshold
                observation_repeat = observation_hits[obs[0]] >= repeat_observation_threshold
                action_repeat = action_hits[action] >= repeat_action_threshold
                if pair_repeat or observation_repeat or action_repeat:
                    circuit_breaker_triggered = True
                    circuit_breaker_reason = "episode_stagnation"
                    stop_reason = "stagnation_detected"
                    break

            kernel_like_state = update_kernel_state(result, obs[0], infos)
        else:
            stop_reason = "max_steps_reached"

        total_tokens = worker_input_tokens + worker_output_tokens + architect_input_tokens + architect_output_tokens
        won_flags = latest_infos.get("won", [False]) if isinstance(latest_infos, dict) else [False]
        success = bool(won_flags[0]) if won_flags else False
        goal_condition = (
            latest_infos.get("goal_condition_success_rate", [1.0 if success else 0.0])[0]
            if isinstance(latest_infos, dict)
            else (1.0 if success else 0.0)
        )
        if success:
            final_status = "success"
        elif waiting_for_user:
            final_status = "waiting_for_user"
        elif circuit_breaker_triggered:
            final_status = "stagnation" if stop_reason == "stagnation_detected" else "circuit_breaker"
        elif stop_reason == "max_steps_reached":
            final_status = "incomplete"
        else:
            final_status = "terminated"

        return {
            "episode_id": episode_id,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "gamefile": gamefile,
            "task_goal": task_desc,
            "ablation_config": {},
            "trajectory": trajectory,
            "success": success,
            "steps": len(trajectory),
            "goal_condition_rate": goal_condition,
            "fallback_action_count": fallback_action_count,
            "patch_error_count": patch_error_count,
            "no_update_event_count": no_update_event_count,
            "repeat_action_count": repeat_action_count,
            "repeat_observation_count": repeat_observation_count,
            "stagnation_event_count": stagnation_event_count,
            "worker_input_tokens": worker_input_tokens,
            "worker_output_tokens": worker_output_tokens,
            "architect_input_tokens": architect_input_tokens,
            "architect_output_tokens": architect_output_tokens,
            "total_tokens": total_tokens,
            "stop_reason": stop_reason,
            "waiting_for_user": waiting_for_user,
            "circuit_breaker_triggered": circuit_breaker_triggered,
            "circuit_breaker_reason": circuit_breaker_reason,
            "final_status": final_status,
            "workflow_final_status": workflow_final_status,
            "captured_states": captured_states,
        }

