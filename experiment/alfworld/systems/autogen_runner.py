"""AutoGen-based ALFWorld baseline runner."""
from __future__ import annotations

import asyncio
import copy
from typing import Any, Callable, Dict, Optional

from experiment.alfworld.utils.autogen_bootstrap import bootstrap_local_autogen

bootstrap_local_autogen()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, RequestUsage

from experiment.alfworld.core.adapter import ALFWorldKernelAdapter
from experiment.alfworld.core.state_bridge import obs_to_kernel_state, update_kernel_state
from experiment.alfworld.utils.autogen_factory import parse_executor_response


def _usage_or_zero(message: Any) -> RequestUsage:
    usage = getattr(message, "models_usage", None)
    if usage is None:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    return RequestUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0)),
        completion_tokens=int(getattr(usage, "completion_tokens", 0)),
    )


class ALFWorldAutoGenRunner:
    """Baseline ALFWorld runner using AutoGen AgentChat."""

    def __init__(
        self,
        env,
        model_client_factory: Callable[[str], ChatCompletionClient],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.model_client_factory = model_client_factory
        self.config = config or {}

    @staticmethod
    def _build_turn_prompt(domain_state: Dict[str, Any]) -> str:
        def _bullet_list(values: list[str]) -> str:
            return "\n".join(f"- {value}" for value in values) if values else "- None"

        return (
            "ALFWORLD TURN CONTEXT\n\n"
            "TASK GOAL:\n"
            f"{domain_state.get('task_goal', '')}\n\n"
            "CURRENT OBSERVATION:\n"
            f"{domain_state.get('current_observation', '')}\n\n"
            "AVAILABLE ACTIONS:\n"
            f"{_bullet_list(list(domain_state.get('available_actions', [])))}\n\n"
            "PREVIOUS ACTIONS:\n"
            f"{_bullet_list(list(domain_state.get('action_history', []))[-5:])}\n\n"
            "GAMEFILE:\n"
            f"{domain_state.get('current_gamefile', '')}\n\n"
            "CANONICAL TASK TYPE:\n"
            f"{domain_state.get('canonical_task_type', '')}\n\n"
            "CANONICAL GOAL OBJECT:\n"
            f"{domain_state.get('canonical_goal_object', '')}\n\n"
            "CANONICAL TARGET RECEPTACLE:\n"
            f"{domain_state.get('canonical_target_receptacle', '')}\n"
        )

    @staticmethod
    def _planner_system_message() -> str:
        return (
            "You are the planner agent for an ALFWorld task.\n"
            "Read the task context and produce a compact natural-language plan for the executor.\n"
            "Return plain text only, using exactly these lines:\n"
            "SUBGOAL: ...\n"
            "FOCUS_OBJECT: ...\n"
            "FOCUS_RECEPTACLE: ...\n"
            "RATIONALE: ...\n"
            "SEARCH_GUIDANCE: ...\n"
            "Do not output JSON."
        )

    @staticmethod
    def _executor_system_message() -> str:
        return (
            "You are the executor agent for an ALFWorld task.\n"
            "Read the planner's natural-language message and the task context.\n"
            "Choose exactly one admissible action from AVAILABLE ACTIONS.\n"
            "Return plain text only, using exactly these lines:\n"
            "ACTION: <exact admissible action>\n"
            "REASON: <brief explanation>\n"
            "Do not output JSON."
        )

    @staticmethod
    def _single_agent_system_message() -> str:
        return (
            "You are an ALFWorld agent.\n"
            "Choose exactly one admissible action from AVAILABLE ACTIONS.\n"
            "Return plain text only, using exactly these lines:\n"
            "ACTION: <exact admissible action>\n"
            "REASON: <brief explanation>\n"
            "Do not output JSON."
        )

    async def _run_autogen_turn(self, domain_state: Dict[str, Any]) -> Dict[str, Any]:
        workflow_mode = self.config.get("workflow_mode", "planner_action")
        turn_prompt = self._build_turn_prompt(domain_state)
        clients: list[ChatCompletionClient] = []

        try:
            if workflow_mode == "single_action":
                client = self.model_client_factory("executor")
                clients.append(client)
                agent = AssistantAgent(
                    "executor",
                    model_client=client,
                    system_message=self._single_agent_system_message(),
                )
                result = await agent.run(task=turn_prompt)
            elif workflow_mode in {"planner_action", "planner_llm_action"}:
                planner_client = self.model_client_factory("planner")
                executor_client = self.model_client_factory("executor")
                clients.extend([planner_client, executor_client])
                planner = AssistantAgent(
                    "planner",
                    model_client=planner_client,
                    system_message=self._planner_system_message(),
                )
                executor = AssistantAgent(
                    "executor",
                    model_client=executor_client,
                    system_message=self._executor_system_message(),
                )
                team = RoundRobinGroupChat(
                    [planner, executor],
                    termination_condition=MaxMessageTermination(3),
                )
                result = await team.run(task=turn_prompt)
            else:
                raise ValueError(f"Unsupported AutoGen workflow_mode: {workflow_mode}")

            messages = list(result.messages)
            communication_trace = []
            worker_input_tokens = 0
            worker_output_tokens = 0
            for message in messages:
                content = message.to_text() if hasattr(message, "to_text") else str(message)
                usage = _usage_or_zero(message)
                worker_input_tokens += usage.prompt_tokens
                worker_output_tokens += usage.completion_tokens
                communication_trace.append(
                    {
                        "source": getattr(message, "source", ""),
                        "content": content,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                )

            admissible_actions = list(domain_state.get("available_actions", []))
            final_text = ""
            for message in reversed(messages):
                if isinstance(message, BaseChatMessage) and getattr(message, "source", "") != "user":
                    final_text = message.to_text()
                    break
            selected_action, decision_reason = parse_executor_response(final_text, admissible_actions)
            return {
                "selected_action": selected_action,
                "decision_reason": decision_reason,
                "communication_trace": communication_trace,
                "worker_input_tokens": worker_input_tokens,
                "worker_output_tokens": worker_output_tokens,
            }
        finally:
            for client in clients:
                await client.close()

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
        """Run one ALFWorld episode with the AutoGen baseline."""
        if self.env.env is None and gamefile is None:
            self.env.init(batch_size=1)
        obs, infos = self.env.reset(gamefile=gamefile)
        kernel_like_state = obs_to_kernel_state(obs[0], infos)
        task_desc = kernel_like_state["domain_state"]["task_goal"]
        gamefile = kernel_like_state["domain_state"].get("current_gamefile", "")

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

        import time as _time
        for step in range(max_steps):
            domain_state = copy.deepcopy(kernel_like_state.get("domain_state", {}))
            last_exc = None
            for _attempt in range(3):
                try:
                    turn_result = asyncio.run(self._run_autogen_turn(domain_state))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    _time.sleep(2)
            if last_exc is not None:
                raise last_exc
            _time.sleep(4.5)
            selected_action = turn_result["selected_action"]
            admissible_actions = infos["admissible_commands"][0]
            fallback_used = selected_action not in admissible_actions
            action = selected_action
            if fallback_used:
                action = "look" if "look" in admissible_actions else (admissible_actions[0] if admissible_actions else "")
                fallback_action_count += 1

            worker_input_tokens += int(turn_result["worker_input_tokens"])
            worker_output_tokens += int(turn_result["worker_output_tokens"])
            patch_err = ""
            step_diagnostics = ALFWorldKernelAdapter._build_step_diagnostics(
                domain_state,
                observation_before=kernel_like_state.get("domain_state", {}).get("current_observation", obs[0]),
                available_actions_before=admissible_actions,
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
                    "decision_reason": turn_result["decision_reason"],
                    "patch_error": patch_err,
                    "retry_count": 0,
                    "communication_trace": turn_result["communication_trace"],
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
                workflow_final_status = "completed"
                break

            if use_episode_breaker:
                pair_repeat = pair_hits[pair] >= repeat_pair_threshold
                observation_repeat = observation_hits[obs[0]] >= repeat_observation_threshold
                action_repeat = action_hits[action] >= repeat_action_threshold
                if pair_repeat or observation_repeat or action_repeat:
                    circuit_breaker_triggered = True
                    circuit_breaker_reason = "episode_stagnation"
                    stop_reason = "stagnation_detected"
                    workflow_final_status = "stagnation"
                    break

            next_state = {
                **kernel_like_state,
                "domain_state": {
                    **kernel_like_state.get("domain_state", {}),
                    "selected_action": action,
                    "decision_reason": turn_result["decision_reason"],
                },
            }
            kernel_like_state = update_kernel_state(next_state, obs[0], infos)
        else:
            stop_reason = "max_steps_reached"
            workflow_final_status = "incomplete"

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
            "captured_states": captured_states if capture_states else [],
        }
