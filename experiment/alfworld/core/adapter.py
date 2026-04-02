"""
ALFWorld-Kernel Adapter

Core adapter that bridges ALFWorld Gym interface and Kernel System StateGraph.
This is the main integration point between the two systems.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.kernel.node import kernel_node
from langgraph_kernel.kernel.router import WorkflowRouter
from langgraph_kernel.types import KernelState
from langgraph_kernel.worker.registry import get_registry
from langgraph_kernel.workflow import extract_worker_name, get_matching_workflow_entry

from .state_bridge import extract_action, extract_action_with_meta, obs_to_kernel_state, update_kernel_state


class ALFWorldKernelAdapter:
    """
    Adapter bridging ALFWorld Gym interface and Kernel System StateGraph.

    This adapter:
    1. Manages ALFWorld environment and Kernel System graph lifecycle
    2. Coordinates ALFWorld step loop and Kernel System invoke calls
    3. Handles state conversion and action extraction
    """

    def __init__(self, env, llm: Optional[BaseChatModel], config: Optional[Dict] = None):
        """
        Initialize the adapter.

        Args:
            env: ALFWorld environment (ALFWorldEnvWrapper)
            llm: Language model for Kernel System
            config: Configuration dictionary
        """
        self.env = env
        self.llm = llm
        self.config = config or {}
        self.kernel_graph = None

    def build_kernel_graph(self):
        """
        Build Kernel System graph with ALFWorld-specific workers.

        This method imports and registers ALFWorld workers, then builds
        the Kernel System graph.
        """
        # Import and register ALFWorld workers
        try:
            from ..workers.alfworld_architect import ALFWorldArchitect
            from ..workers.action_worker import ActionWorker
            from ..workers.llm_alfworld_architect import LLMALFWorldArchitect
            from ..workers.llm_action_worker import LLMActionWorker
            from ..workers.planner_worker import PlannerWorker
        except ImportError:
            from workers.alfworld_architect import ALFWorldArchitect
            from workers.action_worker import ActionWorker
            from workers.llm_alfworld_architect import LLMALFWorldArchitect
            from workers.llm_action_worker import LLMActionWorker
            from workers.planner_worker import PlannerWorker

        # Register workers for the selected ALFWorld workflow.
        registry = get_registry()
        registry.register("planner_worker", PlannerWorker)
        registry.register("action_worker", ActionWorker)
        registry.register("llm_action_worker", LLMActionWorker)

        ablation_config = self.config.get("ablation") or AblationConfig.full()
        workflow_mode = self.config.get("workflow_mode", "single_action")
        architect_mode = self.config.get("architect_mode", "deterministic")
        requires_llm = "llm" in workflow_mode or architect_mode == "llm"
        if requires_llm and self.llm is None:
            raise ValueError(
                f"workflow_mode={workflow_mode!r} architect_mode={architect_mode!r} requires a configured llm"
            )

        class WorkerInputSchema(TypedDict):
            domain_state: dict
            selected_workers: list[str]
            current_worker: str
            data_schema: dict
            ablation_config: AblationConfig

        def worker_node(state: dict[str, Any]) -> dict[str, Any]:
            worker_name = state.get("current_worker", "")
            worker_instructions = state.get("worker_instructions", {})
            instruction = worker_instructions.get(worker_name)
            worker = registry.create_worker(worker_name, self.llm, instruction)
            if worker is None:
                return {
                    "pending_patch": [],
                    "patch_error": f"Unknown worker type: {worker_name}",
                }
            worker.input_schema = WorkerInputSchema if ablation_config.use_context_slicing else None

            worker_result = worker(state)
            return {
                **worker_result,
                "turn_worker_input_tokens": int(state.get("turn_worker_input_tokens", 0))
                + int(worker_result.get("worker_input_tokens", 0)),
                "turn_worker_output_tokens": int(state.get("turn_worker_output_tokens", 0))
                + int(worker_result.get("worker_output_tokens", 0)),
            }

        router = WorkflowRouter(
            worker_names=[],
            max_steps=self.config.get("kernel", {}).get("max_steps", 50),
            max_no_update=2,
            loop_detection_window=3,
        )

        def route_to_worker(state: KernelState) -> str:
            if state.get("circuit_breaker_triggered", False):
                return END
            if state.get("waiting_for_user", False):
                return END
            result = router.route(state)
            return "worker" if result != END else END

        def set_current_worker(state: KernelState) -> dict[str, Any]:
            workflow_rules = state.get("workflow_rules", {})
            domain_state = state.get("domain_state", {})

            _, _, entry = get_matching_workflow_entry(domain_state, workflow_rules)
            return {"current_worker": extract_worker_name(entry) or ""}

        def initialize_ablation(_: KernelState) -> dict[str, AblationConfig]:
            return {"ablation_config": ablation_config}

        def null_architect(state: KernelState) -> dict[str, Any]:
            """Fixed-pipeline architect used when C5 (use_architect_agent) is ablated.

            Uses the same schema, workflow, and domain_state as ALFWorldArchitect
            (deterministic), but replaces LLM-generated worker_instructions with
            the task_goal so LLM workers still receive meaningful context.
            """
            base = ALFWorldArchitect(workflow_mode=workflow_mode)(state)
            task_goal = (base.get("domain_state") or {}).get("task_goal", "")
            fixed_instructions = {w: task_goal for w in base.get("selected_workers", [])}
            return {
                **base,
                "worker_instructions": fixed_instructions,
                "turn_architect_input_tokens": 0,
                "turn_architect_output_tokens": 0,
            }

        builder = StateGraph(KernelState)
        builder.add_node("initialize_ablation", initialize_ablation)
        if not ablation_config.use_architect_agent:
            architect = null_architect
        elif architect_mode == "llm":
            architect = LLMALFWorldArchitect(self.llm, workflow_mode=workflow_mode)
        else:
            architect = ALFWorldArchitect(self.llm, workflow_mode=workflow_mode)
        builder.add_node("architect", architect)
        builder.add_node("kernel", kernel_node)
        builder.add_node("set_worker", set_current_worker)
        if ablation_config.use_context_slicing:
            builder.add_node("worker", worker_node, input_schema=WorkerInputSchema)
        else:
            builder.add_node("worker", worker_node)

        builder.add_edge(START, "initialize_ablation")
        builder.add_edge("initialize_ablation", "architect")
        builder.add_edge("architect", "kernel")
        builder.add_conditional_edges(
            "kernel",
            route_to_worker,
            {"worker": "set_worker", END: END},
        )
        builder.add_edge("set_worker", "worker")
        builder.add_edge("worker", "kernel")
        return builder.compile()

    @staticmethod
    def _capture_runtime_state(
        result: Dict[str, Any],
        *,
        episode_id: str,
        gamefile: str,
        step_id: int,
        fallback_used: bool,
        fallback_action: str,
        fallback_reason: str,
        fallback_source: str,
        default_data_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize one runtime state snapshot for later schema-guard evaluation."""
        domain_state = copy.deepcopy(result.get("domain_state", {}))
        domain_state["episode_id"] = episode_id
        domain_state["step_id"] = step_id
        domain_state["current_gamefile"] = gamefile or domain_state.get("current_gamefile", "")
        domain_state["fallback_used"] = fallback_used
        if fallback_used:
            domain_state["fallback_action"] = fallback_action
            domain_state["fallback_reason"] = fallback_reason
            domain_state["fallback_source"] = fallback_source

        data_schema = copy.deepcopy(result.get("data_schema") or default_data_schema or {})
        return {
            "episode_id": episode_id,
            "gamefile": gamefile,
            "step_id": step_id,
            "domain_state": domain_state,
            "data_schema": data_schema,
        }

    @staticmethod
    def _build_step_diagnostics(
        domain_state: Dict[str, Any],
        *,
        observation_before: str,
        available_actions_before: list[str],
    ) -> Dict[str, Any]:
        """Collect compact per-step diagnostics for later experiment analysis."""
        planner_state = {
            "architect_decision": domain_state.get("architect_decision", ""),
            "workflow_status": domain_state.get("workflow_status", ""),
            "planner_summary": domain_state.get("planner_summary", ""),
            "subgoal": domain_state.get("subgoal", ""),
            "focus_object": domain_state.get("focus_object", ""),
            "focus_receptacle": domain_state.get("focus_receptacle", ""),
            "required_transform": domain_state.get("required_transform", ""),
            "goal_object_guidance": domain_state.get("goal_object_guidance", ""),
            "search_guidance": domain_state.get("search_guidance", ""),
            "recommended_actions": list(domain_state.get("recommended_actions", [])),
            "searched_locations": list(domain_state.get("searched_locations", [])),
            "failed_search_locations": list(domain_state.get("failed_search_locations", [])),
        }
        canonical_task = {
            "canonical_task_type": domain_state.get("canonical_task_type", ""),
            "canonical_goal_object": domain_state.get("canonical_goal_object", ""),
            "canonical_target_receptacle": domain_state.get("canonical_target_receptacle", ""),
        }
        return {
            "observation_before": observation_before,
            "available_actions_before": list(available_actions_before),
            "planner_state": planner_state,
            "canonical_task": canonical_task,
        }

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
        """
        Run a complete ALFWorld episode using Kernel System.

        Args:
            max_steps: Maximum number of steps per episode
            episode_id: Unique identifier for this episode
            run_id: Identifier for the batch run
            experiment_id: Experiment configuration label (e.g. "full", "ablate_c2")
            token_callback: Optional callable(result) -> dict with keys
                worker_input_tokens, worker_output_tokens; called after each kernel invoke
            gamefile: Optional path to a specific ALFWorld task file. When provided,
                env_wrapper.reset(gamefile=...) pins the episode to that task.
            capture_states: Whether to capture per-step runtime states for
                Experiment 2 invalid-patch replay.

        Returns:
            Dictionary containing episode-level metrics and trajectory
        """
        # 1. Initialize ALFWorld
        if self.env.env is None and gamefile is None:
            self.env.init(batch_size=1)
        obs, infos = self.env.reset(gamefile=gamefile)
        kernel_state = obs_to_kernel_state(obs[0], infos)
        task_desc = kernel_state["domain_state"]["task_goal"]
        gamefile = kernel_state["domain_state"].get("current_gamefile", "")
        print(f"Task: {task_desc}")
        print(f"Initial observation: {obs[0][:100]}...")

        # 2. Build Kernel Graph
        if self.kernel_graph is None:
            self.kernel_graph = self.build_kernel_graph()

        ablation_config = self.config.get("ablation") or AblationConfig.full()

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
        workflow_final_status = kernel_state["domain_state"].get(
            "workflow_status",
            kernel_state["domain_state"].get("status", ""),
        )
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

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")

            # Kernel System execution
            result = self.kernel_graph.invoke(kernel_state)

            # Token accounting via callback
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

            # Patch error counting
            patch_err = result.get("patch_error", "")
            if patch_err:
                patch_error_count += 1
            if result.get("no_update_count", 0):
                no_update_event_count += 1

            domain_state = result.get("domain_state", {})
            workflow_final_status = domain_state.get("workflow_status", domain_state.get("status", workflow_final_status))
            if result.get("circuit_breaker_triggered", False):
                circuit_breaker_triggered = True
                circuit_breaker_reason = result.get("circuit_breaker_reason", "")
                stop_reason = "circuit_breaker"
                print(f"Stopping episode due to circuit breaker: {circuit_breaker_reason}")
                break
            if result.get("waiting_for_user", False):
                waiting_for_user = True
                stop_reason = "waiting_for_user"
                print("Stopping episode because kernel is waiting for user input.")
                break

            # Extract action with fallback metadata
            action, fallback_used = extract_action_with_meta(result, infos["admissible_commands"][0])
            if fallback_used:
                fallback_action_count += 1
            print(f"Selected action: {action}")
            step_diagnostics = self._build_step_diagnostics(
                domain_state,
                observation_before=kernel_state.get("domain_state", {}).get("current_observation", obs[0]),
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
                    self._capture_runtime_state(
                        result,
                        episode_id=episode_id,
                        gamefile=gamefile,
                        step_id=step,
                        fallback_used=fallback_used,
                        fallback_action=action,
                        fallback_reason=fallback_reason,
                        fallback_source=fallback_source,
                        default_data_schema=kernel_state.get("data_schema", {}),
                    )
                )

            # ALFWorld execution
            obs, reward, done, infos = self.env.step([action])
            latest_infos = infos
            print(f"Observation: {obs[0][:100]}...")

            trajectory.append({
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
            })

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
                print(f"\n✓ Task completed in {step + 1} steps!")
                stop_reason = "environment_done"
                break

            if ablation_config.use_circuit_breaker:
                pair_repeat = pair_hits[pair] >= repeat_pair_threshold
                observation_repeat = observation_hits[obs[0]] >= repeat_observation_threshold
                action_repeat = action_hits[action] >= repeat_action_threshold
                if pair_repeat or observation_repeat or action_repeat:
                    circuit_breaker_triggered = True
                    circuit_breaker_reason = "episode_stagnation"
                    stop_reason = "stagnation_detected"
                    print(
                        "Stopping episode due to stagnation detection: "
                        f"action_repeat={action_hits[action]}, "
                        f"observation_repeat={observation_hits[obs[0]]}, "
                        f"pair_repeat={pair_hits[pair]}"
                    )
                    break

            kernel_state = update_kernel_state(result, obs[0], infos)
            time.sleep(self.config.get("step_sleep", 4.0))
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
            "ablation_config": ablation_config.__dict__ if hasattr(ablation_config, "__dict__") else {},
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
