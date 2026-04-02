"""Factory helpers for AutoGen-based ALFWorld baselines."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Sequence

from .autogen_bootstrap import bootstrap_local_autogen

bootstrap_local_autogen()

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelFamily, RequestUsage
from autogen_core.tools import Tool, ToolSchema
from experiment.alfworld.workers.action_worker import ActionWorker
from experiment.alfworld.workers.planner_worker import PlannerWorker


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _env_value(primary_env: str, *fallback_envs: str) -> str:
    for env_name in (primary_env, *fallback_envs):
        if env_name:
            value = os.environ.get(env_name, "").strip()
            if value:
                return value
    return ""


def parse_autogen_turn_prompt(prompt: str) -> Dict[str, Any]:
    """Parse the ALFWorld turn prompt used by the AutoGen baseline."""
    lines = [line.rstrip() for line in prompt.splitlines()]
    sections: Dict[str, list[str]] = {}
    current_key = ""
    for line in lines:
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1].isupper():
            current_key = stripped[:-1]
            sections[current_key] = []
            continue
        if current_key:
            sections[current_key].append(line)

    def _joined(key: str) -> str:
        return "\n".join(sections.get(key, [])).strip()

    def _list(key: str) -> list[str]:
        values = []
        for line in sections.get(key, []):
            stripped = line.strip()
            if stripped.startswith("- "):
                values.append(stripped[2:].strip())
        return values

    return {
        "task_goal": _joined("TASK GOAL"),
        "current_observation": _joined("CURRENT OBSERVATION"),
        "available_actions": _list("AVAILABLE ACTIONS"),
        "action_history": _list("PREVIOUS ACTIONS"),
        "observation_history": [_joined("CURRENT OBSERVATION")] if _joined("CURRENT OBSERVATION") else [],
        "current_gamefile": _joined("GAMEFILE"),
        "canonical_task_type": _joined("CANONICAL TASK TYPE"),
        "canonical_goal_object": _joined("CANONICAL GOAL OBJECT"),
        "canonical_target_receptacle": _joined("CANONICAL TARGET RECEPTACLE"),
    }


def parse_executor_response(text: str, admissible_actions: Sequence[str]) -> tuple[str, str]:
    """Extract selected action and brief reason from executor output text."""
    action = ""
    reason = ""
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ACTION:"):
            action = stripped.split(":", 1)[1].strip()
        elif upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()

    if action in admissible_actions:
        return action, reason

    normalized = text.lower()
    for candidate in admissible_actions:
        if candidate.lower() in normalized:
            return candidate, reason or "Matched admissible action from executor response."

    fallback = "look" if "look" in admissible_actions else (admissible_actions[0] if admissible_actions else "")
    return fallback, reason or "Executor response did not contain a valid admissible action."


def parse_planner_message(text: str) -> Dict[str, str]:
    """Parse planner message lines into executor-facing hints."""
    fields = {
        "subgoal": "",
        "focus_object": "",
        "focus_receptacle": "",
        "planner_summary": "",
        "search_guidance": "",
    }
    mapping = {
        "SUBGOAL": "subgoal",
        "FOCUS_OBJECT": "focus_object",
        "FOCUS_RECEPTACLE": "focus_receptacle",
        "RATIONALE": "planner_summary",
        "SEARCH_GUIDANCE": "search_guidance",
    }
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().upper()
        if key in mapping:
            fields[mapping[key]] = value.strip()
    return fields


class AutoGenALFWorldMockClient(ChatCompletionClient):
    """Deterministic mock client for AutoGen ALFWorld baselines."""

    def __init__(self, role: str):
        self.role = role
        self._cur_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }

    @staticmethod
    def _token_count_from_messages(messages: Sequence[LLMMessage]) -> int:
        token_count = 0
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                token_count += len(content.split())
            elif isinstance(content, list):
                token_count += sum(len(str(item).split()) for item in content)
            else:
                token_count += len(str(content).split())
        return token_count

    def _build_planner_response(self, prompt: str) -> str:
        context = parse_autogen_turn_prompt(prompt)
        patches = PlannerWorker()._think(context)
        values: Dict[str, Any] = {}
        for patch in patches:
            path = str(patch.get("path", ""))
            if not path.startswith("/"):
                continue
            values[path[1:]] = patch.get("value")
        return "\n".join(
            [
                f"SUBGOAL: {values.get('subgoal', 'acquire_goal_object')}",
                f"FOCUS_OBJECT: {values.get('focus_object', '')}",
                f"FOCUS_RECEPTACLE: {values.get('focus_receptacle', '')}",
                f"RATIONALE: {values.get('planner_summary', 'Advance the task with the next best admissible action.')}",
                f"SEARCH_GUIDANCE: {values.get('search_guidance', '')}",
            ]
        ).strip()

    def _build_executor_response(self, messages: Sequence[LLMMessage]) -> str:
        user_prompt = ""
        planner_message = ""
        for message in messages:
            source = getattr(message, "source", "")
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = str(content)
            if source == "user":
                user_prompt = content
            elif source == "planner":
                planner_message = content

        context = parse_autogen_turn_prompt(user_prompt)
        if planner_message:
            context.update({k: v for k, v in parse_planner_message(planner_message).items() if v})

        patches = ActionWorker()._think(context)
        values: Dict[str, Any] = {}
        for patch in patches:
            path = str(patch.get("path", ""))
            if not path.startswith("/"):
                continue
            values[path[1:]] = patch.get("value")
        return "\n".join(
            [
                f"ACTION: {values.get('selected_action', '')}",
                f"REASON: {values.get('decision_reason', 'Choose the best admissible action.')}",
            ]
        ).strip()

    def _create_text_response(self, messages: Sequence[LLMMessage]) -> str:
        latest_user_prompt = ""
        for message in messages:
            if getattr(message, "source", "") == "user":
                content = getattr(message, "content", "")
                latest_user_prompt = content if isinstance(content, str) else str(content)
        if self.role == "planner":
            return self._build_planner_response(latest_user_prompt)
        return self._build_executor_response(messages)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | str = "auto",
        json_output: bool | type | None = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        del tools, tool_choice, json_output, extra_create_args, cancellation_token
        response = self._create_text_response(messages)
        prompt_tokens = self._token_count_from_messages(messages)
        completion_tokens = len(response.split())
        self._cur_usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )
        return CreateResult(
            finish_reason="stop",
            content=response,
            usage=self._cur_usage,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | str = "auto",
        json_output: bool | type | None = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ):
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._cur_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        del tools
        return self._token_count_from_messages(messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        del tools
        return max(0, 10000 - self._token_count_from_messages(messages))

    @property
    def capabilities(self):  # type: ignore[override]
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        }

    @property
    def model_info(self):  # type: ignore[override]
        return self._model_info


def build_autogen_model_client_factory(
    *,
    llm_backend: str,
    llm_model: str = "",
    llm_model_env: str = "",
    llm_base_url: str = "",
    llm_base_url_env: str = "",
    llm_api_key_env: str = "",
    llm_temperature: float = 0.0,
    llm_timeout: float = 60.0,
) -> Callable[[str], ChatCompletionClient]:
    """Build a role-aware AutoGen model-client factory."""
    if llm_backend == "mock":
        return lambda role: AutoGenALFWorldMockClient(role=role)

    if llm_backend != "openai_compatible":
        raise ValueError("AutoGen baseline requires --llm-backend mock or --llm-backend openai_compatible.")

    resolved_model = _first_nonempty(
        llm_model.strip(),
        _env_value(llm_model_env, "ALFWORLD_LLM_MODEL", "OPENAI_MODEL_NAME", "OPENAI_MODEL"),
    )
    resolved_base_url = _first_nonempty(
        llm_base_url.strip(),
        _env_value(llm_base_url_env, "ALFWORLD_LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    resolved_api_key = _env_value(llm_api_key_env, "ALFWORLD_LLM_API_KEY", "OPENAI_API_KEY")

    missing: list[str] = []
    if not resolved_model:
        missing.append("model (--llm-model or env ALFWORLD_LLM_MODEL)")
    if not resolved_base_url:
        missing.append("base_url (--llm-base-url or env ALFWORLD_LLM_BASE_URL/OPENAI_BASE_URL)")
    if not resolved_api_key:
        missing.append("api_key (env ALFWORLD_LLM_API_KEY/OPENAI_API_KEY or --llm-api-key-env)")
    if missing:
        raise ValueError("openai_compatible backend is missing required config: " + ", ".join(missing))

    model_info = {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    }

    def _factory(_: str) -> ChatCompletionClient:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            model_info=model_info,
            temperature=llm_temperature,
            timeout=llm_timeout,
        )

    return _factory
