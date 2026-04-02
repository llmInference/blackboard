"""Runtime LLM factory for ALFWorld experiments."""
from __future__ import annotations

import os
from typing import Any

from .mock_llm import ALFWorldMockActionLLM


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


def build_runtime_llm(
    *,
    workflow_mode: str,
    architect_mode: str = "deterministic",
    llm_backend: str,
    llm_model: str = "",
    llm_model_env: str = "",
    llm_base_url: str = "",
    llm_base_url_env: str = "",
    llm_api_key_env: str = "",
    llm_temperature: float = 0.0,
    llm_timeout: float = 60.0,
) -> Any:
    """Build the chat model required by the selected workflow mode."""
    requires_llm = "llm" in workflow_mode or architect_mode == "llm"
    if not requires_llm:
        return None

    if llm_backend == "mock":
        return ALFWorldMockActionLLM()

    if llm_backend != "openai_compatible":
        raise ValueError(
            f"workflow_mode={workflow_mode!r} requires a supported LLM backend; "
            "use --llm-backend mock or --llm-backend openai_compatible."
        )

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
        raise ValueError(
            "openai_compatible backend is missing required config: " + ", ".join(missing)
        )

    from langgraph_kernel.llm_wrapper import SimpleChatModel

    return SimpleChatModel(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=llm_temperature,
        timeout=llm_timeout,
    )
