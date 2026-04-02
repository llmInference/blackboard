"""Runtime LLM factory for WebArena blackboard experiments."""
from __future__ import annotations

import os
from typing import Any

from experiment.webarena.utils.runtime_imports import ensure_webarena_runtime_imports


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
    llm_backend: str,
    llm_model: str = "",
    llm_model_env: str = "",
    llm_base_url: str = "",
    llm_base_url_env: str = "",
    llm_api_key_env: str = "",
    llm_temperature: float = 0.0,
    llm_timeout: float = 60.0,
) -> Any:
    """Build the chat model required by WebArena LLM workers."""
    if llm_backend != "openai_compatible":
        raise ValueError(
            f"Unsupported llm_backend={llm_backend!r}; only 'openai_compatible' is supported."
        )

    resolved_model = _first_nonempty(
        llm_model.strip(),
        _env_value(
            llm_model_env,
            "WEBARENA_LLM_MODEL",
            "MODEL_NAME",
            "OPENAI_MODEL_NAME",
            "OPENAI_MODEL",
        ),
    )
    resolved_base_url = _first_nonempty(
        llm_base_url.strip(),
        _env_value(
            llm_base_url_env,
            "WEBARENA_LLM_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
        ),
    )
    resolved_api_key = _env_value(
        llm_api_key_env,
        "WEBARENA_LLM_API_KEY",
        "OPENAI_API_KEY",
    )

    missing: list[str] = []
    if not resolved_model:
        missing.append("model (--llm-model or env MODEL_NAME/WEBARENA_LLM_MODEL)")
    if not resolved_base_url:
        missing.append("base_url (--llm-base-url or env OPENAI_API_BASE/WEBARENA_LLM_BASE_URL)")
    if not resolved_api_key:
        missing.append("api_key (env OPENAI_API_KEY/WEBARENA_LLM_API_KEY)")
    if missing:
        raise ValueError(
            "openai_compatible backend is missing required config: " + ", ".join(missing)
        )

    ensure_webarena_runtime_imports()
    from langgraph_kernel.llm_wrapper import SimpleChatModel

    return SimpleChatModel(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=llm_temperature,
        timeout=llm_timeout,
    )
