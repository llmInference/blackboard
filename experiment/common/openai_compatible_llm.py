"""Helpers for local OpenAI-compatible model calls."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


def _first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if str(value or "").strip():
            return str(value).strip()
    return ""


def resolve_openai_compatible_config(
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    timeout: float = 60.0,
    model_envs: tuple[str, ...] = ("MODEL_NAME", "OPENAI_MODEL_NAME", "OPENAI_MODEL", "ALFWORLD_LLM_MODEL"),
    base_url_envs: tuple[str, ...] = ("OPENAI_API_BASE", "OPENAI_BASE_URL", "ALFWORLD_LLM_BASE_URL"),
    api_key_envs: tuple[str, ...] = ("OPENAI_API_KEY", "ALFWORLD_LLM_API_KEY"),
) -> Dict[str, Any]:
    resolved_model = _first_nonempty([model] + [os.getenv(env_name, "") for env_name in model_envs])
    resolved_base_url = _first_nonempty([base_url] + [os.getenv(env_name, "") for env_name in base_url_envs])
    resolved_api_key = _first_nonempty([api_key] + [os.getenv(env_name, "") for env_name in api_key_envs])
    if not resolved_model:
        raise ValueError("Missing OpenAI-compatible model name.")
    if not resolved_base_url:
        raise ValueError("Missing OpenAI-compatible base URL.")
    if not resolved_api_key:
        raise ValueError("Missing OpenAI-compatible API key.")
    return {
        "model": resolved_model,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
        "timeout": float(timeout),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("LLM returned empty content.")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response is not valid JSON: {raw[:200]}")
    candidate = raw[start : end + 1]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object.")
    return payload


def _create_chat_completion(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    force_json_mode: bool,
):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json_mode:
        try:
            return client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            message = str(exc or "").lower()
            if "response_format" not in message and "json_object" not in message and "unsupported" not in message:
                raise
    return client.chat.completions.create(**kwargs)


def chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    timeout: float = 60.0,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
    force_json_mode: bool = False,
) -> Dict[str, Any]:
    cfg = resolve_openai_compatible_config(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=cfg["timeout"],
    )
    response = None
    last_error: Exception | None = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            response = _create_chat_completion(
                client=client,
                model=cfg["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                force_json_mode=force_json_mode,
            )
            break
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as exc:
            last_error = exc
            if attempt >= max(0, int(max_retries)):
                raise
            sleep_s = float(retry_backoff) * (2 ** attempt)
            print(f"⚠️  OpenAI-compatible 调用失败，准备重试 {attempt + 1}/{max_retries}: {exc}")
            time.sleep(sleep_s)

    if response is None:
        raise RuntimeError(f"OpenAI-compatible request failed without response: {last_error}")
    content = str(response.choices[0].message.content or "")
    usage = getattr(response, "usage", None)
    return {
        "payload": extract_json_object(content),
        "raw_text": content,
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
        "model": cfg["model"],
        "base_url": cfg["base_url"],
    }


def chat_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    timeout: float = 60.0,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 2,
    retry_backoff: float = 1.5,
    force_json_mode: bool = True,
) -> Dict[str, Any]:
    response = chat_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        force_json_mode=force_json_mode,
    )
    return {
        **response,
        "payload": extract_json_object(str(response.get("raw_text", "") or "")),
    }
