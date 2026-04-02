"""
简单的 LLM 包装器，直接使用 openai 库调用第三方 API。
兼容 langchain 的 BaseChatModel 接口。
"""
from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


class SimpleChatModel(BaseChatModel):
    """
    简单的聊天模型包装器，使用 openai 库直接调用 API。
    避免 langchain-openai 的兼容性问题。
    """

    client: Any = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    timeout: float = 60.0  # 默认 60 秒超时
    max_retries: int = 2
    retry_backoff: float = 1.5
    request_delay_seconds: float = 0.5

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.5,
        request_delay_seconds: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 转换消息格式
        openai_messages = []
        for msg in messages:
            if hasattr(msg, "role"):
                role = msg.role
            else:
                role = msg.__class__.__name__.replace("Message", "").lower()
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"

            openai_messages.append({"role": role, "content": msg.content})

        # 调用 API
        response = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    stop=stop,
                    **kwargs,
                )
                break
            except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                sleep_s = self.retry_backoff * (2 ** attempt)
                print(f"⚠️  LLM 调用失败，准备重试 {attempt + 1}/{self.max_retries}: {exc}")
                time.sleep(sleep_s)

        if response is None:
            raise RuntimeError(f"LLM request failed without response: {last_error}")

        # 转换响应
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        response_metadata = {}
        if usage is not None:
            response_metadata["token_usage"] = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
        message = AIMessage(content=content, response_metadata=response_metadata)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "simple_chat_model"
