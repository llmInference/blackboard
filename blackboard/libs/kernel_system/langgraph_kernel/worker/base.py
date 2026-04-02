from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from langgraph_kernel.types import JsonPatch, WorkerPayload


class BaseWorkerAgent(ABC):
    """
    Worker 基类。子类需要：
    1. 声明 `input_schema`（TypedDict 类），Kernel 用它裁剪状态切片
    2. 实现 `_think(context)`，返回 JSON Patch 或自然语言结果
    """

    # 子类覆盖：声明该 worker 需要的状态字段
    input_schema: type | None = None

    def _build_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ablation = state.get("ablation_config")

        if ablation and not ablation.use_context_slicing:
            return {
                **state.get("domain_state", {}),
                "error_feedback": state.get("error_feedback", ""),
                "retry_count": state.get("retry_count", 0),
                "_data_schema": state.get("data_schema", {}),
                "step_count": state.get("step_count", 0),
                "status_history": state.get("status_history", []),
                "task_flow": state.get("task_flow", []),
                "selected_workers": state.get("selected_workers", []),
                "worker_instructions": state.get("worker_instructions", {}),
            }

        # 传递 domain_state 以及 Layer 5 的反馈信息
        # 同时传递 data_schema 用于类型检查
        return {
            **state.get("domain_state", {}),
            "error_feedback": state.get("error_feedback", ""),
            "retry_count": state.get("retry_count", 0),
            "_data_schema": state.get("data_schema", {}),  # 添加 schema 信息
        }

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        context = self._build_context(state)
        worker_output = self._think(context)
        return {"pending_patch": worker_output}

    @abstractmethod
    def _think(self, context: dict[str, Any]) -> WorkerPayload:
        """核心逻辑：根据状态切片生成结构化或自然语言结果。"""


# ── LLM Worker ────────────────────────────────────────────────────────────────

class LLMWorkerAgent(BaseWorkerAgent):
    """
    调用 LLM 生成 JSON Patch 的 worker。

    LLM 接收状态切片作为 prompt，输出 JSON Patch 列表。
    子类可覆盖 `system_prompt` 来定制角色。
    """

    system_prompt: str = (
        "You are a worker agent. You receive a partial view of the system state "
        "and must return a JSON Patch (RFC 6902) to update it. "
        "Output ONLY a valid JSON array of patch operations, no other text.\n"
        "Example: [{\"op\": \"add\", \"path\": \"/field\", \"value\": \"data\"}]"
    )
    STRUCTURED_OUTPUT_INSTRUCTIONS = """You may also return a JSON object instead of JSON Patch.

Preferred JSON object shape:
{
  "state_updates": {
    "<existing_field>": <value>
  },
  "schema_extensions": {
    "<new_field_name>": {
      "type": "string|number|integer|boolean|object|array",
      "description": "Why this extra field is needed",
      "value": <value>
    }
  }
}

Rules for structured JSON:
- Reuse existing fields whenever possible.
- Only propose schema_extensions when no existing field can hold the information.
- Never create a synonymous duplicate of an existing field.
- schema_extensions are optional-only proposals; do not assume they become required fields.
"""

    def __init__(self, llm: BaseChatModel, instruction: str | None = None) -> None:
        self._llm = llm
        self._instruction = instruction  # 来自 Architect 的动态指令
        self._last_token_usage = {
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
        }

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        self._last_token_usage = {
            "worker_input_tokens": 0,
            "worker_output_tokens": 0,
        }
        context = self._build_context(state)
        ablation = state.get("ablation_config")
        use_json_patch = True if ablation is None else ablation.use_json_patch
        worker_output = self._think(context, use_json_patch=use_json_patch)
        return {"pending_patch": worker_output, **self._last_token_usage}

    @staticmethod
    def _normalize_response_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _strip_reasoning_sections(content: str) -> str:
        """Remove common reasoning wrappers such as <think>...</think>."""
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    @classmethod
    def _extract_json_candidates(cls, content: str) -> list[str]:
        cleaned = cls._strip_reasoning_sections(content)
        candidates: list[str] = []

        if cleaned:
            candidates.append(cleaned)

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        for block in fenced_blocks:
            block = block.strip()
            if block:
                candidates.append(block)

        for opener, closer in (("[", "]"), ("{", "}")):
            start = cleaned.find(opener)
            end = cleaned.rfind(closer)
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1].strip()
                if snippet:
                    candidates.append(snippet)

        # Preserve order while removing duplicates.
        unique_candidates: list[str] = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _normalize_semantic_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(key or "").lower())

    @classmethod
    def _resolve_state_key(
        cls,
        key: str,
        existing_keys: set[str],
    ) -> str:
        alias_map = {
            "reply": "assistant_response",
            "response": "assistant_response",
            "assistantreply": "assistant_response",
            "toolname": "selected_tool_name",
            "selectedtool": "selected_tool_name",
            "toolargs": "selected_tool_args",
            "selectedtoolargs": "selected_tool_args",
            "nextworker": "requested_next_worker",
            "requestednextworker": "requested_next_worker",
            "question": "pending_user_question",
            "pendingquestion": "pending_user_question",
            "readersummary": "reader_summary",
            "writersummary": "writer_summary",
            "latesttool": "latest_tool_result",
            "latesttoolresult": "latest_tool_result",
            "actionresult": "action_result",
            "actionerror": "action_error",
            "reason": "decision_reason",
        }
        if key in existing_keys:
            return key

        normalized_key = cls._normalize_semantic_key(key)
        exact_matches = {
            cls._normalize_semantic_key(existing_key): existing_key
            for existing_key in existing_keys
        }
        if normalized_key in exact_matches:
            return exact_matches[normalized_key]
        return alias_map.get(normalized_key, key)

    @staticmethod
    def _json_pointer_escape(segment: str) -> str:
        return str(segment).replace("~", "~0").replace("/", "~1")

    @staticmethod
    def _infer_json_schema_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int) and not isinstance(value, bool):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, dict):
            return "object"
        if isinstance(value, list):
            return "array"
        return "string"

    @classmethod
    def _structured_payload_to_patch(
        cls,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> JsonPatch:
        existing_keys = set((context or {}).keys())
        existing_keys.update({"extensions", "extension_schema"})
        patch: JsonPatch = []

        state_updates = payload.get("state_updates")
        if not isinstance(state_updates, dict):
            state_updates = {}

        tool_request = payload.get("tool_request")
        if isinstance(tool_request, dict):
            tool_name = tool_request.get("name", "")
            tool_args = tool_request.get("arguments", {})
            if "selected_tool_name" not in state_updates and str(tool_name or "").strip():
                state_updates["selected_tool_name"] = str(tool_name).strip()
            if "selected_tool_args" not in state_updates and isinstance(tool_args, dict):
                state_updates["selected_tool_args"] = dict(tool_args)

        if "response" in payload and "assistant_response" not in state_updates:
            state_updates["assistant_response"] = payload.get("response")
        if "decision_reason" in payload and "decision_reason" not in state_updates:
            state_updates["decision_reason"] = payload.get("decision_reason")
        elif "reason" in payload and "decision_reason" not in state_updates:
            state_updates["decision_reason"] = payload.get("reason")

        known_state_fields = {
            "selected_tool_name",
            "selected_tool_args",
            "assistant_response",
            "reader_summary",
            "writer_summary",
            "requested_next_worker",
            "pending_user_question",
            "latest_tool_result",
            "action_result",
            "action_error",
            "decision_reason",
            "extensions",
            "extension_schema",
        }
        for key, value in payload.items():
            if key in {"state_updates", "schema_extensions", "tool_request", "response", "decision_reason", "reason"}:
                continue
            resolved_key = cls._resolve_state_key(key, existing_keys | known_state_fields)
            if resolved_key in known_state_fields and resolved_key not in state_updates:
                state_updates[resolved_key] = value

        for key, value in state_updates.items():
            resolved_key = cls._resolve_state_key(str(key), existing_keys | known_state_fields)
            if resolved_key in existing_keys or resolved_key in known_state_fields:
                op = "replace" if resolved_key in existing_keys or resolved_key in known_state_fields else "add"
                patch.append({"op": op, "path": f"/{cls._json_pointer_escape(resolved_key)}", "value": value})
            else:
                extension_key = cls._json_pointer_escape(str(key))
                patch.append({"op": "add", "path": f"/extensions/{extension_key}", "value": value})
                patch.append(
                    {
                        "op": "add",
                        "path": f"/extension_schema/{extension_key}",
                        "value": {
                            "type": cls._infer_json_schema_type(value),
                            "description": f"Worker-proposed extension for {key}.",
                        },
                    }
                )

        schema_extensions = payload.get("schema_extensions")
        if not isinstance(schema_extensions, dict):
            schema_extensions = {}
        for key, extension_payload in schema_extensions.items():
            resolved_key = cls._resolve_state_key(str(key), existing_keys | known_state_fields)
            if isinstance(extension_payload, dict):
                extension_value = extension_payload.get("value")
                extension_type = str(extension_payload.get("type", "") or "").strip()
                extension_description = str(extension_payload.get("description", "") or "").strip()
            else:
                extension_value = extension_payload
                extension_type = ""
                extension_description = ""

            if resolved_key in existing_keys or resolved_key in known_state_fields:
                if extension_value is not None:
                    patch.append(
                        {
                            "op": "replace",
                            "path": f"/{cls._json_pointer_escape(resolved_key)}",
                            "value": extension_value,
                        }
                    )
                continue

            extension_key = cls._json_pointer_escape(str(key))
            patch.append({"op": "add", "path": f"/extensions/{extension_key}", "value": extension_value})
            patch.append(
                {
                    "op": "add",
                    "path": f"/extension_schema/{extension_key}",
                    "value": {
                        "type": extension_type or cls._infer_json_schema_type(extension_value),
                        "description": extension_description or f"Worker-proposed extension for {key}.",
                    },
                }
            )
        return patch

    @classmethod
    def _coerce_dict_to_patch(
        cls,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> JsonPatch:
        """Best-effort conversion from common object outputs to JSON Patch."""
        if isinstance(payload.get("patch"), list):
            return payload["patch"]
        if isinstance(payload.get("pending_patch"), list):
            return payload["pending_patch"]
        if any(
            key in payload
            for key in ("state_updates", "schema_extensions", "tool_request", "response")
        ):
            return cls._structured_payload_to_patch(payload, context=context)

        selected_action = payload.get("selected_action", payload.get("action", ""))
        decision_reason = payload.get("decision_reason", payload.get("reason", ""))

        patch: JsonPatch = []
        if str(selected_action or "").strip():
            patch.append(
                {
                    "op": "replace",
                    "path": "/selected_action",
                    "value": str(selected_action).strip(),
                }
            )
        if str(decision_reason or "").strip():
            patch.append(
                {
                    "op": "replace",
                    "path": "/decision_reason",
                    "value": str(decision_reason).strip(),
                }
            )
        elif patch:
            patch.append(
                {
                    "op": "replace",
                    "path": "/decision_reason",
                    "value": "Selected the next action from the current state slice.",
                }
            )
        return patch

    @staticmethod
    def _tokenize_for_match(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    @classmethod
    def _heuristic_action_patch_from_text(
        cls,
        *,
        content: str,
        context: dict[str, Any],
    ) -> JsonPatch:
        available_actions = [str(action) for action in list(context.get("available_actions", []) or []) if str(action or "").strip()]
        if not available_actions:
            return []

        recommended_actions = [
            str(action)
            for action in list(context.get("recommended_actions", []) or [])
            if str(action or "").strip()
        ]
        candidate_actions = recommended_actions + [action for action in available_actions if action not in recommended_actions]

        cleaned = cls._strip_reasoning_sections(content)
        cleaned_lower = cleaned.lower()
        response_tokens = cls._tokenize_for_match(cleaned)
        if not response_tokens:
            return []

        for action in candidate_actions:
            if action.lower() in cleaned_lower:
                return [
                    {"op": "replace", "path": "/selected_action", "value": action},
                    {
                        "op": "replace",
                        "path": "/decision_reason",
                        "value": cleaned.splitlines()[0][:240] or "Recovered action from natural-language response.",
                    },
                ]

        def action_score(action: str) -> tuple[float, int]:
            tokens = cls._tokenize_for_match(action)
            if not tokens:
                return (0.0, 0)
            overlap = len(tokens & response_tokens)
            tail = action.lower()
            for prefix in (
                "go to ",
                "open ",
                "close ",
                "take ",
                "move ",
                "examine ",
                "clean ",
                "heat ",
                "cool ",
                "slice ",
            ):
                if tail.startswith(prefix):
                    tail = tail[len(prefix):]
                    break
            tail_tokens = cls._tokenize_for_match(tail)
            tail_overlap = len(tail_tokens & response_tokens)
            score = overlap + tail_overlap * 1.5
            if tail and tail in cleaned_lower:
                score += 3.0
            if action in recommended_actions:
                score += 0.25
            return (score, len(action))

        best_action = ""
        best_score = 0.0
        best_len = 0
        for action in candidate_actions:
            score, action_len = action_score(action)
            if score > best_score or (score == best_score and action_len > best_len):
                best_action = action
                best_score = score
                best_len = action_len

        if best_action and best_score >= 2.0:
            return [
                {"op": "replace", "path": "/selected_action", "value": best_action},
                {
                    "op": "replace",
                    "path": "/decision_reason",
                    "value": cleaned.splitlines()[0][:240] or "Recovered action from natural-language response.",
                },
            ]

        if recommended_actions and (
            "recommended action" in cleaned_lower
            or "recommended actions" in cleaned_lower
            or "recommended location" in cleaned_lower
            or "recommended locations" in cleaned_lower
        ):
            return [
                {"op": "replace", "path": "/selected_action", "value": recommended_actions[0]},
                {
                    "op": "replace",
                    "path": "/decision_reason",
                    "value": cleaned.splitlines()[0][:240] or "Selected the first recommended action after natural-language response.",
                },
            ]

        return []

    @classmethod
    def _parse_patch_response(
        cls,
        content: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> JsonPatch:
        last_dict_payload: dict[str, Any] | None = None
        last_error: Exception | None = None

        for candidate in cls._extract_json_candidates(content):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

            if isinstance(payload, list):
                return payload

            if isinstance(payload, dict):
                last_dict_payload = payload
                coerced_patch = cls._coerce_dict_to_patch(payload, context=context)
                if coerced_patch:
                    print(f"ℹ️  Worker 返回对象，已自动转换为 patch: {payload}")
                    return coerced_patch

        if last_dict_payload is not None:
            print(f"⚠️  Worker 返回的不是列表: {last_dict_payload}")
            return []
        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("No JSON object or array found", content, 0)

    def _record_token_usage(self, response: Any) -> None:
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        if not isinstance(usage, dict):
            usage = {}

        self._last_token_usage = {
            "worker_input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "worker_output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }

    def _think(self, context: dict[str, Any], use_json_patch: bool = True) -> WorkerPayload:
        from langchain_core.messages import HumanMessage, SystemMessage

        # 如果有动态指令，使用它；否则使用默认 system_prompt
        system_content = self._instruction if self._instruction else self.system_prompt

        # 提取 schema 信息用于类型指导
        data_schema = context.get("_data_schema", {})
        schema_properties = data_schema.get("properties", {})

        # 构建类型提示
        type_hints = ""
        if schema_properties:
            type_hints = "\n\n**CRITICAL - Field Type Requirements:**\n"
            type_hints += "You MUST match these exact types when creating patches:\n"
            for field_name, field_schema in schema_properties.items():
                field_type = field_schema.get("type", "unknown")
                if field_type == "array":
                    items_type = field_schema.get("items", {}).get("type", "any")
                    type_hints += f"  • {field_name}: array of {items_type}\n"
                elif field_type == "object":
                    type_hints += f"  • {field_name}: object (structured data)\n"
                else:
                    type_hints += f"  • {field_name}: {field_type}\n"
            type_hints += "\n**Type Mismatch = Validation Error!**\n"
            type_hints += "- If schema says 'string', return a string (e.g., \"text here\")\n"
            type_hints += "- If schema says 'array', return an array (e.g., [\"item1\", \"item2\"])\n"
            type_hints += "- If schema says 'object', return an object (e.g., {{\"key\": \"value\"}})\n"

        # C1 关闭：退化为自然语言通信，不再要求输出 JSON Patch
        if not use_json_patch:
            natural_language_prompt = f"""{system_content}

Describe your analysis in natural language.
Output plain text only. Do not return JSON, JSON Patch, code fences, or markdown.
Focus on the most useful business result for the next workflow step."""

            response = self._llm.invoke([
                SystemMessage(content=natural_language_prompt),
                HumanMessage(content=f"Current state slice:\n{json.dumps(context, indent=2, ensure_ascii=False)}"),
            ])
            self._record_token_usage(response)
            return self._normalize_response_content(response.content)

        # 增强 system prompt，包含更多上下文
        # 构建增强的 prompt（避免 f-string 中的花括号冲突）
        enhanced_prompt = f"""{system_content}
{type_hints}

IMPORTANT: You must return a JSON Patch array (RFC 6902 format).
Each operation should have: "op" (add/replace/remove), "path" (JSON pointer), "value" (for add/replace).
{self.STRUCTURED_OUTPUT_INSTRUCTIONS}

Guidelines:
- The current state you see is the domain_state
- All JSON Patch paths should start from the root, e.g., "/destination", "/plan", "/result"
- Use "add" for NEW fields that don't exist yet
- Use "replace" for EXISTING fields that need to be updated
- Analyze the current state carefully
- Make incremental, logical changes
- Keep changes minimal and focused
- **DO NOT update the "status" field** - the Kernel will handle state transitions automatically

CRITICAL - ABSOLUTE RULES:
1. **NEVER update the "status" field** - Any status updates will be REJECTED by the Kernel
2. **ONLY update business data fields** relevant to your specific task
3. Your patches MUST contain meaningful business data (content, plan, analysis, results, etc.)
4. If you only return status updates, your work will be COMPLETELY FILTERED OUT

The Kernel controls all state transitions automatically. Your job is ONLY to provide business data.

Formatting rules:
- Return ONLY a JSON array, never a JSON object
- Never include <think>...</think>, analysis text, markdown, or code fences
- Never echo input fields such as user_prompt, task_goal, current_observation, or available_actions
- If you want to set selected_action or decision_reason, do it via JSON Patch operations only

Examples of CORRECT patches (business data):
- Writing task: [{{"op": "add", "path": "/content", "value": "人工智能是计算机科学的一个分支..."}}]
- Planning task: [{{"op": "add", "path": "/plan", "value": [{{"step": 1, "action": "Research"}}]}}]
- Analysis task: [{{"op": "add", "path": "/insights", "value": ["Finding 1", "Finding 2"]}}]

Examples of WRONG patches (will be filtered):
- [{{"op": "replace", "path": "/status", "value": "done"}}]  ❌ REJECTED - No business data!
- [{{"op": "replace", "path": "/status", "value": "planning"}}]  ❌ REJECTED - Status updates forbidden!

Output ONLY the JSON array, no explanations."""

        # Layer 5: 如果有错误反馈，添加到 prompt 中
        error_feedback = context.get("error_feedback", "")
        retry_count = context.get("retry_count", 0)

        if error_feedback and retry_count > 0:
            enhanced_prompt += f"""

⚠️ IMPORTANT: Your previous patch had errors. This is retry attempt #{retry_count}.

Previous Error Report:
{error_feedback}

Please fix these issues:
1. Check that all paths exist before using "replace" (use "add" for new fields)
2. **CRITICAL: Ensure all values match the EXACT types defined in the schema**
   - If schema says "string", provide a string (not object or array)
   - If schema says "array", provide an array
   - If schema says "object", provide an object
   - Do NOT try to be creative with types - follow the schema strictly
3. If using enum fields, use ONLY the allowed values from the schema
4. Double-check the JSON Patch syntax

Generate a corrected patch that addresses these errors."""

        response = self._llm.invoke([
            SystemMessage(content=enhanced_prompt),
            HumanMessage(content=f"Current state slice:\n{json.dumps(context, indent=2, ensure_ascii=False)}"),
        ])
        self._record_token_usage(response)

        # 解析 JSON 响应
        try:
            content = self._normalize_response_content(response.content)
            return self._parse_patch_response(content, context=context)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            fallback_patch = self._heuristic_action_patch_from_text(content=content, context=context)
            if fallback_patch:
                print(f"ℹ️  Worker 自然语言输出已自动转换为 patch: {fallback_patch}")
                return fallback_patch
            print(f"⚠️  Worker 解析失败: {e}")
            print(f"原始响应: {response.content[:200]}")
            return []


# ── Rule Worker ───────────────────────────────────────────────────────────────

class RuleWorkerAgent(BaseWorkerAgent):
    """
    纯规则 worker，不调用 LLM，适合确定性任务。

    子类实现 `_think()` 返回固定逻辑生成的 patch。
    """

    @abstractmethod
    def _think(self, context: dict[str, Any]) -> WorkerPayload: ...
