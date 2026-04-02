from __future__ import annotations

import copy
from typing import Any

import jsonpatch
import jsonschema

from langgraph_kernel.types import DataSchema, JsonPatch


class PatchValidator:
    """验证 JSON Patch 是否合法，并将其应用到当前状态。"""

    def validate(
        self,
        current_state: dict[str, Any],
        patch: JsonPatch,
        schema: DataSchema,
    ) -> tuple[dict[str, Any], str | None]:
        """
        将 patch 应用到 current_state 副本，然后验证结果是否符合 schema。

        返回 (new_state, error_message)。
        error_message 为 None 表示验证通过；否则返回原始 state 和错误描述。
        """
        candidate = copy.deepcopy(current_state)

        # 1. 应用 patch
        try:
            candidate = jsonpatch.apply_patch(candidate, patch)
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
            return current_state, f"patch apply error: {e}"

        # 2. 验证 schema
        try:
            jsonschema.validate(instance=candidate, schema=schema)
        except jsonschema.ValidationError as e:
            return current_state, f"schema validation error: {e.message}"

        return candidate, None
