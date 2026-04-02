"""
JSON Patch 分层自动修复器

实现 5 层修复策略：
1. 结构验证 - 检查 patch 基本结构
2. 路径修复 - 修正路径格式和 add/replace 混淆
3. 类型修复 - 根据 schema 转换类型
4. 应用验证 - 应用并验证
5. 反馈重试 - 生成错误报告供 worker 重试
"""
from __future__ import annotations

import copy
import difflib
from typing import Any, Dict, List, Optional, Tuple

import jsonpatch
import jsonschema

from langgraph_kernel.types import DataSchema, JsonPatch
from langgraph_kernel.kernel.validator import PatchValidator

_validator = PatchValidator()


class FixLog:
    """修复日志"""

    def __init__(
        self,
        level: str,
        layer: str,
        operation_index: int,
        message: str,
        original_value: Any = None,
        fixed_value: Any = None,
    ):
        self.level = level  # "info", "warning", "error"
        self.layer = layer  # "structure", "path", "type", "apply"
        self.operation_index = operation_index
        self.message = message
        self.original_value = original_value
        self.fixed_value = fixed_value

    def __str__(self) -> str:
        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}[self.level]
        return f"{prefix} [{self.layer}] Op#{self.operation_index}: {self.message}"


class PatchFixerConfig:
    """修复器配置"""

    def __init__(
        self,
        fix_structure: bool = True,
        fix_spelling: bool = True,
        fix_path_format: bool = True,
        auto_switch_add_replace: bool = True,
        create_missing_parents: bool = False,
        fix_type_mismatch: bool = True,
        fuzzy_match_enum: bool = True,
        fuzzy_match_threshold: float = 0.8,
        skip_invalid_operations: bool = False,
    ):
        self.fix_structure = fix_structure
        self.fix_spelling = fix_spelling
        self.fix_path_format = fix_path_format
        self.auto_switch_add_replace = auto_switch_add_replace
        self.create_missing_parents = create_missing_parents
        self.fix_type_mismatch = fix_type_mismatch
        self.fuzzy_match_enum = fuzzy_match_enum
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.skip_invalid_operations = skip_invalid_operations


class PatchFixer:
    """分层 JSON Patch 修复器"""

    VALID_OPS = {"add", "remove", "replace", "move", "copy", "test"}

    def __init__(self, config: Optional[PatchFixerConfig] = None):
        self.config = config or PatchFixerConfig()
        self.logs: List[FixLog] = []

    def fix_and_validate(
        self,
        patch: JsonPatch,
        current_state: dict[str, Any],
        schema: DataSchema,
    ) -> Tuple[dict[str, Any], Optional[str], List[FixLog]]:
        """
        执行分层修复并验证

        返回: (新状态, 错误信息, 修复日志)
        """
        self.logs = []

        # 拦截阶段：先对原始 patch 做严格校验，有问题直接拦截，不进入修复流程
        _, intercept_error = _validator.validate(current_state, patch, schema)
        if intercept_error:
            self.logs.append(FixLog("error", "intercept", -1, intercept_error))
            return current_state, intercept_error, self.logs

        # Layer 1: 结构验证
        patch = self._layer1_structure_check(patch)
        if not patch:
            return current_state, "patch structure invalid", self.logs

        # Layer 2: 路径修复
        patch = self._layer2_path_fixer(patch, current_state)

        # Layer 3: 类型修复
        patch = self._layer3_type_fixer(patch, schema)

        # Layer 4: 应用验证
        new_state, error = self._layer4_apply_validate(patch, current_state, schema)

        return new_state, error, self.logs

    def _layer1_structure_check(self, patch: JsonPatch) -> Optional[JsonPatch]:
        """Layer 1: 检查并修复 patch 结构"""
        if not self.config.fix_structure:
            return patch

        # 检查是否为列表
        if not isinstance(patch, list):
            self.logs.append(
                FixLog("error", "structure", -1, "Patch 不是数组，无法修复")
            )
            return None

        fixed_patch = []

        for i, operation in enumerate(patch):
            if not isinstance(operation, dict):
                self.logs.append(
                    FixLog("warning", "structure", i, "操作不是对象，跳过")
                )
                continue

            # 检查必需字段
            if "op" not in operation:
                self.logs.append(
                    FixLog("warning", "structure", i, "缺少 'op' 字段，跳过")
                )
                continue

            if "path" not in operation:
                self.logs.append(
                    FixLog("warning", "structure", i, "缺少 'path' 字段，跳过")
                )
                continue

            # 修正操作拼写
            op = operation["op"]
            if op not in self.VALID_OPS and self.config.fix_spelling:
                corrected_op = self._fuzzy_match(op, list(self.VALID_OPS))
                if corrected_op:
                    operation = copy.copy(operation)
                    operation["op"] = corrected_op
                    self.logs.append(
                        FixLog(
                            "info",
                            "structure",
                            i,
                            f"修正操作拼写: {op} → {corrected_op}",
                            op,
                            corrected_op,
                        )
                    )
                else:
                    self.logs.append(
                        FixLog("warning", "structure", i, f"未知操作: {op}，跳过")
                    )
                    continue

            # 检查 value 字段（add/replace 需要）
            if operation["op"] in ["add", "replace", "test"]:
                if "value" not in operation:
                    self.logs.append(
                        FixLog(
                            "warning",
                            "structure",
                            i,
                            f"{operation['op']} 操作缺少 'value' 字段，跳过",
                        )
                    )
                    continue

            fixed_patch.append(operation)

        return fixed_patch

    def _layer2_path_fixer(
        self, patch: JsonPatch, current_state: dict[str, Any]
    ) -> JsonPatch:
        """Layer 2: 修复路径问题"""
        fixed_patch = []

        for i, operation in enumerate(patch):
            operation = copy.deepcopy(operation)
            path = operation["path"]
            op = operation["op"]

            # 修正路径格式
            if self.config.fix_path_format:
                # 确保以 / 开头
                if path and not path.startswith("/"):
                    operation["path"] = "/" + path
                    self.logs.append(
                        FixLog(
                            "info",
                            "path",
                            i,
                            f"添加路径前缀: {path} → /{path}",
                            path,
                            "/" + path,
                        )
                    )
                    path = operation["path"]

                # 移除多余的斜杠
                if path.startswith("//"):
                    operation["path"] = path[1:]
                    self.logs.append(
                        FixLog(
                            "info",
                            "path",
                            i,
                            f"移除多余斜杠: {path} → {path[1:]}",
                            path,
                            path[1:],
                        )
                    )
                    path = operation["path"]

            # 自动切换 add/replace
            if self.config.auto_switch_add_replace:
                path_exists = self._path_exists(current_state, path)

                if op == "replace" and not path_exists:
                    operation["op"] = "add"
                    self.logs.append(
                        FixLog(
                            "info",
                            "path",
                            i,
                            f"路径不存在，将 replace 改为 add: {path}",
                            "replace",
                            "add",
                        )
                    )

                elif op == "add" and path_exists:
                    operation["op"] = "replace"
                    self.logs.append(
                        FixLog(
                            "info",
                            "path",
                            i,
                            f"路径已存在，将 add 改为 replace: {path}",
                            "add",
                            "replace",
                        )
                    )

            fixed_patch.append(operation)

        return fixed_patch

    def _layer3_type_fixer(self, patch: JsonPatch, schema: DataSchema) -> JsonPatch:
        """Layer 3: 修复类型问题"""
        if not self.config.fix_type_mismatch:
            return patch

        if not schema or "properties" not in schema:
            return patch

        properties = schema.get("properties", {})
        fixed_patch = []

        for i, operation in enumerate(patch):
            operation = copy.deepcopy(operation)
            op = operation["op"]

            if op not in ["add", "replace"]:
                fixed_patch.append(operation)
                continue

            path = operation["path"]
            value = operation.get("value")

            # 提取字段名
            field_name = self._extract_field_name(path)
            if not field_name or field_name not in properties:
                fixed_patch.append(operation)
                continue

            field_schema = properties[field_name]

            # 修复类型
            fixed_value, changed = self._fix_value_type(
                value, field_schema, field_name, i
            )
            if changed:
                operation["value"] = fixed_value

            # 修复 enum
            if self.config.fuzzy_match_enum and "enum" in field_schema:
                fixed_value, changed = self._fix_enum_value(
                    operation["value"], field_schema["enum"], field_name, i
                )
                if changed:
                    operation["value"] = fixed_value

            fixed_patch.append(operation)

        return fixed_patch

    def _layer4_apply_validate(
        self, patch: JsonPatch, current_state: dict[str, Any], schema: DataSchema
    ) -> Tuple[dict[str, Any], Optional[str]]:
        """Layer 4: 应用并验证"""
        candidate = copy.deepcopy(current_state)

        # 应用 patch
        try:
            candidate = jsonpatch.apply_patch(candidate, patch)
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
            error_msg = f"patch apply error: {e}"
            self.logs.append(FixLog("error", "apply", -1, error_msg))
            return current_state, error_msg

        # 验证 schema
        try:
            jsonschema.validate(instance=candidate, schema=schema)
        except jsonschema.ValidationError as e:
            error_msg = f"schema validation error: {e.message}"
            self.logs.append(FixLog("error", "apply", -1, error_msg))
            return current_state, error_msg

        return candidate, None

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _path_exists(self, state: dict[str, Any], path: str) -> bool:
        """检查路径是否存在"""
        if not path or path == "/":
            return True

        try:
            parts = path.split("/")[1:]
            current = state

            for part in parts:
                if isinstance(current, list):
                    try:
                        index = int(part)
                        if 0 <= index < len(current):
                            current = current[index]
                        else:
                            return False
                    except ValueError:
                        return False
                elif isinstance(current, dict):
                    if part in current:
                        current = current[part]
                    else:
                        return False
                else:
                    return False

            return True
        except Exception:
            return False

    def _extract_field_name(self, path: str) -> str:
        """从路径提取字段名"""
        if path.startswith("/"):
            parts = path.split("/")
            return parts[1] if len(parts) > 1 else ""
        return ""

    def _fix_value_type(
        self, value: Any, field_schema: dict, field_name: str, op_index: int
    ) -> Tuple[Any, bool]:
        """修复值类型"""
        expected_type = field_schema.get("type")
        if not expected_type:
            return value, False

        try:
            if expected_type == "string" and not isinstance(value, str):
                if isinstance(value, (dict, list)):
                    import json

                    fixed = json.dumps(value, ensure_ascii=False)
                    self.logs.append(
                        FixLog(
                            "info",
                            "type",
                            op_index,
                            f"将 {field_name} 从 {type(value).__name__} 转为 string",
                            value,
                            fixed,
                        )
                    )
                    return fixed, True
                else:
                    fixed = str(value)
                    self.logs.append(
                        FixLog(
                            "info",
                            "type",
                            op_index,
                            f"将 {field_name} 转为 string",
                            value,
                            fixed,
                        )
                    )
                    return fixed, True

            elif expected_type == "array" and not isinstance(value, list):
                fixed = [value]
                self.logs.append(
                    FixLog(
                        "info",
                        "type",
                        op_index,
                        f"将 {field_name} 包装为 array",
                        value,
                        fixed,
                    )
                )
                return fixed, True

            elif expected_type == "object" and isinstance(value, str):
                try:
                    import json

                    fixed = json.loads(value)
                    self.logs.append(
                        FixLog(
                            "info",
                            "type",
                            op_index,
                            f"将 {field_name} 从 JSON 字符串解析为 object",
                            value,
                            fixed,
                        )
                    )
                    return fixed, True
                except json.JSONDecodeError:
                    pass

        except Exception:
            pass

        return value, False

    def _fix_enum_value(
        self, value: Any, enum_values: list, field_name: str, op_index: int
    ) -> Tuple[Any, bool]:
        """修复 enum 值（模糊匹配）"""
        if value in enum_values:
            return value, False

        # 只对字符串进行模糊匹配
        if not isinstance(value, str):
            return value, False

        matched = self._fuzzy_match(value, [str(v) for v in enum_values])
        if matched:
            self.logs.append(
                FixLog(
                    "info",
                    "type",
                    op_index,
                    f"模糊匹配 {field_name} 的 enum 值: {value} → {matched}",
                    value,
                    matched,
                )
            )
            return matched, True

        return value, False

    def _fuzzy_match(self, value: str, candidates: list[str]) -> Optional[str]:
        """模糊匹配字符串"""
        if not value or not candidates:
            return None

        matches = difflib.get_close_matches(
            value, candidates, n=1, cutoff=self.config.fuzzy_match_threshold
        )
        return matches[0] if matches else None

    def generate_error_report(self) -> str:
        """生成详细的错误报告（用于反馈给 worker）"""
        if not self.logs:
            return "No errors"

        report_lines = ["JSON Patch 错误报告:", ""]

        errors = [log for log in self.logs if log.level == "error"]
        warnings = [log for log in self.logs if log.level == "warning"]
        infos = [log for log in self.logs if log.level == "info"]

        if errors:
            report_lines.append("错误:")
            for log in errors:
                report_lines.append(f"  {log}")
            report_lines.append("")

        if warnings:
            report_lines.append("警告:")
            for log in warnings:
                report_lines.append(f"  {log}")
            report_lines.append("")

        if infos:
            report_lines.append("自动修复:")
            for log in infos:
                report_lines.append(f"  {log}")

        return "\n".join(report_lines)
