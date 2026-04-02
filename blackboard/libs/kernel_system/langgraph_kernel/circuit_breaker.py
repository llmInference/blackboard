"""
Kernel 熔断器 - 检测异常情况并触发熔断
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import hashlib
import json


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""

    # Patch 无效
    max_invalid_patch_count: int = 3

    # 振荡检测
    oscillation_window: int = 6
    oscillation_threshold: int = 2

    # 停滞检测
    stagnation_threshold: int = 3
    stagnation_min_change_ratio: float = 0.1  # 最小变化比例

    # 超时
    max_steps: int = 50
    max_worker_time: float = 60.0  # 秒

    # 错误累积
    max_error_rate: float = 0.5
    max_consecutive_errors: int = 5


class CircuitBreaker:
    """Kernel 熔断器 - 统一管理系统终止决策"""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state_history: List[Dict[str, Any]] = []
        self.error_history: List[Dict[str, Any]] = []
        self.patch_history: List[Dict[str, Any]] = []
        self.invalid_patch_count: int = 0
        self.consecutive_errors: int = 0

    def check(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        检查是否应该熔断

        Returns:
            (should_break, reason, details)
        """
        # 按优先级检查各种熔断条件
        checks = [
            self._check_timeout(state),
            self._check_invalid_patch(state),
            self._check_error_accumulation(state),
            self._check_oscillation(state),
            self._check_stagnation(state),
        ]

        for should_break, reason, details in checks:
            if should_break:
                return True, reason, details

        return False, "", {}

    def record_state(self, state: Dict[str, Any]) -> None:
        """记录状态到历史"""
        domain_state = state.get("domain_state", {})

        # 记录状态
        self.state_history.append({
            "step": state.get("step_count", 0),
            "status": domain_state.get("status", "unknown"),
            "domain_state_hash": self._hash_dict(domain_state),
            "worker": state.get("current_worker", ""),
        })

        # 记录 patch
        patch = state.get("pending_patch", [])
        patch_error = state.get("patch_error", "")

        self.patch_history.append({
            "step": state.get("step_count", 0),
            "patch": patch,
            "valid": not bool(patch_error),
            "error": patch_error,
            "has_business_data": self._has_business_data(patch),
        })

        # 更新计数器
        if patch_error:
            self.invalid_patch_count += 1
            self.consecutive_errors += 1
            self.error_history.append({
                "step": state.get("step_count", 0),
                "error": patch_error,
            })
        else:
            self.consecutive_errors = 0

    def _check_timeout(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """检查超时"""
        step_count = state.get("step_count", 0)

        if step_count >= self.config.max_steps:
            return True, "timeout", {
                "step_count": step_count,
                "max_steps": self.config.max_steps,
                "message": f"系统执行步数 ({step_count}) 超过最大限制 ({self.config.max_steps})"
            }

        return False, "", {}

    def _check_invalid_patch(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """检查 Patch 无效"""
        if self.invalid_patch_count >= self.config.max_invalid_patch_count:
            return True, "invalid_patch", {
                "invalid_count": self.invalid_patch_count,
                "threshold": self.config.max_invalid_patch_count,
                "recent_errors": [e["error"] for e in self.error_history[-3:]],
                "message": f"连续 {self.invalid_patch_count} 次 patch 验证失败"
            }

        return False, "", {}

    def _check_error_accumulation(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """检查错误累积"""
        # 检查连续错误
        if self.consecutive_errors >= self.config.max_consecutive_errors:
            return True, "error_accumulation", {
                "consecutive_errors": self.consecutive_errors,
                "threshold": self.config.max_consecutive_errors,
                "message": f"连续 {self.consecutive_errors} 次错误"
            }

        # 检查错误率
        if len(self.patch_history) >= 10:
            recent_patches = self.patch_history[-10:]
            error_count = sum(1 for p in recent_patches if not p["valid"])
            error_rate = error_count / len(recent_patches)

            if error_rate >= self.config.max_error_rate:
                return True, "error_accumulation", {
                    "error_rate": error_rate,
                    "threshold": self.config.max_error_rate,
                    "error_count": error_count,
                    "total_count": len(recent_patches),
                    "message": f"错误率 ({error_rate:.1%}) 超过阈值 ({self.config.max_error_rate:.1%})"
                }

        return False, "", {}

    def _check_oscillation(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """检测振荡模式"""
        window = self.config.oscillation_window

        if len(self.state_history) < window:
            return False, "", {}

        # 提取最近的状态序列
        recent_states = [s["status"] for s in self.state_history[-window:]]

        # 检测循环模式 (A→B→A→B)
        pattern_length = window // 2
        if pattern_length < 2:
            return False, "", {}

        pattern = tuple(recent_states[:pattern_length])
        repeat = tuple(recent_states[pattern_length:pattern_length * 2])

        if pattern == repeat and len(set(pattern)) > 1:
            return True, "oscillation", {
                "pattern": list(pattern),
                "window": window,
                "history": recent_states,
                "message": f"检测到振荡模式: {' → '.join(map(str, pattern))} (重复)"
            }

        return False, "", {}

    def _check_stagnation(self, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """检测状态停滞"""
        threshold = self.config.stagnation_threshold

        if len(self.patch_history) < threshold:
            return False, "", {}

        # 检查最近 N 次 patch 是否都无实质性业务数据
        recent_patches = self.patch_history[-threshold:]

        stagnant_count = sum(1 for p in recent_patches if not p["has_business_data"])

        if stagnant_count >= threshold:
            return True, "stagnation", {
                "stagnant_count": stagnant_count,
                "threshold": threshold,
                "message": f"连续 {stagnant_count} 次无实质性业务数据更新"
            }

        # 检查 domain_state 是否停滞（哈希值不变）
        if len(self.state_history) >= threshold:
            recent_hashes = [s["domain_state_hash"] for s in self.state_history[-threshold:]]
            unique_hashes = len(set(recent_hashes))

            if unique_hashes == 1:
                return True, "stagnation", {
                    "threshold": threshold,
                    "message": f"连续 {threshold} 步 domain_state 完全相同"
                }

        return False, "", {}

    def _has_business_data(self, patch: List[Dict[str, Any]] | str) -> bool:
        """检查 patch 是否包含业务数据（非 status 字段）.

        When C1 is ablated, workers communicate in natural language and
        ``pending_patch`` becomes a plain string instead of JSON Patch.
        Any non-empty natural-language payload counts as business data.
        """
        if not patch:
            return False

        if isinstance(patch, str):
            return bool(patch.strip())

        for op in patch:
            path = op.get("path", "")
            # 排除 status 和其他控制字段
            if path not in ["/status", "/waiting_for_user", "/pending_user_question", "/user_response"]:
                return True

        return False

    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """计算字典的哈希值"""
        # 排除某些动态字段
        filtered = {k: v for k, v in d.items() if k not in ["user_response", "pending_user_question"]}
        json_str = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode()).hexdigest()

    def generate_diagnostic_report(self, reason: str, details: Dict[str, Any]) -> str:
        """生成诊断报告"""
        report = ["=" * 80]
        report.append("  🔴 系统熔断诊断报告")
        report.append("=" * 80)
        report.append("")

        # 熔断原因
        reason_map = {
            "timeout": "⏱️  超时",
            "invalid_patch": "❌ Patch 无效",
            "error_accumulation": "⚠️  错误累积",
            "oscillation": "🔄 振荡模式",
            "stagnation": "⏸️  状态停滞",
        }

        report.append(f"熔断原因: {reason_map.get(reason, reason)}")
        report.append(f"详细信息: {details.get('message', '无')}")
        report.append("")

        # 统计信息
        report.append("📊 执行统计:")
        report.append(f"  • 总步数: {len(self.state_history)}")
        report.append(f"  • 总错误数: {len(self.error_history)}")
        report.append(f"  • 无效 Patch 数: {self.invalid_patch_count}")
        report.append(f"  • 连续错误数: {self.consecutive_errors}")
        report.append("")

        # 最近的状态历史
        if self.state_history:
            report.append("📜 最近的状态历史:")
            for i, s in enumerate(self.state_history[-5:], 1):
                report.append(f"  {i}. 步骤 {s['step']}: {s['worker']} → {s['status']}")
            report.append("")

        # 最近的错误
        if self.error_history:
            report.append("❌ 最近的错误:")
            for i, e in enumerate(self.error_history[-3:], 1):
                error_msg = e['error'][:100] + "..." if len(e['error']) > 100 else e['error']
                report.append(f"  {i}. 步骤 {e['step']}: {error_msg}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def reset(self) -> None:
        """重置熔断器状态"""
        self.state_history.clear()
        self.error_history.clear()
        self.patch_history.clear()
        self.invalid_patch_count = 0
        self.consecutive_errors = 0

