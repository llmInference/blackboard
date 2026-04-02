"""
Extension System - 支持自定义 Worker 注册和 Architect Prompt 扩展

提供插件式扩展机制，允许用户：
1. 注册自定义 Worker 类型
2. 扩展或替换 Architect 的 system prompt
3. 动态加载扩展模块
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from langgraph_kernel.worker.base import LLMWorkerAgent
from langgraph_kernel.worker.registry import get_registry


# ── Architect Prompt Extension ────────────────────────────────────────────────

class ArchitectPromptExtension:
    """Architect Prompt 扩展管理器"""

    def __init__(self) -> None:
        self._custom_prompt: str | None = None
        self._prompt_modifiers: list[Callable[[str], str]] = []

    def set_custom_prompt(self, prompt: str) -> None:
        """
        设置自定义 Architect prompt（完全替换默认 prompt）

        Args:
            prompt: 自定义的 system prompt
        """
        self._custom_prompt = prompt

    def add_prompt_modifier(self, modifier: Callable[[str], str]) -> None:
        """
        添加 prompt 修改器（在默认 prompt 基础上修改）

        Args:
            modifier: 接收原始 prompt，返回修改后的 prompt 的函数
        """
        self._prompt_modifiers.append(modifier)

    def get_prompt(self, default_prompt: str) -> str:
        """
        获取最终的 Architect prompt

        Args:
            default_prompt: 默认的 system prompt

        Returns:
            最终使用的 prompt（自定义或修改后的）
        """
        # 如果有自定义 prompt，直接使用
        if self._custom_prompt:
            return self._custom_prompt

        # 否则应用所有修改器
        result = default_prompt
        for modifier in self._prompt_modifiers:
            result = modifier(result)

        return result

    def clear(self) -> None:
        """清除所有自定义设置"""
        self._custom_prompt = None
        self._prompt_modifiers.clear()


# 全局 Architect Prompt 扩展实例
_global_architect_extension = ArchitectPromptExtension()


def get_architect_extension() -> ArchitectPromptExtension:
    """获取全局 Architect Prompt 扩展实例"""
    return _global_architect_extension


# ── Extension Loader ──────────────────────────────────────────────────────────

class ExtensionLoader:
    """扩展加载器 - 从 Python 模块或文件加载扩展"""

    @staticmethod
    def load_from_module(module_name: str) -> None:
        """
        从 Python 模块加载扩展

        模块应该定义以下函数之一或多个：
        - register_workers(registry: WorkerRegistry) -> None
        - configure_architect(extension: ArchitectPromptExtension) -> None

        Args:
            module_name: Python 模块名（如 'my_extensions.custom_workers'）
        """
        try:
            module = importlib.import_module(module_name)

            # 调用 worker 注册函数
            if hasattr(module, 'register_workers'):
                registry = get_registry()
                module.register_workers(registry)
                print(f"✅ 已从 {module_name} 加载 worker 扩展")

            # 调用 architect 配置函数
            if hasattr(module, 'configure_architect'):
                extension = get_architect_extension()
                module.configure_architect(extension)
                print(f"✅ 已从 {module_name} 加载 architect 扩展")

        except ImportError as e:
            print(f"⚠️  无法加载扩展模块 {module_name}: {e}")
        except Exception as e:
            print(f"⚠️  加载扩展时出错: {e}")

    @staticmethod
    def load_from_file(file_path: str | Path) -> None:
        """
        从 Python 文件加载扩展

        Args:
            file_path: Python 文件路径
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"⚠️  扩展文件不存在: {file_path}")
            return

        # 动态导入文件
        spec = importlib.util.spec_from_file_location("_extension_module", file_path)
        if spec is None or spec.loader is None:
            print(f"⚠️  无法加载扩展文件: {file_path}")
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules["_extension_module"] = module

        try:
            spec.loader.exec_module(module)

            # 调用 worker 注册函数
            if hasattr(module, 'register_workers'):
                registry = get_registry()
                module.register_workers(registry)
                print(f"✅ 已从 {file_path} 加载 worker 扩展")

            # 调用 architect 配置函数
            if hasattr(module, 'configure_architect'):
                extension = get_architect_extension()
                module.configure_architect(extension)
                print(f"✅ 已从 {file_path} 加载 architect 扩展")

        except Exception as e:
            print(f"⚠️  加载扩展时出错: {e}")


# ── Convenience Functions ─────────────────────────────────────────────────────

def register_custom_worker(
    name: str,
    worker_class: type[LLMWorkerAgent] | None = None,
    system_prompt: str | None = None,
) -> None:
    """
    便捷函数：注册自定义 Worker

    Args:
        name: Worker 名称
        worker_class: Worker 类（如果为 None，将创建一个基于 system_prompt 的类）
        system_prompt: Worker 的 system prompt（仅在 worker_class 为 None 时使用）
    """
    registry = get_registry()

    if worker_class is not None:
        registry.register(name, worker_class)
    elif system_prompt is not None:
        # 动态创建 Worker 类
        class CustomWorker(LLMWorkerAgent):
            pass

        CustomWorker.system_prompt = system_prompt
        CustomWorker.__name__ = f"{name.capitalize()}Worker"
        CustomWorker.__doc__ = f"Custom worker: {name}"

        registry.register(name, CustomWorker)
    else:
        raise ValueError("必须提供 worker_class 或 system_prompt")


def set_custom_architect_prompt(prompt: str) -> None:
    """
    便捷函数：设置自定义 Architect prompt

    Args:
        prompt: 自定义的 system prompt
    """
    extension = get_architect_extension()
    extension.set_custom_prompt(prompt)


def add_architect_prompt_section(section: str, position: str = "end") -> None:
    """
    便捷函数：在 Architect prompt 中添加新的部分

    Args:
        section: 要添加的文本
        position: 添加位置（'start' 或 'end'）
    """
    extension = get_architect_extension()

    def modifier(original: str) -> str:
        if position == "start":
            return f"{section}\n\n{original}"
        else:
            return f"{original}\n\n{section}"

    extension.add_prompt_modifier(modifier)


def load_extensions(module_or_file: str) -> None:
    """
    便捷函数：加载扩展（自动判断是模块还是文件）

    Args:
        module_or_file: 模块名或文件路径
    """
    loader = ExtensionLoader()

    # 判断是文件还是模块
    if Path(module_or_file).exists():
        loader.load_from_file(module_or_file)
    else:
        loader.load_from_module(module_or_file)
