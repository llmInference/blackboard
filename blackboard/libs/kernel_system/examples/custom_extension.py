"""
示例：自定义扩展

演示如何：
1. 注册自定义 Worker
2. 修改 Architect prompt
"""
from langchain_core.language_models import BaseChatModel

from langgraph_kernel.extensions import ArchitectPromptExtension
from langgraph_kernel.worker.base import LLMWorkerAgent
from langgraph_kernel.worker.registry import WorkerRegistry


# ── 方式 1: 定义自定义 Worker 类 ──────────────────────────────────────────────

class TranslatorWorker(LLMWorkerAgent):
    """翻译器 Worker - 翻译文本内容"""

    system_prompt = """You are a Translator Worker. Your job is to:
1. Translate text from one language to another
2. Maintain the original meaning and tone
3. Ensure natural and fluent translation
4. Handle cultural context appropriately

Output JSON Patch operations to update translation fields.
Example: [{"op": "add", "path": "/translation", "value": "Translated text..."}]
"""


class CodeGeneratorWorker(LLMWorkerAgent):
    """代码生成器 Worker - 生成代码"""

    system_prompt = """You are a Code Generator Worker. Your job is to:
1. Generate clean, well-structured code
2. Follow best practices and coding standards
3. Include appropriate comments and documentation
4. Ensure code is functional and efficient

Output JSON Patch operations to update code fields.
Example: [{"op": "add", "path": "/code", "value": "def example():\\n    pass"}]
"""


# ── 方式 2: 使用 system_prompt 字符串 ──────────────────────────────────────────

DEBUGGER_PROMPT = """You are a Debugger Worker. Your job is to:
1. Analyze code for bugs and errors
2. Identify root causes of issues
3. Suggest fixes and improvements
4. Provide clear explanations

Output JSON Patch operations to update debug fields.
Example: [{"op": "add", "path": "/debug_report", "value": "Issue found: ..."}]
"""


# ── Worker 注册函数 ───────────────────────────────────────────────────────────

def register_workers(registry: WorkerRegistry) -> None:
    """
    注册自定义 Worker 到 registry

    这个函数会被扩展加载器自动调用
    """
    # 注册方式 1: 使用 Worker 类
    registry.register("translator", TranslatorWorker)
    registry.register("code_generator", CodeGeneratorWorker)

    # 注册方式 2: 使用 system_prompt 创建动态类
    class DebuggerWorker(LLMWorkerAgent):
        system_prompt = DEBUGGER_PROMPT

    registry.register("debugger", DebuggerWorker)

    print("✅ 已注册自定义 workers: translator, code_generator, debugger")


# ── Architect Prompt 配置函数 ─────────────────────────────────────────────────

def configure_architect(extension: ArchitectPromptExtension) -> None:
    """
    配置 Architect prompt

    这个函数会被扩展加载器自动调用
    """
    # 方式 1: 添加额外的指导信息（在原有 prompt 基础上）
    additional_guidance = """
**CUSTOM EXTENSION - Additional Guidelines:**

When dealing with translation tasks:
- Always use the 'translator' worker for language translation
- Specify source and target languages clearly

When dealing with code tasks:
- Use 'code_generator' for creating new code
- Use 'debugger' for analyzing and fixing code issues
- Consider code quality and maintainability
"""

    extension.add_prompt_modifier(lambda prompt: f"{prompt}\n\n{additional_guidance}")

    # 方式 2: 完全替换 prompt（如果需要）
    # custom_prompt = "Your completely custom architect prompt here..."
    # extension.set_custom_prompt(custom_prompt)

    print("✅ 已配置 Architect prompt 扩展")


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 示例 1: 直接使用便捷函数
    from langgraph_kernel.extensions import (
        register_custom_worker,
        add_architect_prompt_section,
    )

    # 注册 worker
    register_custom_worker(
        "my_worker",
        system_prompt="You are a custom worker that does something special."
    )

    # 添加 prompt 部分
    add_architect_prompt_section(
        "**Special Note:** Always consider user preferences.",
        position="end"
    )

    # 示例 2: 使用加载器
    from langgraph_kernel.extensions import load_extensions

    # 从当前模块加载
    load_extensions("examples.custom_extension")

    # 或从文件加载
    # load_extensions("/path/to/custom_extension.py")
