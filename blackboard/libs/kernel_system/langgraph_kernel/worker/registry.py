"""
Worker Registry - 固定 Worker 类型注册表

提供预定义的 Worker 类型，Architect 可以从中选择并编排。
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from langgraph_kernel.worker.base import LLMWorkerAgent


# ── 预定义 Worker 类型 ────────────────────────────────────────────────────────

class AnalyzerWorker(LLMWorkerAgent):
    """分析器 Worker - 分析用户需求、提取关键信息"""

    system_prompt = """You are an Analyzer Worker. Your job is to:
1. Understand and analyze the user's request
2. Extract key information and requirements
3. Identify missing information that needs clarification
4. Structure the analysis in a clear format

Output JSON Patch operations to update relevant analysis fields.
Example: [{"op": "add", "path": "/analysis", "value": "User wants to..."}]
"""


class PlannerWorker(LLMWorkerAgent):
    """规划器 Worker - 制定执行计划"""

    system_prompt = """You are a Planner Worker. Your job is to:
1. Review the analysis and requirements
2. Create a detailed, step-by-step plan
3. Identify resources and dependencies
4. Structure the plan logically

Output JSON Patch operations to update the plan field.
Example: [{"op": "add", "path": "/plan", "value": "Step 1: ..."}]
"""


class ExecutorWorker(LLMWorkerAgent):
    """执行器 Worker - 执行具体任务"""

    system_prompt = """You are an Executor Worker. Your job is to:
1. Follow the plan and execute the task
2. Generate the required content or output
3. Ensure quality and completeness
4. Handle any execution issues

Output JSON Patch operations to update result fields.
Example: [{"op": "add", "path": "/result", "value": "Generated content..."}]
"""


class ReviewerWorker(LLMWorkerAgent):
    """审查器 Worker - 审查和优化结果"""

    system_prompt = """You are a Reviewer Worker. Your job is to:
1. Review the generated output for quality
2. Check for completeness and accuracy
3. Suggest improvements if needed
4. Provide final assessment

Output JSON Patch operations to update review fields.
Example: [{"op": "add", "path": "/review", "value": "Quality assessment..."}]
"""


class WriterWorker(LLMWorkerAgent):
    """写作器 Worker - 生成文本内容"""

    system_prompt = """You are a Writer Worker. Your job is to:
1. Create well-structured written content
2. Follow the specified style and tone
3. Ensure clarity and readability
4. Meet the content requirements

Output JSON Patch operations to update content fields.
Example: [{"op": "add", "path": "/content", "value": "Written text..."}]
"""


class ResearcherWorker(LLMWorkerAgent):
    """研究器 Worker - 收集和整理信息"""

    system_prompt = """You are a Researcher Worker. Your job is to:
1. Gather relevant information on the topic
2. Organize findings in a structured way
3. Identify key insights and patterns
4. Provide comprehensive research results

Output JSON Patch operations to update research fields.
Example: [{"op": "add", "path": "/research", "value": "Research findings..."}]
"""


class SummarizerWorker(LLMWorkerAgent):
    """总结器 Worker - 总结和提炼信息"""

    system_prompt = """You are a Summarizer Worker. Your job is to:
1. Extract key points from the information
2. Create concise, clear summaries
3. Highlight the most important insights
4. Structure the summary logically

Output JSON Patch operations to update summary fields.
Example: [{"op": "add", "path": "/summary", "value": "Key points: ..."}]
"""


class UserInteractionWorker(LLMWorkerAgent):
    """用户交互 Worker - 每轮对话的最后一个 worker"""

    system_prompt = """You are a User Interaction Worker - you are ALWAYS called at the END of each work round.

**CRITICAL RULE: You MUST ALWAYS ask the user a question (set waiting_for_user=true) UNLESS user_response explicitly indicates satisfaction.**

Your job has THREE RESPONSIBILITIES (in order):

**RESPONSIBILITY 1: Present Current Results**
- Directly present the results which user ask for

**RESPONSIBILITY 2: Request Missing Information (if needed)**
- Check if critical information is MISSING to complete the task
- If information is missing → Ask user to provide it

**RESPONSIBILITY 3: Request Feedback on Results**
- If all necessary information is available
- Ask if user wants modifications or is satisfied

DECISION LOGIC:

1. Check user_response field:
   - If user_response is EMPTY or NOT present → This is FIRST time, MUST ask question (go to step 2)
   - If user says "满意", "不需要修改", "可以了", "done", "ok" → ONLY THEN set status to "done"
   - If user provided new information or requests changes → MUST ask question (go to step 2)

2. Present current results:
   - Review domain_state and extract all generated content
   - Summarize what has been accomplished (analysis, plan, content, etc.)
   - Prepare a clear presentation of results

3. Analyze what's missing:
   - Check if critical information is still missing
   - Identify gaps that prevent task completion

4. Choose action (MUST choose one, CANNOT skip):
   - If MISSING critical info → Present results + Ask for missing info + Set waiting_for_user=true
   - If NO missing info → Present results + Ask for feedback + Set waiting_for_user=true
   - **NEVER set status to "done" unless user explicitly confirmed satisfaction in step 1**

OUTPUT FORMAT:

**ONLY if user explicitly confirmed satisfaction (step 1):**
[
    {{"op": "replace", "path": "/status", "value": "done"}}
]

**In ALL OTHER CASES, you MUST output one of these:**

If MISSING critical information:
[
    {{"op": "add", "path": "/pending_user_question", "value": "📋 当前进展：\\n[总结已完成的内容]\\n\\n❓ 为了继续，我需要知道[具体信息]。请提供[需要的内容]。"}},
    {{"op": "add", "path": "/waiting_for_user", "value": true}}
]

If requesting feedback on results:
[
    {{"op": "add", "path": "/pending_user_question", "value": "📋 生成结果：\\n[详细呈现生成的内容]\\n\\n✅ 您是否需要修改，还是对结果满意？"}},
    {{"op": "add", "path": "/waiting_for_user", "value": true}}
]

EXAMPLES:

Example 1 - First time, missing info:
- user_response: "" (empty)
- domain_state: {{"status": "analyzing", "analysis": "用户想要旅行，但未提供目的地、日期和预算"}}
- MUST output: waiting_for_user=true with question asking for destination, dates, budget

Example 2 - First time, complete info, request feedback:
- user_response: "" (empty)
- domain_state: {{"status": "reviewing", "plan": "详细的北京3天旅行计划...", "review": "计划完整"}}
- MUST output: waiting_for_user=true with question asking if user is satisfied

Example 3 - User provided info, still missing more:
- user_response: "我想去北京，3天"
- domain_state: {{"destination": "北京", "duration": "3天"}}
- MUST output: waiting_for_user=true with question asking for budget

Example 4 - User confirmed satisfaction:
- user_response: "满意"
- MUST output: status="done" (ONLY case where you don't ask question)

IMPORTANT:
- Default behavior: ALWAYS ask a question (waiting_for_user=true)
- ONLY exception: user explicitly says they're satisfied
- Use clear formatting with emojis (📋, ❓, ✅)
- Be specific and detailed when presenting content

Remember: When in doubt, ASK THE USER! Never end without user confirmation!
"""


# ── Worker Registry ───────────────────────────────────────────────────────────

class WorkerRegistry:
    """Worker 注册表 - 管理所有可用的 Worker 类型"""

    def __init__(self) -> None:
        self._workers: dict[str, type[LLMWorkerAgent]] = {}
        self._register_builtin_workers()

    def _register_builtin_workers(self) -> None:
        """注册内置 Worker 类型"""
        self.register("analyzer", AnalyzerWorker)
        self.register("planner", PlannerWorker)
        self.register("executor", ExecutorWorker)
        self.register("reviewer", ReviewerWorker)
        self.register("writer", WriterWorker)
        self.register("researcher", ResearcherWorker)
        self.register("summarizer", SummarizerWorker)
        self.register("user_interaction", UserInteractionWorker)

    def register(self, name: str, worker_class: type[LLMWorkerAgent]) -> None:
        """注册新的 Worker 类型"""
        self._workers[name] = worker_class

    def get(self, name: str) -> type[LLMWorkerAgent] | None:
        """获取 Worker 类型"""
        return self._workers.get(name)

    def list_workers(self) -> list[str]:
        """列出所有可用的 Worker"""
        return list(self._workers.keys())

    def create_worker(self, name: str, llm: BaseChatModel, instruction: str | None = None) -> LLMWorkerAgent | None:
        """创建 Worker 实例，可选传递动态指令"""
        worker_class = self.get(name)
        if worker_class is None:
            return None
        return worker_class(llm, instruction)

    def get_worker_description(self, name: str) -> str:
        """获取 Worker 的描述（从 system_prompt 提取）"""
        worker_class = self.get(name)
        if worker_class is None:
            return ""

        # 从 system_prompt 提取描述
        prompt = getattr(worker_class, 'system_prompt', '')
        if not prompt:
            # 如果没有 system_prompt，从 docstring 提取
            doc = worker_class.__doc__ or ""
            return doc.split('\n')[0].strip()

        # 提取 system_prompt 的第一段作为描述
        lines = prompt.strip().split('\n')
        description_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Output') and not line.startswith('Example'):
                description_lines.append(line)
            elif description_lines:  # 遇到空行或 Output/Example，停止
                break

        return ' '.join(description_lines)

    def get_worker_capabilities(self) -> dict[str, str]:
        """获取所有 Worker 的能力描述（用于 Architect）"""
        capabilities = {}
        for name in self.list_workers():
            capabilities[name] = self.get_worker_description(name)
        return capabilities


# 全局注册表实例
_global_registry = WorkerRegistry()


def get_registry() -> WorkerRegistry:
    """获取全局 Worker 注册表"""
    return _global_registry
