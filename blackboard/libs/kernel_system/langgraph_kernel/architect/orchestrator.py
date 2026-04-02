"""
Orchestrator Architect - 从固定 Worker 中选择并编排

动态读取 Worker Registry 中的 worker 信息来了解可用的 workers。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_kernel.types import KernelState
from langgraph_kernel.worker.registry import get_registry
from langgraph_kernel.workflow import derive_selected_workers, sync_workflow_state_enums


def _build_orchestrator_prompt() -> str:
    """动态构建 Orchestrator prompt，包含所有可用 worker 的信息"""
    registry = get_registry()
    capabilities = registry.get_worker_capabilities()

    # 构建 worker 列表
    worker_list = []
    for name, description in capabilities.items():
        worker_list.append(f"- **{name}**: {description}")

    workers_section = "\n".join(worker_list)

    return f"""\
You are an Orchestrator Architect that designs workflows through task decomposition and worker selection.

Your workflow design process (THREE STAGES):

**STAGE 0: Information Completeness Check (NEW - CRITICAL)**
- FIRST, analyze if the user's request contains all necessary information to complete the task
- Check for missing critical information (e.g., destination for travel, topic for article, requirements for plan)
- Decision:
  * If MISSING critical info → Create MINIMAL workflow with ONLY user_interaction worker
  * If ALL info present → Proceed to STAGE 1 (full task decomposition)

**STAGE 1: Task Decomposition**
- Analyze the user's request and break it down into logical subtasks
- Each subtask should be a clear, atomic unit of work
- Subtasks may form a linear flow, a branch, or a converge-and-review pattern depending on the task.
- Consider what information each subtask needs and produces

**STAGE 2: Worker Selection**
- For each subtask, select the most appropriate worker from the available types
- Match the subtask's nature and requirements to worker capabilities

**CRITICAL: Multi-turn Conversation Pattern**
- ALWAYS make user_interaction the LAST worker in EVERY workflow (MANDATORY)
- After all other workers complete their tasks, user_interaction worker MUST review the results
- The user_interaction worker will:
  a) Present the generated content to the user
  b) Ask if the user wants any modifications or is satisfied
  c) Only set status=done when user explicitly confirms satisfaction

**REQUIRED PATTERN - user_interaction is MANDATORY:**
Every workflow MUST end with user_interaction before done. No exceptions.

**TWO WORKFLOW PATTERNS:**

Pattern A - Missing Information (go directly to user_interaction):
{{
  "status": {{
    "requesting_info": "user_interaction",
    "done": null
  }}
}}

Pattern B - Complete Information (full workflow):
{{
  "status": {{
    "analyzing": "analyzer",
    "planning": "planner",
    "executing": "executor",
    "reviewing": "reviewer",
    "user_feedback": "user_interaction",
    "done": null
  }}
}}

The workflow MUST be:
- If missing info: user_interaction → (wait for user) → full workflow
- If complete info: work → work → work → user_interaction → (wait for user) → done

Available Worker Types:
{workers_section}

Given a user prompt, output a JSON object with:

**IMPORTANT: First check if user wants to end the task**
- If this is a follow-up response (user_response exists), analyze user's intent:
  * If user indicates SATISFACTION ("满意", "不需要修改", "可以了", "ok", "done") → Output: {{"should_continue": false}}
  * If user provides NEW INFORMATION or requests MODIFICATIONS → Continue with workflow design
- This will end the system immediately without calling any workers if user is satisfied

**If task should continue (should_continue: true):**

**STEP 1: Analyze Information Completeness**
- Check if user's request contains all necessary information
- Determine if critical information is missing

**STEP 2: Choose Workflow Pattern**
- If MISSING critical info → Pattern A (direct to user_interaction)
- If ALL info present → Pattern B (full workflow with task decomposition)
- If user provided feedback/modifications → Pattern B (process modifications)

1. `task_flow`: List of workflow nodes.
   - Format: [{{"subtask": "description", "worker": "worker_name", "id": "optional_node_id", "depends_on": ["optional_upstream_node_id"]}}, ...]
   - This structure does NOT need to be strictly linear. It can describe branches, joins, or worker re-entry when useful.
   - Each subtask has a clear description and assigned worker
   - Example: [
       {{"subtask": "Analyze user requirements", "worker": "analyzer"}},
       {{"subtask": "Create execution plan", "worker": "planner"}},
       {{"subtask": "Review final output", "worker": "reviewer"}}
     ]

2. `worker_instructions`: Contextual instructions for each worker (NEW).
   - Format: {{"worker_name": "specific instruction based on user request and context"}}
   - Provide task-specific guidance that supplements the worker's base prompt
   - Include relevant details from user's request, previous results, and conversation history
   - Example: {{
       "analyzer": "用户想要规划去北京的3天旅行，预算5000元。请分析这些需求并提取关键信息。",
       "planner": "基于分析结果，制定详细的北京3天旅行计划，确保总预算不超过5000元。",
       "reviewer": "检查旅行计划是否符合用户的预算和时间要求。"
     }}

3. `data_schema`: A JSON Schema (draft-07) defining the domain state structure.
   - Include fields for: task context, current status, intermediate results, final output
   - Use descriptive field names that reflect the task domain
   - Add "description" annotations to clarify field purposes

   **CRITICAL - Schema Design Guidelines (Type Safety):**
   - **Default to "string" type for ALL fields unless absolutely necessary**
   - Only use "object" or "array" if the data MUST be structured
   - **IMPORTANT**: Workers often return simple text - use "string" to avoid type errors
   - Use "string" for: descriptions, summaries, plans (as text), analysis results
   - Use "array" ONLY for: lists of simple items - {{"type": "array", "items": {{"type": "string"}}}}
   - Use "object" ONLY for: truly structured data with known properties
   - **When in doubt, use "string"** - it's safer and more flexible

4. `workflow_rules`: State-based routing rules for worker selection.
   - Format: {{"field_name": {{"state_value": "worker_name" | {{"worker": "worker_name", "next": "next_state" | conditional_transition}}, ...}}}}
   - `next` can be a direct state name or a conditional object such as {{"cases": [{{"when": {{"field": "need_research", "equals": true}}, "target": "researching"}}], "default": "drafting"}}
   - **IMPORTANT**: worker_name MUST be one of the available workers listed above
   - Design the runtime flow from your task graph, and do NOT default to a fixed linear analyze → plan → execute → review chain when the task can skip, branch, or revisit a worker.
   - Use status values that correspond to your subtasks
   - **IMPORTANT**: Use null (or "END" or "") for terminal states
   - Example: {{"status": {{"analyzing": {{"worker": "analyzer", "next": "planning"}}, "planning": {{"worker": "planner", "next": {{"cases": [{{"when": {{"field": "needs_research", "equals": true}}, "target": "researching"}}], "default": "drafting"}}}}, "drafting": "writer", "done": null}}}}

**Example 1 - Missing Information (Pattern A - Direct to user_interaction):**
User: "帮我规划旅行"
Analysis: Missing destination, dates, budget → Go directly to user_interaction
Output:
{{
  "should_continue": true,
  "task_flow": [
    {{"subtask": "Request missing travel information from user", "worker": "user_interaction"}}
  ],
  "data_schema": {{
    "type": "object",
    "properties": {{
      "user_prompt": {{"type": "string"}},
      "status": {{"type": "string", "enum": ["requesting_info", "done"]}},
      "result": {{"type": "string"}}
    }},
    "required": ["status"]
  }},
  "workflow_rules": {{
    "status": {{
      "requesting_info": "user_interaction",
      "done": null
    }}
  }}
}}

**Example 2 - Complete Information (Pattern B - Full workflow):**
User: "帮我规划去北京的3天旅行，预算5000元"
Analysis: All info present (destination, duration, budget) → Full workflow
Output:
{{
  "should_continue": true,
  "task_flow": [
    {{"subtask": "Analyze travel requirements", "worker": "analyzer"}},
    {{"subtask": "Create detailed travel itinerary", "worker": "planner"}},
    {{"subtask": "Review plan for completeness", "worker": "reviewer"}},
    {{"subtask": "Present plan to user and request feedback", "worker": "user_interaction"}}
  ],
  "worker_instructions": {{
    "analyzer": "用户想要规划去北京的3天旅行，预算5000元。请分析这些需求，提取关键信息：目的地（北京）、时长（3天）、预算（5000元）。",
    "planner": "基于分析结果，制定详细的北京3天旅行计划。要求：1) 包含每天的具体行程安排 2) 列出主要景点和活动 3) 估算各项费用 4) 确保总预算不超过5000元。",
    "reviewer": "检查旅行计划是否完整且可行。验证：1) 行程安排是否合理 2) 预算分配是否符合5000元限制 3) 是否包含必要的信息（交通、住宿、餐饮）。",
    "user_interaction": "向用户展示完整的北京3天旅行计划，询问是否需要调整行程、预算或其他方面。"
  }},
  "data_schema": {{
    "type": "object",
    "properties": {{
      "user_prompt": {{"type": "string"}},
      "status": {{"type": "string", "enum": ["analyzing", "planning", "reviewing", "user_feedback", "done"]}},
      "analysis": {{"type": "string", "description": "Travel requirements analysis"}},
      "plan": {{"type": "string", "description": "Travel itinerary plan"}},
      "review": {{"type": "string", "description": "Plan quality review"}},
      "result": {{"type": "string", "description": "Final travel plan"}}
    }},
    "required": ["status"]
  }},
  "workflow_rules": {{
    "status": {{
      "analyzing": "analyzer",
      "planning": "planner",
      "reviewing": "reviewer",
      "user_feedback": "user_interaction",
      "done": null
    }}
  }}
}}

**Example 3 - Content Writing (Missing topic - Pattern A):**
User: "写一篇文章"
Analysis: Missing topic, target audience → Go directly to user_interaction
Output:
{{
  "should_continue": true,
  "task_flow": [
    {{"subtask": "Request article topic and requirements from user", "worker": "user_interaction"}}
  ],
  "data_schema": {{
    "type": "object",
    "properties": {{
      "user_prompt": {{"type": "string"}},
      "status": {{"type": "string", "enum": ["requesting_info", "done"]}},
      "result": {{"type": "string"}}
    }},
    "required": ["status"]
  }},
  "workflow_rules": {{
    "status": {{
      "requesting_info": "user_interaction",
      "done": null
    }}
  }}
}}

**Example 4 - Content Writing (Complete info - Pattern B):**
User: "写一篇关于AI技术发展的文章，面向技术人员"
Analysis: Topic and audience clear → Full workflow
Output:
{{
  "should_continue": true,
  "task_flow": [
    {{"subtask": "Analyze topic and determine article structure", "worker": "analyzer"}},
    {{"subtask": "Research AI concepts and gather information", "worker": "researcher"}},
    {{"subtask": "Write article content based on research", "worker": "writer"}},
    {{"subtask": "Review article for quality and completeness", "worker": "reviewer"}},
    {{"subtask": "Present article to user and request feedback", "worker": "user_interaction"}}
  ],
  "data_schema": {{
    "type": "object",
    "properties": {{
      "user_prompt": {{"type": "string"}},
      "status": {{"type": "string", "enum": ["analyzing", "researching", "writing", "reviewing", "user_feedback", "done"]}},
      "analysis": {{"type": "string"}},
      "research": {{"type": "string"}},
      "content": {{"type": "string"}},
      "review": {{"type": "string"}},
      "result": {{"type": "string"}}
    }},
    "required": ["status"]
  }},
  "workflow_rules": {{
    "status": {{
      "analyzing": "analyzer",
      "researching": "researcher",
      "writing": "writer",
      "reviewing": "reviewer",
      "user_feedback": "user_interaction",
      "done": null
    }}
  }}
}}

**Example 5 - User Requests Modification:**
User response: "文章太长了，请缩短到1000字以内"
Analysis: User wants modification → Full workflow to process changes
Output:
{{
  "should_continue": true,
  "task_flow": [
    {{"subtask": "Analyze modification request", "worker": "analyzer"}},
    {{"subtask": "Revise article to meet length requirement", "worker": "writer"}},
    {{"subtask": "Review revised article", "worker": "reviewer"}},
    {{"subtask": "Present revised article and request feedback", "worker": "user_interaction"}}
  ],
  "worker_instructions": {{
    "analyzer": "用户反馈文章太长，要求缩短到1000字以内。请分析当前文章的长度和结构，确定哪些部分可以精简。",
    "writer": "根据分析结果，将文章缩短到1000字以内。要求：1) 保留核心观点和关键信息 2) 删除冗余内容 3) 确保文章仍然连贯完整 4) 严格控制在1000字以内。",
    "reviewer": "检查修改后的文章。验证：1) 字数是否在1000字以内 2) 核心内容是否保留 3) 文章是否仍然连贯 4) 是否满足用户的缩短要求。",
    "user_interaction": "向用户展示缩短后的文章（已控制在1000字以内），询问是否满意或需要进一步调整。"
  }},
  "data_schema": {{
    "type": "object",
    "properties": {{
      "user_prompt": {{"type": "string"}},
      "status": {{"type": "string", "enum": ["analyzing", "writing", "reviewing", "user_feedback", "done"]}},
      "analysis": {{"type": "string"}},
      "content": {{"type": "string"}},
      "review": {{"type": "string"}},
      "result": {{"type": "string"}}
    }},
    "required": ["status"]
  }},
  "workflow_rules": {{
    "status": {{
      "analyzing": "analyzer",
      "writing": "writer",
      "reviewing": "reviewer",
      "user_feedback": "user_interaction",
      "done": null
    }}
  }}
}}

**Example 6 - User Satisfied:**
User response: "满意"
Analysis: User is satisfied → End system
Output:
{{
  "should_continue": false
}}

**Output Format:**
Output a valid JSON object with these keys:

**CRITICAL: Check user_response first**
- If user_response exists AND user indicates satisfaction/no modifications needed:
  {{"should_continue": false}}
- If user_response is EMPTY or NOT present (first time): MUST set should_continue to true
- If user_response contains new information or modification requests: MUST set should_continue to true

If should_continue is false:
{{"should_continue": false}}

If should_continue is true (REQUIRED for first-time requests and modifications):
{{
  "should_continue": true,
  "task_flow": [...],
  "worker_instructions": {{"worker_name": "contextual instruction", ...}},
  "data_schema": {{...}},
  "workflow_rules": {{...}}
}}

Examples of when should_continue should be false:
- User says: "满意", "不需要修改", "可以了", "很好", "没问题", "ok", "done"
- User indicates they are happy with the results
- User has no more requests or changes

Examples of when should_continue should be true:
- **First time user request (no user_response) - ALWAYS true**
- User provides new information (e.g., "我想去北京")
- User requests modifications (e.g., "文章太长了，请缩短")
- User asks questions or needs clarification
"""


class OrchestratorArchitect:
    """
    编排型 Architect - 从固定 Worker 中选择并编排工作流

    动态读取 Worker Registry 来了解可用的 workers。
    支持通过扩展系统自定义 prompt。
    """

    def __init__(self, llm: BaseChatModel, use_extensions: bool = True) -> None:
        self._llm = llm
        self._use_extensions = use_extensions
        self._base_prompt = _build_orchestrator_prompt()

    def _get_prompt(self) -> str:
        """获取最终的 prompt（考虑扩展）"""
        if not self._use_extensions:
            return self._base_prompt

        # 使用扩展系统
        try:
            from langgraph_kernel.extensions import get_architect_extension
            extension = get_architect_extension()
            return extension.get_prompt(self._base_prompt)
        except ImportError:
            # 如果扩展系统不可用，使用默认 prompt
            return self._base_prompt

    def __call__(self, state: KernelState) -> dict[str, Any]:
        user_prompt = (state.get("domain_state") or {}).get("user_prompt", "")
        user_response = state.get("user_response", "")

        # 构建完整的上下文给 LLM
        if user_response:
            context = f"User prompt: {user_prompt}\n\nUser response: {user_response}"
        else:
            context = f"User prompt: {user_prompt}"

        # 获取最终的 prompt（可能被扩展修改）
        final_prompt = self._get_prompt()

        response = self._llm.invoke([
            SystemMessage(content=final_prompt),
            HumanMessage(content=context),
        ])

        # 解析 JSON 响应
        try:
            content = response.content
            # 尝试提取 JSON（处理可能的 markdown 代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # 检查是否应该继续
            should_continue = result.get("should_continue", True)

            if not should_continue:
                # 用户满意，直接结束系统
                print("✅ Architect 判断：用户表示满意，系统直接结束")

                # 保留现有的 domain_state，只更新 status
                current_domain_state = state.get("domain_state", {})
                current_domain_state["status"] = "done"

                return {
                    "task_flow": [],
                    "data_schema": state.get("data_schema", {}),
                    "workflow_rules": state.get("workflow_rules", {}),
                    "selected_workers": [],
                    "domain_state": current_domain_state,
                    "pending_patch": [],
                    "patch_error": "",
                    "waiting_for_user": False,
                }

            # 继续正常流程
            task_flow = result.get("task_flow", [])
            data_schema = result.get("data_schema", {})
            workflow_rules = result.get("workflow_rules", {})
            worker_instructions = result.get("worker_instructions", {})  # 新增：worker 指令

            # 从 task_flow 中提取 selected_workers
            data_schema = sync_workflow_state_enums(data_schema, workflow_rules)
            selected_workers = derive_selected_workers(task_flow, workflow_rules)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            # 如果解析失败，返回空的 schema 和 rules
            print(f"⚠️  Orchestrator 解析失败: {e}")
            print(f"原始响应: {response.content[:200]}")
            task_flow = []
            data_schema = {}
            workflow_rules = {}
            worker_instructions = {}  # 新增
            selected_workers = []

        # 初始化 domain_state，包括 user_prompt 和初始状态
        domain_state = {"user_prompt": user_prompt}

        # 从 workflow_rules 中找到第一个状态字段和初始值
        if workflow_rules:
            for field_name, transitions in workflow_rules.items():
                if transitions:
                    # 使用第一个转换的状态值作为初始状态
                    initial_value = list(transitions.keys())[0]
                    domain_state[field_name] = initial_value
                    break

        return {
            "task_flow": task_flow,  # 新增：任务流
            "data_schema": data_schema,
            "workflow_rules": workflow_rules,
            "worker_instructions": worker_instructions,  # 新增：worker 指令
            "selected_workers": selected_workers,  # 选中的 workers
            "domain_state": domain_state,
            "pending_patch": [],
            "patch_error": "",
            "step_count": 0,
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
            "conversation_history": [],  # 初始化对话历史
            "pending_user_question": "",  # 初始化待回答问题
            "user_response": "",  # 初始化用户响应
            "waiting_for_user": False,  # 初始化等待标志
        }
