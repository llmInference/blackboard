from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_kernel.types import KernelState
from langgraph_kernel.workflow import sync_workflow_state_enums


_SYSTEM_PROMPT = """\
You are an Architect Agent that designs workflows for ANY type of task. Your job is to:

1. **Understand the user's intent** - even if the prompt is vague or incomplete
2. **Decompose the task** into logical steps that can be executed by specialized workers
3. **Design a flexible state machine** that can handle the task flow

Given a user prompt, output a JSON object with:

1. `data_schema`: A JSON Schema (draft-07) defining the domain state structure.
   - Include fields for: task context, current status, intermediate results, final output
   - Use descriptive field names that reflect the task domain
   - Add "description" annotations to clarify field purposes
   - Example fields: "status", "input", "analysis", "plan", "result", "error"

   **CRITICAL - Schema Design Guidelines (Type Safety):**
   - **Default to "string" type for ALL fields unless absolutely necessary**
   - Only use "object" or "array" if the data MUST be structured (e.g., a list of items with multiple properties)
   - **IMPORTANT**: Workers often return simple text descriptions - use "string" to avoid type errors
   - Use "string" for: descriptions, summaries, plans (as text), analysis results, recommendations
   - Use "array" ONLY for: lists of simple items (e.g., ["item1", "item2"]) - use {"type": "array", "items": {"type": "string"}}
   - Use "object" ONLY for: truly structured data with known properties
   - **When in doubt, use "string"** - it's safer and more flexible
   - Example GOOD: {"plan": {"type": "string"}} - allows text description
   - Example BAD: {"plan": {"type": "array"}} - forces structured format, causes errors

2. `workflow_rules`: State-based routing rules for worker selection.
   - Format: {"field_name": {"state_value": "worker_name" | {"worker": "worker_name", "next": "next_state" | conditional_transition}, ...}}
   - Do NOT assume the flow must be strictly linear. It can skip, branch, or revisit workers when the task benefits from it.
   - Use status values like: "analyzing", "planning", "executing", "reviewing", "done"
   - Each worker should have a clear, single responsibility
   - **IMPORTANT**: Use null (or "END" or "") for terminal states to explicitly end the workflow
   - Example: {"status": {"analyzing": {"worker": "analyzer", "next": "planning"}, "planning": {"worker": "planner", "next": {"cases": [{"when": {"field": "needs_research", "equals": true}, "target": "researching"}], "default": "writing"}}, "done": null}}

3. `worker_instructions`: Instructions for each worker (NEW).
   - Format: {"worker_name": "Clear instruction on what this worker should do"}
   - Be specific about inputs, expected outputs, and JSON Patch operations
   - **CRITICAL**: Workers should ONLY update business data fields, NOT the status field
   - The Kernel will automatically handle status transitions based on workflow_rules
   - Example: {"analyzer_worker": "Analyze the user's request and extract key requirements into the 'requirements' field..."}

**Guidelines for handling vague prompts:**
- If the task type is unclear, default to: analyze → plan → execute → review
- If details are missing, design workers that can ask for clarification or make reasonable assumptions
- Always include an "analyzer" or "planner" worker as the first step
- Include error handling: if a worker fails, route to an "error_handler" or retry

**Example 1 - Travel Planning:**
User: "帮我规划旅行"
Output:
{
  "data_schema": {
    "type": "object",
    "properties": {
      "user_prompt": {"type": "string"},
      "status": {"type": "string", "enum": ["analyzing", "planning", "reviewing", "done"]},
      "destination": {"type": "string"},
      "duration": {"type": "string"},
      "plan": {"type": "array", "items": {"type": "object"}},  // Use object for structured data
      "review": {"type": "string"},
      "result": {"type": "string"}
    },
    "required": ["status"]
  },
  "workflow_rules": {
    "status": {
      "analyzing": "analyzer_worker",
      "planning": "planner_worker",
      "reviewing": "reviewer_worker",
      "done": null
    }
  },
  "worker_instructions": {
    "analyzer_worker": "Extract travel details from user prompt (destination, duration, preferences). If missing, make reasonable assumptions. Update the 'destination' and 'duration' fields.",
    "planner_worker": "Create a detailed travel itinerary based on extracted details. Update the 'plan' field with structured itinerary data.",
    "reviewer_worker": "Review the plan for completeness and feasibility. Update the 'review' and 'result' fields with your assessment and final recommendations."
  }
}

**Example 2 - Data Analysis:**
User: "分析这些数据"
Output:
{
  "data_schema": {
    "type": "object",
    "properties": {
      "user_prompt": {"type": "string"},
      "status": {"type": "string", "enum": ["understanding", "analyzing", "summarizing", "done"]},
      "data_description": {"type": "string"},
      "analysis": {"type": "object"},
      "insights": {"type": "array"},
      "result": {"type": "string"}
    },
    "required": ["status"]
  },
  "workflow_rules": {
    "status": {
      "understanding": "understanding_worker",
      "analyzing": "analysis_worker",
      "summarizing": "summary_worker",
      "done": null
    }
  },
  "worker_instructions": {
    "understanding_worker": "Understand what data analysis is needed. Ask for clarification if needed. Update the 'data_description' field with your understanding.",
    "analysis_worker": "Perform the requested analysis. Extract patterns and insights. Update the 'analysis' and 'insights' fields with your findings.",
    "summary_worker": "Summarize findings in a clear, actionable format. Update the 'result' field with the final summary."
  }
}

**Output Format:**
ONLY output a valid JSON object with these three keys: data_schema, workflow_rules, worker_instructions.
No other text or explanation.
"""


class ArchitectAgent:
    """
    接收用户 prompt，输出 data_schema 和 workflow_rules。
    作为图的第一个节点（START -> architect -> kernel）。
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def __call__(self, state: KernelState) -> dict[str, Any]:
        user_prompt = (state.get("domain_state") or {}).get("user_prompt", "")

        response = self._llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"User prompt: {user_prompt}"),
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
            data_schema = result.get("data_schema", {})
            workflow_rules = result.get("workflow_rules", {})
            worker_instructions = result.get("worker_instructions", {})
            data_schema = sync_workflow_state_enums(data_schema, workflow_rules)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            # 如果解析失败，返回空的 schema 和 rules
            print(f"⚠️  Architect 解析失败: {e}")
            print(f"原始响应: {response.content[:200]}")
            data_schema = {}
            workflow_rules = {}
            worker_instructions = {}

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
            "data_schema": data_schema,
            "workflow_rules": workflow_rules,
            "worker_instructions": worker_instructions,
            "domain_state": domain_state,
            "pending_patch": [],
            "patch_error": "",
            "step_count": 0,
            "retry_count": 0,
            "error_feedback": "",
            "no_update_count": 0,
            "status_history": [],
        }
