# Architect-Kernel-Worker System

基于 LangGraph 构建的三层智能体系统。

## 架构

- **Architect Agent**: 将用户 prompt 转化为 JSON Schema 和 Workflow Rules
- **Kernel**: 验证 JSON Patch、更新状态、调度 worker
- **Worker Agents**: 互相隔离，接收状态切片，提交 JSON Patch

## 安装

```bash
cd libs/kernel_system
pip install -e .
```

## 快速开始

```python
from langchain_openai import ChatOpenAI
from langgraph_kernel import build_kernel_graph
from langgraph_kernel.worker.base import LLMWorkerAgent
from typing_extensions import TypedDict

# 配置模型
llm = ChatOpenAI(
    model="your-model-name",
    api_key="your-api-key",
    base_url="https://your-api-endpoint/v1",
)

# 定义 worker
class PlannerWorker(LLMWorkerAgent):
    class InputSchema(TypedDict):
        domain_state: dict
    input_schema = InputSchema
    system_prompt = "You are a planner. Generate a JSON Patch to add a plan."

# 构建图
graph = build_kernel_graph(llm, workers={"planner": PlannerWorker(llm)})

# 执行
result = graph.invoke({
    "domain_state": {"user_prompt": "帮我规划一个旅行"},
    "data_schema": {},
    "workflow_rules": {},
    "pending_patch": [],
    "patch_error": "",
    "step_count": 0,
})

print(result["domain_state"])
```

## 运行测试

```bash
cd libs/kernel_system
pytest tests/
```

## 核心特性

- **JSON Schema 约束**: Architect 自动生成 schema，Kernel 强制验证
- **状态驱动路由**: 基于 domain_state 值变化触发对应 worker
- **最小化上下文**: 每个 worker 只看到 input_schema 声明的字段
- **第三方模型支持**: 兼容任意 OpenAI 格式 API
