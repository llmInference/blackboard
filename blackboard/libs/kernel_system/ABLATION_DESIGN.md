# 消融实验改造方案

## 概述

本文档描述如何将 `kernel_system` 改造为支持快速拆装五个核心组件的消融实验框架。
通过一个 `AblationConfig` 配置类，可以独立开关每个组件，无需修改主流程代码。

---

## 配置类设计

```python
# langgraph_kernel/ablation.py
from dataclasses import dataclass, field

@dataclass
class AblationConfig:
    use_json_patch: bool = True        # C1: JSON Patch 通信
    use_schema_validation: bool = True  # C2: Schema 约束校验
    use_circuit_breaker: bool = True    # C3: 确定性内核
    use_context_slicing: bool = True    # C4: Context Slicing
    use_architect_agent: bool = True    # C5: Architect Agent

    # 兜底保护（C2/C3 关闭时仍然生效）
    hard_max_steps: int = 50

    @classmethod
    def full(cls) -> "AblationConfig":
        """实验组：所有组件开启"""
        return cls()

    @classmethod
    def ablate(cls, *components: str) -> "AblationConfig":
        """关闭指定组件，其余保持开启

        用法：AblationConfig.ablate("C1", "C3")
        """
        cfg = cls()
        mapping = {
            "C1": "use_json_patch",
            "C2": "use_schema_validation",
            "C3": "use_circuit_breaker",
            "C4": "use_context_slicing",
            "C5": "use_architect_agent",
        }
        for c in components:
            if c in mapping:
                setattr(cfg, mapping[c], False)
        return cfg
```

**传递路径：**

```
build_kernel_graph(llm, ablation=AblationConfig(...))
    ↓
KernelState 增加 ablation_config: Any 字段
    ↓
各节点读取 state.get("ablation_config") 决定行为
```

---

## C1：JSON Patch 通信

**核心贡献**：结构化、低熵的通信机制，替代自然语言。

### 实验组（开启）

Worker 返回 RFC 6902 JSON Patch 数组，Kernel apply 到 `domain_state`。

### 对照组（关闭）

Worker 退化为自然语言输出，Kernel 将回复追加到 `domain_state["messages"]`。

### 修改点

| 文件 | 修改内容 |
|------|---------|
| `worker/base.py` | `_think()` 根据配置切换 prompt 模板，返回 `str` 而非 `list[dict]` |
| `kernel/node.py` | patch 应用逻辑加分支：关闭时直接存消息列表 |

```python
# worker/base.py - __call__() 关闭时
if not ablation.use_json_patch:
    # prompt 改为：请用自然语言描述你的分析结果
    # 返回 str，kernel 侧存入 messages
    ...

# kernel/node.py - 关闭时
if not ablation.use_json_patch:
    messages = domain_state.get("messages", [])
    messages.append({"worker": current_worker, "content": pending_patch})
    domain_state["messages"] = messages
```

---

## C2：Schema 约束校验

**核心贡献**：运行时类型检查和结构验证，拦截无效 Patch。

### 实验组（开启）

5 层 PatchFixer 自动修复 → PatchValidator schema 校验 → 失败则重试并反馈错误。

### 对照组（关闭）

跳过所有校验，直接 `jsonpatch.apply_patch()`，不管格式和类型是否正确。
遇到错误记录日志后继续执行，不重试。

### 保护机制

C2 关闭时，`hard_max_steps` 硬限制**独立生效**，防止坏 patch 导致卡死。
每个 test case 独立初始化 `KernelState`，错误不跨 case 传播。

### 修改点

| 文件 | 修改内容 |
|------|---------|
| `kernel/node.py` | patch 处理分支：关闭时跳过 PatchFixer/PatchValidator，直接 apply |
| `kernel/node.py` | 重试逻辑：关闭时 `max_retries=0` |

```python
# kernel/node.py
if not ablation.use_schema_validation:
    try:
        new_state = jsonpatch.apply_patch(domain_state, filtered_patch)
        domain_state = new_state
    except Exception as e:
        error_log.append(str(e))  # 记录但不重试，继续执行
else:
    # 原有 PatchFixer + PatchValidator 流程
    new_state, error, logs = _fixer.fix_and_validate(...)
```

---

## C3：确定性内核（熔断器）

**核心贡献**：检测超时、无效 Patch、错误累积、振荡、停滞五种异常并终止执行。

### 实验组（开启）

`CircuitBreaker.check()` 每步检查五种熔断条件，任一触发则终止。

### 对照组（关闭）

跳过熔断器所有智能检测，只保留 `hard_max_steps` 硬性步数上限。

### 修改点

| 文件 | 修改内容 |
|------|---------|
| `kernel/node.py` | 第 0 步加分支：关闭时只检查 `step_count >= hard_max_steps` |

```python
# kernel/node.py - 第 0 步
ablation = state.get("ablation_config")
if ablation and not ablation.use_circuit_breaker:
    if state.get("step_count", 0) >= ablation.hard_max_steps:
        return {"circuit_breaker_triggered": True, "circuit_breaker_reason": "hard_limit"}
else:
    circuit_breaker = state.get("circuit_breaker")
    if circuit_breaker:
        should_break, reason, details = circuit_breaker.check(state)
        ...
```

**注意**：`hard_max_steps` 默认值与 `CircuitBreakerConfig.max_steps`（默认 50）保持一致，
确保实验组和对照组在步数上限上可比。

---

## C4：Context Slicing

**核心贡献**：动态上下文过滤，Worker 只接收 `input_schema` 声明的状态切片。

### 实验组（开启）

两层过滤：
1. LangGraph 层：`input_schema=WorkerInputSchema` 限制传入 worker 节点的字段
2. Worker 层：`__call__()` 只展开 `domain_state` 业务字段 + `error_feedback` + `_data_schema`

### 对照组（关闭）

Worker 接收完整 `domain_state` + 所有历史信息，不做任何过滤。

### 修改点

| 文件 | 修改内容 |
|------|---------|
| `worker/base.py` | `__call__()` 加分支：关闭时传完整状态 |
| `graph.py` | `add_node("worker", ...)` 加分支：关闭时不传 `input_schema` |

```python
# worker/base.py - __call__() 关闭时
if not ablation.use_context_slicing:
    context = {
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

# graph.py
if ablation.use_context_slicing:
    builder.add_node("worker", worker_node, input_schema=WorkerInputSchema)
else:
    builder.add_node("worker", worker_node)
```

---

## C5：Architect Agent

**核心贡献**：LLM 动态设计 `data_schema` 和 `workflow_rules`，任务自适应路由。

### 实验组（开启）

`OrchestratorArchitect` 调用 LLM，动态生成 schema、workflow_rules、worker_instructions。

### 对照组（关闭）：固定流水线

不调用 LLM，硬编码通用流水线 `analyze → plan → execute → review → done`。
`worker_instructions` 为空，worker 只靠原始 `user_prompt` 工作，无任务专属指导。

### 修改点

| 文件 | 修改内容 |
|------|---------|
| `architect/fixed_pipeline.py` | 新增文件，实现 `FixedPipelineArchitect` |
| `graph.py` | `architect` 节点根据配置切换实现 |

```python
# architect/fixed_pipeline.py
class FixedPipelineArchitect:
    FIXED_WORKFLOW = {
        "status": {
            "analyzing": "analyzer",
            "planning":  "planner",
            "executing": "executor",
            "reviewing": "reviewer",
            "done":      None,
        }
    }
    FIXED_SCHEMA = {
        "type": "object",
        "properties": {
            "status":      {"type": "string"},
            "user_prompt": {"type": "string"},
            "result":      {"type": "string"},
            "plan":        {"type": "array", "items": {"type": "string"}},
            "analysis":    {"type": "string"},
            "review":      {"type": "string"},
        },
        "required": ["status", "user_prompt"],
    }

    def __call__(self, state: KernelState) -> dict:
        user_prompt = (state.get("domain_state") or {}).get("user_prompt", "")
        return {
            "task_flow": [
                {"subtask": "analyze", "worker": "analyzer"},
                {"subtask": "plan",    "worker": "planner"},
                {"subtask": "execute", "worker": "executor"},
                {"subtask": "review",  "worker": "reviewer"},
            ],
            "data_schema":         self.FIXED_SCHEMA,
            "workflow_rules":      self.FIXED_WORKFLOW,
            "worker_instructions": {},  # 无任务专属指令
            "selected_workers":    ["analyzer", "planner", "executor", "reviewer"],
            "domain_state":        {"user_prompt": user_prompt, "status": "analyzing"},
            "pending_patch": [], "patch_error": "", "step_count": 0,
            "retry_count": 0, "error_feedback": "", "no_update_count": 0,
            "status_history": [], "conversation_history": [],
            "pending_user_question": "", "user_response": "", "waiting_for_user": False,
        }

# graph.py
architect = (
    OrchestratorArchitect(llm)
    if ablation.use_architect_agent
    else FixedPipelineArchitect()
)
builder.add_node("architect", architect)
```

---

## 使用示例

```python
from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.graph import build_kernel_graph

# 实验组：全部开启
graph = build_kernel_graph(llm, ablation=AblationConfig.full())

# 关闭单个组件
graph = build_kernel_graph(llm, ablation=AblationConfig.ablate("C2"))

# 关闭多个组件
graph = build_kernel_graph(llm, ablation=AblationConfig.ablate("C1", "C5"))

# 手动配置
graph = build_kernel_graph(llm, ablation=AblationConfig(
    use_json_patch=True,
    use_schema_validation=False,
    use_circuit_breaker=False,
    use_context_slicing=True,
    use_architect_agent=True,
))
```

---

## 消融矩阵

| 实验 ID | C1 JSON Patch | C2 Schema 校验 | C3 熔断器 | C4 Context Slicing | C5 Architect |
|--------|:---:|:---:|:---:|:---:|:---:|
| Full   | ✅ | ✅ | ✅ | ✅ | ✅ |
| -C1    | ❌ | ✅ | ✅ | ✅ | ✅ |
| -C2    | ✅ | ❌ | ✅ | ✅ | ✅ |
| -C3    | ✅ | ✅ | ❌ | ✅ | ✅ |
| -C4    | ✅ | ✅ | ✅ | ❌ | ✅ |
| -C5    | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## 文件改动汇总

| 文件 | 操作 | 说明 |
|------|------|------|
| `langgraph_kernel/ablation.py` | 新增 | `AblationConfig` 数据类 |
| `langgraph_kernel/architect/fixed_pipeline.py` | 新增 | C5 对照组：固定流水线 |
| `langgraph_kernel/types.py` | 修改 | `KernelState` 增加 `ablation_config: Any` 字段 |
| `langgraph_kernel/graph.py` | 修改 | 接收 `ablation` 参数，切换 architect 节点和 worker input_schema |
| `langgraph_kernel/kernel/node.py` | 修改 | C2 校验分支、C3 熔断分支 |
| `langgraph_kernel/worker/base.py` | 修改 | C1 通信分支、C4 context 分支 |
