# 扩展系统文档

kernel_system 支持通过扩展系统自定义 Worker 和 Architect prompt。

## 功能概述

1. **自定义 Worker 注册** - 添加新的 Worker 类型到系统
2. **Architect Prompt 扩展** - 修改或替换 Architect 的 system prompt
3. **动态加载** - 从 Python 模块或文件加载扩展

## 快速开始

### 1. 注册自定义 Worker

#### 方式 A: 使用便捷函数

```python
from langgraph_kernel.extensions import register_custom_worker

# 使用 system_prompt 注册
register_custom_worker(
    name="translator",
    system_prompt="""You are a Translator Worker.
    Translate text between languages while maintaining meaning and tone.
    Output JSON Patch: [{"op": "add", "path": "/translation", "value": "..."}]
    """
)
```

#### 方式 B: 使用 Worker 类

```python
from langgraph_kernel.worker.base import LLMWorkerAgent
from langgraph_kernel.extensions import register_custom_worker

class TranslatorWorker(LLMWorkerAgent):
    system_prompt = """You are a Translator Worker..."""

register_custom_worker("translator", worker_class=TranslatorWorker)
```

#### 方式 C: 直接使用 Registry

```python
from langgraph_kernel.worker.registry import get_registry

registry = get_registry()
registry.register("translator", TranslatorWorker)
```

### 2. 扩展 Architect Prompt

#### 方式 A: 添加新的部分

```python
from langgraph_kernel.extensions import add_architect_prompt_section

add_architect_prompt_section(
    """
    **Translation Guidelines:**
    - Use 'translator' worker for language translation
    - Specify source and target languages
    """,
    position="end"  # 或 "start"
)
```

#### 方式 B: 完全替换 Prompt

```python
from langgraph_kernel.extensions import set_custom_architect_prompt

custom_prompt = """
Your completely custom architect prompt here...
"""

set_custom_architect_prompt(custom_prompt)
```

#### 方式 C: 使用 Prompt Modifier

```python
from langgraph_kernel.extensions import get_architect_extension

extension = get_architect_extension()

def my_modifier(original_prompt: str) -> str:
    # 在原有 prompt 基础上修改
    return original_prompt.replace("old_text", "new_text")

extension.add_prompt_modifier(my_modifier)
```

### 3. 创建扩展模块

创建一个 Python 文件（如 `my_extensions.py`）：

```python
"""我的自定义扩展"""

from langgraph_kernel.worker.base import LLMWorkerAgent
from langgraph_kernel.worker.registry import WorkerRegistry
from langgraph_kernel.extensions import ArchitectPromptExtension


# 定义自定义 Worker
class MyCustomWorker(LLMWorkerAgent):
    system_prompt = """You are a custom worker..."""


# Worker 注册函数（必须命名为 register_workers）
def register_workers(registry: WorkerRegistry) -> None:
    registry.register("my_custom", MyCustomWorker)
    print("✅ 已注册自定义 worker: my_custom")


# Architect 配置函数（必须命名为 configure_architect）
def configure_architect(extension: ArchitectPromptExtension) -> None:
    additional_info = """
    **Custom Guidelines:**
    - Use 'my_custom' worker for special tasks
    """
    extension.add_prompt_modifier(lambda p: f"{p}\n\n{additional_info}")
    print("✅ 已配置 Architect prompt")
```

### 4. 加载扩展

```python
from langgraph_kernel.extensions import load_extensions

# 从模块加载
load_extensions("my_extensions")

# 或从文件加载
load_extensions("/path/to/my_extensions.py")
```

## 完整示例

### 示例 1: 添加翻译功能

```python
from langgraph_kernel.extensions import (
    register_custom_worker,
    add_architect_prompt_section,
)

# 1. 注册翻译 Worker
register_custom_worker(
    "translator",
    system_prompt="""You are a Translator Worker.
    Translate text between languages.
    Output: [{"op": "add", "path": "/translation", "value": "translated text"}]
    """
)

# 2. 更新 Architect prompt
add_architect_prompt_section(
    """
    **Translation Tasks:**
    When user requests translation:
    - Use 'translator' worker
    - Workflow: analyzing → translating → reviewing → user_feedback → done
    """,
    position="end"
)

# 3. 使用系统
from langgraph_kernel.graph import build_kernel_graph

graph = build_kernel_graph(llm)
result = graph.invoke({
    "domain_state": {"user_prompt": "请将这段文字翻译成英文：你好世界"},
    # ... 其他字段
})
```

### 示例 2: 创建扩展包

项目结构：
```
my_project/
├── extensions/
│   ├── __init__.py
│   ├── workers.py      # 自定义 Worker 定义
│   └── prompts.py      # Prompt 配置
└── main.py             # 主程序
```

`extensions/workers.py`:
```python
from langgraph_kernel.worker.base import LLMWorkerAgent

class DataAnalyzerWorker(LLMWorkerAgent):
    system_prompt = """Analyze data and extract insights..."""

class VisualizerWorker(LLMWorkerAgent):
    system_prompt = """Create data visualizations..."""
```

`extensions/__init__.py`:
```python
from langgraph_kernel.worker.registry import WorkerRegistry
from langgraph_kernel.extensions import ArchitectPromptExtension
from .workers import DataAnalyzerWorker, VisualizerWorker

def register_workers(registry: WorkerRegistry) -> None:
    registry.register("data_analyzer", DataAnalyzerWorker)
    registry.register("visualizer", VisualizerWorker)

def configure_architect(extension: ArchitectPromptExtension) -> None:
    guidance = """
    **Data Analysis Tasks:**
    - Use 'data_analyzer' for data analysis
    - Use 'visualizer' for creating charts
    """
    extension.add_prompt_modifier(lambda p: f"{p}\n\n{guidance}")
```

`main.py`:
```python
from langgraph_kernel.extensions import load_extensions
from langgraph_kernel.graph import build_kernel_graph

# 加载扩展
load_extensions("extensions")

# 构建图
graph = build_kernel_graph(llm)

# 使用
result = graph.invoke({
    "domain_state": {"user_prompt": "分析这些销售数据"},
    # ...
})
```

## API 参考

### Worker 注册

#### `register_custom_worker(name, worker_class=None, system_prompt=None)`
注册自定义 Worker。

**参数：**
- `name` (str): Worker 名称
- `worker_class` (type[LLMWorkerAgent], 可选): Worker 类
- `system_prompt` (str, 可选): System prompt（当 worker_class 为 None 时使用）

**示例：**
```python
register_custom_worker("my_worker", system_prompt="You are...")
```

#### `WorkerRegistry.register(name, worker_class)`
直接注册到 registry。

**参数：**
- `name` (str): Worker 名称
- `worker_class` (type[LLMWorkerAgent]): Worker 类

### Architect Prompt 扩展

#### `set_custom_architect_prompt(prompt)`
完全替换 Architect prompt。

**参数：**
- `prompt` (str): 自定义 prompt

#### `add_architect_prompt_section(section, position='end')`
添加新的 prompt 部分。

**参数：**
- `section` (str): 要添加的文本
- `position` (str): 位置（'start' 或 'end'）

#### `ArchitectPromptExtension.add_prompt_modifier(modifier)`
添加 prompt 修改器。

**参数：**
- `modifier` (Callable[[str], str]): 修改函数

### 扩展加载

#### `load_extensions(module_or_file)`
加载扩展（自动判断模块或文件）。

**参数：**
- `module_or_file` (str): 模块名或文件路径

#### `ExtensionLoader.load_from_module(module_name)`
从 Python 模块加载。

#### `ExtensionLoader.load_from_file(file_path)`
从 Python 文件加载。

## 最佳实践

1. **Worker 命名** - 使用描述性的名称（如 `translator`, `code_generator`）
2. **System Prompt** - 清晰说明 Worker 的职责和输出格式
3. **Prompt 扩展** - 使用 modifier 而不是完全替换（保持兼容性）
4. **模块化** - 将相关的 Worker 组织到同一个扩展模块
5. **文档** - 为自定义 Worker 添加 docstring 和注释

## 注意事项

1. Worker 名称必须唯一，重复注册会覆盖之前的定义
2. Architect prompt 修改会影响所有任务的行为
3. 扩展加载顺序可能影响最终结果
4. 自定义 Worker 必须继承 `LLMWorkerAgent` 或 `BaseWorkerAgent`
5. System prompt 必须包含 JSON Patch 输出格式说明

## 故障排除

### Worker 未被识别
- 检查是否正确注册到 registry
- 确认 Worker 名称拼写正确
- 验证扩展是否成功加载

### Prompt 修改未生效
- 确认在构建图之前加载扩展
- 检查 `OrchestratorArchitect` 是否启用了扩展（`use_extensions=True`）
- 验证 modifier 函数是否正确返回修改后的 prompt

### 扩展加载失败
- 检查模块路径是否正确
- 确认文件存在且可访问
- 查看错误信息中的具体原因
