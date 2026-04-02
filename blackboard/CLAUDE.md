# AGENTS Instructions

This repository is a monorepo. Each library lives in a subdirectory under `libs/`.

When you modify code in any library, run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters
- `make lint` – run the linter
- `make test` – execute the test suite

To run a particular test file or to pass additional pytest options you can specify the `TEST` variable:

```
TEST=path/to/test.py make test
```

Other pytest arguments can also be supplied inside the `TEST` variable.

## Libraries

The repository contains several Python and JavaScript/TypeScript libraries.
Below is a high-level overview:

- **checkpoint** – base interfaces for LangGraph checkpointers.
- **checkpoint-postgres** – Postgres implementation of the checkpoint saver.
- **checkpoint-sqlite** – SQLite implementation of the checkpoint saver.
- **cli** – official command-line interface for LangGraph.
- **langgraph** – core framework for building stateful, multi-actor agents.
- **prebuilt** – high-level APIs for creating and running agents and tools.
- **sdk-js** – JS/TS SDK for interacting with the LangGraph REST API.
- **sdk-py** – Python SDK for the LangGraph Server API.
- **kernel_system** – 基于 LangGraph 的三层智能体系统（Architect-Kernel-Worker），包含以下核心模块：
  - `langgraph_kernel/architect/` – 将用户 prompt 转化为 JSON Schema (`data_schema`) 和 Workflow Rules (`workflow_rules`)
  - `langgraph_kernel/kernel/` – 验证 worker 提交的 RFC 6902 JSON Patch、更新 `domain_state`、按规则路由到下一个 worker
  - `langgraph_kernel/worker/` – 互相隔离的 worker，只接收 `input_schema` 声明的状态切片，提交 JSON Patch
  - `langgraph_kernel/circuit_breaker.py` – 熔断器，检测超时/无效 Patch/错误累积/振荡/停滞并终止执行
  - `langgraph_kernel/state_tree.py` – 全局状态树，记录完整执行历史和中间结果，支持分支和序列化
  - `langgraph_kernel/types.py` – 核心类型：`KernelState`（含多轮对话、熔断器、状态树字段）
  - 安装：`cd libs/kernel_system && pip install -e .`；测试：`pytest tests/`

### Dependency map

The diagram below lists downstream libraries for each production dependency as
declared in that library's `pyproject.toml` (or `package.json`).

```text
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (standalone)
```

Changes to a library may impact all of its dependents shown above.

- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.
