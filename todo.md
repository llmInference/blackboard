# WebArena 集成 TODO

## 总目标

- 将 WebArena 作为 Blackboard 的下一个重点 benchmark。
- 保持实现方式与现有 neutral protocol、blackboard kernel 架构一致。
- 用 WebArena 验证 Blackboard 的“真实多智能体”结构价值，而不是退化成单智能体浏览器代理：
  - 去掉 `architect` 后，性能应明显下降
  - 固定 worker 顺序后，性能应明显下降
  - worker 角色应是能力单元，而不是一个“大一统 browser agent”

## 当前仓库现状

- 仓库里已经存在 WebArena 任务清单：
  - `experiment/common/task_sets/webarena_debug.json`
  - `experiment/common/task_sets/webarena_live_debug.json`
  - `experiment/common/task_sets/webarena_formal.json`
  - `experiment/common/task_sets/webarena_script_browser_smoke.json`
- 这些 manifest 已经改为指向本地 `webarena-verified/assets/dataset/webarena-verified.json`。
- 仓库里已经有若干地方“假定” WebArena 集成存在，但对应模块其实还没实现：
  - `experiment/common/task_registry.py` 依赖 `experiment.webarena.utils.task_loader.load_task_config`
  - `experiment/common/cross_dataset_analysis.py` 依赖 `experiment.webarena.utils.result_analysis`
  - `experiment/run/run_full_pipeline.sh` 依赖 `experiment/run/run_webarena_experiments.sh`
- 结论：
  - WebArena 目前已经完成数据入口与本地 smoke 环境验证
  - 但 Blackboard 主链路、runtime wrapper 和 workers 仍未实现

## 设计约束

- `webarena` 与 Blackboard 逻辑解耦。
- 只有 WebArena adapter / runtime 层可以直接 import WebArena 专有环境类。
- `blackboard`、`langgraph`、`autogen` 必须继续共用同一套 neutral protocol。
- 优先实现并验证 `blackboard`；在 Blackboard 端到端跑通前，不做其他系统扩展。
- 第一版优先暴露“结构化浏览器动作工具”，不要一开始走通用代码执行。
- 不允许把 WebArena 集成做成“单个 browser worker + 一个装饰性 architect”。
- Blackboard 的职责边界必须保持清晰：
  - `architect` 是 LLM worker，负责按任务生成 `workflow` 和 `data_schema`
  - `kernel` 只负责执行 workflow、合并 patch、做 circuit breaking
  - worker 只负责局部能力，并且只能在自己的职责范围内调用工具
  - 原始任务只传给 `architect`，不直接传给各个 worker
  - Blackboard 的全局状态负责保留历史信息，并按当前 step 动态过滤上下文；不单独设置 `memory_worker`

## 面向 Blackboard 的结构约束

- `workflow` 必须是 task-scoped，并且由 architect 按具体 WebArena 任务生成。
- `data_schema` 必须是 task-scoped，并且在 worker 执行前由 architect 生成。
- worker 之间只能通过 shared blackboard state 协作，不能依赖隐式点对点传递。
- WebArena 上的 worker 编排必须支持：
  - 同一 worker 被重复调用
  - 某些 worker 在部分任务中被跳过
  - 浏览器动作失败后分支与回退
  - 拿到新 DOM / 页面证据后重新解释任务
- 用以下回归检查防止系统退化成“伪多智能体”：
  - 如果去掉 `architect` 后系统几乎不受影响，说明 workflow 太固定
  - 如果一个 worker 同时负责读页面、选动作、执行动作、判断完成，说明拆分太弱
  - 如果搜索、表单填写、多站点跳转等任务的 `data_schema` 基本相同，说明任务分解太弱

## 目标 worker 集合

- 第一版 WebArena worker 集合保持“按能力拆分”：
  - `page_state_worker`
  - `argument_grounding_worker`
  - `browser_action_worker`
  - `verification_worker`
  - `response_worker`
- 各 worker 角色：
  - `page_state_worker`：整理当前 URL、页面文本、DOM 摘要、tab 状态
  - `argument_grounding_worker`：根据当前 workflow step 绑定 element id、输入文本、URL、tab 目标
  - `browser_action_worker`：通过环境实际执行一个浏览器动作
  - `verification_worker`：判断当前页面状态是否意味着任务推进或完成
  - `response_worker`：仅在任务要求最终文本答案时输出结果
- 第一版 worker 明细表：

| worker | 是否 LLM | 输入 | 输出 | 可写 blackboard 字段 | 是否调工具 |
| --- | --- | --- | --- | --- | --- |
| `page_state_worker` | 是 | 当前 workflow step、当前页面 observation、最近一次动作结果、已过滤的全局 blackboard 状态 | 标准化页面语义摘要、候选可交互元素摘要、与任务相关的页面证据 | `shared_data.current_page`、`shared_data.page_evidence`、`shared_data.open_tabs` | 否 |
| `argument_grounding_worker` | 否 | 当前 workflow step、`shared_data.current_page`、最近一次环境 observation、已有元素定位信息 | 可执行动作参数，如 element id、输入文本、目标 URL、tab 目标 | `shared_data.grounded_action`、`shared_data.action_arguments` | 否 |
| `browser_action_worker` | 否 | 当前 workflow step、`shared_data.grounded_action` | 动作执行结果、执行后 observation、动作级错误信息 | `shared_data.last_action`、`shared_data.last_observation`、`shared_data.action_history`、`shared_data.execution_error` | 是 |
| `verification_worker` | 是 | 当前 workflow step、`shared_data.current_page`、`shared_data.page_evidence`、`shared_data.last_observation`、`shared_data.action_history`、已过滤的全局 blackboard 状态 | 任务是否推进、是否完成、是否偏航、是否需要回退或继续 | `shared_data.verification`、`shared_data.finish`、`shared_data.finish_reason` | 否 |
| `response_worker` | 是 | 当前 workflow step、`shared_data.verification`、`shared_data.page_evidence`、`shared_data.current_page`、已过滤的全局 blackboard 状态 | 最终文本回答 | `shared_data.assistant_response` | 否 |

- 第一版实现约束：
  - `page_state_worker`、`verification_worker`、`response_worker` 是 LLM worker
  - `argument_grounding_worker`、`browser_action_worker` 不是 LLM worker
  - 只有 `browser_action_worker` 可以直接调用浏览器工具
  - `argument_grounding_worker` 只做参数绑定，不做动作选择
  - `verification_worker` 只做状态判断，不负责重新规划 worker 编排
- 第一版明确避免固定流水线，例如：
  - `read -> act -> verify -> respond`
- architect 必须能按任务决定是否需要：
  - 多轮浏览循环
  - 检索型行为
  - 最终文本回答
  - 无最终文本直接结束
- architect 产出的 `workflow.next` 必须包含显式路由条件，kernel 按条件路由执行，不再硬编码固定 worker 跳转

## Phase 0：环境与最小基线

- [x] 补充或记录本地 `webarena-verified` checkout 的获取方式。
- [x] 记录 WebArena 所需运行环境：
  - [x] Python 环境：统一使用 `conda` 环境 `exp`
  - [x] 浏览器依赖：`playwright` / `chromium`
  - [x] Playwright 或等价浏览器自动化依赖
  - [x] WebArena 特有的站点启动步骤：`webarena-verified env start --site gitlab --timeout 300`
- [x] 验证现有 WebArena task manifest 中的路径能在本地解析成功。
- [x] 决定第一版执行后端：
  - [x] 官方 browser env
  - [x] script/browser smoke backend 作为后续选项保留
  - [x] 对 reference browser runner 的本地包装暂缓到后续 Phase
- [x] 选择第一个最小 smoke slice：
  - [x] 官方 GitLab demo task `44`
  - [x] `webarena_script_browser_smoke.json` 保留给后续 task loader 接入后使用
- [x] 记录一条可重复执行的 smoke 命令。

Phase 0 说明：

- 已完成的 Phase 0 结论：
  - `webarena-verified` 已安装到 `exp`
  - CLI、`agent-input-get`、Playwright/Chromium 均可用
  - 本地 manifest 已对齐到 `webarena-verified` 数据集
  - GitLab 默认端口 `8023` 的 live smoke 已验证可用
  - 官方 `ui_login` helper 在本地模板配置下可成功生成 storage state
- 关键坑位：
  - 不要使用 `webarena-verified env start --site gitlab --port 8012`
  - 对当前环境，应使用 GitLab 默认端口 `8023`
  - 本地模板里的 GitLab 凭据应为 `byteblaze / hello1234`
- 当前推荐 smoke 命令：
  - `conda run -n exp webarena-verified env start --site gitlab --timeout 300`
  - `conda run -n exp webarena-verified agent-input-get --task-ids 44 --config /home/syq/Documents/blackboard/experiment/common/configs/webarena_verified_local.example.json --output /tmp/webarena_task44_local.json`

## Phase 1：补齐缺失的 WebArena 包骨架

- [x] 新建 `experiment/webarena/__init__.py`
- [x] 新建 `experiment/webarena/bridge`
- [x] 新建 `experiment/webarena/core`
- [x] 新建 `experiment/webarena/systems`
- [x] 新建 `experiment/webarena/workers`
- [x] 新建 `experiment/webarena/utils`
- [x] 新建 `experiment/webarena/tests`
- [x] 实现 `experiment/webarena/utils/task_loader.py`
- [x] 实现 `experiment/webarena/utils/result_analysis.py`
- [x] 先让仓库中已经存在的依赖引用变成有效导入：
  - [x] `experiment.webarena.utils.task_loader.load_task_config`
  - [x] `experiment.webarena.utils.result_analysis.analyze_ablation_summary`
  - [x] `experiment.webarena.utils.result_analysis.analyze_system_compare_summary`

Phase 1 说明：

- 这一阶段的目标不是先解 benchmark，而是先修复仓库层面的“悬空依赖”。
- 当前已落地内容：
  - 提供了 WebArena package skeleton
  - 提供了单任务 / 数据集级 JSON 的最小 task loader
  - 提供了与 cross-dataset aggregation 兼容的 result analysis
  - 增加了对应单元测试

## Phase 2：定义 WebArena 的 neutral bridge

- [x] 实现 task adapter，把单个 WebArena config 转成 neutral `TaskSpec`。
- [x] 在 `TaskSpec.metadata` 中保留 WebArena 任务元信息，包括：
  - [x] config 路径
  - [x] site 列表
  - [x] task intent / rubric 字段
  - [x] 是否需要最终文本答案
  - [x] 任务类别，如 navigation / retrieval / form / multi-site
- [x] 设计稳定的浏览器工具面，映射到 neutral `ToolSpec`。
- [x] 第一版工具面保持最小且显式。候选工具：
  - [x] `browser__goto`
  - [x] `browser__click`
  - [x] `browser__type`
  - [x] `browser__select_option`
  - [x] `browser__press`
  - [x] `browser__scroll`
  - [x] `browser__go_back`
  - [x] `browser__new_tab`
  - [x] `browser__switch_tab`
  - [x] `browser__close_tab`
  - [x] `browser__finish`
- [x] 实现 result adapter，把浏览器 observation 转成 neutral `ToolResult`。
- [x] 对 observation 做标准化，避免 worker 直接依赖 WebArena 原始类。

Phase 2 说明：

- 第一版应该暴露结构化浏览器动作，而不是允许 worker 任意生成 Python。
- worker 最好消费一个标准化页面状态，例如：
  - `url`
  - `page_title`
  - `visible_text`
  - `interactive_elements`
  - `last_action`
  - `last_observation`
  - `open_tabs`
- 如果标准化摘要足够，就不要把底层 DOM 实现细节直接塞进 prompt。
- 当前已落地文件：
  - `experiment/webarena/bridge/task_adapter.py`
  - `experiment/webarena/bridge/tool_adapter.py`
  - `experiment/webarena/bridge/output_adapter.py`
- 当前已验证能力：
  - navigation / retrieval 任务可以稳定转成 `TaskSpec`
  - 最小浏览器工具面可以导出为 neutral `ToolSpec`
  - 浏览器动作结果可以同时转成 neutral `ToolResult` 与标准化 `Observation`

## Phase 3：构建 WebArena runtime wrapper

- [x] 新建 `experiment/webarena/core/browser_runtime.py`
- [x] 新建 `experiment/webarena/core/world_wrapper.py` 或等价 task runner
- [x] 实现一个 task-scoped runtime wrapper，负责：
  - [x] 加载单个 WebArena task config
  - [x] 创建浏览器环境
  - [x] 将浏览器动作暴露为 neutral tool 执行
  - [x] 每次动作后采集 observation
  - [x] 记录完成状态 / reward / evaluator 输出
- [x] 明确第一版统一 runtime 抽象：
  - [x] 类似 `AppWorldTaskRunner` 的 task runner
  - [x] runtime / wrapper 层之外不允许直接 import WebArena 环境类
- [x] 定义 worker 所需的最小可序列化 observation schema。

Phase 3 说明：

- runtime wrapper 只负责与官方环境交互。
- Blackboard 不应该知道底层是 Playwright、Selenium 还是 WebArena 自己的环境类。
- 官方 evaluator 应由 wrapper / runner 触发，而不是塞进 worker 逻辑中。
- 当前已落地文件：
  - `experiment/webarena/core/browser_runtime.py`
  - `experiment/webarena/core/world_wrapper.py`
- 当前 runtime 能力：
  - 基于 Playwright 的 task-scoped 浏览器 runtime
  - 默认复用本地 config 中的站点 URL 与认证信息
  - 浏览器动作执行后同时产出 neutral `ToolResult` / `Observation`
  - 通过 `WebArenaVerified.evaluate_task(...)` 调官方 evaluator
  - `WebArenaTaskRunner` 已能驱动通用 neutral turn loop
- 当前验证状态：
  - WebArena 单元测试 `12 passed`
  - 使用真实 task `44` 与本地 config 的半真实 smoke 已验证 task loading / tool registry / task spec 正常

## Phase 4：实现 WebArena Blackboard 主链路

- [x] 新建 `experiment/webarena/systems/blackboard_system.py`
- [x] 新建 `experiment/webarena/systems/blackboard_runner.py`
- [x] 在 `experiment/webarena/workers` 下实现 WebArena 专用 workers
- [x] 实现 WebArena architect，负责生成 task-scoped workflow 和 data schema
- [x] 将 neutral `TurnInput` 映射为 WebArena blackboard runtime state
- [x] 复用当前 kernel 模式：
  - [x] bounded patch writes
  - [x] 可检查的 shared state
  - [x] no-progress circuit breaker
  - [x] 明确 finish condition
- [x] 为 WebArena episode 添加 communication trace 采集，确保现有 communication analysis 能继续使用

Phase 4 说明：

- 第一版 WebArena Blackboard 不要试图覆盖所有任务类型。
- 优先做一个窄切片：
  - [x] 单站点任务
  - [ ] 非登录任务
  - [x] 完成条件清晰可见的任务
- 只有这个切片稳定后，再考虑：
  - 多站点任务
  - 登录敏感任务
  - 更复杂的长链路任务
- 当前已落地文件：
  - `experiment/webarena/workers/base.py`
  - `experiment/webarena/workers/architect.py`
  - `experiment/webarena/workers/builtin.py`
  - `experiment/webarena/systems/blackboard_system.py`
  - `experiment/webarena/systems/blackboard_runner.py`
- 当前最小 Blackboard 行为：
  - architect 作为 LLM worker 为每个 task 生成 task-scoped `workflow`、`data_schema` 和 `task_constraints`（LLM 异常时回退启发式）
  - session 会在一次 neutral `step()` 里推进多个内部 worker，直到产出工具调用或 finish
  - `page_state_worker` / `verification_worker` / `response_worker` 维持 LLM worker 角色边界
  - `argument_grounding_worker` 只做参数绑定
  - `browser_action_worker` 只发起结构化浏览器动作，不直接承担规划
  - kernel 按 `workflow.next` 的条件路由执行 step（保留 legacy 路由作为兼容兜底）
- 当前已验证的最小回归：
  - navigation 任务会先请求 `browser__goto`，收到新 observation 后可正常 finish
  - 需要最终文本答案的任务会进入 `response_worker`
  - `run_blackboard_task(...)` 已可通过 `WebArenaTaskRunner` 驱动整条 neutral 链路

## Phase 4 合约：最小 workflow 结构

- 每个 workflow step 至少包含：
  - `id`
  - `worker`
  - `purpose`
  - `reads`
  - `writes`
  - `exit_conditions`
  - `next`
- WebArena 上的典型 step 示例：
  - `inspect_current_page`
  - `bind_action_arguments`
  - `execute_browser_action`
  - `verify_goal_progress`
  - `compose_final_answer`
- workflow 必须支持：
  - inspect-act-verify 的重复循环
  - 动作失败后的分支处理
  - 导航后重新读取页面
  - 仅在需要时输出最终回答

## Phase 4 合约：最小 data schema 结构

- `data_schema` 描述的是 task-specific shared slots，而不是通用浏览器数据库。
- 候选 shared-state slots：
  - `task`
  - `task_constraints`
  - `current_page`
  - `navigation_history`
  - `open_tabs`
  - `candidate_actions`
  - `grounded_action`
  - `action_history`
  - `extracted_entities`
  - `evidence`
  - `verification`
  - `assistant_response`
  - `finish`
- 每个 schema entry 至少包含：
  - `type`
  - `required`
  - `producer`
  - `description`

## Phase 5：补齐 runner 与实验脚本

- [x] 新建 `experiment/webarena/examples/run_blackboard_smoke.py`
- [x] 新建 `experiment/webarena/examples/run_ablation.py`
- [x] 新建 `experiment/webarena/examples/run_system_compare.py`
- [x] 新建 `experiment/run/run_webarena_experiments.sh`
- [x] 让 `experiment/run/run_full_pipeline.sh` 中对 WebArena 的调用重新有效
- [x] 支持现有 WebArena task manifest：
  - [x] `webarena_script_browser_smoke.json`
  - [x] `webarena_debug.json`
  - [x] `webarena_live_debug.json`
  - [x] `webarena_formal.json`
- [x] 复用 common task manifest 解析逻辑，不额外造一套选择器

Phase 5 说明：

- 第一版 smoke runner 只需要证明：
  - [x] 一个 task config 能加载
  - [x] 一个浏览器动作能执行
  - [x] 一个 observation 能回流到 Blackboard
  - [x] 一个 evaluator 输出能被记录
- 在 smoke 跑通之前，不要急着加 formal / ablation / system compare
- 当前已落地内容：
  - `run_blackboard_smoke.py` 可直接读取 common task manifest，并按 `metadata.task_ids` 从 `webarena-verified.json` 中取任务
  - `run_ablation.py` 可在同一 task slice 上运行 `full` / `no_architect` 等 Blackboard 模式
  - `run_system_compare.py` 可在同一 task slice 上比较 `blackboard` / `blackboard_no_architect`
  - `run_webarena_experiments.sh` 当前提供 `smoke` / `compare` / `ablation` / `all`
  - smoke 输出会写入：
    - `episode_results.jsonl`
    - `summary.json`
    - 每个 task 子目录下的 `episode.json`
  - `webarena_script_browser_smoke.json` 默认已切到 GitLab task `44`
  - `webarena_debug.json` / `webarena_live_debug.json` 默认已切到 GitLab `44/45/46`
- 当前验证状态：
  - WebArena 单元测试 `17 passed`
  - 真实 GitLab smoke 已执行 through `run_blackboard_smoke.py`，task `44` 官方评估已通过
  - 真实 `run_system_compare.py` 已执行 through GitLab task `44`：
    - `blackboard = 1.0`
    - `blackboard_no_architect = 0.0`
  - 真实 `run_ablation.py` 已执行 through GitLab task `44`：
    - `full = 1.0`
    - `no_architect = 0.0`
  - `result_analysis.py` 已验证可以消费：
    - `/tmp/webarena_compare_phase5/system_compare_summary.json`
    - `/tmp/webarena_ablation_phase5/ablation_summary.json`

## Phase 6：结果格式与分析兼容

- [x] 定义与现有 common analysis 兼容的 WebArena episode result schema
- [x] 确保 episode records 包含现有测试和分析里已经使用的字段：
  - [x] `task_id`
  - [x] `success`
  - [x] `steps`
  - [x] `total_tokens`
  - [x] `goal_condition_rate`
  - [x] `worker_input_tokens`
  - [x] `worker_output_tokens`
  - [x] `architect_input_tokens`
  - [x] `architect_output_tokens`
  - [x] `fallback_action_count`
  - [x] `patch_error_count`
  - [x] `retrieval_precision`
  - [x] `metadata.sites`
  - [x] `trajectory`
  - [x] `communication_trace`
- [x] 实现 `result_analysis.py`，让 cross-dataset aggregation 可以消费 WebArena summary
- [x] 输出以下 summary：
  - [x] system compare
  - [x] ablation
  - [x] communication analysis

Phase 6 说明：

- 现在仓库里已经有不少测试和分析逻辑是“假定 WebArena summary 存在”的。
- 因此集成工作不算完成，直到这些路径能吃真实输出，而不只是测试夹具。
- 当前已完成内容：
  - `episode_results.jsonl` 现在写入了 common analysis 需要的标准字段，包括 token 统计、goal 指标、fallback / patch 计数、`trajectory`、`communication_trace`
  - `summary.json` 现在写入了 `mean_goal_condition_rate`、`mean_total_tokens`、`mean_fallback_rate`、`mean_patch_error_rate`、`mean_retrieval_precision`、context fragment 统计等聚合字段
  - `run_blackboard_smoke.py`、`run_system_compare.py`、`run_ablation.py` 现在都会额外导出：
    - `comm_summary.json`
    - `comm_trace.jsonl`
    - `comm_judge.jsonl`
  - `WebArenaBlackboardSession` 现在会在 metadata 中保留 step-level `trajectory`，并把 architect / grounding / action 的结构化通信记录挂到每个 step 上
  - `result_analysis.summarize_episode_results(...)` 已对齐为 rate 语义：
    - `mean_fallback_rate = fallback_action_count / steps`
    - `mean_patch_error_rate = patch_error_count / steps`
- 当前验证状态：
  - WebArena 单元测试：`19 passed`
  - 真实 GitLab smoke 已重新验证 through `run_blackboard_smoke.py`
  - `/tmp/webarena_phase6_smoke/summary.json` 已包含新的 mean 指标
  - `/tmp/webarena_phase6_smoke/episode_results.jsonl` 已包含 `trajectory` 与标准 episode 字段
  - `/tmp/webarena_phase6_smoke/comm_summary.json`、`comm_trace.jsonl`、`comm_judge.jsonl` 已成功生成

## Phase 7：测试

- [x] 为 `task_loader` 添加单元测试
- [x] 为 task adapter / tool adapter / output adapter 添加单元测试
- [x] 为 runtime wrapper 的 observation 标准化添加单元测试
- [x] 为 WebArena Blackboard workers 添加 fake observation 测试
- [x] 为 task-scoped 端到端流程添加 smoke test，优先使用 fake 或 scripted browser backend
- [x] 添加以下回归测试：
  - [x] architect fallback
  - [x] inspect-act-verify 的重复循环
  - [x] bounded worker write scope
  - [x] 浏览器动作失败恢复
  - [x] 无最终文本答案的完成
  - [x] 有最终文本答案的完成

Phase 7 说明：

- 当前已补充的测试文件：
  - `experiment/webarena/tests/test_task_loader.py`
  - `experiment/webarena/tests/test_bridge.py`
  - `experiment/webarena/tests/test_browser_runtime.py`
  - `experiment/webarena/tests/test_workers.py`
  - `experiment/webarena/tests/test_blackboard_system.py`
  - `experiment/webarena/tests/test_examples_common.py`
  - `experiment/webarena/tests/test_world_wrapper.py`
  - `experiment/webarena/tests/test_result_analysis.py`
- 当前新增覆盖点：
  - `WebArenaBrowserRuntime.current_observation()` 的 tab / element 标准化
  - `PageStateWorker`、`VerificationWorker`、`ResponseWorker` 的 fake observation 输入
  - `run_blackboard_batch(...)` 的 fake smoke artifact 落盘
  - `use_architect=False` 时的 fallback metadata
  - inspect-act-verify 重复循环直到目标页
  - 首次浏览器动作报错后的恢复与继续执行
  - worker 越权写入触发 `patch_error`
- 当前验证状态：
  - WebArena 单元测试：`28 passed`
  - 这一阶段还顺手修复了一个真实 helper 缺口：
    - `run_blackboard_batch(...)` 现在会主动创建 `task_output_dir`，不再隐式依赖底层 runner 创建目录

## 建议实施顺序

- [ ] 第一步：把缺失包和缺失导入补成真实模块
  - `experiment/webarena/utils/task_loader.py`
  - `experiment/webarena/utils/result_analysis.py`
- [ ] 第二步：建立本地 `webarena` checkout，并让最小 smoke manifest 可解析
- [ ] 第三步：实现 runtime wrapper，并先打通一个结构化浏览器动作
- [ ] 第四步：实现 neutral bridge
- [ ] 第五步：实现 Blackboard system + workers，先支持一个窄切片
- [ ] 第六步：补 smoke runner
- [ ] 第七步：补 formal runner / ablation / system compare
- [ ] 第八步：接入 common analysis

## 第一阶段完成标准

- [ ] `experiment/webarena` 包存在且可正常导入
- [ ] `experiment/common/*` 中对 WebArena 的既有引用不再悬空
- [ ] 一条本地 smoke 任务可通过 Blackboard 端到端跑通
- [ ] 这条运行能产出：
  - [ ] task config metadata
  - [ ] 浏览器动作轨迹
  - [ ] communication trace
  - [ ] 官方 completion / evaluator 信号
  - [ ] episode result JSONL
  - [ ] summary JSON
- [ ] 去掉 `architect` 后，smoke slice 的表现明显恶化，从而证明该 WebArena 接入不是“单智能体伪装成多智能体”
