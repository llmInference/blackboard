"""
交互式 Architect-Kernel-Worker 系统演示

清晰展示系统执行过程：
【步骤 N】模块名：执行内容
"""
import sys
from typing import Any

sys.path.insert(0, '/home/syq/Documents/ApexLab/langgraph/libs/kernel_system')

from langgraph_kernel import build_kernel_graph
from langgraph_kernel.llm_wrapper import SimpleChatModel
from langgraph_kernel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from langgraph_kernel.state_tree import StateTree


def run_interactive_system(prompt: str, model: str = "deepseek-v3", verbose: bool = True):
    """
    运行交互式系统（支持多轮对话）

    Args:
        prompt: 用户输入的 prompt
        model: 使用的模型名称
        verbose: 是否显示详细信息
    """
    print("=" * 80)
    print("  🚀 Architect-Kernel-Worker 系统（多轮对话模式）")
    print("=" * 80)
    print(f"\n📝 用户输入: {prompt}")
    print(f"🤖 使用模型: {model}\n")

    # 配置 LLM
    llm = SimpleChatModel(
        model=model,
        api_key="sk-2OsXSnW0fVfpFVnT4HeOkGDue8bxgRflUSc9KqCx7mnNOTgb",
        base_url="https://tb.api.mkeai.com/v1",
        temperature=0.7,
        timeout=60.0,
    )

    # 构建图
    graph = build_kernel_graph(llm, max_steps=15)

    # 初始化熔断器
    circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
        max_steps=15,
        max_invalid_patch_count=3,
        oscillation_window=6,
        stagnation_threshold=3,
    ))

    # 初始化状态树
    state_tree = StateTree()

    # 初始状态
    current_state = {
        "domain_state": {"user_prompt": prompt},
        "task_flow": [],
        "data_schema": {},
        "workflow_rules": {},
        "worker_instructions": {},
        "selected_workers": [],
        "pending_patch": [],
        "patch_error": "",
        "step_count": 0,
        "current_worker": "",
        "retry_count": 0,
        "error_feedback": "",
        "conversation_history": [],
        "pending_user_question": "",
        "user_response": "",
        "waiting_for_user": False,
        "no_update_count": 0,
        "status_history": [],
        # 新增：熔断器和状态树
        "circuit_breaker": circuit_breaker,
        "circuit_breaker_triggered": False,
        "circuit_breaker_reason": "",
        "circuit_breaker_details": {},
        "state_tree": state_tree,
        "current_node_id": "",
    }

    # 创建状态树的根节点
    root_id = state_tree.create_root(current_state["domain_state"])
    current_state["current_node_id"] = root_id

    print("=" * 80)
    print("  执行过程")
    print("=" * 80)

    business_step = 0
    max_conversation_turns = 10  # 最多 10 轮对话

    try:
        # 主执行循环：支持多轮对话
        for conversation_turn in range(max_conversation_turns + 1):
            if conversation_turn > 0:
                # 这是继续对话的轮次
                if not current_state.get("waiting_for_user", False):
                    # 不需要用户输入，结束
                    break

                pending_question = current_state.get("pending_user_question", "")
                if not pending_question:
                    break

                print("\n" + "=" * 80)
                print(f"  💬 第 {conversation_turn} 轮对话")
                print("=" * 80)
                print(f"\n❓ 系统: {pending_question}")

                # 获取用户输入
                user_input = input("\n👤 您的回复: ").strip()

                if not user_input:
                    print("⚠️  未输入内容，结束对话")
                    break

                # 更新状态
                current_state["user_response"] = user_input
                current_state["waiting_for_user"] = False

                print("\n" + "=" * 80)
                print("  继续执行")
                print("=" * 80)

            # 执行图（stream 模式）
            for event in graph.stream(current_state, stream_mode="updates"):
                node_name = list(event.keys())[0]
                node_output = event[node_name]

                # 跳过内部节点
                if node_name == "set_worker":
                    current_state.update(node_output)
                    continue

                # 根据节点类型显示信息
                if node_name == "handle_user_response":
                    print(f"\n  → 处理用户响应并更新对话历史")

                elif node_name == "architect":
                    business_step += 1
                    if conversation_turn == 0:
                        print(f"\n【步骤 {business_step}】Architect: 分析用户需求并设计系统架构")
                    else:
                        print(f"\n【步骤 {business_step}】Architect: 根据新信息重新规划")

                    if verbose and "task_flow" in node_output:
                        task_flow = node_output["task_flow"]
                        if task_flow:
                            print(f"  • 任务分解: {len(task_flow)} 个子任务")
                            for i, task in enumerate(task_flow, 1):
                                subtask = task.get("subtask", "")
                                worker = task.get("worker", "")
                                print(f"    {i}. {subtask[:60]}... → {worker}")

                elif node_name == "kernel":
                    old_status = current_state.get("domain_state", {}).get("status", "")
                    new_status = node_output.get("domain_state", {}).get("status", "")

                    if new_status and new_status != old_status:
                        print(f"  → Kernel: 验证通过，状态转换 {old_status} → {new_status}")

                    if node_output.get("patch_error"):
                        print(f"  → Kernel: ❌ 验证失败 - {node_output['patch_error'][:80]}...")

                elif node_name == "worker":
                    worker_name = current_state.get("current_worker", "unknown")
                    business_step += 1

                    # 从 task_flow 获取任务描述
                    task_flow = current_state.get("task_flow", [])
                    task_desc = "执行任务"
                    for task in task_flow:
                        if task.get("worker") == worker_name:
                            task_desc = task.get("subtask", "执行任务")
                            break

                    if len(task_desc) > 60:
                        task_desc = task_desc[:60] + "..."

                    print(f"\n【步骤 {business_step}】{worker_name}: {task_desc}")

                    if verbose and "pending_patch" in node_output:
                        patch = node_output["pending_patch"]
                        if patch:
                            print(f"  • 提交了 {len(patch)} 个状态更新")
                            print(f"  • Worker 提交的完整内容:")

                            for i, op in enumerate(patch, 1):
                                op_type = op.get("op", "unknown")
                                path = op.get("path", "")
                                value = op.get("value")
                                field_name = path.split("/")[-1] if "/" in path else path

                                # 显示完整内容，不截断
                                print(f"\n    [{i}] {op_type} {field_name}:")

                                if isinstance(value, str):
                                    # 字符串类型，按行显示
                                    if "\n" in value:
                                        print("    " + "-" * 60)
                                        for line in value.split("\n"):
                                            print(f"    {line}")
                                        print("    " + "-" * 60)
                                    else:
                                        print(f"    {value}")
                                elif isinstance(value, (dict, list)):
                                    # 字典或列表，格式化显示
                                    import json
                                    formatted = json.dumps(value, indent=6, ensure_ascii=False)
                                    print(f"    {formatted}")
                                else:
                                    # 其他类型
                                    print(f"    {value}")

                # 更新当前状态
                current_state.update(node_output)

            # 检查是否需要继续对话
            if not current_state.get("waiting_for_user", False):
                # 不需要用户输入，结束循环
                break

        # 检查是否达到最大轮数
        if current_state.get("waiting_for_user", False):
            print(f"\n⚠️  已达到最大对话轮数 ({max_conversation_turns})，结束对话")

        # 显示最终结果
        print("\n" + "=" * 80)
        print("  最终结果")
        print("=" * 80)

        final_state = current_state
        domain_state = final_state.get("domain_state", {})

        # 显示执行统计
        print(f"\n📊 执行统计:")
        print(f"  • 总步数: {final_state.get('step_count', 0)}")
        print(f"  • 最终状态: {domain_state.get('status', '未知')}")
        error = final_state.get('patch_error')
        print(f"  • 错误: {error if error else '无'}")

        # 显示对话历史（如果有）
        conversation_history = final_state.get("conversation_history", [])
        if conversation_history:
            print(f"\n💬 对话历史 ({len(conversation_history)} 条):")
            for i, msg in enumerate(conversation_history, 1):
                role = "🤖 系统" if msg["role"] == "system" else "👤 用户"
                content = msg["content"]
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  {i}. {role}: {content}")

        # 显示业务数据（排除 user_prompt 和 status）
        print(f"\n📋 生成的内容:")
        has_content = False
        for key, value in domain_state.items():
            if key in ["user_prompt", "status", "error_feedback", "retry_count"]:
                continue

            has_content = True
            print(f"\n  {key}:")

            if isinstance(value, dict):
                # 格式化显示字典
                print(json.dumps(value, indent=4, ensure_ascii=False))
            elif isinstance(value, list):
                if len(value) > 0:
                    # 如果是对象列表，格式化显示
                    if isinstance(value[0], dict):
                        print(json.dumps(value, indent=4, ensure_ascii=False))
                    else:
                        # 简单列表，逐行显示
                        for i, item in enumerate(value, 1):
                            print(f"    {i}. {item}")
                else:
                    print("    (空)")
            else:
                # 字符串或其他类型
                value_str = str(value)
                if len(value_str) > 500:
                    print(f"    {value_str[:500]}...")
                else:
                    print(f"    {value_str}")

        if not has_content:
            print("  (无业务数据)")

        print("\n" + "=" * 80)

        return final_state

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        return current_state
    except Exception as e:
        error_msg = str(e)
        print(f"\n\n❌ 执行出错: {error_msg}")

        # 针对不同类型的错误给出建议
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print("\n💡 建议:")
            print("  1. 检查网络连接是否正常")
            print("  2. API 服务可能繁忙，请稍后重试")
            print("  3. 尝试使用更简单的 prompt")
        elif "connection" in error_msg.lower():
            print("\n💡 建议:")
            print("  1. 检查网络连接")
            print("  2. 检查 API 地址是否正确")
            print("  3. 检查防火墙设置")
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            print("\n💡 建议:")
            print("  1. 检查 API Key 是否有效")
            print("  2. 检查 API 配额是否用尽")

        if verbose:
            print("\n详细错误信息:")
            import traceback
            traceback.print_exc()

        return current_state


def main():
    """主函数"""
    # 检查是否是测试模式（暂时禁用）
    # if len(sys.argv) > 1 and sys.argv[1] == "test":
    #     test_multi_turn()
    #     return

    print("=" * 80)
    print("  🤖 Architect-Kernel-Worker 交互式演示系统")
    print("=" * 80)
    print("\n这个系统可以处理任意模糊的用户 prompt，自动设计工作流并执行。")
    print("支持多轮对话：系统会在需要时向您请求更多信息。")
    print("\n示例 prompt:")
    print("  • 帮我规划一次旅行")
    print("  • 分析销售数据")
    print("  • 写一篇文章")
    print("  • 设计一个网站")
    print("  • 制定学习计划")

    # 获取用户输入
    print("\n" + "-" * 80)
    user_prompt = input("\n请输入你的 prompt (按 Ctrl+C 退出): ").strip()

    if not user_prompt:
        print("❌ Prompt 不能为空")
        return

    # 询问是否显示详细信息
    verbose_input = input("\n是否显示详细执行过程? (y/n, 默认 y): ").strip().lower()
    verbose = verbose_input != 'n'

    print()  # 空行

    # 运行系统
    run_interactive_system(user_prompt, verbose=verbose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见!")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
