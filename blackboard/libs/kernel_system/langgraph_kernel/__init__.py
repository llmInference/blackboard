from langgraph_kernel.ablation import AblationConfig
from langgraph_kernel.types import DataSchema, JsonPatch, KernelState, WorkflowRules

# 延迟导入，避免在模块加载时导入 langgraph
def __getattr__(name):
    if name == "build_kernel_graph":
        from langgraph_kernel.graph import build_kernel_graph
        return build_kernel_graph
    elif name == "register_custom_worker":
        from langgraph_kernel.extensions import register_custom_worker
        return register_custom_worker
    elif name == "set_custom_architect_prompt":
        from langgraph_kernel.extensions import set_custom_architect_prompt
        return set_custom_architect_prompt
    elif name == "add_architect_prompt_section":
        from langgraph_kernel.extensions import add_architect_prompt_section
        return add_architect_prompt_section
    elif name == "load_extensions":
        from langgraph_kernel.extensions import load_extensions
        return load_extensions
    elif name == "get_registry":
        from langgraph_kernel.worker.registry import get_registry
        return get_registry
    elif name == "get_architect_extension":
        from langgraph_kernel.extensions import get_architect_extension
        return get_architect_extension
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataSchema",
    "JsonPatch",
    "KernelState",
    "WorkflowRules",
    "AblationConfig",
    "build_kernel_graph",
    # Extension system
    "register_custom_worker",
    "set_custom_architect_prompt",
    "add_architect_prompt_section",
    "load_extensions",
    "get_registry",
    "get_architect_extension",
]
