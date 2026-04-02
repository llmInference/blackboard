"""Bridge helpers between WebArena primitives and the neutral protocol."""

from experiment.webarena.bridge.output_adapter import (
    adapt_browser_result,
    browser_result_to_tool_result,
    observation_to_neutral,
    standardize_browser_observation,
)
from experiment.webarena.bridge.task_adapter import (
    classify_task,
    requires_final_response,
    task_to_spec,
)
from experiment.webarena.bridge.tool_adapter import browser_tools_to_specs

__all__ = [
    "adapt_browser_result",
    "browser_result_to_tool_result",
    "browser_tools_to_specs",
    "classify_task",
    "observation_to_neutral",
    "requires_final_response",
    "standardize_browser_observation",
    "task_to_spec",
]
