"""Tests for ALFWorld runtime LLM factory."""
from __future__ import annotations

import pytest

from experiment.alfworld.utils.llm_factory import build_runtime_llm
from experiment.alfworld.utils.mock_llm import ALFWorldMockActionLLM
from langgraph_kernel.llm_wrapper import SimpleChatModel


def test_llm_factory_returns_none_for_non_llm_workflow():
    assert build_runtime_llm(workflow_mode="single_action", llm_backend="none") is None


def test_llm_factory_supports_mock_backend():
    llm = build_runtime_llm(workflow_mode="planner_llm_action", llm_backend="mock")
    assert isinstance(llm, ALFWorldMockActionLLM)


def test_llm_factory_supports_llm_architect_on_non_llm_workflow():
    llm = build_runtime_llm(workflow_mode="single_action", architect_mode="llm", llm_backend="mock")
    assert isinstance(llm, ALFWorldMockActionLLM)


def test_llm_factory_builds_openai_compatible_model_from_env(monkeypatch):
    monkeypatch.setenv("ALFWORLD_LLM_MODEL", "test-model")
    monkeypatch.setenv("ALFWORLD_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ALFWORLD_LLM_API_KEY", "test-key")

    llm = build_runtime_llm(
        workflow_mode="planner_llm_action",
        llm_backend="openai_compatible",
        llm_temperature=0.2,
        llm_timeout=15.0,
    )

    assert isinstance(llm, SimpleChatModel)
    assert llm.model == "test-model"
    assert llm.temperature == 0.2
    assert llm.timeout == 15.0


def test_llm_factory_reports_missing_openai_compatible_config(monkeypatch):
    monkeypatch.delenv("ALFWORLD_LLM_MODEL", raising=False)
    monkeypatch.delenv("ALFWORLD_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ALFWORLD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="missing required config"):
        build_runtime_llm(workflow_mode="planner_llm_action", llm_backend="openai_compatible")
