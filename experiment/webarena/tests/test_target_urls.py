from __future__ import annotations

from experiment.webarena.core.target_urls import (
    extract_expected_target_urls,
    iter_expected_urls,
    normalize_navigable_url,
)


def test_iter_expected_urls_supports_string_without_character_split():
    assert iter_expected_urls("__GITLAB__/dashboard/todos") == ["__GITLAB__/dashboard/todos"]


def test_normalize_navigable_url_handles_regex_style_target():
    assert normalize_navigable_url("^http://localhost:8023/repo/-/issues.*$") == "http://localhost:8023/repo/-/issues"


def test_extract_expected_target_urls_renders_and_filters_candidates():
    evaluators = [
        {"evaluator": "OtherEvaluator", "expected": {"url": "__GITLAB__/ignored"}},
        {"evaluator": "NetworkEventEvaluator", "expected": {"url": "__GITLAB__/a11yproject/a11yproject.com/-/issues"}},
        {"evaluator": "NetworkEventEvaluator", "expected": {"url": "^__GITLAB__/byteblaze/a11y-syntax-highlighting/-/issues.*$"}},
    ]

    targets = extract_expected_target_urls(
        evaluators,
        render_urls=lambda urls: [str(url).replace("__GITLAB__", "http://localhost:8023") for url in urls],
    )

    assert targets == [
        "http://localhost:8023/a11yproject/a11yproject.com/-/issues",
        "http://localhost:8023/byteblaze/a11y-syntax-highlighting/-/issues",
    ]
