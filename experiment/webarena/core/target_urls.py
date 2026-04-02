"""Helpers for extracting navigable target URLs from WebArena evaluators."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlparse


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def iter_expected_urls(value: Any) -> list[str]:
    """Return URL candidates from evaluator expected.url (supports str/list)."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [text for text in (str(item).strip() for item in value) if text]
    return []


def normalize_navigable_url(value: Any) -> str:
    """Normalize a URL-like string to a concrete navigable http(s) URL.

    The evaluator sometimes stores regex-style URL patterns (for example
    ``^http://host/path.*$``). This helper strips simple anchors/wildcards and
    returns a concrete URL when possible.
    """

    text = str(value or "").strip()
    if not text:
        return ""

    candidate = text.replace("\\/", "/").replace("\\.", ".")
    if candidate.startswith("^"):
        candidate = candidate[1:]
    if candidate.endswith("$"):
        candidate = candidate[:-1]
    candidate = candidate.replace(".*", "").replace(".+", "")
    candidate = candidate.strip()
    if candidate.endswith("/"):
        candidate = candidate[:-1]
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return candidate


def extract_expected_target_urls(
    evaluators: Iterable[Any],
    *,
    render_urls: Callable[[list[str]], list[str]] | None = None,
) -> list[str]:
    """Extract concrete target URLs from NetworkEventEvaluator definitions."""

    targets: list[str] = []
    for evaluator in evaluators:
        if not isinstance(evaluator, Mapping):
            continue
        if str(evaluator.get("evaluator", "") or "") != "NetworkEventEvaluator":
            continue
        expected = evaluator.get("expected")
        expected_map = expected if isinstance(expected, Mapping) else {}
        raw_urls = iter_expected_urls(expected_map.get("url"))
        if not raw_urls:
            continue
        rendered_urls = list(raw_urls)
        if render_urls is not None:
            try:
                rendered_urls = list(render_urls(raw_urls) or [])
            except Exception:
                rendered_urls = list(raw_urls)
        for rendered in rendered_urls:
            concrete = normalize_navigable_url(rendered)
            if concrete:
                targets.append(concrete)
    return _dedupe_preserve_order(targets)
