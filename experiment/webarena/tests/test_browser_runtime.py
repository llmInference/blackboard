from __future__ import annotations

from experiment.webarena.core.browser_runtime import WebArenaBrowserRuntime


class _FakeLocator:
    def __init__(self, text: str) -> None:
        self._text = text

    def inner_text(self, timeout: int = 3000) -> str:
        del timeout
        return self._text


class _FakePage:
    def __init__(self, *, url: str, title: str, body_text: str, elements: list[dict]) -> None:
        self.url = url
        self._title = title
        self._body_text = body_text
        self._elements = elements

    def is_closed(self) -> bool:
        return False

    def title(self) -> str:
        return self._title

    def locator(self, selector: str) -> _FakeLocator:
        assert selector == "body"
        return _FakeLocator(self._body_text)

    def evaluate(self, script: str):
        del script
        return self._elements


class _FakeContext:
    def __init__(self, pages) -> None:
        self.pages = pages


class _VisibleCandidate:
    def __init__(self, locator: "_VisibilityLocator", index: int) -> None:
        self._locator = locator
        self._index = index

    def is_visible(self) -> bool:
        return bool(self._locator._visible_flags[self._index])

    @property
    def index(self) -> int:
        return self._index


class _VisibilityLocator:
    def __init__(self, visible_flags: list[bool]) -> None:
        self._visible_flags = visible_flags

    def count(self) -> int:
        return len(self._visible_flags)

    def nth(self, index: int) -> _VisibleCandidate:
        return _VisibleCandidate(self, index)

    @property
    def first(self) -> _VisibleCandidate:
        return _VisibleCandidate(self, 0)


def test_current_observation_normalizes_page_state(tmp_path):
    active = _FakePage(
        url="http://localhost:8023/dashboard/todos",
        title="Your To-Do List",
        body_text="Todo item one\nTodo item two",
        elements=[
            {"element_id": "mark-done", "tag": "button", "role": "button", "text": "Mark as done", "enabled": True},
            {"element_id": "search", "tag": "input", "role": "", "text": "Search", "enabled": True},
        ],
    )
    inactive = _FakePage(
        url="http://localhost:8023/",
        title="Home",
        body_text="Welcome",
        elements=[],
    )
    runtime = WebArenaBrowserRuntime(
        task=object(),
        config={},
        evaluator=object(),
        output_dir=tmp_path,
    )
    runtime._context = _FakeContext([inactive, active])
    runtime._active_page = active

    observation = runtime.current_observation()

    assert observation["url"] == "http://localhost:8023/dashboard/todos"
    assert observation["title"] == "Your To-Do List"
    assert observation["active_tab_index"] == 1
    assert len(observation["tabs"]) == 2
    assert observation["tabs"][1]["active"] is True
    assert observation["elements"][0]["element_id"] == "mark-done"
    assert observation["elements"][1]["element_id"] == "search"


def test_first_visible_locator_prefers_visible_candidate(tmp_path):
    runtime = WebArenaBrowserRuntime(
        task=object(),
        config={},
        evaluator=object(),
        output_dir=tmp_path,
    )
    locator = _VisibilityLocator([False, False, True, True])

    selected = runtime._first_visible_locator(locator)

    assert selected.index == 2
