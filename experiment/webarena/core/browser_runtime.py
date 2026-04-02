"""Browser runtime wrapper for WebArena tasks."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from experiment.common.neutral import ToolCall
from experiment.webarena.bridge import adapt_browser_result


def _looks_like_selector(value: str) -> bool:
    return value.startswith(("#", ".", "[", "/", "xpath=", "css=", "text="))


async def default_ui_login(sites: list[str], config: dict[str, Any], storage_state_file: Path) -> None:
    """Perform minimal UI login for WebArena sites and save storage state."""
    from playwright.async_api import async_playwright

    def _resolve_env(site_name: str) -> tuple[str, str, str]:
        environments = config.get("environments", {})
        candidates = [
            site_name.lower(),
            site_name.upper(),
            f"__{site_name.upper()}__",
            f"__{site_name.lower()}__",
        ]
        env_config = next((environments.get(candidate) for candidate in candidates if environments.get(candidate)), None)
        if env_config is None:
            raise ValueError(f"Environment config for site '{site_name}' not found")
        urls = list(env_config.get("urls", []) or [])
        active_url_idx = int(env_config.get("active_url_idx", 0) or 0)
        base_url = str(urls[active_url_idx] if urls else "")
        credentials = dict(env_config.get("credentials") or {})
        return base_url, str(credentials.get("username", "") or ""), str(credentials.get("password", "") or "")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        try:
            for site_name in sites:
                base_url, username, password = _resolve_env(site_name)
                page = await context.new_page()
                if site_name == "shopping":
                    await page.goto(f"{base_url}/customer/account/login/")
                    await page.get_by_label("Email", exact=True).fill(username)
                    await page.get_by_label("Password", exact=True).fill(password)
                    await page.get_by_role("button", name="Sign In").click()
                elif site_name == "shopping_admin":
                    await page.goto(base_url)
                    await page.get_by_label("Username").fill(username)
                    await page.get_by_label("Password").fill(password)
                    await page.get_by_role("button", name="Sign in").click()
                elif site_name == "gitlab":
                    await page.goto(f"{base_url}/users/sign_in")
                    if username == "root":
                        await page.get_by_test_id("username-field").fill(username)
                        await page.get_by_test_id("password-field").fill(password)
                        await page.get_by_test_id("sign-in-button").click()
                    else:
                        await page.get_by_label("Username or email").fill(username, timeout=3000)
                        await page.get_by_label("Password").fill(password)
                        await page.get_by_role("button", name="Sign in").click()
                elif site_name == "reddit":
                    await page.goto(base_url)
                    await page.get_by_role("link", name="Log in").click()
                    await page.get_by_label("Username").fill(username)
                    await page.get_by_label("Password").fill(password)
                    await page.get_by_role("button", name="Log in").click()
                elif site_name in {"map", "wikipedia"}:
                    pass
                else:
                    raise ValueError(f"No default UI login handler for site '{site_name}'")
                await page.close()
            storage_state_file.parent.mkdir(parents=True, exist_ok=True)
            await context.storage_state(path=str(storage_state_file))
        finally:
            await context.close()
            await browser.close()


class WebArenaBrowserRuntime:
    """Task-scoped browser runtime for one WebArena task."""

    def __init__(
        self,
        *,
        task: Any,
        config: Any,
        evaluator: Any,
        output_dir: str | Path,
        headless: bool = True,
        slow_mo_ms: int = 0,
        ui_login_func: Any | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.evaluator = evaluator
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = bool(headless)
        self.slow_mo_ms = int(slow_mo_ms)
        self.ui_login_func = ui_login_func

        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._active_page: Any | None = None
        self._finished = False
        self._final_response: str = ""
        self._network_entries: list[dict[str, Any]] = []

    @property
    def har_path(self) -> Path:
        return self.output_dir / "network.har"

    @property
    def storage_state_path(self) -> Path:
        return self.output_dir / ".storage_state.json"

    @property
    def eval_result_path(self) -> Path:
        return self.output_dir / "eval_result.json"

    @property
    def agent_response_path(self) -> Path:
        return self.output_dir / "agent_response.json"

    @property
    def network_entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._network_entries)

    @property
    def final_response(self) -> str:
        return self._final_response

    @property
    def finished(self) -> bool:
        return self._finished

    def _task_sites(self) -> list[str]:
        return [str(site.value if hasattr(site, "value") else site) for site in getattr(self.task, "sites", ())]

    def _task_start_urls(self) -> list[str]:
        rendered = self.config.render_url(list(self.task.start_urls), self.task.sites)
        return [str(url) for url in list(rendered or [])]

    def _config_dict(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            return dict(self.config.model_dump(mode="json"))
        return dict(self.config)

    def _ensure_storage_state(self) -> Path | None:
        if self.storage_state_path.exists():
            return self.storage_state_path
        if self.ui_login_func is None:
            return None

        environments = self._config_dict().get("environments", {})
        if not environments:
            return None

        asyncio.run(self.ui_login_func(self._task_sites(), self._config_dict(), self.storage_state_path))
        return self.storage_state_path if self.storage_state_path.exists() else None

    def _context_headers_file(self, storage_state_file: Path | None) -> Path | None:
        if storage_state_file is None:
            return None
        candidate = Path(str(storage_state_file) + ".headers.json")
        return candidate if candidate.exists() else None

    def _attach_page(self, page: Any) -> None:
        page.on("requestfinished", lambda request: self._record_request(request, failed=False))
        page.on("requestfailed", lambda request: self._record_request(request, failed=True))
        self._active_page = page

    def _headers_dict_to_list(self, headers: dict[str, str] | None) -> list[dict[str, str]]:
        headers = dict(headers or {})
        return [{"name": str(name), "value": str(value)} for name, value in sorted(headers.items())]

    def _request_post_data(self, request: Any) -> dict[str, Any] | None:
        text = request.post_data or ""
        if not text:
            return None
        content_type = ""
        try:
            headers = request.headers or {}
            content_type = str(headers.get("content-type", "") or "")
        except Exception:
            content_type = ""
        return {
            "mimeType": content_type,
            "text": text,
        }

    def _response_content(self, response: Any) -> dict[str, Any]:
        if response is None:
            return {"mimeType": "", "text": ""}
        try:
            headers = response.headers or {}
        except Exception:
            headers = {}
        mime_type = str(headers.get("content-type", "") or "")
        text = ""
        try:
            if any(token in mime_type for token in ("json", "text", "html", "xml", "javascript")):
                text = response.text() or ""
        except Exception:
            text = ""
        return {"mimeType": mime_type, "text": text}

    def _record_request(self, request: Any, *, failed: bool) -> None:
        try:
            response = None if failed else request.response()
        except Exception:
            response = None
        try:
            request_headers = dict(request.headers or {})
        except Exception:
            request_headers = {}
        try:
            response_headers = dict(response.headers or {}) if response is not None else {}
        except Exception:
            response_headers = {}
        status = 0 if response is None else int(response.status)
        redirect_url = str(response_headers.get("location", "") or "") if status in range(300, 400) else ""
        entry = {
            "request": {
                "url": str(request.url),
                "method": str(request.method),
                "headers": self._headers_dict_to_list(request_headers),
                "postData": self._request_post_data(request),
            },
            "response": {
                "status": status,
                "headers": self._headers_dict_to_list(response_headers),
                "content": self._response_content(response),
                "redirectURL": redirect_url,
                "cookies": [],
            },
        }
        self._network_entries.append(entry)

    def _page(self) -> Any:
        self.open()
        assert self._context is not None
        if self._active_page is not None and not self._active_page.is_closed():
            return self._active_page
        pages = [page for page in self._context.pages if not page.is_closed()]
        if not pages:
            self._active_page = self._context.new_page()
            self._attach_page(self._active_page)
        else:
            self._active_page = pages[-1]
        return self._active_page

    def _locator(self, page: Any, element_id: str) -> Any:
        element_id = str(element_id or "").strip()
        if not element_id:
            raise ValueError("element_id is required")
        if _looks_like_selector(element_id):
            return page.locator(element_id)

        candidates = [
            f'[data-testid="{element_id}"]',
            f'[bid="{element_id}"]',
            f'[id="{element_id}"]',
            f'[name="{element_id}"]',
            f'#{element_id}',
        ]
        for selector in candidates:
            locator = page.locator(selector)
            try:
                if locator.count() > 0:
                    return locator
            except Exception:
                continue
        return page.locator(element_id)

    def _collect_interactive_elements(self, page: Any) -> list[dict[str, Any]]:
        script = """
        () => {
          const selectors = ['a', 'button', 'input', 'select', 'textarea', '[role]', '[data-testid]', '[bid]'];
          const nodes = Array.from(document.querySelectorAll(selectors.join(',')));
          return nodes.slice(0, 80).map((node) => ({
            element_id: node.getAttribute('data-testid') || node.getAttribute('bid') || node.id || node.name || '',
            tag: (node.tagName || '').toLowerCase(),
            role: node.getAttribute('role') || '',
            text: (node.innerText || node.textContent || node.value || '').trim().slice(0, 160),
            enabled: !node.disabled,
            visible: (() => {
              const style = window.getComputedStyle(node);
              const rect = node.getBoundingClientRect();
              return !!(
                rect.width > 0 &&
                rect.height > 0 &&
                style.visibility !== 'hidden' &&
                style.display !== 'none' &&
                style.opacity !== '0'
              );
            })(),
          }));
        }
        """
        try:
            result = page.evaluate(script)
            return list(result or [])
        except Exception:
            return []

    def _first_visible_locator(self, locator: Any) -> Any:
        try:
            total = int(locator.count())
        except Exception:
            total = 0
        for index in range(max(total, 0)):
            candidate = locator.nth(index)
            try:
                if bool(candidate.is_visible()):
                    return candidate
            except Exception:
                continue
        return locator.first

    def current_observation(self) -> dict[str, Any]:
        page = self._page()
        open_tabs: list[dict[str, Any]] = []
        if self._context is not None:
            for index, candidate in enumerate(self._context.pages):
                if candidate.is_closed():
                    continue
                try:
                    title = candidate.title()
                except Exception:
                    title = ""
                open_tabs.append(
                    {
                        "index": index,
                        "url": str(candidate.url or ""),
                        "title": str(title or ""),
                        "active": candidate == page,
                    }
                )
        try:
            title = page.title()
        except Exception:
            title = ""
        try:
            visible_text = page.locator("body").inner_text(timeout=3000)
        except Exception:
            visible_text = ""
        return {
            "url": str(page.url or ""),
            "title": str(title or ""),
            "text": str(visible_text or ""),
            "tabs": open_tabs,
            "active_tab_index": next((tab["index"] for tab in open_tabs if tab["active"]), 0),
            "elements": self._collect_interactive_elements(page),
        }

    def open(self) -> "WebArenaBrowserRuntime":
        if self._context is not None:
            return self

        from playwright.sync_api import sync_playwright

        storage_state_file = self._ensure_storage_state()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo_ms,
        )
        context_kwargs: dict[str, Any] = {
            "record_har_path": str(self.har_path),
            "record_har_content": "embed",
        }
        if storage_state_file is not None and storage_state_file.exists():
            context_kwargs["storage_state"] = str(storage_state_file)
        self._context = self._browser.new_context(**context_kwargs)

        headers_file = self._context_headers_file(storage_state_file)
        if headers_file is not None:
            headers = json.loads(headers_file.read_text(encoding="utf-8"))
            self._context.set_extra_http_headers(headers)

        self._context.on("page", self._attach_page)
        page = self._context.new_page()
        self._attach_page(page)
        start_urls = self._task_start_urls()
        if start_urls:
            page.goto(start_urls[0], wait_until="load")
        return self

    def execute_action(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = dict(arguments or {})
        page = self._page()
        done = False
        error_message = ""
        try:
            if tool_name == "browser__goto":
                page.goto(str(arguments["url"]), wait_until="load")
            elif tool_name == "browser__click":
                locator = self._locator(page, str(arguments["element_id"]))
                self._first_visible_locator(locator).click()
            elif tool_name == "browser__type":
                locator = self._locator(page, str(arguments["element_id"])).first
                if bool(arguments.get("clear_first", True)):
                    locator.fill("")
                locator.fill(str(arguments.get("text", "")))
            elif tool_name == "browser__select_option":
                self._locator(page, str(arguments["element_id"])).first.select_option(str(arguments["value"]))
            elif tool_name == "browser__press":
                key = str(arguments["key"])
                if arguments.get("element_id"):
                    self._locator(page, str(arguments["element_id"])).first.press(key)
                else:
                    page.keyboard.press(key)
            elif tool_name == "browser__scroll":
                page.mouse.wheel(float(arguments.get("x", 0) or 0), float(arguments.get("y", 800) or 0))
            elif tool_name == "browser__go_back":
                page.go_back(wait_until="load")
            elif tool_name == "browser__new_tab":
                page = self._context.new_page()
                self._attach_page(page)
                url = str(arguments.get("url", "") or "")
                if url:
                    page.goto(url, wait_until="load")
            elif tool_name == "browser__switch_tab":
                index = int(arguments["tab_index"])
                pages = [candidate for candidate in self._context.pages if not candidate.is_closed()]
                self._active_page = pages[index]
                page = self._active_page
            elif tool_name == "browser__close_tab":
                index = arguments.get("tab_index")
                if index is None:
                    page.close()
                else:
                    pages = [candidate for candidate in self._context.pages if not candidate.is_closed()]
                    pages[int(index)].close()
                page = self._page()
            elif tool_name == "browser__finish":
                self._finished = True
                self._final_response = str(arguments.get("response", "") or "")
                done = True
            else:
                raise ValueError(f"Unsupported WebArena browser action: {tool_name}")
        except Exception as exc:
            error_message = str(exc)

        observation = self.current_observation()
        observation["last_action"] = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        return {
            "done": done or self._finished,
            "reward": 1.0 if done and not error_message else 0.0,
            "error_message": error_message,
            "observation": observation,
        }

    def execute_tool_call(self, tool_call: ToolCall) -> tuple[Any, Any]:
        raw_result = self.execute_action(tool_call.tool_name, dict(tool_call.arguments))
        return adapt_browser_result(
            tool_call.tool_name,
            raw_result,
            call_id=tool_call.call_id,
        )

    def evaluate(self, *, agent_response: Any | None = None) -> dict[str, Any]:
        response_payload = agent_response if agent_response is not None else self._final_response
        if isinstance(response_payload, (dict, list)):
            self.agent_response_path.write_text(json.dumps(response_payload, ensure_ascii=False), encoding="utf-8")
        else:
            self.agent_response_path.write_text(str(response_payload or ""), encoding="utf-8")

        network_trace_input: Any = list(self._network_entries)
        if self._context is not None:
            self._context.close()
            self._context = None
            self._active_page = None
        if self.har_path.exists():
            network_trace_input = self.har_path

        result = self.evaluator.evaluate_task(
            task_id=int(self.task.task_id),
            agent_response=response_payload,
            network_trace=network_trace_input,
        )
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else {"raw_evaluation": result}
        self.eval_result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._active_page = None
