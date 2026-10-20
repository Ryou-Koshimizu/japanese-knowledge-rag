from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from knowledge_rag.app import create_app
from knowledge_rag.config import Settings

PORT = 8104
BASE = f"http://127.0.0.1:{PORT}"


def _port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


@pytest.fixture(scope="session")
def server() -> Iterator[str]:
    """公開デモと同じ設定（記録の再生・閲覧者を選ぶ）で、本物の HTTP サーバーを立てる。"""
    assert _port_free(), f"ポート {PORT} が使用中です"
    config = uvicorn.Config(
        create_app(Settings.from_env()), host="127.0.0.1", port=PORT, log_level="warning"
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not srv.started:
        if time.time() > deadline:
            raise RuntimeError("サーバーが起動しませんでした")
        time.sleep(0.05)
    yield BASE
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="session")
def pw() -> Iterator[Playwright]:
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session", params=["chromium", "webkit"])
def browser(request: pytest.FixtureRequest, pw: Playwright) -> Iterator[Browser]:
    b = getattr(pw, request.param).launch()
    yield b
    b.close()


@pytest.fixture
def page(browser: Browser, server: str) -> Iterator[Page]:
    context = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ja-JP")
    pg = context.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    yield pg
    context.close()
    assert errors == [], errors
