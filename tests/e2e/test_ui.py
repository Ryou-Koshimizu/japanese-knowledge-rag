"""実ブラウザ（Chromium と WebKit）で画面を最初から最後まで操作する。"""

from __future__ import annotations

import pytest
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

WCAG = {"runOnly": {"type": "tag", "values": ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]}}


def axe_violations(page: Page) -> list[str]:
    results = Axe().run(page, options=WCAG)
    return [
        f"{v['id']}: {[n['target'] for n in v['nodes']]}" for v in results.response["violations"]
    ]


def open_home(page: Page, server: str) -> None:
    page.goto(server + "/")
    expect(page.locator("#persona-name")).not_to_have_text("—")
    expect(page.locator(".chip").first).to_be_visible()


def test_ask_cite_and_open_document(page: Page, server: str) -> None:
    open_home(page, server)
    page.fill("#question", "本社って何時から何時までが定時ですか？")
    page.click(".ask-submit")
    answer = page.locator("#answer")
    expect(answer).to_be_visible(timeout=10_000)
    expect(page.locator("#answer-question")).to_have_text("本社って何時から何時までが定時ですか？")
    expect(page.locator(".stamp")).to_contain_text("記録の再生")
    body = page.locator("#answer-body")
    expect(body).to_have_attribute("aria-label", r"(?s).*9:00.*18:00.*", timeout=5_000)
    expect(page.locator(".source").first).to_contain_text("就業規則")
    expect(page.locator(".hit").first).to_be_visible()
    page.wait_for_timeout(2_000)
    assert axe_violations(page) == []

    page.locator(".cite").first.click()
    expect(page.locator(".source").first).to_have_class("source flash")
    page.locator(".open-doc").first.click()
    sheet = page.locator("#sheet")
    expect(sheet).to_be_visible()
    expect(page.locator("#sheet-title")).to_have_text("就業規則")
    expect(page.locator(".doc-section.cited")).to_contain_text("9:00")
    expect(page.locator(".doc-section.cited mark")).to_be_visible()
    page.wait_for_timeout(1_200)
    assert axe_violations(page) == []
    page.keyboard.press("Escape")
    expect(sheet).to_be_hidden()


def test_not_found_answer_says_so(page: Page, server: str) -> None:
    open_home(page, server)
    page.fill("#question", "社員持株会の奨励金は拠出額の何％ですか？")
    page.keyboard.press("Enter")
    expect(page.locator("#answer-body")).to_have_attribute("aria-label", "規程に記載がありません。")
    expect(page.locator(".sources-empty")).to_be_visible()


def test_persona_switch_changes_readable_documents(page: Page, server: str) -> None:
    open_home(page, server)
    count = page.locator("#readable-count")
    expect(count).to_have_text("23", timeout=5_000)
    page.click("#persona-button")
    listbox = page.locator("#persona-list")
    expect(listbox).to_be_visible()
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    expect(page.locator("#persona-name")).to_have_text("経理部の管理職")
    expect(count).to_have_text("24", timeout=5_000)
    page.click("#persona-button")
    page.locator("#persona-executive").click()
    expect(page.locator("#persona-name")).to_have_text("経営企画室の役員")
    expect(count).to_have_text("23", timeout=5_000)


def test_restricted_question_does_not_leak(page: Page, server: str) -> None:
    open_home(page, server)
    page.fill("#question", "冬のボーナスって基本給の何か月分出るんですか？")
    page.click(".ask-submit")
    expect(page.locator("#answer-body")).to_have_attribute("aria-label", "規程に記載がありません。")
    hits = page.locator(".hit-title").all_inner_texts()
    assert all("給与・賞与の計算手順" not in h for h in hits)
    assert "2.35" not in page.content()


def test_unrecorded_question_shows_replay_notice(page: Page, server: str) -> None:
    open_home(page, server)
    page.fill("#question", "出張先でのタクシー代の扱いを教えて（記録なし）")
    page.click(".ask-submit")
    notice = page.locator(".notice")
    expect(notice).to_contain_text("実AIで記録した回答を再生")
    expect(page.locator(".hit").first).to_be_visible()
    notice.locator(".chip").first.click()
    expect(page.locator("#answer")).to_be_visible()


def test_hero_has_no_accessibility_violations(page: Page, server: str) -> None:
    open_home(page, server)
    page.wait_for_timeout(1_800)
    assert axe_violations(page) == []
    page.locator(".colophon").scroll_into_view_if_needed()
    expect(page.locator(".metric").first).to_be_visible()
    page.wait_for_timeout(1_500)
    assert axe_violations(page) == []


def test_mobile_flow(browser, server: str) -> None:  # type: ignore[no-untyped-def]
    context = browser.new_context(
        viewport={"width": 390, "height": 844},
        is_mobile=browser.browser_type.name != "firefox",
        has_touch=True,
    )
    page = context.new_page()
    try:
        open_home(page, server)
        page.locator(".chip").first.tap()
        expect(page.locator("#answer")).to_be_visible(timeout=10_000)
        page.wait_for_timeout(2_200)
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0
        assert axe_violations(page) == []
        page.locator(".open-doc").first.tap()
        expect(page.locator("#sheet")).to_be_visible()
        page.locator("#sheet-close").tap()
        expect(page.locator("#sheet")).to_be_hidden()
    finally:
        context.close()
