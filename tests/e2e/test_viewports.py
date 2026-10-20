"""すべての画面の大きさで、横はみ出し0・ヘッダー80px・ヒーローの中身が最初の画面に収まることを確かめる。"""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, expect

pytestmark = pytest.mark.e2e

SIZES = [
    (280, 653), (320, 568), (360, 640), (375, 667), (390, 844), (393, 852), (412, 915),
    (430, 932), (568, 320), (667, 375), (844, 390), (932, 430), (600, 960), (768, 1024),
    (820, 1180), (1024, 768), (1024, 1366), (1180, 820), (1280, 720), (1366, 768),
    (1440, 900), (1536, 864), (1920, 1080), (2560, 1440), (3840, 2160),
]  # fmt: skip

MEASURE = """
() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const rect = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, left: r.left, right: r.right, width: r.width, height: r.height}; };
  const header = rect(document.querySelector('.masthead'));
  const hero = rect(document.querySelector('.hero'));
  const parts = ['.kicker', '.headline', '.deck', '#ask', '.suggest', '.hero-aside']
    .map((s) => [s, document.querySelector('.hero ' + s)])
    .filter(([, el]) => el && getComputedStyle(el).display !== 'none')
    .map(([s, el]) => [s, rect(el)]);
  const clipped = [...document.querySelectorAll('.headline .line, #persona-name, .brand-name, .deck, .kicker, .ask-submit, .ask input')]
    .filter((el) => getComputedStyle(el).display !== 'none' && el.scrollWidth > el.clientWidth + 1)
    .map((el) => el.className || el.id || el.tagName);
  const headerParts = ['.brand', '.persona-button', '.issue']
    .map((s) => document.querySelector(s))
    .filter((el) => el && getComputedStyle(el).display !== 'none')
    .map((el) => rect(el));
  const input = document.querySelector('#question');
  const ctx = document.createElement('canvas').getContext('2d');
  const cs = getComputedStyle(input);
  ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const placeholderOverflow = ctx.measureText(input.placeholder).width - input.clientWidth;
  return {vw, vh, placeholderOverflow, overflow: document.documentElement.scrollWidth - vw, header, hero, parts, clipped, headerParts};
}
"""


def overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    return (
        min(a["right"], b["right"]) - max(a["left"], b["left"]) > 1
        and min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]) > 1
    )


@pytest.mark.parametrize(("width", "height"), SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_first_view_fits(browser: Browser, server: str, width: int, height: int) -> None:
    context = browser.new_context(
        viewport={"width": width, "height": height},
        reduced_motion="reduce",
    )
    page = context.new_page()
    try:
        page.goto(server + "/")
        expect(page.locator("#persona-name")).not_to_have_text("—")
        page.evaluate("() => document.fonts.ready.then(() => true)")
        m = page.evaluate(MEASURE)
        assert m["overflow"] <= 0, f"横にはみ出し {m['overflow']}px"
        assert abs(m["header"]["height"] - 80) < 0.5, m["header"]
        assert abs(m["hero"]["height"] - (height - 80)) < 1, m["hero"]
        assert abs(m["hero"]["top"] - 80) < 0.5
        for name, r in m["parts"]:
            assert r["top"] >= 80 - 0.5 and r["bottom"] <= height + 0.5, f"{name} が画面外 {r}"
            assert r["left"] >= -0.5 and r["right"] <= width + 0.5, f"{name} が横にはみ出し {r}"
        parts = m["parts"]
        for i, (name_a, a) in enumerate(parts):
            for name_b, b in parts[i + 1 :]:
                assert not overlaps(a, b), f"{name_a} と {name_b} が重なっている"
        heads = m["headerParts"]
        for i, a in enumerate(heads):
            for b in heads[i + 1 :]:
                assert not overlaps(a, b), "ヘッダーの部品が重なっている"
            assert a["top"] >= 0 and a["bottom"] <= 80
        assert m["clipped"] == [], f"文字が欠けている: {m['clipped']}"
        assert m["placeholderOverflow"] <= 0, "質問欄の例文が欠けている"
    finally:
        context.close()
