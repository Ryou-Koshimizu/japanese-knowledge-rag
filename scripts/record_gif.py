"""README 用の GIF と画面写真を、実ブラウザ（Playwright）で撮る。

    uv run uvicorn knowledge_rag.asgi:app --port 8104 &
    uv run python scripts/record_gif.py

連続で画面を撮り、撮った時刻の間隔をそのままコマ送りの長さにする。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
BASE = "http://127.0.0.1:8104"
SIZE = {"width": 1100, "height": 690}
OUT_WIDTH = 880
LEAVE = "使わなかった有給って次の年に持ち越せますか？最大で何日まで持てますか？"


class Recorder:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.frames: list[tuple[float, Image.Image]] = []

    def grab(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            png = self.page.screenshot(caret="initial", animations="allow")
            image = Image.open(io.BytesIO(png)).convert("RGB")
            ratio = OUT_WIDTH / image.width
            image = image.resize((OUT_WIDTH, round(image.height * ratio)), Image.Resampling.LANCZOS)
            self.frames.append((time.monotonic(), image))

    def save(self, path: Path) -> None:
        durations = [
            max(40, round((b[0] - a[0]) * 1000)) for a, b in zip(self.frames, self.frames[1:])
        ]
        durations.append(1800)
        images = [f[1].quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in self.frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=1,
        )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=1, locale="ja-JP")
        rec = Recorder(page)
        page.goto(BASE + "/")
        rec.grab(2.2)
        page.click("#question")
        page.keyboard.type(LEAVE, delay=45)
        rec.grab(0.4)
        page.keyboard.press("Enter")
        rec.grab(4.2)
        page.mouse.wheel(0, 260)
        rec.grab(1.2)
        page.locator(".open-doc").first.click()
        rec.grab(2.4)
        page.keyboard.press("Escape")
        rec.grab(0.8)
        page.fill("#question", "冬のボーナスって基本給の何か月分出るんですか？")
        page.keyboard.press("Enter")
        rec.grab(3.0)
        rec.save(ASSETS / "demo.gif")
        browser.close()

    with sync_playwright() as p:
        for engine, viewport, name in (
            (p.chromium, {"width": 1440, "height": 900}, "answer-desktop.png"),
            (p.webkit, {"width": 390, "height": 844}, "answer-mobile.png"),
        ):
            browser = engine.launch()
            page = browser.new_page(viewport=viewport, device_scale_factor=2, locale="ja-JP")
            page.goto(BASE + "/")
            page.wait_for_timeout(1500)
            page.fill("#question", LEAVE)
            page.keyboard.press("Enter")
            page.wait_for_timeout(4200)
            page.locator("#answer").screenshot(path=str(ASSETS / name), caret="initial")
            browser.close()
    print("docs/assets に書きました")


if __name__ == "__main__":
    main()
