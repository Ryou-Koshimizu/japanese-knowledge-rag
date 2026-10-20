"""実機の通し確認：実AI（Codex 経由）で動くサーバーを、実ブラウザ（Chromium と WebKit）で操作する。

    uv run python scripts/live_check.py

サーバーは RAG_MODE=live（実AI）・閲覧者を選べる設定で 8104 番に立てる。質問の種類ごとに
評価セットから選んだ質問を画面から入力し、画面が受け取った回答を評価と同じ基準で採点する。
結果は docs/evidence/live-browser.json に保存する。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_rag.answer import AnswerResult, ResolvedCitation  # noqa: E402
from knowledge_rag.evaluation import Question, grade_answer, load_questions  # noqa: E402

BASE = "http://127.0.0.1:8104"
PICK = [
    "q-hr-annual-leave-2",  # 答えがある（Markdown）
    "q-hr-childcare-leave-1",  # 答えがある（PDF）
    "q-fin-travel-1",  # 答えがある（Word）
    "q-injection-it-system-guide-1",  # 文書の中に乗っ取りの文
    "q-absent-12",  # 近い話題はあるが書かれていない
    "q-restricted-hr-payroll-ops-1",  # 読めない文書にある
    "q-attack-1",  # 質問からの乗っ取り
]


def wait_for_server() -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=2):  # noqa: S310
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("サーバーが起動しませんでした")


def to_result(body: dict[str, Any]) -> AnswerResult:
    return AnswerResult(
        question=body["question"],
        found=body["found"],
        answer=body["answer"],
        citations=[ResolvedCitation(**c) for c in body["citations"]],
        hits=[],
        model=body["model"],
        rejected=body["rejected"],
    )


def ask(page: Page, persona: str, question: str) -> dict[str, Any]:
    page.evaluate(f"() => localStorage.setItem('persona', {json.dumps(persona)})")
    page.goto(BASE + "/")
    page.wait_for_selector(".chip, #suggest-list", state="attached")
    page.fill("#question", question)
    with page.expect_response(lambda r: r.url.endswith("/api/ask"), timeout=240_000) as info:
        page.click(".ask-submit")
    body: dict[str, Any] = info.value.json()
    page.wait_for_selector("#answer:not([hidden])", timeout=10_000)
    shown = page.get_attribute("#answer-body", "aria-label") or ""
    body["shown_on_screen"] = shown
    if body.get("found"):
        page.locator(".open-doc").first.click()
        page.wait_for_selector(".doc-section.cited", timeout=10_000)
        body["sheet_cited_section"] = page.inner_text(".doc-section.cited h3")
        page.keyboard.press("Escape")
    return body


def check_one(page: Page, q: Question, engine: str, errors: list[str]) -> dict[str, Any]:
    body = ask(page, q.persona, q.question)
    grade = grade_answer(q, to_result(body))
    expected = re.sub(r"\[(\d+)\]", r"（根拠\1）", body["answer"])
    screen_ok = body["shown_on_screen"] == expected
    problems = grade.problems + ([] if screen_ok else ["画面の表示が違う"]) + errors
    return {
        "browser": engine,
        "id": q.id,
        "type": q.type,
        "persona": q.persona,
        "ok": not problems,
        "problems": problems,
        "answer": body["answer"],
        "citations": [{"doc_id": c["doc_id"], "heading": c["heading"]} for c in body["citations"]],
        "sheet_cited_section": body.get("sheet_cited_section"),
        "seconds": body["seconds"],
        "model": body["model"],
    }


def main() -> int:
    questions = {q.id: q for q in load_questions(ROOT / "eval" / "questions.yaml")}
    env = {
        **os.environ,
        "RAG_MODE": "live",
        "RAG_AUTH": "demo",
        "RAG_ANSWERER": "codex",
        "RAG_EMBEDDER": "local",
        "RAG_DB_PATH": ":memory:",
        "RAG_ASK_PER_MINUTE": "100",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "knowledge_rag.asgi:app", "--port", "8104"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rows: list[dict[str, Any]] = []
    try:
        wait_for_server()
        with sync_playwright() as p:
            for engine in ("chromium", "webkit"):
                browser = getattr(p, engine).launch()
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))  # noqa: B023
                page.goto(BASE + "/")
                for qid in PICK:
                    row = check_one(page, questions[qid], engine, errors)
                    rows.append(row)
                    print(f"{'OK' if row['ok'] else 'NG'} {engine} {qid} {row['seconds']}s")
                if errors:
                    rows.append({"browser": engine, "id": "page-errors", "ok": False})
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)
    out = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": "RAG_MODE=live RAG_ANSWERER=codex RAG_EMBEDDER=local（実AI・実ブラウザ）",
        "checks": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "rows": rows,
    }
    path = ROOT / "docs" / "evidence" / "live-browser.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out['ok']}/{out['checks']} 件が正しい")
    return 0 if out["ok"] == out["checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
