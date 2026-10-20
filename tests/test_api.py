from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_rag.app import create_app
from knowledge_rag.config import Settings
from knowledge_rag.engine import build_engine
from knowledge_rag.models import Citation, ModelAnswer
from knowledge_rag.providers import AnswerError, Generation, ReplayAnswerer

from .conftest import CORPUS, ROOT, ScriptedAnswerer, first_source_sentence


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "mode": "demo",
        "auth": "demo",
        "corpus_dir": CORPUS,
        "db_path": ":memory:",
        "recordings": ROOT / "eval" / "recordings.json",
        "people_file": CORPUS / "_personas.yaml",
        "answerer": "claude",
        "ask_per_minute": 20,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def demo() -> TestClient:
    return TestClient(create_app(settings()))


def as_(persona: str) -> dict[str, str]:
    return {"X-Demo-Persona": persona}


def test_meta_describes_demo(demo: TestClient) -> None:
    body = demo.get("/api/meta").json()
    assert body["mode"] == "demo" and body["documents"] == 32
    assert body["recorded_at"] and body["model"].startswith("codex:")
    assert {p["id"] for p in body["personas"]} >= {"sales-staff", "executive"}


def test_security_headers(demo: TestClient) -> None:
    response = demo.get("/")
    assert response.status_code == 200
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert demo.get("/api/meta").headers["cache-control"] == "no-store"


def test_requires_persona(demo: TestClient) -> None:
    assert demo.post("/api/search", json={"query": "経費"}).status_code == 401
    assert demo.post("/api/search", json={"query": "経費"}, headers=as_("ghost")).status_code == 401


def test_library_differs_by_persona(demo: TestClient) -> None:
    sales = demo.get("/api/library", headers=as_("sales-staff")).json()
    executive = demo.get("/api/library", headers=as_("executive")).json()
    sales_ids = {d["id"] for d in sales["documents"]}
    assert "sales-discount" in sales_ids and "exec-compensation" not in sales_ids
    assert "exec-compensation" in {d["id"] for d in executive["documents"]}
    assert sales["suggestions"]


def test_search_returns_ranked_hits(demo: TestClient) -> None:
    body = demo.post("/api/search", json={"query": "出張の日当"}, headers=as_("dev-manager")).json()
    assert body["hits"][0]["doc_id"] == "fin-travel"
    assert body["hits"][0]["ranks"]


def test_ask_replays_recorded_answer(demo: TestClient) -> None:
    question = demo.get("/api/library", headers=as_("sales-staff")).json()["suggestions"][0]
    body = demo.post("/api/ask", json={"question": question}, headers=as_("sales-staff")).json()
    assert body["status"] in ("answered", "not_found")
    assert body["replayed_at"] and body["hits"]


def test_ask_unrecorded_question_still_searches(demo: TestClient) -> None:
    body = demo.post(
        "/api/ask",
        json={"question": "出張の日当はいくら？（記録にない言い方）"},
        headers=as_("sales-staff"),
    ).json()
    assert body["status"] == "not_recorded"
    assert body["hits"] and body["suggestions"]


def test_document_endpoint_hides_unreadable(demo: TestClient) -> None:
    assert (
        demo.get("/api/documents/exec-compensation", headers=as_("sales-staff")).status_code == 404
    )
    assert demo.get("/api/documents/no-such-doc", headers=as_("sales-staff")).status_code == 404
    doc = demo.get("/api/documents/exec-compensation", headers=as_("executive")).json()
    assert doc["title"] and doc["sections"]


def test_input_validation(demo: TestClient) -> None:
    assert (
        demo.post("/api/ask", json={"question": ""}, headers=as_("sales-staff")).status_code == 422
    )
    long = "あ" * 401
    assert (
        demo.post("/api/ask", json={"question": long}, headers=as_("sales-staff")).status_code
        == 422
    )


def test_health(demo: TestClient) -> None:
    assert demo.get("/api/health").json() == {"ok": True, "documents": 32}


def test_static_assets_are_served(demo: TestClient) -> None:
    assert demo.get("/static/app.js").status_code == 200
    assert "i-search" in demo.get("/static/icons.svg").text


def live_client(answerer: object, tmp_path: Path, per_minute: int = 20) -> TestClient:
    users = tmp_path / "users.yaml"
    digest = hashlib.sha256(b"secret-token").hexdigest()
    users.write_text(
        "- id: u1\n  name: 営業の人\n  department: 営業部\n  role: 一般社員\n"
        f"  token_sha256: {digest}\n",
        encoding="utf-8",
    )
    engine = build_engine(CORPUS, answerer)  # type: ignore[arg-type]
    app = create_app(
        settings(mode="live", auth="token", people_file=users, ask_per_minute=per_minute), engine
    )
    return TestClient(app)


def test_live_mode_with_token_auth(tmp_path: Path) -> None:
    def reply(message: str) -> ModelAnswer:
        return ModelAnswer(
            found=True,
            answer="答え [S1]",
            citations=[Citation(source="S1", quote=first_source_sentence(message))],
        )

    client = live_client(ScriptedAnswerer(reply), tmp_path)
    assert client.post("/api/ask", json={"question": "日当"}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/api/ask", json={"question": "日当"}, headers=wrong).status_code == 401
    ok = {"Authorization": "Bearer secret-token"}
    body = client.post("/api/ask", json={"question": "出張の日当はいくら？"}, headers=ok).json()
    assert body["status"] == "answered" and body["replayed_at"] is None
    assert body["citations"][0]["number"] == 1
    assert client.get("/api/meta").json()["personas"] == []


def test_live_mode_rate_limit(tmp_path: Path) -> None:
    client = live_client(ScriptedAnswerer(), tmp_path, per_minute=2)
    ok = {"Authorization": "Bearer secret-token"}
    codes = [
        client.post("/api/ask", json={"question": "日当"}, headers=ok).status_code for _ in range(3)
    ]
    assert codes == [200, 200, 429]


def test_ai_failure_becomes_502(tmp_path: Path) -> None:
    class Broken:
        model = "broken"

        def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
            raise AnswerError("AIの呼び出しに失敗しました")

    client = live_client(Broken(), tmp_path)
    response = client.post(
        "/api/ask", json={"question": "日当"}, headers={"Authorization": "Bearer secret-token"}
    )
    assert response.status_code == 502 and "失敗" in response.json()["detail"]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MODE", "live")
    monkeypatch.setenv("RAG_AUTH", "token")
    s = Settings.from_env()
    assert s.db_path == "data/index.db" and s.people_file.name == "_users.yaml"
    monkeypatch.setenv("RAG_MODE", "bad")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("RAG_MODE", "demo")
    monkeypatch.setenv("RAG_AUTH", "bad")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_replay_recordings_cover_every_persona() -> None:
    replay = ReplayAnswerer(ROOT / "eval" / "recordings.json")
    personas = {e["persona"] for e in replay.entries.values()}
    assert personas == {
        "sales-staff",
        "hr-staff",
        "acct-manager",
        "it-staff",
        "dev-manager",
        "executive",
    }
