from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import httpx2
import pytest

from knowledge_rag.embeddings import (
    HashingEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    embedder_from_env,
)
from knowledge_rag.providers import (
    AnswerError,
    ClaudeAnswerer,
    CodexAnswerer,
    NotRecorded,
    ReplayAnswerer,
    parse_codex_events,
    parse_model_json,
)

from .conftest import ROOT

ANSWER = {"found": True, "answer": "a [S1]", "citations": [{"source": "S1", "quote": "q"}]}


def events(*items: dict[str, Any]) -> str:
    return "\n".join(json.dumps(i) for i in items)


def test_parse_codex_events_returns_only_final_message() -> None:
    out = events(
        {"type": "thread.started"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "考え中"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(ANSWER)}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    )
    message, tokens_in, tokens_out = parse_codex_events(out)
    assert json.loads(message) == ANSWER and (tokens_in, tokens_out) == (10, 5)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (events({"type": "turn.failed", "error": "x"}), "失敗"),
        (events({"type": "item.started", "item": {"type": "command_execution"}}), "想定外"),
        (events({"type": "item.completed", "item": {"type": "error"}}), "失敗"),
        (events({"type": "thread.started"}), "途中で終わり"),
    ],
)
def test_parse_codex_events_rejects_failures_and_tool_use(payload: str, message: str) -> None:
    with pytest.raises(AnswerError, match=message):
        parse_codex_events(payload)


def test_parse_model_json_validates_shape() -> None:
    assert parse_model_json(json.dumps(ANSWER)).found
    with pytest.raises(AnswerError):
        parse_model_json('{"found": "maybe"}')
    with pytest.raises(AnswerError):
        parse_model_json("not json")


def _fake_codex(tmp_path: Path, script: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "codex"
    exe.write_text("#!/bin/sh\n" + script)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def test_codex_answerer_runs_the_cli_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "args.txt"
    reply = events(
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(ANSWER)}},
        {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}},
    )
    script = (
        f'printf "%s\\n" "$@" > {record}\ncat > {tmp_path}/stdin.txt\ncat <<"EOF"\n{reply}\nEOF\n'
    )
    bin_dir = _fake_codex(tmp_path, script)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    result = CodexAnswerer(model="gpt-test").generate("q", "資料と質問", "p")
    assert result.answer.found and result.model == "codex:gpt-test"
    args = record.read_text().splitlines()
    assert "read-only" in args and "--output-schema" in args and "features.shell_tool=false" in args
    assert (tmp_path / "stdin.txt").read_text() == "資料と質問"


def test_codex_answerer_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = _fake_codex(tmp_path, 'echo "unauthorized" >&2\nexit 3\n')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(AnswerError, match="失敗"):
        CodexAnswerer().generate("q", "m", "p")


def test_codex_answerer_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = _fake_codex(tmp_path, "sleep 5\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(AnswerError, match="時間内"):
        CodexAnswerer(timeout=1).generate("q", "m", "p")


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_client(messages: FakeMessages) -> Any:
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_claude_answerer_uses_structured_output() -> None:
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="text", text=json.dumps(ANSWER))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )
    messages = FakeMessages(response)
    result = ClaudeAnswerer(model="claude-sonnet-5-5", client=fake_client(messages)).generate(
        "q", "資料", "p"
    )
    assert result.answer.found and result.input_tokens == 100
    sent = messages.kwargs
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["fallbacks"] == "default"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_answerer_handles_refusal_and_errors() -> None:
    refusal = SimpleNamespace(stop_reason="refusal", content=[], model="m", usage=None)
    with pytest.raises(AnswerError, match="控えました"):
        ClaudeAnswerer(client=fake_client(FakeMessages(refusal))).generate("q", "m", "p")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError(
        "limit", response=httpx2.Response(429, request=request), body=None
    )
    with pytest.raises(AnswerError, match="上限"):
        ClaudeAnswerer(client=fake_client(FakeMessages(error=error))).generate("q", "m", "p")
    auth = anthropic.AuthenticationError(
        "auth", response=httpx2.Response(401, request=request), body=None
    )
    with pytest.raises(AnswerError, match="鍵"):
        ClaudeAnswerer(client=fake_client(FakeMessages(error=auth))).generate("q", "m", "p")


def test_claude_model_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ANSWER_MODEL", "claude-opus-5-5")
    assert ClaudeAnswerer(client=object()).model == "claude-opus-5-5"


def test_replay_answerer_returns_recorded_answers_with_timestamp() -> None:
    replay = ReplayAnswerer(ROOT / "eval" / "recordings.json")
    entry = next(iter(replay.entries.values()))
    result = replay.generate(entry["question"], "", entry["persona"])
    assert result.replayed_at == replay.recorded_at
    assert result.context_ids and result.model.startswith("codex:")
    # 全角・半角の違いは同じ質問として扱う
    assert replay.generate(entry["question"].replace("?", "？"), "", entry["persona"])
    with pytest.raises(NotRecorded):
        replay.generate("記録にない質問", "", entry["persona"])
    assert replay.questions_for(entry["persona"])


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed_documents(["育児休業の期間", "育児休業の期間"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1) < 1e-9
    near = sum(x * y for x, y in zip(emb.embed_query("育児休業"), a, strict=True))
    far = sum(x * y for x, y in zip(emb.embed_query("パスワード"), a, strict=True))
    assert near > far


def _transport(captured: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(
            {"url": str(request.url), "body": body, "auth": request.headers["authorization"]}
        )
        data = [{"index": i, "embedding": [1.0, float(i)]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("cls", "host"), [(VoyageEmbedder, "voyageai"), (OpenAIEmbedder, "openai")]
)
def test_api_embedders_batch_and_normalize(cls: type, host: str) -> None:
    captured: list[dict[str, Any]] = []
    emb = cls("key", "m", client=httpx.Client(transport=_transport(captured)))
    emb.batch_size = 2
    vectors = emb.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and len(captured) == 2
    assert host in captured[0]["url"] and captured[0]["auth"] == "Bearer key"
    assert abs(vectors[1][0] ** 2 + vectors[1][1] ** 2 - 1) < 1e-9
    if cls is VoyageEmbedder:
        assert captured[0]["body"]["input_type"] == "document"
        emb.embed_query("q")
        assert captured[-1]["body"]["input_type"] == "query"


def test_api_embedder_requires_key() -> None:
    with pytest.raises(ValueError, match="鍵"):
        VoyageEmbedder("", "m")


def test_embedder_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_EMBEDDER", raising=False)
    assert isinstance(embedder_from_env(), HashingEmbedder)
    monkeypatch.setenv("RAG_EMBEDDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    assert isinstance(embedder_from_env(), VoyageEmbedder)
    monkeypatch.setenv("RAG_EMBEDDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(embedder_from_env(), OpenAIEmbedder)
    monkeypatch.setenv("RAG_EMBEDDER", "nope")
    with pytest.raises(ValueError):
        embedder_from_env()


def test_local_multilingual_embedder_understands_meaning() -> None:
    emb = LocalEmbedder()
    doc = emb.embed_documents(["社員またはその配偶者が出産したときは、出産祝金を支給する。"])[0]
    related = sum(
        a * b
        for a, b in zip(emb.embed_query("子どもが生まれたらお祝い金は出る？"), doc, strict=True)
    )
    unrelated = sum(
        a * b for a, b in zip(emb.embed_query("パスワードは何文字以上？"), doc, strict=True)
    )
    assert related > unrelated
    assert emb.id.startswith("local:")
