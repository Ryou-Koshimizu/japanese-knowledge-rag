"""回答を作るAIの差し替え境界。

- `ClaudeAnswerer`：本番。Claude API（構造化出力）。モデルは RAG_ANSWER_MODEL で変更。
- `CodexAnswerer`：測定用。ChatGPT のログインで動く Codex CLI（読み取り専用・道具なし）。
- `ReplayAnswerer`：公開デモ。実AIで事前に記録した回答を再生する（再生であることを返す）。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .models import ModelAnswer
from .prompt import INSTRUCTIONS, SCHEMA
from .text import normalize


@dataclass(frozen=True)
class Generation:
    answer: ModelAnswer
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    replayed_at: str | None = None  # 記録の再生なら、記録した日時
    context_ids: tuple[str, ...] | None = None  # 記録の再生なら、記録時に渡した資料のチャンク


class AnswerError(RuntimeError):
    """AIの呼び出しに失敗した。メッセージは利用者に見せられる日本語。"""


class Answerer(Protocol):
    @property
    def model(self) -> str: ...

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation: ...


def parse_model_json(text: str) -> ModelAnswer:
    try:
        return ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise AnswerError("AIの応答を読み取れませんでした") from exc


class ClaudeAnswerer:
    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self._model = model or os.environ.get("RAG_ANSWER_MODEL", "claude-sonnet-5-5")
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        # 試験では同じ形の偽物を渡せるよう、型は緩めに持つ
        self.client: Any = client

    @property
    def model(self) -> str:
        return self._model

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
        import anthropic

        started = time.monotonic()
        try:
            response = self.client.beta.messages.create(
                model=self._model,
                max_tokens=4000,
                system=[
                    {"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": user_message}],
                output_config={
                    "effort": os.environ.get("RAG_ANSWER_EFFORT", "low"),
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                # 安全上の判定で断られたときは、サーバー側で既定の代わりのモデルが続きを答える
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise AnswerError("AIの利用上限に達しました。少し待って再度お試しください") from exc
        except anthropic.AuthenticationError as exc:
            raise AnswerError("Claude APIの鍵を確認してください") from exc
        except anthropic.APIError as exc:
            raise AnswerError("AIの呼び出しに失敗しました") from exc
        if response.stop_reason == "refusal":
            raise AnswerError("AIがこの質問への回答を控えました")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return Generation(
            answer=parse_model_json(text),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            seconds=time.monotonic() - started,
        )


_CODEX_ENV_KEYS = (
    "PATH",
    "HOME",
    "CODEX_HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)


def parse_codex_events(output: str) -> tuple[str, int, int]:
    """Codex CLI の --json の出力から、最終の応答と使ったトークン数だけを取り出す。"""
    message: str | None = None
    usage: tuple[int, int] | None = None
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind in ("error", "turn.failed"):
            raise AnswerError(f"Codex の推論に失敗しました: {str(event)[:200]}")
        if kind in ("item.started", "item.completed"):
            item = event.get("item") or {}
            if item.get("type") == "error":
                raise AnswerError("Codex の推論に失敗しました")
            if item.get("type") not in ("agent_message", "reasoning"):
                raise AnswerError(f"Codex が想定外の操作をしました: {item.get('type')}")
            if kind == "item.completed" and item.get("type") == "agent_message":
                message = item.get("text")
        if kind == "turn.completed":
            u = event.get("usage") or {}
            usage = (int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))
    if message is None or usage is None:
        raise AnswerError("Codex の応答が途中で終わりました")
    return message, usage[0], usage[1]


class CodexAnswerer:
    """測定用。ChatGPT のログインで動く Codex CLI を、道具も書き込みも使えない状態で呼ぶ。"""

    def __init__(self, model: str = "gpt-6.1-sol", effort: str = "low", timeout: int = 180):
        self._model = model
        self.effort = effort
        self.timeout = timeout

    @property
    def model(self) -> str:
        return f"codex:{self._model}"

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rag-codex-") as tmp:
            directory = Path(tmp)
            schema_path = directory / "schema.json"
            schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
            instructions_path = directory / "instructions.md"
            instructions_path.write_text(
                INSTRUCTIONS + "\n道具・ファイル・コマンド・ウェブ検索は使わず、JSON だけを返す。",
                encoding="utf-8",
            )
            env = {k: os.environ[k] for k in _CODEX_ENV_KEYS if k in os.environ}
            features = [
                "shell_tool",
                "apps",
                "plugins",
                "hooks",
                "multi_agent",
                "browser_use",
                "computer_use",
                "image_generation",
                "view_image",
            ]
            args = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(directory),
                "--model",
                self._model,
                "--json",
                "--output-schema",
                str(schema_path),
                "-c",
                f'model_reasoning_effort="{self.effort}"',
                "-c",
                f"model_instructions_file={json.dumps(str(instructions_path))}",
                *[x for f in features for x in ("-c", f"features.{f}=false")],
                "-c",
                'web_search="disabled"',
                "--",
                "-",
            ]
            proc = subprocess.Popen(  # noqa: S603 - 引数は固定で、利用者の入力は標準入力だけ
                args,
                cwd=directory,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                out, err = proc.communicate(user_message.encode("utf-8"), timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise AnswerError("Codex の応答が時間内に返りませんでした") from exc
            if proc.returncode != 0:
                text = out.decode("utf-8", "replace")
                errors = [
                    line for line in text.splitlines() if '"error' in line or "failed" in line
                ]
                detail = (errors[-1] if errors else err.decode("utf-8", "replace"))[-300:]
                raise AnswerError(f"Codex の実行に失敗しました: {detail}")
        message, tokens_in, tokens_out = parse_codex_events(out.decode("utf-8"))
        return Generation(
            answer=parse_model_json(message),
            model=self.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            seconds=time.monotonic() - started,
        )


def recording_key(principal_id: str, question: str) -> str:
    return f"{principal_id}␟{normalize(question)}"


class NotRecorded(LookupError):
    """公開デモで、記録にない質問が来た。"""


class ReplayAnswerer:
    """実AIで記録した回答を再生する。記録はモデル名・記録日時つき。"""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.recorded_model = str(data["model"])
        self.recorded_at = str(data["recorded_at"])
        self.entries: dict[str, dict[str, Any]] = {
            recording_key(e["persona"], e["question"]): e for e in data["entries"]
        }

    @property
    def model(self) -> str:
        return self.recorded_model

    def questions_for(self, principal_id: str) -> list[str]:
        return [e["question"] for e in self.entries.values() if e["persona"] == principal_id]

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
        entry = self.entries.get(recording_key(principal_id, question))
        if entry is None:
            raise NotRecorded(question)
        return Generation(
            answer=ModelAnswer.model_validate(entry["raw"]),
            model=self.recorded_model,
            input_tokens=int(entry.get("input_tokens", 0)),
            output_tokens=int(entry.get("output_tokens", 0)),
            seconds=float(entry.get("seconds", 0.0)),
            replayed_at=self.recorded_at,
            context_ids=tuple(entry["context"]),
        )
