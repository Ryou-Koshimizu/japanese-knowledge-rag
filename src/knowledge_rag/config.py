"""環境変数からの設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    mode: str  # demo（記録の再生）/ live（実AI）
    auth: str  # demo / token
    corpus_dir: Path
    db_path: str
    recordings: Path
    people_file: Path
    answerer: str  # claude / codex（live のとき）
    ask_per_minute: int

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.environ.get("RAG_MODE", "demo")
        if mode not in ("demo", "live"):
            raise ValueError("RAG_MODE は demo か live です")
        auth = os.environ.get("RAG_AUTH", "demo")
        if auth not in ("demo", "token"):
            raise ValueError("RAG_AUTH は demo か token です")
        corpus = Path(os.environ.get("RAG_CORPUS_DIR", ROOT / "corpus"))
        default_people = corpus / ("_personas.yaml" if auth == "demo" else "_users.yaml")
        return cls(
            mode=mode,
            auth=auth,
            corpus_dir=corpus,
            db_path=os.environ.get(
                "RAG_DB_PATH", ":memory:" if mode == "demo" else "data/index.db"
            ),
            recordings=Path(os.environ.get("RAG_RECORDINGS", ROOT / "eval" / "recordings.json")),
            people_file=Path(os.environ.get("RAG_PEOPLE_FILE", default_people)),
            answerer=os.environ.get("RAG_ANSWERER", "claude"),
            ask_per_minute=int(os.environ.get("RAG_ASK_PER_MINUTE", "20")),
        )
