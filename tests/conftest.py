from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from knowledge_rag.auth import DemoResolver
from knowledge_rag.engine import Engine, build_engine
from knowledge_rag.models import ModelAnswer, Principal
from knowledge_rag.providers import Generation

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


@dataclass
class ScriptedAnswerer:
    """試験用の差し替え。渡された資料を記録し、決めた返事を返す（本番の画面では使わない）。"""

    reply: Callable[[str], ModelAnswer] = field(
        default=lambda message: ModelAnswer(found=False, answer="", citations=[])
    )
    messages: list[str] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "scripted"

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
        self.messages.append(user_message)
        return Generation(answer=self.reply(user_message), model=self.model)


def first_source_sentence(message: str) -> str:
    """資料 S1 の本文の最初の文（引用の試験用）。"""
    match = re.search(r"<source-[0-9a-f]+ [^>]*\"S1\"[^>]*>\n(.+?)\n</source-", message, re.S)
    assert match, message[:300]
    return match.group(1).split("。")[0] + "。"


@pytest.fixture(scope="session")
def personas() -> dict[str, Principal]:
    return {p.id: p for p in DemoResolver.from_file(CORPUS / "_personas.yaml").personas}


@pytest.fixture(scope="session")
def engine() -> Engine:
    return build_engine(CORPUS, ScriptedAnswerer())


@pytest.fixture
def corpus_copy(tmp_path: Path) -> Path:
    target = tmp_path / "corpus"
    shutil.copytree(CORPUS, target)
    return target
