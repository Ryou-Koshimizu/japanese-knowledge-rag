"""誰が聞いているか（部署・役職）を決める部分。

- demo：公開デモ用。架空の社員の中から画面で選ぶ（ヘッダー X-Demo-Persona）。
- token：実運用の最小構成。利用者ごとのトークン（SHA-256 のハッシュだけを保存）で識別する。
  社内のシングルサインオンにつなぐ場合は、この `PrincipalResolver` を差し替える。
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from .models import Principal, Role


class Unauthorized(Exception):
    pass


class PrincipalResolver(Protocol):
    def resolve(self, headers: dict[str, str]) -> Principal: ...


def load_people(path: Path) -> list[tuple[Principal, dict[str, str]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    people = []
    for item in data:
        principal = Principal(
            id=str(item["id"]),
            name=str(item["name"]),
            department=str(item["department"]),
            role=Role.from_label(str(item["role"])),
        )
        people.append((principal, {k: str(v) for k, v in item.items()}))
    return people


@dataclass
class DemoResolver:
    personas: list[Principal]

    @classmethod
    def from_file(cls, path: Path) -> DemoResolver:
        return cls([p for p, _ in load_people(path)])

    def resolve(self, headers: dict[str, str]) -> Principal:
        wanted = headers.get("x-demo-persona", "")
        for persona in self.personas:
            if persona.id == wanted:
                return persona
        raise Unauthorized("閲覧者を選んでください")


@dataclass
class TokenResolver:
    users: list[tuple[Principal, str]]

    @classmethod
    def from_file(cls, path: Path) -> TokenResolver:
        return cls([(p, extra["token_sha256"]) for p, extra in load_people(path)])

    def resolve(self, headers: dict[str, str]) -> Principal:
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise Unauthorized("認証が必要です")
        digest = hashlib.sha256(auth[7:].strip().encode()).hexdigest()
        for principal, expected in self.users:
            if hmac.compare_digest(digest, expected):
                return principal
        raise Unauthorized("認証が必要です")
