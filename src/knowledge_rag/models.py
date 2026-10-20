"""ドメインの型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field

ALL_DEPARTMENTS = "*"


class Role(IntEnum):
    """役職。数が大きいほど広く読める。"""

    STAFF = 1
    MANAGER = 2
    EXECUTIVE = 3

    @classmethod
    def from_label(cls, label: str) -> Role:
        try:
            return _ROLE_LABELS[label]
        except KeyError as exc:
            raise ValueError(f"未知の役職です: {label}") from exc

    @property
    def label(self) -> str:
        return {Role.STAFF: "一般社員", Role.MANAGER: "管理職", Role.EXECUTIVE: "役員"}[self]


_ROLE_LABELS = {"一般社員": Role.STAFF, "管理職": Role.MANAGER, "役員": Role.EXECUTIVE}


@dataclass(frozen=True)
class Principal:
    """検索・回答を求める人。部署と役職で読める文書が決まる。"""

    id: str
    name: str
    department: str
    role: Role


@dataclass(frozen=True)
class AccessRule:
    departments: tuple[str, ...]
    min_role: Role

    def allows(self, principal: Principal) -> bool:
        dept_ok = ALL_DEPARTMENTS in self.departments or principal.department in self.departments
        return dept_ok and principal.role >= self.min_role


@dataclass(frozen=True)
class DocumentMeta:
    id: str
    title: str
    access: AccessRule
    owner: str = ""
    updated: str = ""


@dataclass(frozen=True)
class Section:
    """見出しで区切られた本文。heading_path は文書の見出しから順にたどった道筋。"""

    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    meta: DocumentMeta
    source_path: str
    format: Literal["markdown", "pdf", "docx", "html"]
    sections: list[Section]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    body: str

    @property
    def heading(self) -> str:
        return " › ".join(self.heading_path)


@dataclass
class Hit:
    """検索で見つかった1か所。ranks は検索方式ごとの順位（1始まり）。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: tuple[str, ...]
    body: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return " › ".join(self.heading_path)


class Citation(BaseModel):
    source: str = Field(description="資料の番号（S1 など）")
    quote: str = Field(description="資料の本文から一字一句そのまま抜き出した文")


class ModelAnswer(BaseModel):
    """AIが返す回答の形。"""

    found: bool
    answer: str
    citations: list[Citation]


NOT_FOUND_MESSAGE = "規程に記載がありません。"
