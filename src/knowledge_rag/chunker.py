"""見出しの単位を崩さずに分割する。

1つの節（条・項目）は原則そのまま1チャンク。長い節だけ、行・文の切れ目で分け、
前のかたまりの最後の1文を次の頭に重ねる（文脈が切れないように）。
"""

from __future__ import annotations

import re

from .models import Chunk, ParsedDocument

MAX_CHARS = 700
_SENTENCE_END = re.compile(r"(?<=[。！？])")


def _units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if len(line) <= MAX_CHARS:
            units.append(line)
        else:
            units.extend(s for s in _SENTENCE_END.split(line) if s)
    return units


def split_section(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in _units(text):
        if current and size + len(unit) + 1 > max_chars:
            pieces.append("\n".join(current))
            overlap = current[-1] if len(current[-1]) < max_chars // 3 else ""
            current = [overlap] if overlap else []
            size = len(overlap)
        current.append(unit)
        size += len(unit) + 1
    if current:
        pieces.append("\n".join(current))
    return [p.strip() for p in pieces if p.strip()]


def chunk_document(doc: ParsedDocument, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        for piece in split_section(section.text, max_chars):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc.meta.id}#{ordinal}",
                    doc_id=doc.meta.id,
                    ordinal=ordinal,
                    heading_path=section.heading_path,
                    body=piece,
                )
            )
    return chunks


def search_text(title: str, chunk: Chunk) -> str:
    """索引に入れる文字列。文書名と見出しを本文の前に付け、見出しの語でも当たるようにする。"""
    return "\n".join([title, *chunk.heading_path, chunk.body])
