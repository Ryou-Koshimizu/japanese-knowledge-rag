"""検索 → AI → 引用の照合までをまとめる。

AIの返した引用は、渡した資料の本文と一字一句照合する。合わない引用は捨て、
根拠が1つも残らない回答は「規程に記載がありません」に置き換える（捏造を表に出さない）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import NOT_FOUND_MESSAGE, Hit, Principal
from .prompt import build_user_message, source_label
from .providers import Answerer, Generation
from .search import Searcher
from .store import Store
from .text import compact

CONTEXT_SIZE = 8
_MARKER = re.compile(r"\[\s*(S\d+(?:\s*[,、]\s*S\d+)*)\s*\]")
_URL = re.compile(r"https?://[^\s)）」』\]]+")


@dataclass
class ResolvedCitation:
    number: int
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    quote: str


@dataclass
class AnswerResult:
    question: str
    found: bool
    answer: str
    citations: list[ResolvedCitation]
    hits: list[Hit]
    model: str | None
    replayed_at: str | None = None
    rejected: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def not_found(question: str, hits: list[Hit], gen: Generation | None, **kw: object) -> AnswerResult:
    return AnswerResult(
        question=question,
        found=False,
        answer=NOT_FOUND_MESSAGE,
        citations=[],
        hits=hits,
        model=gen.model if gen else None,
        replayed_at=gen.replayed_at if gen else None,
        input_tokens=gen.input_tokens if gen else 0,
        output_tokens=gen.output_tokens if gen else 0,
        seconds=gen.seconds if gen else 0.0,
        **kw,  # type: ignore[arg-type]
    )


def ground(question: str, context: list[Hit], gen: Generation, hits: list[Hit]) -> AnswerResult:
    """AIの回答を資料と照合し、表に出せる形にする。"""
    by_label = {source_label(i): hit for i, hit in enumerate(context)}
    rejected: list[str] = []
    valid: dict[str, list[str]] = {}
    for citation in gen.answer.citations:
        label = citation.source.strip().upper()
        hit = by_label.get(label)
        if hit is None:
            rejected.append(f"存在しない資料番号 {citation.source}")
            continue
        quote = citation.quote.strip()
        if len(compact(quote)) < 4 or compact(quote) not in compact(hit.body):
            rejected.append(f"{label} の本文と一致しない引用")
            continue
        valid.setdefault(label, [])
        if quote not in valid[label]:
            valid[label].append(quote)

    if not gen.answer.found or not valid:
        if gen.answer.found:
            rejected.append("根拠を本文で確認できる引用がない")
        return not_found(question, hits, gen, rejected=rejected)

    numbers: dict[str, int] = {}

    def renumber(match: re.Match[str]) -> str:
        labels = [s.strip().upper() for s in re.split(r"[,、]", match.group(1))]
        out = []
        for label in labels:
            if label in valid:
                numbers.setdefault(label, len(numbers) + 1)
                out.append(f"[{numbers[label]}]")
        return "".join(out)

    text = _MARKER.sub(renumber, gen.answer.answer).strip()
    for label in valid:  # 本文で触れられなかった根拠も番号を振る
        numbers.setdefault(label, len(numbers) + 1)

    allowed_text = "".join(by_label[label].body for label in valid)
    for url in _URL.findall(text):
        if url not in allowed_text:
            text = text.replace(url, "")
            rejected.append("根拠にないURLを除いた")

    citations = [
        ResolvedCitation(
            number=numbers[label],
            chunk_id=by_label[label].chunk_id,
            doc_id=by_label[label].doc_id,
            doc_title=by_label[label].doc_title,
            heading=by_label[label].heading,
            quote=quote,
        )
        for label in sorted(valid, key=lambda lb: numbers[lb])
        for quote in valid[label]
    ]
    return AnswerResult(
        question=question,
        found=True,
        answer=text,
        citations=citations,
        hits=hits,
        model=gen.model,
        replayed_at=gen.replayed_at,
        rejected=rejected,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        seconds=gen.seconds,
    )


class AnswerService:
    def __init__(self, store: Store, searcher: Searcher, answerer: Answerer) -> None:
        self.store = store
        self.searcher = searcher
        self.answerer = answerer

    def retrieve(self, question: str, principal: Principal) -> list[Hit]:
        return self.searcher.search(question, principal, top_k=8)

    def answer(self, question: str, principal: Principal) -> AnswerResult:
        hits = self.retrieve(question, principal)
        context = hits[:CONTEXT_SIZE]
        if not context:
            return not_found(question, hits, None)
        message = build_user_message(question, context)
        gen = self.answerer.generate(question, message, principal.id)
        if gen.context_ids is not None:
            # 記録の再生：記録したときに渡した資料で照合する（権限で絞った上で）
            context = self.store.hits_by_chunk_ids(list(gen.context_ids), principal)
        return ground(question, context, gen, hits)
