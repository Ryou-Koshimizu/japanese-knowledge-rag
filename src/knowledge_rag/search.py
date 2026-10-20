"""日本語のハイブリッド検索。

3つの順位を Reciprocal Rank Fusion（RRF）で1つにまとめる。
- trigram：FTS5 の trigram 索引による部分一致（3文字以上の語を3文字ずつに分けて照合）
- 語の一致：2文字以上の語がそのまま含まれるか（「残業」「賞与」など2文字の語を拾う）
- ベクトル：埋め込みの近さ（差し替え可能）
質問の語は社内用語の言い換え辞書で広げる（「育休」→「育児休業」など）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .embeddings import Embedder
from .models import Hit, Principal
from .store import Store
from .text import dedupe, normalize, query_terms, trigrams

RRF_K = 60
# 方式ごとの重み。語の一致と見出しの一致は日本語の規程で当たりやすく、
# 字面のハッシュ埋め込みは補助にとどめる（docs/adr/002 と docs/evidence/retrieval.json）。
DEFAULT_WEIGHTS: dict[str, float] = {"terms": 1.0, "heading": 0.25, "trigram": 0.25}


@dataclass(frozen=True)
class Glossary:
    """社内用語の言い換え。1行が同じ意味の語のまとまり。"""

    groups: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def load(cls, path: Path) -> Glossary:
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        groups = tuple(tuple(normalize(str(t)) for t in group) for group in data if group)
        return cls(groups)

    def expand(self, query: str) -> list[str]:
        q = normalize(query)
        extra: list[str] = []
        for group in self.groups:
            if any(term in q for term in group):
                extra.extend(t for t in group if t not in q)
        return extra


@dataclass(frozen=True)
class QueryPlan:
    terms: list[str]
    fts_match: str | None

    @classmethod
    def build(cls, query: str, glossary: Glossary) -> QueryPlan:
        terms = dedupe([*query_terms(query), *glossary.expand(query)])
        grams = dedupe(g for t in terms for g in trigrams(t))
        match = " OR ".join('"' + g.replace('"', '""') + '"' for g in grams) or None
        return cls(terms, match)


def _bm25_terms(
    candidates: list[tuple[int, str]], terms: list[str], total: int
) -> list[tuple[int, float]]:
    """語の一致を BM25 の式で採点する（k1=1.2, b=0.75）。"""
    if not candidates:
        return []
    k1, b = 1.2, 0.75
    avg_len = sum(len(text) for _, text in candidates) / len(candidates)
    df = {t: sum(1 for _, text in candidates if t in text) for t in terms}
    scored: list[tuple[int, float]] = []
    for rowid, text in candidates:
        score = 0.0
        for term in terms:
            tf = text.count(term)
            if not tf:
                continue
            idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(text) / avg_len))
        scored.append((rowid, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


class Searcher:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        glossary: Glossary | None = None,
        depth: int = 50,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.glossary = glossary or Glossary()
        self.depth = depth

    def rankings(self, query: str, principal: Principal) -> dict[str, list[int]]:
        """方式ごとの順位（rowid の並び）。どれも権限で絞った後の結果。"""
        plan = QueryPlan.build(query, self.glossary)
        result: dict[str, list[int]] = {}
        result["trigram"] = (
            [r for r, _ in self.store.fts_search(plan.fts_match, principal, self.depth)]
            if plan.fts_match
            else []
        )
        candidates = self.store.term_candidates(plan.terms, plan.fts_match, principal)
        total = self.store.count_readable_chunks(principal)
        bodies = [(rowid, text) for rowid, text, _ in candidates]
        headings = [(rowid, heading) for rowid, _, heading in candidates]
        result["terms"] = [r for r, s in _bm25_terms(bodies, plan.terms, total) if s > 0][
            : self.depth
        ]
        result["heading"] = [r for r, s in _bm25_terms(headings, plan.terms, total) if s > 0][
            : self.depth
        ]
        vector = self.embedder.embed_query(normalize(query))
        result["vector"] = [r for r, _ in self.store.vector_search(vector, principal, self.depth)]
        return result

    def search(
        self,
        query: str,
        principal: Principal,
        top_k: int = 8,
        methods: tuple[str, ...] = ("terms", "heading", "trigram", "vector"),
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        rankings = self.rankings(query, principal)
        weights = weights or {**DEFAULT_WEIGHTS, "vector": self.embedder.fusion_weight}
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for method in methods:
            weight = weights.get(method, 1.0) if len(methods) > 1 else 1.0
            for position, rowid in enumerate(rankings.get(method, []), start=1):
                fused[rowid] = fused.get(rowid, 0.0) + weight / (RRF_K + position)
                ranks.setdefault(rowid, {})[method] = position
        ordered = sorted(fused, key=lambda r: (-fused[r], r))[:top_k]
        chunks = self.store.chunks_by_rowid(ordered, principal)
        hits: list[Hit] = []
        for rowid in ordered:
            stored = chunks.get(rowid)
            if stored is None:  # 権限で絞った SQL を通らなかったものは出さない
                continue
            hits.append(
                Hit(
                    chunk_id=stored.chunk.id,
                    doc_id=stored.chunk.doc_id,
                    doc_title=stored.doc_title,
                    heading_path=stored.chunk.heading_path,
                    body=stored.chunk.body,
                    score=fused[rowid],
                    ranks=ranks[rowid],
                )
            )
        return hits
