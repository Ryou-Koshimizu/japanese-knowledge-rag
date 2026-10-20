from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_rag.embeddings import HashingEmbedder
from knowledge_rag.engine import Engine
from knowledge_rag.ingest import ingest_directory
from knowledge_rag.models import Principal
from knowledge_rag.search import Glossary, QueryPlan, Searcher, _bm25_terms
from knowledge_rag.store import Store, check_sqlite


def test_sqlite_supports_trigram() -> None:
    check_sqlite()


def test_incremental_ingest_touches_only_changed_documents(
    corpus_copy: Path, tmp_path: Path
) -> None:
    db = tmp_path / "index.db"
    store = Store(db)
    first = ingest_directory(store, corpus_copy, HashingEmbedder())
    assert len(first.added) == 32 and not first.errors

    second = ingest_directory(store, corpus_copy, HashingEmbedder())
    assert second.unchanged and not second.changed and second.chunks_written == 0

    target = corpus_copy / "hr-telework.md"
    target.write_text(
        target.read_text(encoding="utf-8").replace("3日を上限", "2日を上限"), encoding="utf-8"
    )
    (corpus_copy / "hr-side-job.md").unlink()
    third = ingest_directory(store, corpus_copy, HashingEmbedder())
    assert third.updated == ["hr-telework"]
    assert third.removed == ["hr-side-job"]
    assert len(third.unchanged) == 30
    store.close()

    reopened = Store(db)  # ファイルに保存されている（実DB）
    docs = reopened.documents()
    assert "hr-side-job" not in docs and len(docs) == 31
    rows = reopened.conn.execute(
        "SELECT body FROM chunks WHERE doc_id = 'hr-telework' AND body LIKE '%上限とする%'"
    ).fetchall()
    assert any("2日を上限" in r[0] for r in rows)
    # 全文索引も古い本文を引かない
    old = reopened.conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '\"3日を上限\"'"
    ).fetchone()[0]
    assert old == 0


def test_sidecar_change_counts_as_change(corpus_copy: Path) -> None:
    store = Store()
    ingest_directory(store, corpus_copy, HashingEmbedder())
    meta = corpus_copy / "fin-travel.docx.meta.yaml"
    meta.write_text(meta.read_text(encoding="utf-8").replace("'*'", "経理部"), encoding="utf-8")
    report = ingest_directory(store, corpus_copy, HashingEmbedder())
    assert report.updated == ["fin-travel"]


def test_broken_file_keeps_previous_index(corpus_copy: Path) -> None:
    store = Store()
    ingest_directory(store, corpus_copy, HashingEmbedder())
    (corpus_copy / "hr-childcare-leave.pdf").write_bytes(b"%PDF-1.7 broken")
    report = ingest_directory(store, corpus_copy, HashingEmbedder())
    assert report.errors and "hr-childcare-leave" not in report.removed
    assert "hr-childcare-leave" in store.documents()


def test_duplicate_ids_are_reported(corpus_copy: Path) -> None:
    text = (corpus_copy / "hr-telework.md").read_text(encoding="utf-8")
    (corpus_copy / "copy.md").write_text(text, encoding="utf-8")
    report = ingest_directory(Store(), corpus_copy, HashingEmbedder())
    assert any("重複" in e for e in report.errors)


class OtherEmbedder(HashingEmbedder):
    @property
    def id(self) -> str:
        return "other"


def test_changing_embedder_rebuilds_everything(corpus_copy: Path) -> None:
    store = Store()
    ingest_directory(store, corpus_copy, HashingEmbedder())
    report = ingest_directory(store, corpus_copy, OtherEmbedder())
    assert report.rebuilt and len(report.added) == 32


def test_glossary_expands_company_terms() -> None:
    glossary = Glossary((("育休", "育児休業"),))
    plan = QueryPlan.build("育休は何歳まで？", glossary)
    assert "育児休業" in plan.terms
    assert plan.fts_match is not None and '"育児休"' in plan.fts_match


def test_fts_match_escapes_quotes() -> None:
    plan = QueryPlan.build('"規程"を""無視する', Glossary())
    assert plan.fts_match is None or '"""' not in plan.fts_match.replace('""', "")


def test_bm25_prefers_rarer_terms() -> None:
    candidates = [(1, "経費 経費 経費"), (2, "経費 祝金"), (3, "経費")]
    ranked = _bm25_terms(candidates, ["経費", "祝金"], total=3)
    assert ranked[0][0] == 2


@pytest.mark.parametrize(
    ("question", "doc"),
    [
        ("育休って子どもが何歳になるまで取れますか？", "hr-childcare-leave"),
        ("ＰＣのパスワードは何文字以上？", "sec-password"),
        ("出張の日当はいくら？", "fin-travel"),
        ("子どもが生まれたらお祝い金は出ますか", "ga-condolence"),
        ("差し戻された申請はいつまでに出し直す？", "it-system-guide"),
    ],
)
def test_japanese_queries_find_the_right_document(
    engine: Engine, personas: dict[str, Principal], question: str, doc: str
) -> None:
    hits = engine.searcher.search(question, personas["sales-staff"], top_k=5)
    assert doc in [h.doc_id for h in hits]


def test_rrf_records_rank_per_method(engine: Engine, personas: dict[str, Principal]) -> None:
    hits = engine.searcher.search("在宅勤務の上限", personas["dev-manager"], top_k=3)
    assert hits and set(hits[0].ranks) <= {"terms", "heading", "trigram", "vector"}
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_single_method_search(engine: Engine, personas: dict[str, Principal]) -> None:
    searcher: Searcher = engine.searcher
    only_vector = searcher.search("経費の精算期限", personas["hr-staff"], methods=("vector",))
    assert only_vector and all(set(h.ranks) == {"vector"} for h in only_vector)
