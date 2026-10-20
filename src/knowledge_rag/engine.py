"""部品の組み立て。API・評価・コマンドから同じ形で使う。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .answer import AnswerService
from .embeddings import Embedder, HashingEmbedder
from .ingest import IngestReport, ingest_directory
from .providers import Answerer, ReplayAnswerer
from .search import Glossary, Searcher
from .store import Store


@dataclass
class Engine:
    store: Store
    searcher: Searcher
    service: AnswerService
    report: IngestReport


def build_engine(
    corpus_dir: Path,
    answerer: Answerer,
    db_path: str = ":memory:",
    embedder: Embedder | None = None,
) -> Engine:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    embedder = embedder or HashingEmbedder()
    report = ingest_directory(store, corpus_dir, embedder)
    searcher = Searcher(store, embedder, Glossary.load(corpus_dir / "_glossary.yaml"))
    return Engine(store, searcher, AnswerService(store, searcher, answerer), report)


def replay_answerer(path: Path) -> ReplayAnswerer:
    return ReplayAnswerer(path)
