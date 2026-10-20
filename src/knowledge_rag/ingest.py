"""差分取り込み。中身（と権限の情報）が変わった文書だけを分割・埋め込みし直す。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_document, search_text
from .embeddings import Embedder
from .loaders import DocumentError, iter_corpus, load_document
from .models import ParsedDocument
from .store import Store
from .text import normalize

PIPELINE_VERSION = "4"  # 分割・正規化の手順を変えたら上げる（全件を作り直す）


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chunks_written: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def _fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ingest_directory(store: Store, root: Path, embedder: Embedder) -> IngestReport:
    report = IngestReport()
    signature = f"{PIPELINE_VERSION}:{embedder.id}"
    rebuild = store.get_meta("pipeline") != signature
    report.rebuilt = rebuild and bool(store.documents())
    existing = {} if rebuild else store.documents()

    parsed: list[tuple[ParsedDocument, str]] = []
    seen_ids: dict[str, str] = {}
    failed_paths: set[str] = set()
    for path in iter_corpus(root):
        rel = str(path.relative_to(root))
        try:
            doc, raw = load_document(path)
        except DocumentError as exc:
            report.errors.append(str(exc))
            failed_paths.add(rel)
            continue
        if doc.meta.id in seen_ids:
            report.errors.append(f"{rel}: id「{doc.meta.id}」が {seen_ids[doc.meta.id]} と重複")
            continue
        seen_ids[doc.meta.id] = rel
        parsed.append((doc, _fingerprint(raw)))

    with store.transaction() as conn:
        if rebuild:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM doc_access")
            conn.execute("DELETE FROM documents")
        for doc, fingerprint in parsed:
            before = existing.get(doc.meta.id)
            if before is not None and before.fingerprint == fingerprint:
                report.unchanged.append(doc.meta.id)
                continue
            chunks = chunk_document(doc)
            texts = [normalize(search_text(doc.meta.title, c)) for c in chunks]
            vectors = embedder.embed_documents(texts)
            store.put_document(
                conn,
                meta=doc.meta,
                source_path=str(Path(doc.source_path).relative_to(root)),
                fmt=doc.format,
                fingerprint=fingerprint,
                rows=list(zip(chunks, texts, vectors, strict=True)),
            )
            report.chunks_written += len(chunks)
            (report.updated if before is not None else report.added).append(doc.meta.id)
        # 読めなかったファイルの文書は消さない（壊れた上書きで既存の索引を失わないため）
        for doc_id, stored in existing.items():
            if doc_id not in seen_ids and stored.source_path not in failed_paths:
                store.delete_document(conn, doc_id)
                report.removed.append(doc_id)
        store.set_meta(conn, "pipeline", signature)
    return report
