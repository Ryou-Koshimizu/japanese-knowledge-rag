"""SQLite の保存先。全文索引（FTS5 trigram）・ベクトル・権限を1つのファイルに持つ。

権限は検索の SQL の段階で絞る。読めない文書のチャンクは、全文検索・語の一致・ベクトル
検索のどれにも候補として現れない。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .models import ALL_DEPARTMENTS, Chunk, DocumentMeta, Hit, Principal
from .text import normalize

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT NOT NULL,
  format TEXT NOT NULL,
  owner TEXT NOT NULL,
  updated TEXT NOT NULL,
  min_role INTEGER NOT NULL,
  fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_access (
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  department TEXT NOT NULL,
  PRIMARY KEY (doc_id, department)
);
CREATE TABLE IF NOT EXISTS chunks (
  rowid INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  heading_path TEXT NOT NULL,
  body TEXT NOT NULL,
  search_text TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  search_text, content='chunks', content_rowid='rowid', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, search_text)
  VALUES ('delete', old.rowid, old.search_text);
END;
"""

# 権限の条件。:role と :dept を受け取り、documents d に対して使う。
ACL_SQL = (
    "d.min_role <= :role AND EXISTS (SELECT 1 FROM doc_access a "
    "WHERE a.doc_id = d.id AND a.department IN (:all, :dept))"
)


class SqliteFeatureError(RuntimeError):
    pass


def check_sqlite() -> None:
    """FTS5 の trigram トークナイザ（SQLite 3.34 以上）が使えるかを確かめる。"""
    if sqlite3.sqlite_version_info < (3, 34, 0):
        raise SqliteFeatureError(
            f"SQLite {sqlite3.sqlite_version} では trigram 索引を作れません（3.34 以上が必要）"
        )
    with sqlite3.connect(":memory:") as probe:
        try:
            probe.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        except sqlite3.OperationalError as exc:
            raise SqliteFeatureError("この SQLite は FTS5 の trigram に対応していません") from exc


@dataclass(frozen=True)
class StoredDocument:
    id: str
    title: str
    format: str
    owner: str
    updated: str
    fingerprint: str
    source_path: str


@dataclass(frozen=True)
class StoredChunk:
    rowid: int
    chunk: Chunk
    doc_title: str
    search_text: str


def pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> array[float]:
    values: array[float] = array("f")
    values.frombytes(blob)
    return values


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        check_sqlite()
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._vector_cache: tuple[int, dict[int, array[float]]] | None = None
        self.generation = 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            finally:
                self.generation += 1
                self._vector_cache = None

    # --- メタ情報 ---
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # --- 文書 ---
    def documents(self) -> dict[str, StoredDocument]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, title, format, owner, updated, fingerprint, source_path FROM documents"
            ).fetchall()
        return {r["id"]: StoredDocument(**dict(r)) for r in rows}

    def delete_document(self, conn: sqlite3.Connection, doc_id: str) -> None:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM doc_access WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def put_document(
        self,
        conn: sqlite3.Connection,
        *,
        meta: DocumentMeta,
        source_path: str,
        fmt: str,
        fingerprint: str,
        rows: list[tuple[Chunk, str, list[float]]],
    ) -> None:
        self.delete_document(conn, meta.id)
        conn.execute(
            "INSERT INTO documents(id, title, source_path, format, owner, updated, min_role,"
            " fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                meta.id,
                meta.title,
                source_path,
                fmt,
                meta.owner,
                meta.updated,
                int(meta.access.min_role),
                fingerprint,
            ),
        )
        conn.executemany(
            "INSERT INTO doc_access(doc_id, department) VALUES (?, ?)",
            [(meta.id, d) for d in dict.fromkeys(meta.access.departments)],
        )
        conn.executemany(
            "INSERT INTO chunks(id, doc_id, ordinal, heading_path, body, search_text, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.id,
                    c.doc_id,
                    c.ordinal,
                    json.dumps(list(c.heading_path), ensure_ascii=False),
                    c.body,
                    text,
                    pack(vec),
                )
                for c, text, vec in rows
            ],
        )

    # --- 権限つきの読み出し ---
    @staticmethod
    def _acl(principal: Principal) -> dict[str, object]:
        return {"role": int(principal.role), "dept": principal.department, "all": ALL_DEPARTMENTS}

    def readable_documents(self, principal: Principal) -> list[StoredDocument]:
        sql = (
            "SELECT id, title, format, owner, updated, fingerprint, source_path FROM documents d "
            f"WHERE {ACL_SQL} ORDER BY title"
        )
        with self._lock:
            rows = self.conn.execute(sql, self._acl(principal)).fetchall()
        return [StoredDocument(**dict(r)) for r in rows]

    def _row_to_chunk(self, row: sqlite3.Row) -> StoredChunk:
        return StoredChunk(
            rowid=int(row["rowid"]),
            chunk=Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                ordinal=int(row["ordinal"]),
                heading_path=tuple(json.loads(row["heading_path"])),
                body=row["body"],
            ),
            doc_title=row["title"],
            search_text=row["search_text"],
        )

    _CHUNK_COLUMNS = (
        "c.rowid AS rowid, c.id AS id, c.doc_id AS doc_id, c.ordinal AS ordinal, "
        "c.heading_path AS heading_path, c.body AS body, c.search_text AS search_text, "
        "d.title AS title"
    )

    def chunks_by_rowid(self, rowids: list[int], principal: Principal) -> dict[int, StoredChunk]:
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        acl = ACL_SQL.replace(":role", "?").replace(":all", "?").replace(":dept", "?")
        sql = (
            f"SELECT {self._CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.rowid IN ({marks}) AND {acl}"
        )
        params = [*rowids, int(principal.role), ALL_DEPARTMENTS, principal.department]
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {int(r["rowid"]): self._row_to_chunk(r) for r in rows}

    def hits_by_chunk_ids(self, chunk_ids: list[str], principal: Principal) -> list[Hit]:
        """チャンクIDの順に、読めるものだけを Hit にして返す。"""
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        acl = ACL_SQL.replace(":role", "?").replace(":all", "?").replace(":dept", "?")
        sql = (
            f"SELECT {self._CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.id IN ({marks}) AND {acl}"
        )
        params = [*chunk_ids, int(principal.role), ALL_DEPARTMENTS, principal.department]
        with self._lock:
            rows = {r["id"]: self._row_to_chunk(r) for r in self.conn.execute(sql, params)}
        return [
            Hit(
                chunk_id=s.chunk.id,
                doc_id=s.chunk.doc_id,
                doc_title=s.doc_title,
                heading_path=s.chunk.heading_path,
                body=s.chunk.body,
                score=0.0,
            )
            for cid in chunk_ids
            if (s := rows.get(cid)) is not None
        ]

    def document_chunks(self, doc_id: str, principal: Principal) -> list[StoredChunk]:
        sql = (
            f"SELECT {self._CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.doc_id = :doc AND {ACL_SQL} ORDER BY c.ordinal"
        )
        with self._lock:
            rows = self.conn.execute(sql, {"doc": doc_id, **self._acl(principal)}).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def fts_search(self, match: str, principal: Principal, limit: int) -> list[tuple[int, float]]:
        """FTS5 の trigram 索引で検索し、(rowid, bm25) を良い順に返す。"""
        sql = (
            "SELECT c.rowid AS rowid, bm25(chunks_fts) AS score FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid JOIN documents d ON d.id = c.doc_id "
            f"WHERE chunks_fts MATCH :match AND {ACL_SQL} ORDER BY score LIMIT :limit"
        )
        with self._lock:
            rows = self.conn.execute(
                sql, {"match": match, "limit": limit, **self._acl(principal)}
            ).fetchall()
        return [(int(r["rowid"]), float(r["score"])) for r in rows]

    def term_candidates(
        self, terms: list[str], fts_match: str | None, principal: Principal
    ) -> list[tuple[int, str, str]]:
        """いずれかの語を含む、読めるチャンクの (rowid, 索引用の文字列, 文書名と見出し)。

        3文字以上の語は FTS5 の trigram 索引で引き、2文字以下の語は文字列の照合で拾う。
        """
        if not terms:
            return []
        params: dict[str, object] = {**self._acl(principal)}
        ors = []
        for i, term in enumerate(t for t in terms if len(t) < 3):
            params[f"t{i}"] = term
            ors.append(f"instr(c.search_text, :t{i}) > 0")
        if fts_match:
            params["match"] = fts_match
            ors.append("c.rowid IN (SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH :match)")
        if not ors:
            return []
        sql = (
            "SELECT c.rowid AS rowid, c.search_text AS search_text, "
            "d.title || ' ' || c.heading_path AS heading FROM chunks c "
            f"JOIN documents d ON d.id = c.doc_id WHERE {ACL_SQL} AND ({' OR '.join(ors)})"
        )
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [
            (int(r["rowid"]), str(r["search_text"]), normalize(str(r["heading"]))) for r in rows
        ]

    def count_readable_chunks(self, principal: Principal) -> int:
        sql = f"SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE {ACL_SQL}"
        with self._lock:
            return int(self.conn.execute(sql, self._acl(principal)).fetchone()[0])

    def _all_vectors(self) -> dict[int, array[float]]:
        with self._lock:
            cache = self._vector_cache
            if cache is None or cache[0] != self.generation:
                rows = self.conn.execute("SELECT rowid, embedding FROM chunks").fetchall()
                cache = (self.generation, {int(r[0]): unpack(r[1]) for r in rows})
                self._vector_cache = cache
            return cache[1]

    def readable_rowids(self, principal: Principal) -> set[int]:
        sql = f"SELECT c.rowid FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE {ACL_SQL}"
        with self._lock:
            return {int(r[0]) for r in self.conn.execute(sql, self._acl(principal))}

    def vector_search(
        self, query: list[float], principal: Principal, limit: int
    ) -> list[tuple[int, float]]:
        """読めるチャンクだけを対象に、内積（正規化済みなのでコサイン類似度）で並べる。"""
        allowed = self.readable_rowids(principal)
        vectors = self._all_vectors()
        scored = [
            (rowid, sum(q * v for q, v in zip(query, vectors[rowid], strict=True)))
            for rowid in allowed
            if rowid in vectors
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
