"""FastAPI の API と画面の配信。"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from .answer import AnswerResult
from .auth import DemoResolver, PrincipalResolver, TokenResolver, Unauthorized
from .config import ROOT, Settings
from .embeddings import embedder_from_env
from .engine import Engine, build_engine
from .models import Hit, Principal
from .providers import (
    Answerer,
    AnswerError,
    ClaudeAnswerer,
    CodexAnswerer,
    NotRecorded,
    ReplayAnswerer,
)

WEB_DIR = Path(__file__).parent / "web"
SUMMARY_PATH = ROOT / "docs" / "evidence" / "summary.json"

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response


class RateLimiter:
    """利用者ごとの1分あたりの回数制限（1プロセス内）。"""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.events: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            queue = self.events.setdefault(key, deque())
            while queue and now - queue[0] > 60:
                queue.popleft()
            if len(queue) >= self.per_minute:
                raise HTTPException(429, "質問の回数が多すぎます。1分ほど待ってからお試しください")
            queue.append(now)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=400)


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=400)


def hit_json(hit: Hit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "doc_title": hit.doc_title,
        "heading": hit.heading,
        "body": hit.body,
        "score": round(hit.score, 5),
        "ranks": hit.ranks,
    }


def answer_json(result: AnswerResult) -> dict[str, Any]:
    return {
        "status": "answered" if result.found else "not_found",
        "question": result.question,
        "found": result.found,
        "answer": result.answer,
        "citations": [asdict(c) for c in result.citations],
        "hits": [hit_json(h) for h in result.hits],
        "model": result.model,
        "replayed_at": result.replayed_at,
        "rejected": result.rejected,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        "seconds": round(result.seconds, 2),
    }


def current_principal(request: Request) -> Principal:
    resolver: PrincipalResolver = request.app.state.resolver
    return resolver.resolve({k.lower(): v for k, v in request.headers.items()})


Who = Annotated[Principal, Depends(current_principal)]


def make_answerer(settings: Settings) -> Answerer:
    if settings.mode == "demo":
        return ReplayAnswerer(settings.recordings)
    if settings.answerer == "codex":
        return CodexAnswerer()
    return ClaudeAnswerer()


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    answerer = engine.service.answerer if engine else make_answerer(settings)
    engine = engine or build_engine(
        settings.corpus_dir, answerer, settings.db_path, embedder_from_env()
    )
    resolver: PrincipalResolver = (
        DemoResolver.from_file(settings.people_file)
        if settings.auth == "demo"
        else TokenResolver.from_file(settings.people_file)
    )
    limiter = RateLimiter(settings.ask_per_minute)
    summary = json.loads(SUMMARY_PATH.read_text("utf-8")) if SUMMARY_PATH.exists() else None

    app = FastAPI(title="社内ナレッジ検索", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeaders)

    @app.exception_handler(Unauthorized)
    async def unauthorized(_: Request, exc: Unauthorized) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=401)

    app.state.resolver = resolver

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "documents": len(engine.store.documents())}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        replay = answerer if isinstance(answerer, ReplayAnswerer) else None
        return {
            "mode": settings.mode,
            "auth": settings.auth,
            "model": answerer.model,
            "recorded_at": replay.recorded_at if replay else None,
            "documents": len(engine.store.documents()),
            "personas": [
                {
                    "id": p.id,
                    "name": p.name,
                    "department": p.department,
                    "role": p.role.label,
                }
                for p in (resolver.personas if isinstance(resolver, DemoResolver) else [])
            ],
            "summary": summary,
        }

    @app.get("/api/library")
    def library(who: Who) -> dict[str, Any]:
        docs = engine.store.readable_documents(who)
        suggestions = answerer.questions_for(who.id) if isinstance(answerer, ReplayAnswerer) else []
        return {
            "documents": [
                {"id": d.id, "title": d.title, "format": d.format, "owner": d.owner} for d in docs
            ],
            "suggestions": suggestions,
        }

    @app.post("/api/search")
    def search(body: SearchBody, who: Who) -> dict[str, Any]:
        hits = engine.searcher.search(body.query, who, top_k=10)
        return {"hits": [hit_json(h) for h in hits]}

    @app.post("/api/ask")
    def ask(body: AskBody, who: Who, request: Request) -> dict[str, Any]:
        if settings.mode == "live":
            limiter.check(who.id)
        try:
            result = engine.service.answer(body.question, who)
        except NotRecorded:
            hits = engine.searcher.search(body.question, who, top_k=10)
            return {
                "status": "not_recorded",
                "question": body.question,
                "hits": [hit_json(h) for h in hits],
                "suggestions": answerer.questions_for(who.id)
                if isinstance(answerer, ReplayAnswerer)
                else [],
            }
        except AnswerError as exc:
            raise HTTPException(502, str(exc)) from exc
        return answer_json(result)

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str, who: Who) -> dict[str, Any]:
        chunks = engine.store.document_chunks(doc_id, who)
        if not chunks:  # 読めない文書は「無い」と同じ応答にして、存在も知らせない
            raise HTTPException(404, "文書が見つかりません")
        stored = engine.store.documents()[doc_id]
        return {
            "id": doc_id,
            "title": chunks[0].doc_title,
            "format": stored.format,
            "owner": stored.owner,
            "updated": stored.updated,
            "sections": [
                {"chunk_id": c.chunk.id, "heading": c.chunk.heading, "body": c.chunk.body}
                for c in chunks
            ],
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
