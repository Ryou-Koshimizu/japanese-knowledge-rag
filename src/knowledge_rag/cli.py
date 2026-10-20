"""コマンド。`uv run knowledge-rag --help`。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT


def _ingest(args: argparse.Namespace) -> int:
    from .embeddings import embedder_from_env
    from .ingest import ingest_directory
    from .store import Store

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.db)
    report = ingest_directory(store, Path(args.corpus), embedder_from_env())
    print(
        f"追加 {len(report.added)} / 更新 {len(report.updated)} / 変更なし {len(report.unchanged)}"
        f" / 削除 {len(report.removed)} / 書いたチャンク {report.chunks_written}"
    )
    for error in report.errors:
        print(f"読めなかった文書: {error}", file=sys.stderr)
    return 1 if report.errors else 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("knowledge_rag.asgi:app", host=args.host, port=args.port, log_level="info")
    return 0


def _search(args: argparse.Namespace) -> int:
    from .auth import DemoResolver
    from .engine import build_engine
    from .providers import ReplayAnswerer

    corpus = Path(args.corpus)
    engine = build_engine(corpus, ReplayAnswerer(ROOT / "eval" / "recordings.json"))
    who = DemoResolver.from_file(corpus / "_personas.yaml").resolve(
        {"x-demo-persona": args.persona}
    )
    for i, hit in enumerate(engine.searcher.search(args.query, who, top_k=args.k), start=1):
        print(f"{i:>2}. {hit.doc_title} › {hit.heading}  {json.dumps(hit.ranks)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowledge-rag", description="社内ナレッジ検索")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="文書のフォルダを差分で取り込む")
    p.add_argument("--corpus", default=str(ROOT / "corpus"))
    p.add_argument("--db", default="data/index.db")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("serve", help="API と画面を起動する")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8104)
    p.set_defaults(func=_serve)

    p = sub.add_parser("search", help="検索だけを試す（AIなし）")
    p.add_argument("query")
    p.add_argument("--persona", default="sales-staff")
    p.add_argument("--corpus", default=str(ROOT / "corpus"))
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=_search)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
