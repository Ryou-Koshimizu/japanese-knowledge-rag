"""評価の実行。

    uv run python scripts/evaluate.py retrieval            # 検索（AIなし）と権限の漏れの検査
    uv run python scripts/evaluate.py answers --answerer codex --workers 4
    uv run python scripts/evaluate.py answers --answerer claude
    uv run python scripts/evaluate.py replay     # 記録の再生で採点し直す（AIなし）

answers は回答の記録を eval/recordings.json（公開デモの再生に使う）と
docs/evidence/answers.json に保存する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_rag.answer import CONTEXT_SIZE, AnswerResult, ground  # noqa: E402
from knowledge_rag.auth import DemoResolver  # noqa: E402
from knowledge_rag.embeddings import (  # noqa: E402
    Embedder,
    HashingEmbedder,
    LocalEmbedder,
    embedder_from_env,
)
from knowledge_rag.engine import Engine, build_engine  # noqa: E402
from knowledge_rag.evaluation import (  # noqa: E402
    Question,
    evaluate_retrieval,
    grade_answer,
    load_questions,
    summarize_grades,
)
from knowledge_rag.ingest import PIPELINE_VERSION  # noqa: E402
from knowledge_rag.loaders import iter_corpus, load_document  # noqa: E402
from knowledge_rag.models import Principal  # noqa: E402
from knowledge_rag.prompt import INSTRUCTIONS, build_user_message  # noqa: E402
from knowledge_rag.providers import (  # noqa: E402
    Answerer,
    AnswerError,
    ClaudeAnswerer,
    CodexAnswerer,
    Generation,
    ReplayAnswerer,
)

CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "eval" / "questions.yaml"
EVIDENCE = ROOT / "docs" / "evidence"
RECORDINGS = ROOT / "eval" / "recordings.json"
CHECKPOINTS = ROOT / "reports"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, check=True
        )
        return out.stdout.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def personas() -> dict[str, Principal]:
    return {p.id: p for p in DemoResolver.from_file(CORPUS / "_personas.yaml").personas}


class _NoAnswer:
    model = "none"

    def generate(self, question: str, user_message: str, principal_id: str) -> Generation:
        raise AnswerError("検索の評価ではAIを呼びません")


def acl_audit(
    engine: Engine, questions: list[Question], people: dict[str, Principal]
) -> dict[str, int]:
    """全質問 × 全員で検索し、返った文書を権限の定義（ファイル側）と独立に照合する。"""
    rules = {}
    for path in iter_corpus(CORPUS):
        doc, _ = load_document(path)
        rules[doc.meta.id] = doc.meta.access
    searches = leaks = hits_checked = 0
    for q in questions:
        for person in people.values():
            for hit in engine.searcher.search(q.question, person, top_k=10):
                hits_checked += 1
                if not rules[hit.doc_id].allows(person):
                    leaks += 1
            searches += 1
    return {"searches": searches, "hits_checked": hits_checked, "leaks": leaks}


def make_embedder(kind: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "local":
        return LocalEmbedder()
    raise SystemExit(f"未知の埋め込み: {kind}")


def run_retrieval(embedders: list[str]) -> int:
    questions = load_questions(QUESTIONS)
    people = personas()
    out: dict[str, Any] = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "questions_sha": digest(QUESTIONS),
        "questions": sum(q.type in ("answer", "injection") for q in questions),
        "embedders": {},
    }
    leaks = 0
    for kind in embedders:
        embedder = make_embedder(kind)
        engine = build_engine(CORPUS, _NoAnswer(), embedder=embedder)
        out["documents"] = len(engine.store.documents())
        out["chunks"] = engine.store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        reports = []
        for methods in (
            ("trigram",),
            ("terms",),
            ("heading",),
            ("vector",),
            ("terms", "heading", "trigram"),
            ("terms", "heading", "trigram", "vector"),
        ):
            started = time.perf_counter()
            report = evaluate_retrieval(engine.searcher, questions, people, methods)
            elapsed = (time.perf_counter() - started) / max(report.questions, 1)
            reports.append({**asdict(report), "ms_per_query": round(elapsed * 1000, 1)})
            recall = " ".join(f"R@{k}={v:.3f}" for k, v in report.recall.items())
            print(f"[{kind}] {report.method:<32} {recall} MRR={report.mrr:.3f}")
        audit = acl_audit(engine, questions, people)
        leaks += audit["leaks"]
        print(f"[{kind}] 権限の検査: {audit}  5位以内に無い: {reports[-1]['misses']}")
        out["embedders"][kind] = {"id": embedder.id, "methods": reports, "acl_audit": audit}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 1 if leaks else 0


class UsageLimit(RuntimeError):
    """AIの利用上限。途中まで保存して止める（続きは同じ条件で再開できる）。"""


def answer_one(
    engine: Engine, answerer: Answerer, q: Question, person: Principal, retries: int = 2
) -> tuple[AnswerResult, Generation | None, list[str]]:
    hits = engine.service.retrieve(q.question, person)
    context = hits[:CONTEXT_SIZE]
    context_ids = [h.chunk_id for h in context]
    if not context:
        return engine.service.answer(q.question, person), None, context_ids
    message = build_user_message(q.question, context)
    error = ""
    for attempt in range(retries + 1):
        try:
            gen = answerer.generate(q.question, message, person.id)
            return ground(q.question, context, gen, hits), gen, context_ids
        except AnswerError as exc:
            error = str(exc)
            if "usage limit" in error or "上限" in error:
                raise UsageLimit(error) from exc
            time.sleep(10 * (attempt + 1))
    raise AnswerError(f"{q.id}: {error}")


def run_fingerprint(answerer: Answerer, embedder: Embedder) -> str:
    """同じ条件かどうかの印。指示文・質問・規程集・検索の設定・モデルが同じなら同じ値。"""
    parts = [
        INSTRUCTIONS,
        QUESTIONS.read_text(encoding="utf-8"),
        answerer.model,
        embedder.id,
        str(CONTEXT_SIZE),
        PIPELINE_VERSION,
        (ROOT / "src" / "knowledge_rag" / "search.py").read_text(encoding="utf-8"),
        (ROOT / "src" / "knowledge_rag" / "answer.py").read_text(encoding="utf-8"),
        (ROOT / "src" / "knowledge_rag" / "text.py").read_text(encoding="utf-8"),
        (ROOT / "src" / "knowledge_rag" / "loaders.py").read_text(encoding="utf-8"),
        (CORPUS / "_glossary.yaml").read_text(encoding="utf-8"),
        *(
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CORPUS.glob("*"))
            if not p.name.endswith(".pdf")  # PDF は作成日時が入るので、文字情報で比べる
        ),
        *(
            "".join(s.text for s in load_document(p)[0].sections)
            for p in sorted(CORPUS.glob("*.pdf"))
        ),
    ]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def run_answers(kind: str, workers: int, only: list[str] | None) -> int:
    questions = load_questions(QUESTIONS)
    if only:
        questions = [q for q in questions if q.id in only]
    people = personas()
    answerer: Answerer = CodexAnswerer() if kind == "codex" else ClaudeAnswerer()
    embedder = embedder_from_env()
    engine = build_engine(CORPUS, answerer, embedder=embedder)
    fingerprint = run_fingerprint(answerer, embedder)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"answers-{fingerprint}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
        print(f"同じ条件の記録 {len(done)} 件を再利用します（{checkpoint.name}）")
    lock = threading.Lock()

    def task(q: Question) -> dict[str, Any] | None:
        if q.id in done:
            return done[q.id]
        try:
            result, gen, context_ids = answer_one(engine, answerer, q, people[q.persona])
        except UsageLimit as exc:
            print(f"利用上限で中断: {exc}", flush=True)
            return None
        grade = grade_answer(q, result)
        mark = "OK " if grade.ok else "NG "
        print(f"{mark}{q.id} {result.seconds:.1f}s {'; '.join(grade.problems)}", flush=True)
        row = {
            "id": q.id,
            "persona": q.persona,
            "type": q.type,
            "question": q.question,
            "context": context_ids,
            "raw": gen.answer.model_dump() if gen else None,
            "found": result.found,
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "rejected": result.rejected,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "seconds": round(result.seconds, 2),
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with lock, checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        maybe_rows = list(pool.map(task, questions))
    if any(r is None for r in maybe_rows):
        left = sum(r is None for r in maybe_rows)
        print(f"未回答 {left} 件。上限が戻ったら同じコマンドで続きから測ります。")
        return 2
    rows = [r for r in maybe_rows if r is not None]
    grades = [grade_answer(q, _result_from_row(r)) for q, r in zip(questions, rows, strict=True)]
    for row, grade in zip(rows, grades, strict=True):
        row["grade"] = asdict(grade)
    summary = summarize_grades(grades)
    seconds = sorted(r["seconds"] for r in rows if r["raw"] is not None)
    times = sorted(r["measured_at"] for r in rows if r.get("measured_at"))
    out = {
        "measured_at": times[0] if times else None,
        "measured_until": times[-1] if times else None,
        "commit": git_commit(),
        "model": answerer.model,
        "embedder": embedder.id,
        "context_size": CONTEXT_SIZE,
        "fingerprint": fingerprint,
        "questions_sha": digest(QUESTIONS),
        "instructions_sha": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()[:12],
        "summary": summary,
        "median_seconds": seconds[len(seconds) // 2] if seconds else None,
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "rows": rows,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    suffix = "-partial" if only else ""
    (EVIDENCE / f"answers{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    if not only:
        keys = ("persona", "question", "context", "raw", "input_tokens", "output_tokens", "seconds")
        recordings = {
            "model": answerer.model,
            "recorded_at": out["measured_until"],
            "note": "実AIで記録した回答。公開デモはこれを再生する（AIを呼ばない）。",
            "entries": [{k: r[k] for k in keys} for r in rows if r["raw"] is not None],
        }
        RECORDINGS.write_text(json.dumps(recordings, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False))
    for failure in summary["failures"]:
        print("NG", failure)
    return 0 if summary["all_ok"] == summary["questions"] else 1


def _result_from_row(row: dict[str, Any]) -> AnswerResult:
    from knowledge_rag.answer import ResolvedCitation

    return AnswerResult(
        question=row["question"],
        found=row["found"],
        answer=row["answer"],
        citations=[ResolvedCitation(**c) for c in row["citations"]],
        hits=[],
        model=None,
        rejected=row["rejected"],
    )


def run_replay() -> int:
    """記録を再生して、今のコード（検索・照合）で採点し直す。AIは呼ばない。"""
    questions = load_questions(QUESTIONS)
    people = personas()
    answerer = ReplayAnswerer(RECORDINGS)
    engine = build_engine(CORPUS, answerer)
    grades = []
    for q in questions:
        result = engine.service.answer(q.question, people[q.persona])
        grades.append(grade_answer(q, result))
    summary = summarize_grades(grades)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False))
    for failure in summary["failures"]:
        print("NG", failure)
    return 0 if summary["all_ok"] == summary["questions"] else 1


def run_summary() -> int:
    """画面の奥付と README に載せる数字を、測定の記録から1つにまとめる。"""
    retrieval = json.loads((EVIDENCE / "retrieval.json").read_text())
    answers = json.loads((EVIDENCE / "answers.json").read_text())
    local = retrieval["embedders"].get("local") or next(iter(retrieval["embedders"].values()))
    hybrid = local["methods"][-1]
    summary = answers["summary"]
    out = {
        "retrieval": {
            "label": "ハイブリッド・手元の多言語埋め込み",
            "questions": hybrid["questions"],
            "recall_at_5": round(hybrid["recall"]["5"], 3),
            "recall_at_10": round(hybrid["recall"]["10"], 3),
            "mrr": round(hybrid["mrr"], 3),
            "measured_at": retrieval["measured_at"],
        },
        "acl": local["acl_audit"],
        "answers": {
            "model": answers["model"].removeprefix("codex:"),
            "questions": summary["questions"],
            "all_ok": summary["all_ok"],
            "answerable": summary["answerable"],
            "facts_ok": summary["facts_ok"],
            "grounded": summary["grounded"],
            "citations": summary["citations"],
            "citations_wrong": summary["citations_wrong"],
            "unanswerable": summary["unanswerable"],
            "not_found_ok": summary["not_found_ok"],
            "median_seconds": answers["median_seconds"],
            "measured_at": answers["measured_at"],
        },
        "note": (
            f"架空の規程{retrieval['documents']}篇・質問{summary['questions']}問で測定。"
            f"回答は {answers['model'].removeprefix('codex:')}（ChatGPT の Codex CLI 経由）、"
            "測定日 " + answers["measured_at"][:10] + "。公開デモはこの記録を再生しています。"
        ),
    }
    (EVIDENCE / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("retrieval")
    p.add_argument("--embedders", nargs="+", default=["hashing", "local"])
    p = sub.add_parser("answers")
    p.add_argument("--answerer", choices=["codex", "claude"], default="codex")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*")
    sub.add_parser("replay")
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.cmd == "retrieval":
        return run_retrieval(args.embedders)
    if args.cmd == "answers":
        return run_answers(args.answerer, args.workers, args.only)
    if args.cmd == "summary":
        return run_summary()
    return run_replay()


if __name__ == "__main__":
    raise SystemExit(main())
