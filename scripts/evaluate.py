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


def run_regrade(note: str) -> int:
    """記録した回答（AIの出力・照合の結果）はそのままに、今の評価セットの基準で採点し直す。

    AIを呼び直さない。質問の文と聞いた人が記録と一致することを確かめ、
    採点し直した日時と理由を記録に残す。
    """
    path = EVIDENCE / "answers.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    grades = []
    for row in data["rows"]:
        q = questions[row["id"]]
        if (q.question, q.persona) != (row["question"], row["persona"]):
            raise SystemExit(f"{q.id}: 質問の文か聞いた人が記録と違うため、採点し直せません")
        grade = grade_answer(q, _result_from_row(row))
        row["grade"] = asdict(grade)
        grades.append(grade)
    if len(grades) != len(questions):
        raise SystemExit("記録の問数が評価セットと違います")
    summary = summarize_grades(grades)
    data["summary"] = summary
    data["questions_sha"] = digest(QUESTIONS)
    data.setdefault("regrades", []).append(
        {"at": datetime.now(UTC).isoformat(timespec="seconds"), "note": note}
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    write_report_blocks(retrieval, answers)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


METHOD_NAMES = {
    "trigram": "部分一致（FTS5 trigram）だけ",
    "terms": "語の一致（BM25）だけ",
    "heading": "見出しだけ",
    "vector": "埋め込みだけ",
    "terms+heading+trigram": "語・見出し・部分一致の統合",
    "terms+heading+trigram+vector": "**4方式の統合（既定）**",
}
EMBEDDER_NAMES = {
    "hashing": "軽量（文字 n-gram のハッシュ、CI・公開デモ）",
    "local": "多言語の文埋め込み（MiniLM、手元で動く）",
}
TYPE_NAMES = {
    "answer": "答えがある",
    "injection": "文書に乗っ取りの文",
    "absent": "書かれていない",
    "restricted": "読めない文書にある",
    "attack": "質問からの乗っ取り",
}


def replace_block(path: Path, name: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{body.strip()}\n{end}{tail}", encoding="utf-8")


def write_report_blocks(retrieval: dict[str, Any], answers: dict[str, Any]) -> None:
    """README と検証記録の表を、測定の記録から書き直す（手で数字を写さない）。"""
    lines = [
        "| 埋め込み | 方式 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | 1問あたり |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind, data in retrieval["embedders"].items():
        for m in data["methods"]:
            r = m["recall"]
            lines.append(
                f"| {EMBEDDER_NAMES.get(kind, kind)} | {METHOD_NAMES.get(m['method'], m['method'])} "
                f"| {r['1']:.3f} | {r['3']:.3f} | {r['5']:.3f} | {r['10']:.3f} | {m['mrr']:.3f} "
                f"| {m['ms_per_query']:.0f} ms |"
            )
    replace_block(ROOT / "docs" / "verification.md", "retrieval", "\n".join(lines))

    s = answers["summary"]
    model = answers["model"].removeprefix("codex:")
    rows = ["| 種類 | 問数 | 正しかった |", "| --- | --- | --- |"]
    for kind, v in s["by_type"].items():
        rows.append(f"| {TYPE_NAMES.get(kind, kind)}（{kind}） | {v['total']} | {v['ok']} |")
    detail = "\n".join(
        [
            f"測定モデル：**{model}**（ChatGPT の Codex CLI 経由、推論は low）。"
            f"検索は多言語の文埋め込みを使う構成、上位{answers.get('context_size', CONTEXT_SIZE)}件を渡す。"
            f"記録：[answers.json](evidence/answers.json)。",
            "",
            *rows,
            "",
            f"- 全体：**{s['all_ok']}/{s['questions']} 問が正しい**（誤答・誤った引用・「記載なし」の取り違え・エラーの合計 "
            f"{s['questions'] - s['all_ok']} 件）",
            f"- 答えがある{s['answerable']}問：正しい事実 {s['facts_ok']}/{s['answerable']}、"
            f"正解の見出しを引用 {s['grounded']}/{s['answerable']}",
            f"- 表示した引用 {s['citations']} 件のうち、正解の文書以外を引いたもの {s['citations_wrong']} 件",
            f"- 照合で表示しなかった引用（AIの引用ミスを仕組みで止めた数）：{s['rejected_by_guard']} 件",
            f"- 「記載なし」が正解の{s['unanswerable']}問：{s['not_found_ok']}/{s['unanswerable']}",
            f"- 1問あたりの時間（中央値）：{answers['median_seconds']} 秒、"
            f"入力トークン合計 {answers['input_tokens']:,}・出力 {answers['output_tokens']:,}",
            f"- 指示文のハッシュ `{answers['instructions_sha']}`・質問のハッシュ `{answers['questions_sha']}`",
        ]
    )
    replace_block(ROOT / "docs" / "verification.md", "answers", detail)

    local = retrieval["embedders"].get("local") or next(iter(retrieval["embedders"].values()))
    hybrid = local["methods"][-1]
    hashing = retrieval["embedders"].get("hashing", local)["methods"][-1]
    acl = local["acl_audit"]
    readme = "\n".join(
        [
            "| 指標 | 結果 | 条件 |",
            "| --- | --- | --- |",
            f"| **回答の正答（実AI）** | **{s['all_ok']}/{s['questions']} 問** | {model}（ChatGPT の Codex CLI 経由）。"
            "答えがある・書かれていない・読めない・乗っ取りの5種類 |",
            f"| 誤った引用 | {s['citations_wrong']} 件 | 表示した引用 {s['citations']} 件。すべて本文と一字一句一致 |",
            f"| 「記載なし」の正答 | {s['not_found_ok']}/{s['unanswerable']} 問 | 書かれていない13・読めない11・質問からの乗っ取り4 |",
            f"| 検索 Recall@5 / MRR | {hybrid['recall']['5']:.3f} / {hybrid['mrr']:.3f} | 77問、4方式の統合・多言語の文埋め込み。"
            f"軽量な埋め込み（公開デモ）では {hashing['recall']['5']:.3f} / {hashing['mrr']:.3f} |",
            f"| 検索 Recall@10 | {hybrid['recall']['10']:.3f} | 同上 |",
            f"| 権限の漏れ | {acl['leaks']} 件 | {acl['searches']}回の検索・{acl['hits_checked']:,}件の結果を権限の定義と照合 |",
            f"| 1問あたりの回答時間 | 中央値 {answers['median_seconds']} 秒 | 検索は十数ミリ秒、残りは AI |",
            *live_rows(),
        ]
    )
    replace_block(ROOT / "README.md", "summary", readme)
    live_path = EVIDENCE / "live-browser.json"
    if live_path.exists():
        live = json.loads(live_path.read_text(encoding="utf-8"))
        lines = [
            f"実AI（{model}）で動くサーバーを、Chromium と WebKit で画面から操作しました（{live['measured_at'][:10]}）。"
            "質問を入力して回答・根拠を表示し、回答があるときは根拠の文書を開いて該当の条が印つきで表示されることまで確かめます。"
            f"記録：[live-browser.json](evidence/live-browser.json)。",
            "",
            "| ブラウザ | 質問 | 種類 | 結果 | 秒 |",
            "| --- | --- | --- | --- | --- |",
            *(
                f"| {r['browser']} | `{r['id']}` | {TYPE_NAMES.get(r.get('type', ''), r.get('type', ''))} "
                f"| {'正しい' if r['ok'] else '誤り：' + '・'.join(r.get('problems', []))} | {r.get('seconds', '')} |"
                for r in live["rows"]
            ),
            "",
            f"**{live['ok']}/{live['checks']} 件が正しい**（誤答・誤った引用・画面の表示の違い・ページのエラー 0 件）。",
        ]
        replace_block(ROOT / "docs" / "verification.md", "live", "\n".join(lines))


def live_rows() -> list[str]:
    path = EVIDENCE / "live-browser.json"
    if not path.exists():
        return []
    live = json.loads(path.read_text(encoding="utf-8"))
    return [
        f"| 実AI×実ブラウザの通し確認 | {live['ok']}/{live['checks']} 件 | Chromium と WebKit で画面から7種類の質問を入力し、回答・根拠・文書の表示まで確認 |"
    ]


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
    p = sub.add_parser("regrade")
    p.add_argument("--note", required=True)
    args = parser.parse_args()
    if args.cmd == "retrieval":
        return run_retrieval(args.embedders)
    if args.cmd == "answers":
        return run_answers(args.answerer, args.workers, args.only)
    if args.cmd == "summary":
        return run_summary()
    if args.cmd == "regrade":
        return run_regrade(args.note)
    return run_replay()


if __name__ == "__main__":
    raise SystemExit(main())
