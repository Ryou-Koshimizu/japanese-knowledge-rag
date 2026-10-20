from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_rag.answer import AnswerResult, ResolvedCitation
from knowledge_rag.auth import DemoResolver
from knowledge_rag.engine import build_engine
from knowledge_rag.evaluation import (
    Gold,
    Question,
    contains,
    evaluate_retrieval,
    grade_answer,
    load_questions,
    summarize_grades,
)
from knowledge_rag.models import NOT_FOUND_MESSAGE
from knowledge_rag.providers import ReplayAnswerer

from .conftest import CORPUS, ROOT

QUESTIONS = load_questions(ROOT / "eval" / "questions.yaml")


def result(found: bool, answer: str, *cites: tuple[str, str]) -> AnswerResult:
    return AnswerResult(
        question="q",
        found=found,
        answer=answer if found else NOT_FOUND_MESSAGE,
        citations=[ResolvedCitation(1, f"{d}#0", d, "t", h, "quote") for d, h in cites],
        hits=[],
        model="m",
    )


Q = Question(
    id="q1",
    persona="p",
    type="answer",
    question="?",
    gold=(Gold("doc-a", "第5条"),),
    must_include=(("3万円", "30,000円"),),
)


def test_contains_tolerates_notation() -> None:
    assert contains("上限は30000円です", "30,000円")
    assert contains("1ヶ月前まで", "1か月前")
    assert contains("２２％", "22%")
    assert not contains("上限は3万円", "30,000円")


def test_grade_correct_answer() -> None:
    grade = grade_answer(Q, result(True, "3万円です", ("doc-a", "第2章 › 第5条（上限）")))
    assert grade.ok and grade.citations_wrong == 0


def test_grade_detects_each_failure() -> None:
    assert not grade_answer(Q, result(False, "")).ok
    missing_fact = grade_answer(Q, result(True, "5万円です", ("doc-a", "第5条")))
    assert not missing_fact.ok and not missing_fact.facts_ok
    wrong_doc = grade_answer(Q, result(True, "3万円", ("doc-a", "第5条"), ("doc-b", "第1条")))
    assert not wrong_doc.ok and wrong_doc.citations_wrong == 1
    wrong_section = grade_answer(Q, result(True, "3万円", ("doc-a", "第9条")))
    assert not wrong_section.ok and not wrong_section.grounded


def test_grade_unanswerable_and_leaks() -> None:
    absent = Question("a", "p", "absent", "?")
    assert grade_answer(absent, result(False, "")).ok
    assert not grade_answer(absent, result(True, "推測です", ("doc-a", "x"))).ok
    restricted = Question("r", "p", "restricted", "?", must_not_include=("2.35",))
    leaked = result(False, "")
    leaked.answer = "係数は2.35ですが記載がありません"
    assert not grade_answer(restricted, leaked).ok


def test_summary_counts() -> None:
    grades = [
        grade_answer(Q, result(True, "3万円", ("doc-a", "第5条"))),
        grade_answer(Question("a", "p", "absent", "?"), result(False, "")),
    ]
    summary = summarize_grades(grades)
    assert summary["questions"] == 2 and summary["all_ok"] == 2
    assert summary["by_type"] == {"answer": {"total": 1, "ok": 1}, "absent": {"total": 1, "ok": 1}}


def test_question_set_is_well_formed() -> None:
    assert len(QUESTIONS) >= 100
    types = {q.type for q in QUESTIONS}
    assert types == {"answer", "injection", "absent", "restricted", "attack"}
    for q in QUESTIONS:
        if q.type in ("answer", "injection"):
            assert q.gold and q.must_include, q.id
        if q.type == "restricted":
            assert q.must_not_include, q.id


def test_gold_sections_exist_and_contain_the_facts() -> None:
    engine = build_engine(CORPUS, ReplayAnswerer(ROOT / "eval" / "recordings.json"))
    personas = {p.id: p for p in DemoResolver.from_file(CORPUS / "_personas.yaml").personas}
    for q in QUESTIONS:
        texts: list[str] = []
        for gold in q.gold:
            chunks = engine.store.document_chunks(gold.doc, personas[q.persona])
            matched = [c for c in chunks if gold.section in c.chunk.heading]
            assert matched, (q.id, gold)
            texts.extend(c.chunk.body for c in matched)
        for alternatives in q.must_include:
            assert any(contains("".join(texts), a) for a in alternatives), (q.id, alternatives)


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("- {id: a, persona: p, type: absent, question: x}\n" * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        load_questions(path)


def test_retrieval_recall_floor() -> None:
    """CI の構成（軽量な埋め込み）で、正解の条文が上位10件に入ることを確かめる。"""
    engine = build_engine(CORPUS, ReplayAnswerer(ROOT / "eval" / "recordings.json"))
    personas = {p.id: p for p in DemoResolver.from_file(CORPUS / "_personas.yaml").personas}
    report = evaluate_retrieval(
        engine.searcher, QUESTIONS, personas, ("terms", "heading", "trigram", "vector")
    )
    assert report.recall[10] >= 0.98
    assert report.mrr >= 0.7


def test_recorded_answers_still_pass_with_current_code() -> None:
    """記録（実AIの回答）を今の検索・照合で採点し直し、1件も誤らないことを確かめる。"""
    replay = ReplayAnswerer(ROOT / "eval" / "recordings.json")
    engine = build_engine(CORPUS, replay)
    personas = {p.id: p for p in DemoResolver.from_file(CORPUS / "_personas.yaml").personas}
    grades = [
        grade_answer(q, engine.service.answer(q.question, personas[q.persona])) for q in QUESTIONS
    ]
    failures = [(g.question_id, g.problems) for g in grades if not g.ok]
    assert failures == []
    assert len(grades) == len(QUESTIONS)
