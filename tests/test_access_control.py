"""権限：回答に他部署・上位役職の文書が混ざらないことを保証する。"""

from __future__ import annotations

import pytest

from knowledge_rag.engine import Engine, build_engine
from knowledge_rag.evaluation import load_questions
from knowledge_rag.loaders import iter_corpus, load_document
from knowledge_rag.models import AccessRule, ModelAnswer, Principal, Role

from .conftest import CORPUS, ROOT, ScriptedAnswerer

QUESTIONS = load_questions(ROOT / "eval" / "questions.yaml")


@pytest.fixture(scope="module")
def rules() -> dict[str, AccessRule]:
    rules = {}
    for path in iter_corpus(CORPUS):
        doc, _ = load_document(path)
        rules[doc.meta.id] = doc.meta.access
    return rules


@pytest.fixture(scope="module")
def restricted_texts() -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    for path in iter_corpus(CORPUS):
        doc, _ = load_document(path)
        if doc.meta.access.departments != ("*",) or doc.meta.access.min_role > 1:
            texts[doc.meta.id] = [s.text[:40] for s in doc.sections]
    return texts


def test_every_question_for_every_persona_returns_only_readable_documents(
    engine: Engine, personas: dict[str, Principal], rules: dict[str, AccessRule]
) -> None:
    checked = 0
    for question in QUESTIONS:
        for person in personas.values():
            for hit in engine.searcher.search(question.question, person, top_k=10):
                assert rules[hit.doc_id].allows(person), (question.id, person.id, hit.doc_id)
                checked += 1
    assert checked > 5000


def test_each_retriever_is_filtered_individually(
    engine: Engine, personas: dict[str, Principal], rules: dict[str, AccessRule]
) -> None:
    for method in ("terms", "heading", "trigram", "vector"):
        for person in personas.values():
            hits = engine.searcher.search(
                "賞与の算定係数 役員報酬 値引き率 障害レベル", person, top_k=50, methods=(method,)
            )
            assert all(rules[h.doc_id].allows(person) for h in hits)


def test_restricted_documents_never_reach_the_ai(
    personas: dict[str, Principal],
    restricted_texts: dict[str, list[str]],
    rules: dict[str, AccessRule],
) -> None:
    answerer = ScriptedAnswerer()
    engine = build_engine(CORPUS, answerer)
    restricted = [q for q in QUESTIONS if q.type in ("restricted", "attack")]
    assert len(restricted) >= 10
    for question in restricted:
        person = personas[question.persona]
        answerer.messages.clear()
        result = engine.service.answer(question.question, person)
        assert not result.found
        sent = "\n".join(answerer.messages)
        for doc_id, snippets in restricted_texts.items():
            if rules[doc_id].allows(person):
                continue
            assert all(s not in sent for s in snippets), (question.id, doc_id)


def test_documents_endpoint_data_is_filtered(
    engine: Engine, personas: dict[str, Principal]
) -> None:
    assert engine.store.document_chunks("exec-compensation", personas["sales-staff"]) == []
    assert engine.store.document_chunks("exec-compensation", personas["executive"])
    assert engine.store.hits_by_chunk_ids(["exec-compensation#0"], personas["sales-staff"]) == []


def test_role_and_department_rules() -> None:
    rule = AccessRule(("人事部",), min_role=Role.MANAGER)
    assert rule.allows(Principal("a", "a", "人事部", Role.MANAGER))
    assert not rule.allows(Principal("b", "b", "人事部", Role.STAFF))
    assert not rule.allows(Principal("c", "c", "経理部", Role.EXECUTIVE))
    assert AccessRule(("*",), Role.STAFF).allows(Principal("d", "d", "営業部", Role.STAFF))


def test_answer_cannot_cite_a_document_outside_its_context(
    personas: dict[str, Principal],
) -> None:
    def reply(message: str) -> ModelAnswer:
        # 存在しない資料番号で、読めない文書の文を引用しようとする
        return ModelAnswer.model_validate(
            {
                "found": True,
                "answer": "係数は2.35です。[S99]",
                "citations": [{"source": "S99", "quote": "賞与の算定係数"}],
            }
        )

    engine = build_engine(CORPUS, ScriptedAnswerer(reply))
    result = engine.service.answer("冬の賞与は何か月分？", personas["sales-staff"])
    assert not result.found and result.citations == []
    assert "2.35" not in result.answer
