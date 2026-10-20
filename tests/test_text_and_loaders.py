from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document

from knowledge_rag.chunker import MAX_CHARS, chunk_document, split_section
from knowledge_rag.loaders import (
    DocumentError,
    fix_radicals,
    iter_corpus,
    load_document,
    parse_html,
    parse_markdown,
    parse_meta,
)
from knowledge_rag.models import Role
from knowledge_rag.text import compact, normalize, query_terms, script_runs

from .conftest import CORPUS

FRONT = """---
id: test-doc
title: 試験規程
departments: ["人事部"]
min_role: 管理職
---
"""


def test_normalize_unifies_width_and_months() -> None:
    assert normalize("ＡＢＣ１２３　テスト") == "abc123 テスト"
    assert normalize("1ヶ月と2ケ月と3カ月") == "1か月と2か月と3か月"


def test_compact_ignores_spaces_and_thousands_separators() -> None:
    assert compact("上限は 50,000 円\nとする") == "上限は50000円とする"


def test_script_runs_split_by_character_kind() -> None:
    assert script_runs("育休はPC2台") == [
        ("kanji", "育休"),
        ("hira", "は"),
        ("alnum", "pc2"),
        ("kanji", "台"),
    ]


def test_query_terms_keep_single_kanji_and_split_compounds() -> None:
    terms = query_terms("子どもが生まれたら会社からお祝い金は出ますか？")
    assert "祝" in terms and "金" in terms
    assert "会社" not in terms  # 質問にだけ出る語は除く
    assert "勤務" in query_terms("時短勤務は何歳まで？")
    assert "何" not in query_terms("何日まで？")


def test_markdown_keeps_heading_path_and_cleans_tables() -> None:
    raw = (
        FRONT + "# 試験規程\n\n## 第1章 総則\n\n### 第1条（目的）\n目的を定める。\n\n"
        "### 第2条（表）\n| 区分 | 金額 |\n|---|---|\n| A | 1,000円 |\n\n"
        "```\n# コード内の見出しではない\n```\n"
    ).encode()
    meta, sections = parse_markdown(raw, "x.md")
    assert meta.access.departments == ("人事部",)
    assert meta.access.min_role is Role.MANAGER
    assert sections[0].heading_path == ("第1章 総則", "第1条（目的）")
    assert "区分 | 金額\nA | 1,000円" in sections[1].text
    assert "---" not in sections[1].text
    assert "# コード内の見出しではない" in sections[1].text


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "文書の情報"),
        ({"id": "x", "title": "t"}, "id は英小文字"),
        ({"id": "abc", "title": "t", "min_role": "社長"}, "未知の役職"),
        ({"id": "abc", "title": "t", "departments": "人事部"}, "departments"),
        ({"title": "t"}, "文書の情報が不正"),
    ],
)
def test_parse_meta_rejects_bad_metadata(data: object, message: str) -> None:
    with pytest.raises(DocumentError, match=message):
        parse_meta(data, "x")


def test_markdown_without_front_matter_is_rejected() -> None:
    with pytest.raises(DocumentError, match="フロントマター"):
        parse_markdown(b"# title\n", "x.md")


def test_html_skips_navigation_scripts_and_reads_tables() -> None:
    meta, _ = parse_markdown(FRONT.encode() + b"# t\nbody\n", "x.md")
    html = """<html><head><title>無視</title><script>alert(1)</script></head><body>
    <nav>メニュー</nav><main><h1>試験規程</h1><h2>1. 総則</h2><p>本文 A</p>
    <ul><li>項目1</li><li>項目2</li></ul>
    <table><tr><th>区分</th><th>金額</th></tr><tr><td>A</td><td>1円</td></tr></table>
    </main><footer>フッター</footer></body></html>"""
    sections = parse_html(html.encode(), meta)
    text = sections[0].text
    assert sections[0].heading_path == ("1. 総則",)
    assert "本文 A" in text and "- 項目1" in text and "区分 | 金額" in text
    assert "メニュー" not in text and "alert" not in text and "フッター" not in text


def test_docx_headings_lists_and_tables(tmp_path: Path) -> None:
    doc = Document()
    doc.add_heading("試験規程", level=0)
    doc.add_heading("第1章 総則", level=1)
    doc.add_heading("第1条（目的）", level=2)
    doc.add_paragraph("目的を定める。")
    doc.add_paragraph("箇条", style="List Bullet")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "区分", "金額"
    table.cell(1, 0).text, table.cell(1, 1).text = "A", "1円"
    path = tmp_path / "x.docx"
    buffer = io.BytesIO()
    doc.save(buffer)
    path.write_bytes(buffer.getvalue())
    (tmp_path / "x.docx.meta.yaml").write_text("id: x-doc\ntitle: 試験規程\n", encoding="utf-8")
    parsed, _ = load_document(path)
    assert parsed.format == "docx"
    assert parsed.sections[0].heading_path == ("第1章 総則", "第1条（目的）")
    assert "- 箇条" in parsed.sections[0].text
    assert "区分 | 金額" in parsed.sections[0].text


def test_pdf_articles_become_headings_and_radicals_are_fixed() -> None:
    parsed, _ = load_document(CORPUS / "hr-childcare-leave.pdf")
    headings = [s.heading_path for s in parsed.sections]
    assert ("第2章 育児休業", "第5条（休業の期間）") in headings
    article5 = next(s for s in parsed.sections if "第5条" in s.heading_path[-1])
    assert "1歳6か月まで" in article5.text  # 行の折り返しを戻している
    assert all(not ("⺀" <= ch <= "⿟") for s in parsed.sections for ch in s.text)


def test_pdf_numbered_guide_headings() -> None:
    parsed, _ = load_document(CORPUS / "hr-evaluation.pdf")
    assert any(
        s.heading_path == ("3. 評価の段階と基準", "3.3 評価の分布の目安") for s in parsed.sections
    )


def test_fix_radicals_maps_lookalike_characters() -> None:
    assert fix_radicals("⼈事部⻑") == "人事部長"


def test_sidecar_is_required_for_non_markdown(tmp_path: Path) -> None:
    (tmp_path / "a.html").write_text("<h1>x</h1><p>y</p>", encoding="utf-8")
    with pytest.raises(DocumentError, match=r"meta\.yaml"):
        load_document(tmp_path / "a.html")


def test_unsupported_and_broken_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(DocumentError, match="対応する形式"):
        load_document(tmp_path / "a.txt")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-broken")
    (tmp_path / "b.pdf.meta.yaml").write_text("id: b-doc\ntitle: b\n")
    with pytest.raises(DocumentError, match="読み取れませんでした"):
        load_document(tmp_path / "b.pdf")


def test_corpus_has_all_four_formats() -> None:
    suffixes = {p.suffix for p in iter_corpus(CORPUS)}
    assert suffixes == {".md", ".pdf", ".docx", ".html"}
    assert len(iter_corpus(CORPUS)) == 32


def test_long_section_is_split_with_overlap() -> None:
    sentence = "これは長い規程の文です。" * 10
    text = "\n".join([sentence] * 8)
    pieces = split_section(text)
    assert len(pieces) > 1
    assert all(len(p) <= MAX_CHARS + len(sentence) for p in pieces)
    assert pieces[1].startswith(pieces[0].splitlines()[-1])


def test_chunks_keep_section_boundaries() -> None:
    parsed, _ = load_document(CORPUS / "hr-work-rules.md")
    chunks = chunk_document(parsed)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.heading_path for c in chunks)
    assert chunks[0].id == "hr-work-rules#0"
