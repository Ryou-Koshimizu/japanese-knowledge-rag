"""Markdown / PDF / Word / HTML を読み、見出しの構造を保ったまま節に分ける。

権限などの情報は、Markdown はフロントマター、それ以外は同じ名前の `.meta.yaml` に書く。
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

import yaml

from .models import AccessRule, DocumentMeta, ParsedDocument, Role, Section

SUPPORTED_SUFFIXES = {".md": "markdown", ".pdf": "pdf", ".docx": "docx", ".html": "html"}
META_SUFFIX = ".meta.yaml"

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_RULE = re.compile(r":?-{2,}:?")


class DocumentError(ValueError):
    """取り込めない文書。メッセージは利用者にそのまま見せられる日本語。"""


def parse_meta(data: Any, source: str) -> DocumentMeta:
    if not isinstance(data, dict):
        raise DocumentError(f"{source}: 文書の情報（id・title・departments・min_role）がありません")
    try:
        doc_id = str(data["id"])
        title = str(data["title"])
        departments = data.get("departments", ["*"])
        min_role = Role.from_label(str(data.get("min_role", "一般社員")))
    except (KeyError, ValueError) as exc:
        raise DocumentError(f"{source}: 文書の情報が不正です（{exc}）") from exc
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,80}", doc_id):
        raise DocumentError(f"{source}: id は英小文字・数字・ハイフンで書いてください")
    if not isinstance(departments, list) or not departments:
        raise DocumentError(f"{source}: departments は部署名のリストで書いてください")
    return DocumentMeta(
        id=doc_id,
        title=title,
        access=AccessRule(tuple(str(d) for d in departments), min_role),
        owner=str(data.get("owner", "")),
        updated=str(data.get("updated", "")),
    )


def read_sidecar(path: Path) -> bytes:
    sidecar = path.with_name(path.name + META_SUFFIX)
    if not sidecar.exists():
        raise DocumentError(f"{path.name}: 権限を書いた {sidecar.name} がありません")
    return sidecar.read_bytes()


class _SectionBuilder:
    """見出しの深さを追いながら節を組み立てる。"""

    def __init__(self, title: str) -> None:
        self.title = title
        self.stack: list[tuple[int, str]] = []
        self.buffer: list[str] = []
        self.sections: list[Section] = []

    def heading(self, level: int, text: str) -> None:
        self.flush()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    def line(self, text: str) -> None:
        self.buffer.append(text)

    def flush(self) -> None:
        body = "\n".join(self.buffer).strip()
        self.buffer = []
        if not body:
            return
        path = tuple(h for _, h in self.stack)
        if path and path[0] == self.title:
            path = path[1:]
        self.sections.append(Section(path, body))


def parse_markdown(raw: bytes, source: str) -> tuple[DocumentMeta, list[Section]]:
    text = raw.decode("utf-8-sig")
    match = _FRONT_MATTER.match(text)
    if not match:
        raise DocumentError(f"{source}: 先頭に --- で囲んだフロントマターがありません")
    meta = parse_meta(yaml.safe_load(match.group(1)), source)
    builder = _SectionBuilder(meta.title)
    in_code = False
    for line in text[match.end() :].splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            builder.line(line)
            continue
        heading = None if in_code else _MD_HEADING.match(line)
        if heading:
            builder.heading(len(heading.group(1)), heading.group(2))
        elif not in_code and _MD_TABLE_ROW.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(_MD_TABLE_RULE.fullmatch(c) for c in cells):  # 区切りの行は捨てる
                builder.line(" | ".join(cells))
        else:
            builder.line(line.rstrip())
    builder.flush()
    return meta, builder.sections


class _HtmlSections(HTMLParser):
    _SKIP = frozenset(
        {"head", "script", "style", "nav", "footer", "header", "template", "noscript"}
    )
    _BLOCK = frozenset({"p", "li", "tr", "div", "section", "article", "br", "dd", "dt", "pre"})

    def __init__(self, builder: _SectionBuilder) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = builder
        self.skip = 0
        self.heading_level: int | None = None
        self.heading_text: list[str] = []
        self.line: list[str] = []
        self.cells: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif re.fullmatch(r"h[1-6]", tag):
            self._end_line()
            self.heading_level = int(tag[1])
            self.heading_text = []
        elif tag == "tr":
            self._end_line()
            self.cells = []
        elif tag in ("td", "th") and self.cells is not None:
            self.cells.append("")
        elif tag == "li":
            self._end_line()
            self.line.append("- ")
        elif tag in self._BLOCK:
            self._end_line()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif self.heading_level is not None and re.fullmatch(r"h[1-6]", tag):
            self.builder.heading(self.heading_level, "".join(self.heading_text))
            self.heading_level = None
        elif tag == "tr" and self.cells is not None:
            row = " | ".join(c.strip() for c in self.cells if c.strip())
            if row:
                self.builder.line(row)
            self.cells = None
        elif tag in self._BLOCK:
            self._end_line()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.heading_level is not None:
            self.heading_text.append(data)
        elif self.cells is not None and self.cells:
            self.cells[-1] += data
        elif self.cells is None:
            self.line.append(data)

    def _end_line(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.line)).strip()
        self.line = []
        if text and text != "-":
            self.builder.line(text)

    def close(self) -> None:
        super().close()
        self._end_line()
        self.builder.flush()


def parse_html(raw: bytes, meta: DocumentMeta) -> list[Section]:
    builder = _SectionBuilder(meta.title)
    parser = _HtmlSections(builder)
    parser.feed(raw.decode("utf-8-sig"))
    parser.close()
    return builder.sections


def parse_docx(raw: bytes, meta: DocumentMeta) -> list[Section]:
    import docx

    document = docx.Document(io.BytesIO(raw))
    builder = _SectionBuilder(meta.title)
    body = document.element.body
    paragraphs = {p._p: p for p in document.paragraphs}
    tables = {t._tbl: t for t in document.tables}
    for child in body.iterchildren():
        if child in paragraphs:
            para = paragraphs[child]
            style = (para.style.name if para.style is not None else "") or ""
            text = para.text.strip()
            level = _docx_heading_level(style)
            if level is not None:
                builder.heading(level, text)
            elif text:
                prefix = "- " if "List" in style else ""
                builder.line(prefix + text)
        elif child in tables:
            for row in tables[child].rows:
                cells = [c.text.strip() for c in row.cells]
                builder.line(" | ".join(dict.fromkeys(c for c in cells if c)))
    builder.flush()
    return builder.sections


def _docx_heading_level(style: str) -> int | None:
    if style == "Title":
        return 1
    match = re.fullmatch(r"(?:Heading|見出し)\s*(\d)", style)
    return int(match.group(1)) if match else None


_PDF_CHAPTER = re.compile(r"^(第[0-9０-９一二三四五六七八九十百]+章)\s*(.*)$")
_PDF_ARTICLE = re.compile(r"^(第[0-9０-９一二三四五六七八九十百]+条)\s*(（[^）]{1,40}）)?\s*(.*)$")
_PDF_SUPPLEMENT = re.compile(r"^附\s*則$")
_PDF_SECTION = re.compile(r"^(\d{1,2})\.\s*([^。]{1,30})$")
_PDF_SUBSECTION = re.compile(r"^(\d{1,2}\.\d{1,2})\s+([^。]{1,40})$")
_CJK_JOIN = re.compile(r"(?<=[^\x00-\x7f])\n(?=[^\x00-\x7f・■●①-⑳]|\d+[^\x00-\x7f\s%％])")
# PDF の文字情報では「人」が康熙部首の「⼈」(U+2F08) になるなど、見た目が同じ別の文字が混ざる
_RADICALS = re.compile(r"[\u2e80-\u2fdf]")


# NFKC で戻らない CJK 部首補助の文字（PDF で実際に現れるもの）
_RADICAL_SUPPLEMENT = str.maketrans(
    "⻑⻘⻝⻄⻤⻩⻁⻆⻭⻫⻯⻲⻨⻖⻏⻌⻊⺟⺠⻣⻑⻟⻑⺣⺭⺾⻂⻊⺮",
    "長青食西鬼黄虎角歯斉竜亀麦阝阝辶足母民骨長食長灬示艹衤足竹",
)


def fix_radicals(text: str) -> str:
    text = text.translate(_RADICAL_SUPPLEMENT)
    return _RADICALS.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)


def paragraph_is_list(paragraph: list[str]) -> bool:
    return bool(paragraph) and bool(re.match(r"^\d+[.．)]", paragraph[-1]))


def parse_pdf(raw: bytes, meta: DocumentMeta) -> list[Section]:
    """PDF は文字情報を取り出し、「第N章」「第N条」「附則」を見出しとして扱う。"""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    lines: list[str] = []
    for page in reader.pages:
        lines.extend(fix_radicals(page.extract_text() or "").splitlines())
    builder = _SectionBuilder(meta.title)
    paragraph: list[str] = []

    def end_paragraph() -> None:
        if paragraph:
            builder.line(_CJK_JOIN.sub("", "\n".join(paragraph)))
            paragraph.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == meta.title:
            end_paragraph()
            continue
        if chapter := _PDF_CHAPTER.match(line):
            end_paragraph()
            builder.heading(2, f"{chapter.group(1)} {chapter.group(2)}".strip())
        elif article := _PDF_ARTICLE.match(line):
            end_paragraph()
            builder.heading(3, article.group(1) + (article.group(2) or ""))
            if article.group(3):
                paragraph.append(article.group(3))
        elif sub := _PDF_SUBSECTION.match(line):
            end_paragraph()
            builder.heading(3, f"{sub.group(1)} {sub.group(2)}")
        elif (sec := _PDF_SECTION.match(line)) and not paragraph_is_list(paragraph):
            end_paragraph()
            builder.heading(2, f"{sec.group(1)}. {sec.group(2)}")
        elif _PDF_SUPPLEMENT.match(line):
            end_paragraph()
            builder.heading(2, "附則")
        elif line.startswith(("- ", "・", "(", "（")) or re.match(r"^\d+[.．)]", line):
            end_paragraph()
            paragraph.append(line)
        else:
            paragraph.append(line)
    end_paragraph()
    builder.flush()
    return builder.sections


_Parser = Callable[[bytes, DocumentMeta], list[Section]]
_PARSERS: dict[str, _Parser] = {"pdf": parse_pdf, "docx": parse_docx, "html": parse_html}


def load_document(path: Path) -> tuple[ParsedDocument, bytes]:
    """文書を読み、(解析結果, 変更検知用の生バイト列) を返す。"""
    fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        raise DocumentError(f"{path.name}: 対応する形式は Markdown・PDF・Word(docx)・HTML です")
    raw = path.read_bytes()
    kind: Literal["markdown", "pdf", "docx", "html"]
    if fmt == "markdown":
        meta, sections = parse_markdown(raw, path.name)
        fingerprint = raw
        kind = "markdown"
    else:
        meta_raw = read_sidecar(path)
        meta = parse_meta(yaml.safe_load(meta_raw), path.name + META_SUFFIX)
        try:
            sections = _PARSERS[fmt](raw, meta)
        except DocumentError:
            raise
        except Exception as exc:  # 壊れたファイルは1件だけ飛ばし、取り込み全体は止めない
            raise DocumentError(
                f"{path.name}: 読み取れませんでした（{type(exc).__name__}）"
            ) from exc
        fingerprint = raw + b"\0" + meta_raw
        kind = "pdf" if fmt == "pdf" else "docx" if fmt == "docx" else "html"
    if not sections:
        raise DocumentError(f"{path.name}: 本文が見つかりません")
    return ParsedDocument(meta, str(path), kind, sections), fingerprint


def iter_corpus(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith((".", "_"))
    )
