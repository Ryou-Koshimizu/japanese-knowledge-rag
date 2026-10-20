"""評価用の架空の規程集を、原稿（fixtures/corpus-src/*.md）から corpus/ に組み立てる。

実際の社内文書の形式の混在を再現するため、一部を PDF・Word・HTML に変換する。
PDF は Chromium（Playwright）で印刷し、Word は python-docx で見出しスタイル付きで書く。

    uv run python scripts/build_corpus.py                    # 文書と質問
    uv run python scripts/build_corpus.py --questions-only   # 質問だけ
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fixtures" / "corpus-src"
OUT = ROOT / "corpus"

PDF = {"hr-childcare-leave", "cmp-harassment", "hr-evaluation"}
DOCX = {"hr-nursing-care-leave", "fin-travel", "hr-training-support", "fin-monthly-close"}
HTML = {"sec-policy", "ga-condolence", "ga-records", "it-system-guide", "it-incident-runbook"}

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_front(text: str) -> tuple[dict[str, object], str]:
    match = _FRONT.match(text)
    if not match:
        raise SystemExit("フロントマターがありません")
    return yaml.safe_load(match.group(1)), text[match.end() :]


def inline(text: str) -> str:
    text = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def markdown_blocks(body: str) -> list[tuple[str, Any]]:
    """見出し・段落・箇条書き・表のかたまりに分ける。"""
    blocks: list[tuple[str, Any]] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if m := re.match(r"^(#{1,6})\s+(.*)$", line):
            blocks.append(("h", (len(m.group(1)), m.group(2).strip())))
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif re.match(r"^\s*([-*]|\d+[.)])\s+", line):
            items = []
            ordered = bool(re.match(r"^\s*\d+[.)]", line))
            while i < len(lines) and re.match(r"^\s*([-*]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+[.)])\s+", "", lines[i]).strip())
                i += 1
            blocks.append(("ol" if ordered else "ul", items))
        else:
            para = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"^(#{1,6}\s|\s*\||\s*([-*]|\d+[.)])\s)", lines[i])
            ):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("p", "".join(para)))
    return blocks


def to_html_body(blocks: list[tuple[str, Any]]) -> str:
    out: list[str] = []
    for kind, value in blocks:
        if kind == "h":
            level, text = value
            out.append(f"<h{level}>{inline(text)}</h{level}>")
        elif kind == "p":
            out.append(f"<p>{inline(str(value))}</p>")
        elif kind in ("ul", "ol"):
            items = "".join(f"<li>{inline(x)}</li>" for x in value)
            out.append(f"<{kind}>{items}</{kind}>")
        elif kind == "table":
            rows = value
            head, *rest = rows
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            trs = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rest
            )
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
    return "\n".join(out)


def intranet_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{html.escape(title)}｜社内ポータル</title>
<style>body{{font-family:sans-serif;max-width:46rem;margin:2rem auto;line-height:1.8}}
table{{border-collapse:collapse}}td,th{{border:1px solid #999;padding:.3rem .6rem}}</style>
<script>window.portalAnalytics = "これは本文ではありません";</script></head>
<body><header><nav>社内ポータル ＞ 規程・マニュアル ＞ {html.escape(title)}</nav></header>
<main>
{body}
</main>
<footer>株式会社ソラノメ計器 社内ポータル（社外秘）</footer></body></html>
"""


def write_docx(path: Path, blocks: list[tuple[str, Any]]) -> None:
    from docx import Document

    doc = Document()
    doc.core_properties.author = "ソラノメ計器 人事部"
    for kind, value in blocks:
        if kind == "h":
            level, text = value
            doc.add_heading(text, level=0 if level == 1 else min(level - 1, 9))
        elif kind == "p":
            doc.add_paragraph(str(value))
        elif kind in ("ul", "ol"):
            style = "List Bullet" if kind == "ul" else "List Number"
            for item in value:
                doc.add_paragraph(re.sub(r"\*\*(.+?)\*\*", r"\1", item), style=style)
        elif kind == "table":
            rows = value
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
    doc.save(str(path))


def write_pdfs(jobs: list[tuple[Path, str]]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for path, page_html in jobs:
            page.set_content(page_html, wait_until="load")
            page.pdf(path=str(path), format="A4", margin={"top": "18mm", "bottom": "18mm"})
        browser.close()


PRINT_CSS = """<style>
body{font-family:"Hiragino Mincho ProN","Noto Serif CJK JP",serif;font-size:10.5pt;
line-height:1.75;margin:0 6mm}
h1{font-size:16pt;text-align:center;margin:0 0 8mm}h2{font-size:12pt;margin:6mm 0 2mm}
h3{font-size:10.5pt;margin:3mm 0 1mm}p{margin:0 0 1.5mm}table{border-collapse:collapse;margin:2mm 0}
td,th{border:0.5pt solid #444;padding:1mm 2mm;font-size:9.5pt}
</style>"""


def merge_questions() -> None:
    """担当ごとの質問ファイルを1つにまとめる。"""
    items: list[object] = []
    for name in ("A", "B", "C", "D", "absent"):
        items.extend(yaml.safe_load((SRC / f"questions-{name}.yaml").read_text("utf-8")))
    header = (
        "# 評価用の質問と正解の根拠（fixtures/corpus-src/questions-*.yaml から生成）\n"
        "# type: answer / injection / absent / restricted / attack（意味は evaluation.py）\n"
    )
    body = yaml.safe_dump(items, allow_unicode=True, sort_keys=False, width=200)
    (ROOT / "eval" / "questions.yaml").write_text(header + body, encoding="utf-8")


def main() -> int:
    if "--questions-only" in sys.argv:  # 文書（PDF・Word は作成日時が変わる）を作り直さない
        merge_questions()
        return 0
    if OUT.exists():
        for child in OUT.iterdir():
            if not child.name.startswith("_"):
                child.unlink() if child.is_file() else shutil.rmtree(child)
    OUT.mkdir(exist_ok=True)
    pdf_jobs: list[tuple[Path, str]] = []
    for src in sorted(SRC.glob("*.md")):
        text = src.read_text(encoding="utf-8")
        meta, body = split_front(text)
        doc_id = str(meta["id"])
        blocks = markdown_blocks(body)
        sidecar = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
        if doc_id in PDF:
            # 印刷物らしく「第N条（見出し）」を本文の行頭に置く
            page = f'<html lang="ja"><head><meta charset="utf-8">{PRINT_CSS}</head><body>'
            pdf_jobs.append((OUT / f"{doc_id}.pdf", page + to_html_body(blocks) + "</body></html>"))
            (OUT / f"{doc_id}.pdf.meta.yaml").write_text(sidecar, encoding="utf-8")
        elif doc_id in DOCX:
            write_docx(OUT / f"{doc_id}.docx", blocks)
            (OUT / f"{doc_id}.docx.meta.yaml").write_text(sidecar, encoding="utf-8")
        elif doc_id in HTML:
            page = intranet_page(str(meta["title"]), to_html_body(blocks))
            (OUT / f"{doc_id}.html").write_text(page, encoding="utf-8")
            (OUT / f"{doc_id}.html.meta.yaml").write_text(sidecar, encoding="utf-8")
        else:
            shutil.copyfile(src, OUT / f"{doc_id}.md")
    write_pdfs(pdf_jobs)
    merge_questions()
    print(f"{len(list(SRC.glob('*.md')))} 文書を {OUT} に書きました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
