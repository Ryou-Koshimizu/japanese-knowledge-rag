from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_rag.cli import main


def test_cli_ingest_reports_and_is_incremental(
    corpus_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = str(tmp_path / "x" / "index.db")
    assert main(["ingest", "--corpus", str(corpus_copy), "--db", db]) == 0
    assert "追加 32" in capsys.readouterr().out
    assert main(["ingest", "--corpus", str(corpus_copy), "--db", db]) == 0
    assert "変更なし 32" in capsys.readouterr().out
    (corpus_copy / "broken.pdf").write_bytes(b"%PDF-")
    (corpus_copy / "broken.pdf.meta.yaml").write_text("id: broken\ntitle: x\n")
    assert main(["ingest", "--corpus", str(corpus_copy), "--db", db]) == 1
    assert "読めなかった文書" in capsys.readouterr().err


def test_cli_search(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "出張の日当はいくら？", "--persona", "dev-manager", "-k", "3"]) == 0
    out = capsys.readouterr().out
    assert " 1. 旅費規程" in out
