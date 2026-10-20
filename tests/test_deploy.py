"""公開デモ（Vercel の Python ランタイム）の入口と同梱ファイルの確認。"""

from __future__ import annotations

import fnmatch
import json
import runpy

from fastapi.testclient import TestClient

from .conftest import ROOT


def test_vercel_entry_serves_the_demo_without_keys(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for key in ("ANTHROPIC_API_KEY", "RAG_MODE", "RAG_AUTH", "RAG_EMBEDDER"):
        monkeypatch.delenv(key, raising=False)
    module = runpy.run_path(str(ROOT / "api" / "index.py"))
    client = TestClient(module["app"])
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "demo" and meta["documents"] == 32
    assert client.get("/").status_code == 200


def test_vercel_bundle_includes_runtime_files() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())
    pattern = config["functions"]["api/index.py"]["includeFiles"]
    globs = pattern.strip("{}").split(",")
    needed = [
        "src/knowledge_rag/web/index.html",
        "corpus/hr-childcare-leave.pdf",
        "corpus/_personas.yaml",
        "eval/recordings.json",
        "docs/evidence/summary.json",
    ]
    for path in needed:
        assert (ROOT / path).exists(), path
        assert any(fnmatch.fnmatch(path, g.replace("/**", "/*")) for g in globs), path


def test_requirements_match_lockfile() -> None:
    requirements = (ROOT / "requirements.txt").read_text()
    for name in ("fastapi", "uvicorn", "pypdf", "python-docx", "anthropic", "pyyaml"):
        assert f"\n{name}==" in requirements, name
    assert "fastembed" not in requirements  # 公開デモには重い埋め込みを入れない
