#!/usr/bin/env bash
# 全件試験：書式・静的検査・型・ロックファイル・単体・実DB・実ブラウザ（Chromium と WebKit）・axe・全画面サイズ
set -euo pipefail
cd "$(dirname "$0")/.."
uv lock --check
uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -q | diff - requirements.txt
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest "$@"
