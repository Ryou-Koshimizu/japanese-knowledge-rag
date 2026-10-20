"""Vercel の Python ランタイムの入口。公開デモ（記録の再生・鍵なし）として動く。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knowledge_rag.app import create_app

app = create_app()
