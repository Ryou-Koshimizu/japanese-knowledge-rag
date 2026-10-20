"""`uvicorn knowledge_rag.asgi:app` 用。設定は環境変数（config.py）から読む。"""

from .app import create_app

app = create_app()
