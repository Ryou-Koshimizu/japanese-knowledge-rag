"""埋め込み（文をベクトルにする部分）の差し替え境界。

- `HashingEmbedder`：外部サービスなし・依存なしの軽量な方法。文字の2〜3文字の並びを
  ハッシュで固定長のベクトルにする。CI と公開デモで使う。
- `VoyageEmbedder` / `OpenAIEmbedder`：本番向け。API の埋め込みモデルを使う。
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

import httpx

from .text import normalize


class Embedder(Protocol):
    fusion_weight: float

    @property
    def id(self) -> str:
        """モデルの識別子。変わったら索引を作り直す。"""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def _is_hiragana_only(gram: str) -> bool:
    return all("぀" <= ch <= "ゟ" for ch in gram)


class HashingEmbedder:
    """文字 n-gram の特徴ハッシュ。意味の近さではなく字面の近さを測る軽量な方法。"""

    # 検索の統合（RRF）での重み。字面の近さは語の一致と重なるので補助にとどめる
    fusion_weight = 0.1

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @property
    def id(self) -> str:
        return f"hashing-char23-{self.dim}"

    def _vector(self, text: str) -> list[float]:
        text = "".join(ch for ch in normalize(text) if not ch.isspace())
        counts: dict[int, float] = {}
        for n, weight in ((2, 1.0), (3, 1.4)):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                w = weight * (0.25 if _is_hiragana_only(gram) else 1.0)
                digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                counts[index] = counts.get(index, 0.0) + sign * w
        vector = [0.0] * self.dim
        for index, value in counts.items():
            vector[index] = math.copysign(math.log1p(abs(value)), value)
        return l2_normalize(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    """手元で動く多言語の文埋め込み（fastembed / ONNX）。API の鍵なしで意味の近さを測る。"""

    fusion_weight = 0.25

    def __init__(self, model: str = "") -> None:
        model = model or LOCAL_MODEL
        from fastembed import TextEmbedding

        self.model = model
        self._impl = TextEmbedding(model)

    @property
    def id(self) -> str:
        return f"local:{self.model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize([float(x) for x in v]) for v in self._impl.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class _HttpEmbedder:
    url: str
    model: str
    batch_size = 64
    fusion_weight = 0.25

    def __init__(self, api_key: str, model: str, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("埋め込みAPIの鍵が設定されていません")
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.Client(timeout=60)

    @property
    def id(self) -> str:
        return f"{type(self).__name__}:{self.model}"

    def _payload(self, texts: list[str], kind: str) -> dict[str, object]:
        raise NotImplementedError

    def _call(self, texts: list[str], kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._payload(batch, kind),
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            if len(data) != len(batch):
                raise ValueError("埋め込みAPIの返した件数が合いません")
            vectors.extend(l2_normalize([float(x) for x in d["embedding"]]) for d in data)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "query")[0]


class VoyageEmbedder(_HttpEmbedder):
    url = "https://api.voyageai.com/v1/embeddings"

    def _payload(self, texts: list[str], kind: str) -> dict[str, object]:
        return {"model": self.model, "input": texts, "input_type": kind}


class OpenAIEmbedder(_HttpEmbedder):
    url = "https://api.openai.com/v1/embeddings"

    def _payload(self, texts: list[str], kind: str) -> dict[str, object]:
        return {"model": self.model, "input": texts}


def embedder_from_env() -> Embedder:
    """RAG_EMBEDDER=hashing（既定）/ local / voyage / openai。"""
    kind = os.environ.get("RAG_EMBEDDER", "hashing")
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "local":
        return LocalEmbedder(os.environ.get("RAG_EMBEDDING_MODEL", LOCAL_MODEL))
    if kind == "voyage":
        return VoyageEmbedder(
            os.environ.get("VOYAGE_API_KEY", ""),
            os.environ.get("RAG_EMBEDDING_MODEL", "voyage-3.5"),
        )
    if kind == "openai":
        return OpenAIEmbedder(
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    raise ValueError(f"RAG_EMBEDDER の値が不正です: {kind}")
