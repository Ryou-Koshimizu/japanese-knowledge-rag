# ADR 003：埋め込み・回答AI・認証を差し替えられる境界にする

- 状態：採用
- 日付：2026-10-20

## 決定

3つの境界を `Protocol` で定め、実装を環境変数で選ぶ。

| 境界 | 実装 | 選び方 |
| --- | --- | --- |
| `Embedder`（埋め込み） | `HashingEmbedder`（依存なし）／`LocalEmbedder`（fastembed・多言語）／`VoyageEmbedder`／`OpenAIEmbedder` | `RAG_EMBEDDER` |
| `Answerer`（回答AI） | `ClaudeAnswerer`（本番）／`CodexAnswerer`（測定用）／`ReplayAnswerer`（公開デモ） | `RAG_MODE`・`RAG_ANSWERER` |
| `PrincipalResolver`（誰が聞いているか） | `DemoResolver`（架空の社員を選ぶ）／`TokenResolver`（利用者ごとのトークン、SHA-256 だけ保存） | `RAG_AUTH` |

## 理由

- CI と公開デモは鍵なしで動かし、本番は API の埋め込みと Claude を使う、という使い分けを1つのコードで行う。
- 埋め込みのモデルが変わったら索引を作り直す必要がある。`Embedder.id` を索引に記録し、変わったら全件を作り直す（`ingest.py`）。
- 社内のシングルサインオンにつなぐときは `PrincipalResolver` を1つ書けばよい。部署と役職さえ渡せば、検索の権限は同じ仕組みで効く。

## 回答AI

- 本番は Claude API（既定 `claude-sonnet-5-5`、`RAG_ANSWER_MODEL` で変更）。構造化出力（`output_config.format`）で `found / answer / citations` の形を保証し、指示文はプロンプトキャッシュに載せる。安全上の判定で断られたときはサーバー側の代替（`fallbacks: "default"`）が続きを答える。
- 測定は、このMacに Claude API の鍵がないため、ChatGPT の Codex CLI（`gpt-6.1-sol`）を同じ指示文・同じ出力の形で呼んで行った。Codex は読み取り専用のサンドボックスで、シェル・ウェブ検索・画像などの道具をすべて止めて呼ぶ。
