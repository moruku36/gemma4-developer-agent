# Gemma 4 Developer Agent — オフライン実行ハーネス

日本語版のみ（コンペ期間中は英語版の公開を停止しています）

**Gemma 4 Developer Agent Competition**（Google DeepMind 主催、Kaggle）
向けの実行基盤です。Kaggle ノートブックの GPU インスタンス（L4x4 等）上
で、Gemma 4 を搭載した自律型コーディングエージェントを**完全オフライン**
かつ**12時間以内**で動作させ、未知の Python リポジトリを探索してバグ修正
パッチ（diff）を生成します。採点は SWE-bench 形式（非公開テストの
Pass/Fail）です。

ファインチューニング済み・ベースラインどちらの Gemma 4 重みでも差し込む
だけで、ローカル性能検証から Kaggle 提出用バンドル化までを一貫して行える
ことを目的にしています。

📖 **ドキュメント**：コンペ自体を一から学ぶためのガイドは [docs/competition-guide.ja.md](docs/competition-guide.ja.md)、設計の詳細は [docs/architecture.ja.md](docs/architecture.ja.md) を参照してください。

## チーム役割分担

| 役割 | 担当 | 責務 |
| --- | --- | --- |
| インフラ / エージェントシステムリード | 本リポジトリ | オフライン実行ハーネス、vLLM サービング基盤、エージェントのアクションループ、ファイル探索・編集ツール、構文・テスト検証サンドボックス、Kaggle 提出パイプライン |
| ML / モデリングリード | 共同作業者 | Gemma 4 の事後学習（PEFT/LoRA）、プロンプトチューニング、推論最適化。成果物としてチェックポイントを本基盤へ渡す |

両者のインタフェースは意図的に薄くしています：重みディレクトリ
（重み＋トークナイザー）と `configs/` の設定ファイルだけです。

## アーキテクチャ

```text
Kaggle ノートブック（GPU、インターネット OFF、合計12時間以内）
│
├─ submission/notebook_template.ipynb   入口：バンドル特定 → サーバー起動
│                                       → 全タスク実行 → predictions 出力
├─ src/engine/vllm_server.py            vLLM サブプロセスのライフサイクル
│                                       （起動 / /health 監視 / 停止）
├─ src/engine/client.py                 OpenAI 互換ローカルクライアント
│                                       （標準ライブラリのみで実装）
├─ src/agent/core.py                    ReAct / Plan-and-Solve ループ。
│                                       steps・トークン・時間の上限を強制
├─ src/agent/prompts.py                 システムプロンプト＋厳密な JSON
│                                       ツールプロトコル＋ Few-shot 例
├─ src/agent/memory.py                  コンテキスト予算管理と
│                                       ツール出力のプルーニング（圧縮）
├─ src/tools/file_ops.py                読み込み / 編集 / 検索 / AST 概要。
│                                       対象リポジトリ外へのアクセス不可
├─ src/tools/patch.py                   git diff 生成。検証は pristine
│                                       clone への git apply --check と
│                                       変更 .py の compile() チェック
├─ src/tools/execution.py               タイムアウト付きサブプロセス実行
│                                       （プロセスグループごと kill、
│                                       出力は上限付き）
└─ src/pipeline/runner.py               バッチランナー：タスクごとに
                                        リポジトリの使い捨てコピーで実行、
                                        全体で1つの時間予算を管理
```

主な設計判断：

- **ツール呼び出しはテキストプロトコル**。モデルの返答は1ターンに
  JSON オブジェクト1つだけ（ツール呼び出し、または `finish`）。
  ネイティブ function calling に依存しないため、ベースライン重みでも
  そのまま動き、ログも完全に残ります。
- **決定論的なガードレール**。ステップ数・トークン予算・制限時間は
  プロンプトではなくハーネス側（`configs/default_agent.yaml`）で強制。
- **検証を通らないパッチは提出しない**。元の HEAD に clean に適用でき、
  変更した Python ファイルがすべてコンパイルできる場合のみ成功扱い。
- **プロンプトインジェクション対策**。リポジトリや Issue の本文は
  「データ」として扱い、ファイル内に埋め込まれた指示には従わないこと
  をシステムプロンプトで明示。ファイルツールは対象リポジトリの外に
  出られません。

## オフライン制約

スコアリング中はインターネット接続不可です。以下はすべて事前に
バンドル化します。

- モデル重み＋トークナイザー（`models/gemma-4/`、Kaggle データセット化）
- pip wheel（`wheels/`、`pip install --no-index` で導入）
- 本コードと設定（バンドル内 `code/`）

ネットワークを使うのは `scripts/setup_offline_deps.sh` のみで、
これは準備フェーズ専用です（Kaggle 上では実行しません）。

## 12時間予算の内訳（デフォルト設定）

| フェーズ | 予算 |
| --- | --- |
| ノートブック準備＋オフライン install | 約10分 |
| vLLM 起動（L4x4 へのモデルロード） | 最大15分（ヘルスチェック上限） |
| 全タスクのエージェント実行 | 約11時間（全体上限 41,400 秒） |
| タスクごとの上限 | 60 steps / 400k トークン / 30分 |
| predictions 生成＋予備 | 15分を予備として温存 |

## ローカル実行（スモークテストは GPU 不要）

```bash
pip install -e ".[dev]"
pytest                                   # 単体テスト（偽LLMクライアント）

# ハーネス end-to-end スモークテスト（スクリプト駆動クライアント、
# 小さなバグ修正タスク2件で 探索→編集→テスト→パッチ を検証）：
python scripts/run_local_eval.py --demo

# 実モデルでのローカル評価（vLLM サーバー起動後）：
python scripts/run_local_eval.py --tasks tasks.json --output artifacts/local-eval
```

`tasks.json` の形式：

```json
[{"task_id": "demo-1", "repo_path": "/path/to/checkout",
  "issue_text": "バグの説明...", "test_cmd": "python3 -m pytest -q"}]
```

## Kaggle 提出バンドルの作成

```bash
# 1. wheel と重みを取得（インターネット ON、準備フェーズのみ）
MODEL_ID=<ファインチューニング済みチェックポイントのHFリポジトリ> \
  bash scripts/setup_offline_deps.sh
#    ローカル出力を使う場合は MODEL_LOCAL_DIR=/path/to/checkpoint

# 2. バンドルを組み立てる（コード＋ノートブック＋重み＋wheel＋タスク）
python scripts/build_submission.py --output artifacts/bundle \
    --model-dir models/gemma-4 --wheels-dir wheels --tasks tasks.json

# 3. artifacts/bundle を Kaggle データセットとしてアップロードし、
#    submission/notebook_template.ipynb からノートブックを作成、
#    データセットをアタッチ、GPU 有効・インターネット OFF で実行。
#    出力： /kaggle/working/predictions.jsonl
```

## ディレクトリ構成

```text
configs/     エージェントと vLLM の設定（調整面はこの2つ）
src/agent/   ループ・プロンプト・メモリ    src/engine/  サービング＋クライアント
src/tools/   ファイル操作・パッチ・実行    src/pipeline/ ランナー＋エクスポーター
scripts/     オフライン準備・ローカル評価・バンドル作成
tests/       pytest 単体テスト（GPU / vLLM 不要）
submission/  Kaggle 提出用ノートブックテンプレート
```

## ステータス

Scaffold v0.1：ハーネス、検証、パッケージング、テストは実装済みで
グリーンです。今後の反復：リポジトリナビゲーションの強化（シンボル
グラフ）、失敗フィードバック付きリトライ、ML リードからのチェック
ポイント差し替え対応。
