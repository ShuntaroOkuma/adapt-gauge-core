# adapt-gauge-core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/ShuntaroOkuma/adapt-gauge-core/actions/workflows/test.yml/badge.svg)](https://github.com/ShuntaroOkuma/adapt-gauge-core/actions/workflows/test.yml)

[English](README.md)

**LLM に手本を何個見せれば賢くなるか？ — 学習の速さと崩壊を自動で測定します。**

本リポジトリは、LLM の **適応効率（Adaptation Efficiency）** を測定するオープンソースの評価ツールです。プロンプトに含める入出力の手本（few-shot examples）の数を 0, 1, 2, 4, 8 と変えながら性能の伸びを計測し、手本を増やしたときにかえって性能が下がる**崩壊現象（Few-Shot Collapse）** を自動で検出します。

## なぜ適応効率を測るのか？

一般的な LLM ベンチマークは、ある一時点での精度しか測りません。しかし実運用では、プロンプトに手本（入力と期待する出力のペア）を添えてモデルをタスクに適応させるのが一般的です。そこで次の 2 つの疑問が生まれます。

1. **手本は何個あれば十分か？** 2 個で頭打ちになるモデルもあれば、8 個まで伸び続けるモデルもあります。
2. **手本を増やして逆効果になることはないか？** モデルとタスクの組み合わせによっては、手本が増えるほど性能が *下がる* ことがあります。これを **崩壊現象** と呼びます。

adapt-gauge-core は、この両方を自動で明らかにします。

私たちの評価では、手本の数によって **リーダーボードの順位が逆転する** 現象（0 個では劣るモデルが 4 個で首位に立つ）や、手本を増やすと **スコアがほぼゼロまで崩壊する** モデルが確認されました。これらは例外的なケースではなく、通常のベンチマークでは見逃される構造的なパターンです。

### 実際の動作例

**4 タスク × 5 モデルの学習曲線:**

![学習曲線の概要](docs/images/learning-curves-overview.png)

**崩壊検出** — gemini-3-flash-preview が 4-shot でピークに達した後、0-shot レベルまで急落:

![崩壊検出](docs/images/learning-curve-collapse.png)

## クイックスタート

### 前提条件

- Python 3.11 以上
- 以下のいずれかのモデルプロバイダーへの API アクセス:
  - **Google Cloud**（Vertex AI）— Gemini モデル
  - **Anthropic** — Claude モデル
  - **LMStudio** — ローカルモデル

### インストール

```bash
git clone https://github.com/ShuntaroOkuma/adapt-gauge-core.git
cd adapt-gauge-core
pip install -e ".[dev]"
```

### 設定

```bash
cp .env.example .env
# .env を編集して API キーを設定
```

### 評価の実行

```bash
# デフォルトモデル（Gemini 3 Flash, Claude Haiku 4.5）で実行
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# モデルを指定して実行
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001

# TF-IDF による例題選択（デフォルト）または固定順で実行
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf

# 2 つの選択方式を比較実行
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection

# 前回の実行を途中から再開
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --run-id 20260101_120000
```

#### CLI オプション一覧

| オプション | 説明 |
|-----------|------|
| `--task-pack` | タスクパック JSON のパス（必須） |
| `--models` | モデル名をカンマ区切りで指定（省略時はデフォルトリストを使用） |
| `--num-trials` | 試行回数 |
| `--run-id` | 指定した実行 ID から途中再開 |
| `--output-dir` | 出力先ディレクトリ（デフォルト: `results`） |
| `--example-selection` | 例題の選択方式: `tfidf`（デフォルト）または `fixed`（固定順） |
| `--compare-selection` | 両方の選択方式を同時に実行して比較 |

### 結果の閲覧

デモ用の評価結果を同梱しているので、評価を実行しなくてもビューアを試せます。

```bash
# ビューア用の依存パッケージをインストール
pip install -e ".[viewer]"

# デモ結果を閲覧（results/demo/ に同梱）
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results/demo

# 自分の評価結果を閲覧
streamlit run src/adapt_gauge_core/viewer.py
```

## 測定項目

モデルとタスクの組み合わせごとに、shot 数（0, 1, 2, 4, 8）を変えて以下を測定します。

| 指標 | 内容 |
|------|------|
| **改善率（Improvement Rate）** | shot を 1 つ増やしたときのスコア上昇幅 |
| **到達 shot 数（Threshold Shots）** | 目標スコア（デフォルト 0.8）に初めて到達する shot 数 |
| **学習曲線 AUC** | 学習曲線の曲線下面積。大きいほど少ない手本で高い性能に到達している |
| **崩壊検出（Collapse Detection）** | 3 種類の独立した崩壊チェック（後述） |
| **崩壊パターン** | stable / immediate_collapse / gradual_decline / peak_regression に分類 |
| **レジリエンススコア** | モデルごとの崩壊耐性を 0〜1 で数値化 |
| **pass@k** | 複数回の試行における再現性の指標 |
| **トークン使用量** | 各評価の入出力トークン数とレイテンシ |

### 崩壊検出

評価パイプラインでは、以下の 3 種類の崩壊チェックを独立に実行します。

| チェック | 検出条件 |
|---------|---------|
| **Few-Shot Collapse (崩壊現象)** | 最終 shot のスコアが 0-shot から 10% 以上低下 |
| **ピーク回帰** | ピーク時のスコアが最終 shot で 20% 以上低下 |
| **途中の急落** | 連続する shot 間で 30% 以上の急落 |

検出結果は **崩壊パターン**（stable / immediate_collapse / gradual_decline / peak_regression）に分類され、モデルごとの **レジリエンススコア**（0.0〜1.0）として集約されます。

## デモ用タスクパック

同梱の `task_pack_core_demo.json` には、採点方式の異なる 4 つのタスクが含まれています。

| タスク | 採点方式 | 対象領域 |
|--------|---------|---------|
| 分類 | exact_match | メールのカテゴリ分類 |
| コード修正 | contains | バグ修正 |
| 要約 | f1 | テキスト要約 |
| 配送ルート | llm_judge | ルート最適化 |

## プロジェクト構成

```
adapt-gauge-core/
├── src/adapt_gauge_core/
│   ├── runner.py              # CLI エントリポイント
│   ├── viewer.py              # Streamlit 結果ビューア
│   ├── prompt_builder.py      # Few-shot プロンプト組み立て
│   ├── example_selector.py    # TF-IDF / 固定順の例題選択
│   ├── task_loader.py         # タスクパック JSON の読み込み
│   ├── efficiency_calc.py     # AUC・改善率・到達 shot 数の算出
│   ├── harness_config.py      # 設定の読み込みと管理
│   ├── domain/                # エンティティ・値オブジェクト・定数
│   ├── scoring/               # 採点: exact_match, contains, f1, llm_judge
│   ├── infrastructure/        # モデルクライアント: Vertex AI, Claude, LMStudio
│   └── use_cases/             # 評価実行・崩壊分析・ヘルスチェック
├── tasks/                     # タスク定義とデモパック
├── results/                   # 評価結果の出力（CSV）
└── tests/                     # テストスイート（264 テスト）
```

## 採点方式

| 方式 | 内容 |
|------|------|
| `exact_match` | 正規化した文字列の完全一致 |
| `contains` | 期待する出力が実際の出力に含まれているか |
| `f1` | トークン単位の F1 スコア（日本語トークナイズにも対応） |
| `llm_judge` | 採点用モデル（grader）による LLM ベースの評価 |

## 設定

すべての設定は環境変数または `.env` ファイルで指定できます。

```bash
# 試行回数
HARNESS_NUM_TRIALS=3           # 各評価の試行回数
HARNESS_AGGREGATION=mean       # 集約方法: mean または median

# LLM Judge
LLM_JUDGE_ENABLED=true
LLM_JUDGE_GRADER_MODEL=gemini-2.5-flash

# 再現性
HARNESS_PASS_AT_K=true
HARNESS_K_VALUES=1,3
```

全設定項目は [.env.example](.env.example) を参照してください。

インストール・設定・例題選択方式・結果の読み方などの詳細は [使い方ガイド](docs/usage-guide_ja.md) を参照してください。

## 開発

```bash
make install         # 開発モードでインストール
make test            # 現在の Python でテスト実行
make test-all        # Python 3.11, 3.12, 3.13 でテスト実行
make run             # デモ用タスクパックで評価を実行
make help            # 全コマンドを表示
```

## Contributing

開発環境のセットアップとガイドラインは [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## 変更履歴

リリース履歴は [CHANGELOG.md](CHANGELOG.md) を参照してください。

## ライセンス

MIT
