# adapt-gauge-core 使い方ガイド

インストール、設定、評価の実行、結果の解釈に関する詳細ガイドです。

## 目次

- [インストール](#インストール)
- [設定](#設定)
- [評価の実行](#評価の実行)
- [例文選択方式](#例文選択方式)
- [タスクパック形式](#タスクパック形式)
- [スコアリング方式](#スコアリング方式)
- [出力ファイル](#出力ファイル)
- [結果の閲覧](#結果の閲覧)
- [崩壊検出と分類](#崩壊検出と分類)
- [評価の再開](#評価の再開)
- [対応モデル](#対応モデル)
- [トラブルシューティング](#トラブルシューティング)

---

## インストール

### 前提条件

- Python 3.11以上
- 以下のいずれかのモデルプロバイダーへのAPIアクセス:
  - **Google Cloud** (Vertex AI): Geminiモデル用
  - **Anthropic**: Claudeモデル用
  - **LMStudio**: ローカルモデル用

### Google Cloudのセットアップ（Geminiモデル使用時）

Vertex AI経由でGeminiモデルを使用する場合、事前にGoogle Cloudの認証が必要です:

```bash
gcloud auth login
gcloud auth application-default login
```

その後、`.env`にプロジェクトIDを設定してください:

```
GCP_PROJECT_ID=your-project-id
```

### インストール手順

```bash
git clone https://github.com/ShuntaroOkuma/adapt-gauge-core.git
cd adapt-gauge-core
pip install -e ".[dev]"
```

Streamlitビューアーを使用する場合:

```bash
pip install -e ".[viewer]"
```

---

## 設定

環境ファイルのサンプルをコピーし、APIキーを設定してください:

```bash
cp .env.example .env
```

### APIキー（必須）

| 変数 | 説明 |
|------|------|
| `GCP_PROJECT_ID` | Google CloudプロジェクトID（Geminiモデル用） |
| `ANTHROPIC_API_KEY` | Anthropic APIキー（Claudeモデル用） |
| `LMSTUDIO_BASE_URL` | LMStudioサーバーURL（デフォルト: `http://localhost:1234/v1`） |
| `LMSTUDIO_API_KEY` | LMStudio APIキー（デフォルト: `lm-studio`） |

### 評価設定

| 変数 | デフォルト値 | 説明 |
|------|------------|------|
| `HARNESS_NUM_TRIALS` | 3 | 評価あたりの試行回数 |
| `HARNESS_AGGREGATION` | mean | 集約方法: `mean`（平均）または `median`（中央値） |
| `HARNESS_SUCCESS_THRESHOLD` | 0.8 | threshold_shots指標の目標スコア |
| `HARNESS_PASS_AT_K` | true | pass@k信頼性指標を計算するか |
| `HARNESS_K_VALUES` | 1,3 | pass@kのK値（カンマ区切り） |

### 分離設定

| 変数 | デフォルト値 | 説明 |
|------|------------|------|
| `HARNESS_NEW_CLIENT_PER_TRIAL` | true | 試行ごとに新しいクライアントを生成（状態の漏洩を防止） |
| `HARNESS_TIMEOUT_SECONDS` | 120 | モデル応答のタイムアウト（秒） |
| `HARNESS_MAX_RETRIES` | 3 | 失敗時の最大リトライ回数 |
| `HARNESS_RETRY_DELAY_SECONDS` | 1.0 | リトライ間の基本遅延（秒） |

### LLM Judge設定

| 変数 | デフォルト値 | 説明 |
|------|------------|------|
| `LLM_JUDGE_ENABLED` | true | LLMベースのスコアリングを有効化 |
| `LLM_JUDGE_GRADER_MODEL` | gemini-2.5-flash | 採点に使用するモデル |
| `LLM_JUDGE_TIMEOUT_SECONDS` | 30 | 採点モデルの応答タイムアウト |
| `LLM_JUDGE_MAX_RETRIES` | 2 | 採点モデルのリトライ回数 |
| `LLM_JUDGE_FALLBACK_METHOD` | f1 | 採点モデルが失敗した場合の代替スコアリング方式 |

---

## 評価の実行

### CLIオプション

```bash
python -m adapt_gauge_core.runner [OPTIONS]
```

| オプション | 必須 | デフォルト値 | 説明 |
|-----------|------|------------|------|
| `--task-pack PATH` | Yes | - | タスクパックJSONファイルのパス |
| `--models LIST` | No | DEFAULT_MODELS | カンマ区切りのモデル名リスト |
| `--num-trials N` | No | .envから取得 | 試行回数 |
| `--run-id ID` | No | 自動生成 | 実行ID（再開時にも使用） |
| `--output-dir DIR` | No | `results` | CSV出力ディレクトリ |
| `--example-selection METHOD` | No | `fixed` | 例文選択方式: `fixed` または `tfidf` |
| `--compare-selection` | No | false | fixedとtfidfの両方を比較実行 |

### 実行例

```bash
# デフォルトモデルで基本実行
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# モデルを指定して実行
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001

# TF-IDF例文選択を使用
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf

# fixed vs TF-IDF の比較モード
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection

# 出力ディレクトリと試行回数をカスタマイズ
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --num-trials 5 \
  --output-dir results/experiment1
```

### 評価フロー

1. **ヘルスチェック** - 各モデルにシンプルなプロンプトを送信してテスト
2. **採点モデルのヘルスチェック** - LLM Judgeの採点モデルをテスト（llm_judgeタスクが存在する場合）
3. **評価実行** - 選択方式 > 試行 > モデル > タスク > ショット数 > テストケースの順に反復
4. **結果集約** - モデル・タスクごとのサマリー指標を算出
5. **崩壊検出** - Few-Shot Collapse、ピーク回帰、中間曲線の落ち込みを識別
6. **パターン分類** - 各モデル・タスクペアを崩壊パターンタイプに分類
7. **結果保存** - 生データとサマリーのCSVを出力

---

## 例文選択方式

### Fixed（デフォルト）

タスクパックJSONに定義された順序で例文を使用します。エグゼンプラー（正例）とディストラクター（ノイズ例）の数はショット設定に従います:

| ショット数 | エグゼンプラー | ディストラクター |
|-----------|-------------|---------------|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 1 | 1 |
| 4 | 2 | 2 |
| 8 | 6 | 2 |

### TF-IDF

TF-IDFコサイン類似度を使用して、テスト入力に最も類似した例文を動的に選択します。文字n-gram（`char_wb`、2-4グラム）を使用することで、日本語を含む多言語でのロバストなマッチングを実現します。

```bash
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --example-selection tfidf
```

### 比較モード

同一設定で`fixed`と`tfidf`の両方を順番に実行し、生データCSVに`example_selection`を記録します。例文選択方式がFew-Shot Collapseの発生に影響するかを調査できます。

```bash
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --compare-selection
```

---

## タスクパック形式

タスクパックは、複数の評価タスクを含むJSONファイルです。

### 構造

```json
{
  "pack_id": "my_pack",
  "pack_name": "評価パック名",
  "description": "パックの説明",
  "version": "1.0",
  "categories": ["classification", "summarization"],
  "tasks": [...]
}
```

### タスク定義

```json
{
  "task_id": "classification_001",
  "category": "classification",
  "version": "1.0",
  "difficulty": "medium",
  "description": "タスクの簡潔な説明",
  "instruction": "モデルへの詳細な指示",
  "measures": ["Acquisition", "Fidelity"],
  "examples": [
    {"input": "入力例", "output": "期待される出力"}
  ],
  "test_cases": [
    {
      "input": "テスト入力",
      "expected_output": "期待される出力",
      "scoring_method": "exact_match",
      "acceptable_variations": ["許容される変形1", "許容される変形2"]
    }
  ],
  "distractors": [
    {"input": "紛らわしい入力", "output": "紛らわしい出力"}
  ]
}
```

### フィールド説明

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `task_id` | Yes | 一意の識別子 |
| `category` | Yes | タスクカテゴリ |
| `difficulty` | Yes | `low`、`medium`、または `hard` |
| `description` | Yes | タスクの簡潔な説明 |
| `instruction` | No | モデルへの詳細な指示（存在する場合プロンプトに使用） |
| `examples` | Yes | Few-shot用の例文（8-shotの場合、最低6件推奨） |
| `test_cases` | Yes | 期待される出力を含む評価ケース |
| `distractors` | No | ロバスト性テスト用のノイズ例 |
| `measures` | No | 評価軸: `Acquisition`, `Resilience-Noise`, `Resilience-Detect`, `Efficiency`, `Agency`, `Fidelity` |
| `scoring_method` | Yes（test_caseごと） | `exact_match`, `contains`, `f1`, または `llm_judge` |
| `acceptable_variations` | No（test_caseごと） | 許容される代替回答（llm_judgeで使用） |

---

## スコアリング方式

| 方式 | 説明 | 適したタスク |
|------|------|------------|
| `exact_match` | 正規化された文字列の一致判定（大文字小文字無視、markdown除去、Unicode NFKC正規化） | 分類、ラベリング |
| `contains` | 正規化後の出力内に期待される出力が含まれるか | コード修正、キーワード抽出 |
| `f1` | トークンレベルのF1スコア（日本語テキスト対応） | 要約、自由形式のテキスト |
| `llm_judge` | 採点モデルを使用したLLMベースの評価 | 複雑な推論、経路最適化 |

### LLM Judgeの詳細

LLM Judgeは3つの`expected_output`形式をサポートします:

1. **キーワードリスト**（文字列）: 出力にキーワードが含まれるかチェック。スコア: 0.0または1.0。
2. **自然文テキスト**（文字列）: ルーブリックベースの6段階評価（0.0, 0.2, 0.4, 0.6, 0.8, 1.0）。
3. **辞書形式ルーブリック**（dict）: カスタム基準によるスコア（0.0, 0.5, 1.0）。

---

## 出力ファイル

### 生データCSV (`raw_results_{run_id}.csv`)

個別の評価結果ごとに1行。

| カラム | 説明 |
|-------|------|
| `run_id` | 実行識別子 |
| `task_id` | タスク識別子 |
| `category` | タスクカテゴリ |
| `model_name` | 使用したモデル |
| `shot_count` | ショット数（0, 1, 2, 4, 8） |
| `input` | テストケースの入力 |
| `expected_output` | 期待される出力 |
| `actual_output` | モデルの実際の応答 |
| `score` | 評価スコア（0.0-1.0） |
| `scoring_method` | 使用したスコアリング方式 |
| `latency_ms` | 応答レイテンシー（ミリ秒） |
| `timestamp` | ISOタイムスタンプ |
| `trial_id` | 試行番号 |
| `input_tokens` | 入力トークン数 |
| `output_tokens` | 出力トークン数 |
| `example_selection` | 使用した選択方式（`fixed` または `tfidf`） |

### サマリーCSV (`summary_{run_id}.csv`)

モデル・タスクの組み合わせごとに1行（試行間で集約）。

| カラム | 説明 |
|-------|------|
| `task_id` | タスク識別子 |
| `category` | タスクカテゴリ |
| `model_name` | モデル名 |
| `score_0shot` - `score_8shot` | 各ショット数での平均/中央値スコア |
| `improvement_rate` | (score_8shot - score_0shot) / 8 |
| `threshold_shots` | 成功閾値（0.8）に到達する最小ショット数 |
| `learning_curve_auc` | 学習曲線の曲線下面積 |
| `num_trials` | 試行回数 |
| `score_variance` | 試行間のスコア分散 |
| `collapse_pattern` | パターン分類（stable, immediate_collapse, gradual_decline, peak_regression） |
| `resilience_score` | モデルの耐性スコア（0.0-1.0） |
| `pass_@1`, `pass_@3` | pass@k信頼性指標（オプション） |

---

## 結果の閲覧

### Streamlitビューアー

```bash
# デモ結果を表示（同梱）
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results/demo

# 自分の結果を表示
streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results
```

### ビューアーのセクション

1. **学習曲線** - ショット数ごとのスコア推移を表示するインタラクティブなPlotlyチャート。Few-Shot Collapse区間は赤色でハイライト表示されます。
2. **崩壊検出** - 3種類のパフォーマンス劣化に対する警告:
   - **Few-Shot Collapse (崩壊現象)**: 最終スコアが0-shotベースラインを下回る
   - **ピーク回帰（Peak Regression）**: 中間ショットでスコアがピークに達した後に低下
   - **中間曲線の落ち込み（Mid-curve Dip）**: 隣接するショット数間での急激なスコア低下
3. **崩壊パターン分類** - 各モデル・タスクペアをstable、immediate_collapse、gradual_decline、peak_regressionに分類したテーブル。
4. **崩壊耐性スコア** - モデルごとのスコア（0.0 = 常に崩壊、1.0 = 完全に安定）。
5. **指標サマリー** - 算出されたすべての指標を含む詳細テーブル。

---

## 崩壊検出と分類

### 検出タイプ

| タイプ | 条件 | 深刻度 |
|-------|------|--------|
| **Few-Shot Collapse (崩壊現象)** | 最終スコア < 0-shotスコアの90% | degradation（10-50%低下）/ collapse（50%以上低下） |
| **ピーク回帰** | ピーク > 0-shotの110% かつ 最終 < ピークの80% | - |
| **中間曲線の落ち込み** | 隣接ショット数間のスコア低下 > 30% | - |

### 崩壊パターン分類

各モデル・タスクペアは以下の4パターンのいずれかに分類されます:

| パターン | 説明 |
|---------|------|
| `stable` | 顕著な劣化なし（単調増加、横ばい、または全体で10%未満の低下） |
| `immediate_collapse` | 0-shot直後の急激な低下が持続（最初の低下が全体の60%以上） |
| `gradual_decline` | ショット数全体にわたる緩やかな低下（全体で10%以上の低下が均等に分散） |
| `peak_regression` | 中間ショットでスコアが向上した後、大幅に低下 |

### 耐性スコア

モデルごとのスコアで、以下の要素を組み合わせて算出:
- **パターンタイプペナルティ**: stable=0.0, gradual_decline=0.5, peak_regression=0.6, immediate_collapse=1.0
- **低下幅**: 実際のパフォーマンス低下率でスケーリング

スコア = 1.0 - (パターンペナルティ × 低下率)、モデルの全タスクの平均値。

---

## 評価の再開

評価が中断された場合、中断箇所から再開できます:

```bash
# 初回実行（run_idが自動生成される、例: 20260205_143000）
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json

# 同じrun_idを指定して再開
python -m adapt_gauge_core.runner \
  --task-pack tasks/task_pack_core_demo.json \
  --run-id 20260205_143000
```

ランナーは既存の`raw_results_{run_id}.csv`を読み込み、完了済みの評価をスキップします。`--compare-selection`使用時も、選択方式がスキップキーに含まれるため正しく動作します。

---

## 対応モデル

### デフォルトモデル

```
gemini-3-flash-preview
gemini-2.5-flash
claude-haiku-4-5-20251001
```

### 全対応モデル

| プロバイダー | モデル | 必要な設定 |
|------------|--------|-----------|
| **Vertex AI** | gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview | `GCP_PROJECT_ID` |
| **Anthropic** | claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929, claude-opus-4-5-20251101 | `ANTHROPIC_API_KEY` |
| **LMStudio** | 任意のローカルモデル（`lmstudio/`プレフィックスを付与） | `LMSTUDIO_BASE_URL` |

### モデルの指定方法

```bash
# 単一モデル
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash

# 複数モデル
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models gemini-2.5-flash,claude-haiku-4-5-20251001,gemini-3-flash-preview

# LMStudio経由のローカルモデル
python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json \
  --models lmstudio/llama-3.1-8b
```

---

## トラブルシューティング

### よくある問題

**モデルのヘルスチェックが失敗する**
- `.env`のAPIキーを確認してください
- ネットワーク接続を確認してください
- Vertex AIの場合: `gcloud auth application-default login`が設定されていることを確認してください

**LLM Judgeのスコアリングが失敗する**
- ランナーは自動的にf1スコアリングにフォールバックします
- `LLM_JUDGE_GRADER_MODEL`にアクセス可能か確認してください
- タイムアウトが発生する場合は`LLM_JUDGE_TIMEOUT_SECONDS`を増やしてください

**Streamlitビューアーが読み込まれない**
- `pip install -e ".[viewer]"`が実行済みであることを確認してください
- 指定した`--results-dir`に結果CSVファイルが存在することを確認してください

### テストの実行

```bash
# 全テスト実行
make test

# 特定のテストファイルを実行
python -m pytest tests/test_example_selector.py -v

# カバレッジ付きで実行
python -m pytest tests/ --cov=adapt_gauge_core
```
