# GitHub API統合の使用方法

## セットアップ

### 1. GitHub Personal Access Tokenの取得

1. GitHubにログイン
2. Settings → Developer settings → Personal access tokens → Tokens (classic)
3. "Generate new token (classic)"をクリック
4. 以下のスコープを選択：
   - `public_repo` (公開リポジトリへのアクセス)
   - `repo` (プライベートリポジトリへのアクセス、必要な場合)
5. トークンを生成し、安全に保存

### 2. 環境変数の設定

```bash
export GITHUB_TOKEN=your_token_here
```

または、`.env`ファイルを作成：

```
GITHUB_TOKEN=your_token_here
```

## 使用方法

### 基本的な使用方法

```bash
# CVE関連のコミットを検索して収集
python -m data.collect_dataset \
    --github-query "CVE" \
    --limit 50 \
    --output-dir ./dataset
```

### より具体的な検索クエリ

```bash
# 特定の年のCVEを検索
python -m data.collect_dataset \
    --github-query "CVE-2023" \
    --limit 100 \
    --output-dir ./dataset

# セキュリティ修正を検索
python -m data.collect_dataset \
    --github-query "security fix" \
    --limit 50 \
    --output-dir ./dataset

# SQLインジェクション関連を検索
python -m data.collect_dataset \
    --github-query "SQL injection" \
    --limit 30 \
    --output-dir ./dataset
```

### 特定のリポジトリから収集

```bash
python -m data.collect_dataset \
    --repo https://github.com/user/repo \
    --commit-before abc123def456 \
    --commit-after 789ghi012jkl \
    --file-path src/vulnerable.py \
    --output-dir ./dataset
```

## レート制限について

GitHub APIにはレート制限があります：

- **認証なし**: 1時間あたり60リクエスト
- **認証あり**: 1時間あたり5,000リクエスト

この実装では、自動的にレート制限をチェックし、必要に応じて待機します。

## 出力形式

収集されたデータは以下の構造で保存されます：

```
dataset/
├── metadata.jsonl          # メタデータ（CVE ID、CWE ID、コミット情報など）
├── processed/
│   ├── {commit_before}_{filename}_before.jsonl  # 脆弱コードのCPGグラフ
│   └── {commit_after}_{filename}_after.jsonl   # 修正コードのCPGグラフ
└── raw/                    # クローンしたリポジトリ（オプション）
```

### metadata.jsonlの形式

各行はJSONオブジェクト：

```json
{
  "cve_id": "CVE-2023-1234",
  "cwe_id": "CWE-79",
  "repo_url": "https://github.com/user/repo",
  "commit_before": "abc123...",
  "commit_after": "def456...",
  "file_path": "src/vulnerable.py",
  "vulnerability_type": "XSS",
  "description": "Fix XSS vulnerability in user input",
  "timestamp": "2024-01-01T12:00:00"
}
```

## トラブルシューティング

### エラー: "GITHUB_TOKEN not set"

環境変数`GITHUB_TOKEN`が設定されていません。上記のセットアップ手順を参照してください。

### エラー: "Rate limit exceeded"

レート制限に達しました。しばらく待ってから再試行してください。実装では自動的に待機しますが、大量のデータを収集する場合は時間がかかる可能性があります。

### エラー: "Not found"

指定したリポジトリやコミットが見つかりません。URLやコミットハッシュが正しいか確認してください。

## 次のステップ

データ収集後、学習データを準備：

```bash
python -m data.prepare_training_data \
    --dataset-dir ./dataset \
    --output-dir ./training_data
```

その後、モデルを学習：

```bash
python -m ml.train \
    --graphs ./training_data/train_graphs.jsonl \
    --labels ./training_data/train_labels.jsonl \
    --output-dir ./checkpoints
```

