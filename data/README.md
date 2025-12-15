# Python Vulnerability Dataset Collection

このディレクトリには、NNASTの学習用Python脆弱性データセットを収集・構築するためのツールが含まれています。

## クイックスタート

### 1. GitHub Tokenの設定

**方法A: セットアップスクリプトを使用（推奨）**

```bash
./data/quick_setup.sh
```

**方法B: 手動設定**

```bash
# 一時的な設定（現在のセッションのみ）
export GITHUB_TOKEN=your_token_here

# 永続的な設定（~/.zshrcに追加）
echo 'export GITHUB_TOKEN=your_token_here' >> ~/.zshrc
source ~/.zshrc

# または、.envファイルを作成
echo "GITHUB_TOKEN=your_token_here" > .env
```

詳細は [`setup_token.md`](setup_token.md) を参照してください。

### 2. データセットの収集

```bash
# CVE関連のコミットを検索して収集
python -m data.collect_dataset \
    --github-query "CVE" \
    --limit 50 \
    --output-dir ./dataset
```

### 3. 学習データの準備

```bash
python -m data.prepare_training_data \
    --dataset-dir ./dataset \
    --output-dir ./training_data
```

## 概要

Python特化の脆弱性検出データセットは、研究コミュニティにとって重要なリソースです。このパイプラインは、以下のソースから自動的にデータを収集します：

1. **GitHub**: CVE参照を含むPythonリポジトリ
2. **CVEデータベース**: NVD、GitHub Security Advisories（今後実装予定）
3. **セキュリティプロジェクト**: セキュリティに焦点を当てたPythonプロジェクト

## データセット構造

```
dataset/
├── raw/                    # 生データ（クローンしたリポジトリなど）
├── processed/              # 処理済みデータ（CPGグラフなど）
├── metadata.jsonl          # メタデータ（CVE ID、CWE ID、コミット情報など）
└── README.md
```

## 使用方法

### 1. 特定のリポジトリから収集

```bash
python -m data.collect_dataset \
    --repo https://github.com/user/repo \
    --commit-before abc123 \
    --commit-after def456 \
    --file-path src/vulnerable.py \
    --output-dir ./dataset
```

### 2. GitHubから自動収集

```bash
python -m data.collect_dataset \
    --github-query "CVE" \
    --limit 100 \
    --output-dir ./dataset
```

### 3. より具体的な検索クエリ

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
```

詳細は [`example_usage.md`](example_usage.md) を参照してください。

## データセットの特徴

- **脆弱コードと修正コードのペア**: 各レコードには、脆弱なバージョンと修正後のバージョンの両方が含まれます
- **CPGグラフ**: 両方のバージョンに対してCPGグラフが自動生成されます
- **メタデータ**: CVE ID、CWE ID、脆弱性タイプ、説明などの情報が含まれます
- **ラベル付きデータ**: 脆弱性の有無とタイプが明確にラベル付けされています

## 論文としての価値

このデータセット構築パイプライン自体が、以下の理由で研究価値があります：

1. **Python特化**: 既存のデータセット（BigVul、Devignなど）は主にC/C++に焦点を当てており、Python特化のデータセットは限られています
2. **自動化**: データ収集からCPG生成まで自動化されたパイプライン
3. **品質管理**: 脆弱性の検証とラベリングのプロセス
4. **再現性**: 他の研究者が同じデータセットを再構築できる

## 実装状況

- [x] GitHub API統合の完全実装
- [x] 脆弱性タイプの自動分類
- [x] データセットの統計情報生成
- [ ] CVEデータベースからの自動収集
- [ ] データ品質の自動検証
- [ ] データセットの公開準備

## 注意事項

- GitHub APIを使用する場合は、レート制限に注意してください（認証時5,000リクエスト/時間）
- リポジトリのクローンには時間がかかる場合があります
- 一部の脆弱性コードは実行に危険が伴う可能性があるため、サンドボックス環境での処理を推奨します
- `.env`ファイルは既に`.gitignore`に含まれているため、Gitにコミットされません

## 関連ファイル

- [`setup_token.md`](setup_token.md) - GitHub Token設定の詳細ガイド
- [`example_usage.md`](example_usage.md) - 使用例とトラブルシューティング
- [`github_api.py`](github_api.py) - GitHub APIクライアント実装
- [`collect_dataset.py`](collect_dataset.py) - データ収集スクリプト
- [`prepare_training_data.py`](prepare_training_data.py) - 学習データ準備スクリプト
