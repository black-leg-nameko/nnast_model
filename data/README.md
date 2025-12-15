# Python Vulnerability Dataset Collection

このディレクトリには、NNASTの学習用Python脆弱性データセットを収集・構築するためのツールが含まれています。

## 概要

Python特化の脆弱性検出データセットは、研究コミュニティにとって重要なリソースです。このパイプラインは、以下のソースから自動的にデータを収集します：

1. **GitHub**: CVE参照を含むPythonリポジトリ
2. **CVEデータベース**: NVD、GitHub Security Advisories
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

### 2. GitHubから自動収集（要APIトークン）

```bash
export GITHUB_TOKEN=your_github_token
python -m data.collect_dataset \
    --github-query "language:python CVE" \
    --limit 100 \
    --output-dir ./dataset
```

### 3. 特定のCVE IDから収集

```bash
python -m data.collect_dataset \
    --cve-list CVE-2023-1234 CVE-2023-5678 \
    --output-dir ./dataset
```

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

## 今後の拡張

- [ ] GitHub API統合の完全実装
- [ ] CVEデータベースからの自動収集
- [ ] 脆弱性タイプの自動分類
- [ ] データ品質の自動検証
- [ ] データセットの統計情報生成
- [ ] データセットの公開準備

## 注意事項

- GitHub APIを使用する場合は、レート制限に注意してください
- リポジトリのクローンには時間がかかる場合があります
- 一部の脆弱性コードは実行に危険が伴う可能性があるため、サンドボックス環境での処理を推奨します

