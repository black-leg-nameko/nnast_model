# データ収集ガイド

## 本格的なデータ収集

### 方法1: 単一クエリで収集（推奨）

```bash
# CVE-2023関連を30件収集
python -m data.collect_dataset \
    --github-query "CVE-2023" \
    --limit 30 \
    --output-dir ./dataset

# CVE-2024関連を30件収集
python -m data.collect_dataset \
    --github-query "CVE-2024" \
    --limit 30 \
    --output-dir ./dataset
```

### 方法2: 複数クエリで自動収集

```bash
# 複数のクエリを順番に実行
./data/collect_full_dataset.sh
```

このスクリプトは以下のクエリを順番に実行します：
- CVE-2023
- CVE-2024
- fix CVE
- security CVE
- vulnerability fix

### 方法3: バックグラウンドで実行

```bash
# バックグラウンドで実行（長時間かかる場合）
nohup python -m data.collect_dataset \
    --github-query "CVE-2023" \
    --limit 100 \
    --output-dir ./dataset \
    > collection.log 2>&1 &

# 進捗確認
tail -f collection.log
```

## 進捗確認

```bash
# 進捗を確認
./data/check_progress.sh

# または手動で確認
wc -l dataset/metadata.jsonl  # レコード数
ls -lh dataset/processed/     # CPGグラフファイル
```

## レート制限について

GitHub APIのレート制限：
- **認証済み**: 5,000リクエスト/時間
- **未認証**: 60リクエスト/時間

大量のデータを収集する場合：
- 各クエリの間に10-30秒の待機時間を設ける
- 複数のクエリを分散して実行する
- レート制限に達した場合は自動的に待機します

## 推奨設定

### 小規模テスト（10-30件）
```bash
python -m data.collect_dataset \
    --github-query "CVE-2023" \
    --limit 20 \
    --output-dir ./dataset_test
```

### 中規模収集（50-100件）
```bash
python -m data.collect_dataset \
    --github-query "CVE-2023" \
    --limit 50 \
    --output-dir ./dataset
```

### 大規模収集（100件以上）
```bash
# 複数のクエリに分けて実行
for query in "CVE-2023" "CVE-2024" "fix CVE"; do
    python -m data.collect_dataset \
        --github-query "$query" \
        --limit 50 \
        --output-dir ./dataset
    sleep 30  # レート制限を避けるため待機
done
```

## データ収集の最適化

1. **クエリの選択**: より具体的なクエリ（例: "CVE-2023"）の方が、関連性の高い結果が得られます
2. **時間帯**: GitHub APIの負荷が低い時間帯（深夜など）に実行すると効率的です
3. **並列実行**: 複数のターミナルで異なるクエリを並列実行できますが、レート制限に注意

## トラブルシューティング

### レート制限に達した場合
- 自動的に待機しますが、長時間かかる場合があります
- 1時間待ってから再実行することを推奨します

### データが少ない場合
- より具体的なクエリを試す（例: "CVE-2023-1234"）
- 異なるクエリを試す（例: "security fix", "vulnerability patch"）

### プロセスが停止した場合
- 既に収集されたデータは保存されています
- 同じ出力ディレクトリで再実行すると、既存のデータに追加されます

