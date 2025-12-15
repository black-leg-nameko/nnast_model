# Dynamic Taint Analysis (DTA) Module

軽量なPython向け動的Taint解析ツール。NNASTの学習データ生成用に設計されています。

## 機能

- **Source/Sinkマーキング**: デコレータで簡単にSource/Sinkを定義
- **実行時追跡**: 自動的にTaintフローを追跡
- **JSONL出力**: 標準フォーマットでTaintレコードを出力

## 使い方

### 基本的な使い方

```python
from dta.tracker import taint_source, taint_sink, get_tracker

# Taint sourceをマーク
@taint_source
def get_user_input():
    return input()

# Taint sinkをマーク
@taint_sink("sql_exec")
def execute_sql(query):
    # SQL実行処理
    pass

# 追跡を有効化
tracker = get_tracker()
tracker.enable()

# コードを実行
user_data = get_user_input()
execute_sql(user_data)  # Taintフローが自動記録される

# レコードを取得
records = tracker.get_records()
```

### CLIツールとして使う

```bash
# スクリプトを実行してTaintログを生成
python -m dta.cli example.py --output taint_log.jsonl
```

### CPGとの統合

生成したTaintログは、既存のCPG生成CLIで直接使用できます：

```bash
# CPG生成 + DDFGマージ
python cli.py /path/to/code --taint-log taint_log.jsonl --out graphs.jsonl
```

## 実装状況

### ✅ 実装済み

- Source/Sinkデコレータ
- 基本的な実行時追跡
- JSONL出力
- CPGマージとの統合

### 🚧 今後の拡張予定

- ASTベースの自動インストルメンテーション
- より詳細なパス追跡（中間ノードの記録）
- パフォーマンス最適化
- より複雑なTaint伝播ルール

## 例

`dta/example.py` を参照してください。

