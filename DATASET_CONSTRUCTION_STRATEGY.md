# 論文レベルのデータセット構築戦略

**目標**: 脆弱性検出に使えるPythonコード集として、論文単独で発表できるレベルのデータセットを構築する

---

## 📊 現状分析

### 現在のデータセット規模
- **総サンプル数**: 約990サンプル
  - 実在データ: 約376ファイル
  - 合成データ: 約450サンプル
  - 学習用: 315サンプル（圧倒的に不足）

### 論文レベルのデータセット規模（参考）
- **BigVul**: 95,000+ サンプル（C/C++）
- **Devign**: 27,000+ サンプル（C/C++）
- **CodeXGLUE**: 10,000+ サンプル（多言語）
- **Python特化**: 現状、大規模データセットが存在しない

### 目標規模
- **最小目標**: 5,000サンプル（論文として発表可能）
- **推奨目標**: 10,000サンプル（高品質な研究）
- **理想目標**: 20,000+サンプル（ベンチマークとしての価値）

---

## 🎯 データセット構築の価値提案

### なぜデータセット構築だけでも論文になるのか？

1. **Python Web特化の大規模データセットが存在しない**
   - 既存データセットは主にC/C++に焦点
   - Python Webアプリケーション特化のデータセットは希少

2. **OWASP Top 10との整合性**
   - 15パターンID × OWASP Top 10マッピング
   - 実用的な脆弱性カテゴリを網羅

3. **再現可能な構築手法**
   - 合成データ + 実在データのハイブリッドアプローチ
   - パターンマッチングベースの品質保証

4. **ベンチマークとしての価値**
   - 他の研究者が使用できる標準データセット
   - 継続的な拡張と改善

---

## 📋 データセット構築戦略

### Phase 1: 大規模データ収集（2-3ヶ月）

#### 1.1 実在データの収集（目標: 5,000-8,000サンプル）

**ソース1: CVE関連コミット**
```bash
# 複数年度にわたって収集
for year in 2020 2021 2022 2023 2024; do
    python -m data.collect_dataset \
        --github-query "CVE-$year" \
        --limit 200 \
        --output-dir ./dataset_cve_$year
done
```
**期待サンプル数**: 1,000-1,500

**ソース2: GitHub Security Advisories**
```bash
# GHSAからPythonプロジェクトを抽出
python -m data.collect_from_ghsa \
    --language python \
    --limit 500 \
    --output-dir ./dataset_ghsa
```
**期待サンプル数**: 500-800

**ソース3: セキュリティ修正コミット（多様なクエリ）**
```bash
# 多様なクエリパターンで収集
QUERIES=(
    "security fix python"
    "vulnerability fix python"
    "fix CVE python"
    "security patch python"
    "vulnerability patch python"
    "CVE fix python"
    "security update python"
    "vulnerability update python"
)

for query in "${QUERIES[@]}"; do
    python -m data.collect_dataset \
        --github-query "$query" \
        --limit 100 \
        --output-dir ./dataset_security_fixes
done
```
**期待サンプル数**: 2,000-3,000

**ソース4: 特定脆弱性タイプの収集**
```bash
# OWASP Top 10カテゴリ別に収集
python -m data.collect_diverse_vulnerabilities \
    --vuln-types sql_injection xss command_injection \
                 path_traversal deserialization ssrf \
                 authentication authorization crypto \
    --limit-per-type 100 \
    --output-dir ./dataset_owasp
```
**期待サンプル数**: 1,500-2,500

**ソース5: セキュリティプロジェクト・テストケース**
```bash
# OWASP関連プロジェクト
python -m data.collect_dataset \
    --github-query "OWASP python" \
    --limit 200 \
    --output-dir ./dataset_owasp_projects

# セキュリティツールのテストケース
python -m data.collect_dataset \
    --github-query "security scanner test python" \
    --limit 200 \
    --output-dir ./dataset_security_tools

# CTF・教育用脆弱アプリ
python -m data.collect_dataset \
    --github-query "CTF python web" \
    --limit 200 \
    --output-dir ./dataset_ctf
```
**期待サンプル数**: 1,000-1,500

**合計期待サンプル数**: 6,000-9,300サンプル

#### 1.2 合成データの生成（目標: 2,000-3,000サンプル）

**戦略**: 実在データで不足しているパターンを補完

```bash
# パターンID別の不足分を分析
python -m data.analyze_gaps \
    --real-dataset ./dataset_real \
    --target-samples-per-pattern 200 \
    --output gaps_analysis.json

# 不足分を合成データで生成
python -m data.generate_balanced_dataset \
    --output-dir ./dataset_synthetic \
    --samples-per-pattern 200 \
    --frameworks flask django fastapi \
    --vulnerable-ratio 0.5 \
    --complexity-levels simple medium complex
```

**期待サンプル数**: 2,000-3,000（15パターン × 約130-200サンプル/パターン）

#### 1.3 データ拡張（目標: 1,000-2,000サンプル）

**コード変換による拡張**
```python
# 変数名の変更
# コメントの追加/削除
# 関数の分割/統合
# 制御フローの変更（if文の追加、ループの変更）
```

**期待サンプル数**: 1,000-2,000（既存データから生成）

---

### Phase 2: データ品質保証（1ヶ月）

#### 2.1 自動品質チェック

```bash
# 包括的な品質検証
python -m data.validate_dataset \
    dataset/metadata.jsonl \
    --labels dataset/labels.jsonl \
    --output validation_results.json \
    --check-all
```

**チェック項目**:
- Graph-Label Alignment
- Pattern Distribution
- Framework Distribution
- Source/Sink Coverage
- CPG Quality
- 重複検出

#### 2.2 手動検証（サンプリング）

- 各パターンIDからランダムに10-20サンプルを抽出
- 専門家による脆弱性の有無の検証
- 誤ラベルの修正

#### 2.3 データクリーニング

- 重複の除去
- 品質の低いサンプルの除去
- パターンIDの再分類

---

### Phase 3: データセットの統合とバランス調整（2週間）

#### 3.1 データセットの統合

```bash
# 実在データ + 合成データ + 拡張データを統合
python -m data.merge_datasets \
    --real-dataset ./dataset_real \
    --synthetic-dataset ./dataset_synthetic \
    --augmented-dataset ./dataset_augmented \
    --output-dir ./dataset_final \
    --real-ratio 0.6 \
    --synthetic-ratio 0.3 \
    --augmented-ratio 0.1
```

#### 3.2 バランス調整

```bash
# パターンID別のバランス調整
python -m data.balance_dataset \
    --input-dataset ./dataset_final \
    --target-samples-per-pattern 300 \
    --output-dir ./dataset_balanced
```

**目標分布**:
- パターンID別: 各パターンで200-400サンプル
- フレームワーク別: Flask 40%, Django 30%, FastAPI 30%
- 複雑度: シンプル 40%, 中程度 40%, 複雑 20%
- ラベル: 脆弱 50%, 安全 50%

#### 3.3 データセット分割

```bash
# Project-split / Time-split戦略
python -m data.split_dataset \
    --input-dataset ./dataset_balanced \
    --split-strategy project_time \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

---

### Phase 4: データセットのドキュメント化（1週間）

#### 4.1 データセット仕様書の作成

- データセットの概要
- 収集方法とソース
- データセット統計
- 品質保証プロセス
- 使用ライセンス

#### 4.2 ベンチマーク評価

- 既存手法との比較
- ベースライン性能の測定
- データセットの難易度分析

---

## 🔧 実装が必要なツール

### 1. 大規模データ収集ツール

**新規実装が必要**:
- `data/collect_from_ghsa.py` - GitHub Security Advisoriesからの収集
- `data/collect_from_nvd.py` - NVDからの収集
- `data/collect_parallel.py` - 並列収集ツール
- `data/collect_scheduled.py` - スケジュール実行ツール

### 2. データ拡張ツール

**新規実装が必要**:
- `data/augment_code.py` - コード変換による拡張
- `data/augment_control_flow.py` - 制御フロー変更による拡張
- `data/augment_variables.py` - 変数名変更による拡張

### 3. データ品質保証ツール

**拡張が必要**:
- `data/validate_dataset.py` - 既存ツールを拡張
- `data/detect_duplicates.py` - 重複検出
- `data/quality_scoring.py` - 品質スコアリング

### 4. データセット統合ツール

**新規実装が必要**:
- `data/merge_datasets.py` - データセット統合
- `data/balance_dataset.py` - バランス調整
- `data/split_dataset.py` - データセット分割（Project-split/Time-split）

---

## 📈 タイムライン

### Month 1-2: 大規模データ収集
- Week 1-2: CVE関連コミットの収集
- Week 3-4: GitHub Security Advisoriesからの収集
- Week 5-6: セキュリティ修正コミットの収集
- Week 7-8: 特定脆弱性タイプの収集

### Month 3: 合成データ生成とデータ拡張
- Week 1-2: 合成データ生成ツールの改善
- Week 3-4: データ拡張ツールの実装と実行

### Month 4: データ品質保証
- Week 1-2: 自動品質チェックの実行
- Week 3-4: 手動検証とデータクリーニング

### Month 5: データセット統合とドキュメント化
- Week 1-2: データセットの統合とバランス調整
- Week 3-4: ドキュメント化とベンチマーク評価

---

## 💡 論文としての価値

### 1. データセットの独自性
- **Python Web特化**: 既存データセットにない特化性
- **OWASP Top 10整合**: 実用的な脆弱性カテゴリを網羅
- **大規模**: 5,000-10,000サンプル（論文レベル）

### 2. 構築手法の再現性
- **ハイブリッドアプローチ**: 実在データ + 合成データ
- **パターンマッチングベース**: 品質保証の自動化
- **Project-split/Time-split**: 適切な評価設計

### 3. ベンチマークとしての価値
- **標準データセット**: 他の研究者が使用可能
- **継続的な拡張**: コミュニティによる改善
- **評価指標**: 標準的な評価指標の定義

### 4. 実用的な価値
- **実在データ中心**: 実際のプロジェクトで使われているコード
- **多様性**: 複数のフレームワーク、複雑度、パターン
- **品質保証**: 手動検証と自動チェックの両立

---

## 🎯 成功指標

### データセット規模
- ✅ **最小目標**: 5,000サンプル
- ✅ **推奨目標**: 10,000サンプル
- ✅ **理想目標**: 20,000+サンプル

### データセット品質
- ✅ Source/Sink coverage: >80%
- ✅ パターンID別分布: 各パターンで200+サンプル
- ✅ フレームワーク分布: Flask/Django/FastAPI均等
- ✅ 手動検証精度: >95%

### データセットの価値
- ✅ 論文として発表可能
- ✅ 他の研究者が使用可能
- ✅ ベンチマークとしての価値
- ✅ 継続的な拡張と改善

---

## 🚀 次のステップ

### 即座に開始可能

1. **大規模データ収集の開始**
   ```bash
   # CVE関連コミットの収集を開始
   python -m data.collect_dataset \
       --github-query "CVE-2023" \
       --limit 200 \
       --output-dir ./dataset_cve_2023
   ```

2. **収集ツールの改善**
   - GitHub Security Advisories対応
   - 並列収集の実装
   - レート制限の自動管理

3. **データ拡張ツールの実装**
   - コード変換ツール
   - 制御フロー変更ツール

### 中期的な実装

1. **データ品質保証の強化**
   - 自動品質チェックの拡張
   - 重複検出の実装
   - 品質スコアリングの実装

2. **データセット統合ツールの実装**
   - データセット統合
   - バランス調整
   - Project-split/Time-split

---

## 📝 結論

**データセット構築だけでも論文レベルの価値がある**

1. **Python Web特化の大規模データセットが存在しない**
2. **OWASP Top 10との整合性**
3. **再現可能な構築手法**
4. **ベンチマークとしての価値**

**推奨アプローチ**:
1. 大規模データ収集を開始（2-3ヶ月）
2. 合成データ生成とデータ拡張（1ヶ月）
3. データ品質保証（1ヶ月）
4. データセット統合とドキュメント化（1ヶ月）

**目標**: 5,000-10,000サンプルの高品質データセットを構築し、論文として発表する。

