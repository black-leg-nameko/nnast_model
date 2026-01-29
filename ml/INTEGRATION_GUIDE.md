# NNASTアーキテクチャ改善統合ガイド

## 統合完了日
2025年1月29日

## 統合内容

### Phase 1: 必須改善（統合済み）

#### 1.1 EdgeTypeWeightedGATLayer ✅
- **統合状態**: `CPGTaintFlowModel`のGNNレイヤーに統合済み
- **使用方法**: `use_edge_types=True`で自動的に使用されます
- **変更点**: `EdgeTypeAwareGATLayer`から`EdgeTypeWeightedGATLayer`に置き換え

#### 1.2 HierarchicalPooling ✅
- **統合状態**: 改善版が統合済み（`num_levels=3`）
- **使用方法**: `use_hierarchical=True`で自動的に使用されます
- **新機能**: `file_ids`パラメータに対応（ファイルレベルのプーリング）

### Phase 2: オプション改善（統合済み）

#### 2.1 SequentialContextEncoder ✅
- **統合状態**: オプション機能として統合済み
- **使用方法**: `use_sequential_context=True`で有効化
- **動作**: CodeBERT埋め込みをシーケンシャルコンテキストで強化

#### 2.2 MultiScaleFeatureFusion ✅
- **統合状態**: オプション機能として統合済み
- **使用方法**: `use_multiscale_fusion=True`で有効化
- **動作**: 各GNNレイヤーの後にマルチスケール融合を適用

### Phase 3: オプション改善（統合済み）

#### 3.1 AdaptiveSliceSelector ✅
- **統合状態**: オプション機能として統合済み
- **使用方法**: `use_adaptive_slice=True`で有効化
- **動作**: ノードエンコーディングの後にスライス選択を適用

## 使用例

### 基本設定（Phase 1のみ）

```python
model = CPGTaintFlowModel(
    input_dim=768,
    hidden_dim=256,
    num_layers=3,
    gnn_type="GAT",
    dropout=0.5,
    use_edge_types=True,  # Phase 1.1: EdgeTypeWeightedGATLayer
    use_hierarchical=True,  # Phase 1.2: Improved HierarchicalPooling
    use_multi_modal=True,
    use_multi_scale=True,
    use_positional=True,
)
```

### フル機能設定（Phase 1 + 2 + 3）

```python
model = CPGTaintFlowModel(
    input_dim=768,
    hidden_dim=256,
    num_layers=3,
    gnn_type="GAT",
    dropout=0.5,
    # Phase 1 (必須)
    use_edge_types=True,
    use_hierarchical=True,
    # Phase 2 (オプション)
    use_sequential_context=True,  # Phase 2.1
    use_multiscale_fusion=True,   # Phase 2.2
    # Phase 3 (オプション)
    use_adaptive_slice=True,      # Phase 3.1
    # その他
    use_multi_modal=True,
    use_multi_scale=True,
    use_positional=True,
)
```

## データ形式の要件

### Phase 1.2 (HierarchicalPooling) で必要なデータ

```python
# Batchオブジェクトに以下を追加
data.function_ids = torch.tensor([0, 0, 1, 1, 2, ...])  # 各ノードの関数ID
data.file_ids = torch.tensor([0, 0, 0, 1, 1, ...])     # 各ノードのファイルID（オプション）
```

### Phase 2.1 (SequentialContextEncoder) で必要なデータ

```python
# Batchオブジェクトに以下を追加（オプション）
data.node_order = torch.tensor([0, 1, 2, 3, ...])  # ノードの順序インデックス
```

### Phase 3.1 (AdaptiveSliceSelector) で必要なデータ

```python
# Batchオブジェクトに以下を追加（オプション）
data.vulnerability_type = torch.tensor([0, 1, 2, ...])  # 脆弱性タイプヒント
```

## パフォーマンス考慮事項

### 計算コスト

| 機能 | 計算コスト増加 | 推奨設定 |
|------|--------------|---------|
| Phase 1.1: EdgeTypeWeightedGAT | +5% | 常時有効 |
| Phase 1.2: HierarchicalPooling | +10% | 常時有効 |
| Phase 2.1: SequentialContext | +20% | 必要時のみ |
| Phase 2.2: MultiScaleFusion | +25% | 必要時のみ |
| Phase 3.1: AdaptiveSlice | +15% | 必要時のみ |

### メモリ使用量

- Phase 2.1とPhase 2.2はメモリ使用量が増加します
- 大きなグラフでは、Phase 2.2の全結合グラフ構築に注意が必要です

## 段階的導入の推奨

1. **Step 1**: Phase 1のみで動作確認
2. **Step 2**: Phase 2.1を追加して評価
3. **Step 3**: Phase 2.2を追加して評価
4. **Step 4**: Phase 3.1を追加して評価

## トラブルシューティング

### エラー: `file_ids`が見つからない

```python
# 解決策: file_idsをNoneに設定（後方互換性あり）
data.file_ids = None
```

### エラー: `node_order`が見つからない

```python
# 解決策: SequentialContextEncoderが自動的に順序を生成
# または、明示的に設定
data.node_order = torch.arange(data.x.size(0))
```

### エラー: MultiScaleFusionでメモリ不足

```python
# 解決策: use_multiscale_fusion=Falseに設定
# または、バッチサイズを減らす
```

## 次のステップ

1. **評価**: 各Phaseの効果をアブレーション研究で検証
2. **最適化**: ハイパーパラメータの調整
3. **ベンチマーク**: 既存ベースラインとの比較

## 参考資料

- `ml/IMPLEMENTATION_SUMMARY.md`: 実装の詳細
- `NNAST_ARCHITECTURE_IMPROVEMENTS.md`: 理論的根拠
