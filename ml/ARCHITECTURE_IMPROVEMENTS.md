# CPG+GNNアーキテクチャ改善提案

## 現状の課題と改善提案

### 🔴 1. エッジ特徴量の未活用（重要度: 高）

**現状**: `edge_attr`は作成されているが、モデルで使用されていない
- CPGにはAST/CFG/DFG/DDFGなど重要なエッジタイプがある
- エッジタイプ情報は脆弱性検出に重要（例: DFGエッジはデータフロー追跡に重要）

**改善案**:
```python
# Edge-aware GNN layer
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import TransformerConv  # Edge-aware attention

# Option 1: Edge-type embedding
self.edge_embedding = nn.Embedding(num_edge_types, edge_dim)

# Option 2: Edge-type specific GNN layers
self.ast_layer = GATConv(hidden_dim, hidden_dim)
self.cfg_layer = GATConv(hidden_dim, hidden_dim)
self.dfg_layer = GATConv(hidden_dim, hidden_dim)
# エッジタイプごとに異なるレイヤーで処理し、後で統合
```

**論文での貢献**: "Edge-type aware vulnerability detection" - CPGの多様なエッジタイプを活用

---

### 🔴 2. ノード特徴量の未活用（重要度: 高）

**現状**: `node_kinds`, `type_hint`, `symbol`などの情報が活用されていない
- CodeBERTのみに依存しているが、構造的情報も重要

**改善案**:
```python
# Multi-modal node features
node_kind_emb = nn.Embedding(num_node_kinds, kind_dim)
type_hint_emb = nn.Embedding(num_types, type_dim)

# Combine: CodeBERT + structural features
x_combined = torch.cat([
    codebert_emb,           # (768-dim)
    node_kind_emb(node_kinds),  # (kind_dim)
    type_hint_emb(type_hints), # (type_dim)
], dim=1)
```

**論文での貢献**: "Hybrid semantic-structural node representation"

---

### 🟡 3. マルチスケール特徴抽出の不足（重要度: 中-高）

**現状**: 各レイヤーの出力を統合していない
- 浅い層: 局所的なパターン
- 深い層: グローバルな依存関係

**改善案**:
```python
# Multi-scale feature aggregation
layer_outputs = []
for layer in self.gnn_layers:
    x = layer(x, edge_index)
    layer_outputs.append(x)

# Hierarchical pooling
x_multi_scale = torch.cat([
    global_mean_pool(layer_outputs[0], batch),  # Local
    global_mean_pool(layer_outputs[-1], batch), # Global
    global_max_pool(layer_outputs[-1], batch),
], dim=1)
```

**論文での貢献**: "Multi-scale vulnerability pattern detection"

---

### 🟡 4. 階層的表現学習の不足（重要度: 中）

**現状**: 関数レベルのみで、ファイル/プロジェクトレベルの情報がない
- 脆弱性は関数間の相互作用で発生することもある

**改善案**:
```python
# Hierarchical GNN
class HierarchicalCPGModel(nn.Module):
    def __init__(self):
        # Function-level GNN
        self.function_gnn = CPGTaintFlowModel(...)
        
        # File-level aggregation
        self.file_gnn = GATConv(...)
        
        # Project-level aggregation
        self.project_gnn = GATConv(...)
```

**論文での貢献**: "Hierarchical code representation for vulnerability detection"

---

### 🟡 5. 位置情報の未活用（重要度: 中）

**現状**: `spans`情報があるが活用されていない
- コードの位置情報は脆弱性パターンと相関がある

**改善案**:
```python
# Positional encoding for code spans
class CodePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        self.line_emb = nn.Embedding(max_lines, d_model // 2)
        self.col_emb = nn.Embedding(max_cols, d_model // 2)
    
    def forward(self, spans):
        line_emb = self.line_emb(spans[:, 0])
        col_emb = self.col_emb(spans[:, 1])
        return torch.cat([line_emb, col_emb], dim=1)
```

---

### 🟢 6. 解釈可能性の不足（重要度: 中-高）

**現状**: どのノード/エッジが重要かを示す機構がない
- 実用化には「なぜ脆弱と判断したか」の説明が必要

**改善案**:
```python
# Attention weights for interpretability
class InterpretableGAT(nn.Module):
    def forward(self, x, edge_index):
        x, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        return x, attention_weights

# Node importance scoring
node_importance = attention_weights.mean(dim=1)
```

**論文での貢献**: "Explainable vulnerability detection with attention visualization"

---

### 🟢 7. 大きなグラフへの対応不足（重要度: 高・実用性）

**現状**: `max_nodes`で切り詰めている
- 実コードでは数千ノードのグラフが一般的

**改善案**:
```python
# Graph sampling strategies
from torch_geometric.loader import NeighborSampler

# Option 1: Subgraph sampling
# Option 2: Hierarchical pooling (先に小さなサブグラフに分割)
# Option 3: Graph Transformer with efficient attention
from torch_geometric.nn import TransformerConv

# Option 4: Graph coarsening
class GraphCoarsening(nn.Module):
    """大きなグラフを小さなグラフに粗化"""
```

**論文での貢献**: "Scalable vulnerability detection for large codebases"

---

### 🟢 8. 転移学習・事前学習の不足（重要度: 高）

**現状**: スクラッチから学習
- コード表現の事前学習が有効

**改善案**:
```python
# Self-supervised pre-training tasks
class PreTrainingTasks:
    # Task 1: Masked code prediction
    # Task 2: Edge prediction (AST/CFG/DFG)
    # Task 3: Node type prediction
    # Task 4: Graph contrastive learning

# Pre-train on large code corpus
# Fine-tune on vulnerability detection
```

**論文での貢献**: "Self-supervised pre-training for code vulnerability detection"

---

### 🟢 9. データ拡張の不足（重要度: 中）

**現状**: データ拡張がない
- コードの多様性を増やす

**改善案**:
```python
# Code augmentation strategies
class CodeAugmentation:
    # 1. Variable renaming (semantic preserving)
    # 2. Dead code insertion
    # 3. Control flow restructuring (equivalent)
    # 4. Graph-level augmentation (edge dropping, node masking)
```

---

### 🟢 10. アテンション機構の改善（重要度: 中）

**現状**: 標準的なマルチヘッドアテンション
- コード特有のパターンに特化したアテンション

**改善案**:
```python
# Code-specific attention mechanisms
class CodeAwareAttention(nn.Module):
    """コードの構造を考慮したアテンション"""
    # 1. Distance-aware attention (近いノードを重視)
    # 2. Edge-type aware attention (DFGエッジを重視)
    # 3. Hierarchical attention (関数内/関数間)
```

---

## 優先度別実装ロードマップ

### Phase 1: 即座に実装すべき（精度向上に直結）
1. ✅ **エッジ特徴量の活用** - Edge-type aware GNN
2. ✅ **ノード特徴量の統合** - Multi-modal node features
3. ✅ **マルチスケール特徴抽出** - Layer-wise aggregation

### Phase 2: 論文の新規性向上
4. ✅ **階層的表現学習** - Function/File/Project levels
5. ✅ **解釈可能性** - Attention visualization
6. ✅ **転移学習** - Self-supervised pre-training

### Phase 3: 実用化
7. ✅ **大きなグラフ対応** - Graph sampling/coarsening
8. ✅ **推論速度最適化** - Model compression, quantization
9. ✅ **データ拡張** - Code augmentation

---

## 論文での差別化ポイント

### 1. **CPGの多様なエッジタイプの活用**
- AST/CFG/DFG/DDFGを区別して処理
- エッジタイプごとの重要度学習

### 2. **マルチモーダル融合**
- CodeBERT（意味） + 構造特徴（ノード種類、型情報）
- 3段階融合（Early/Intermediate/Late）

### 3. **階層的脆弱性検出**
- 関数レベル → ファイルレベル → プロジェクトレベル
- 関数間の相互作用を考慮

### 4. **解釈可能な検出**
- どのノード/エッジが脆弱性に寄与しているか可視化
- 開発者への説明可能性

### 5. **実用的なスケーラビリティ**
- 大規模コードベースへの対応
- 効率的な推論

---

## ベースライン比較の提案

### 比較すべき手法:
1. **Rule-based**: Bandit, SonarQube
2. **Classical ML**: Random Forest (token features)
3. **Sequence models**: Transformer encoder (code as sequence)
4. **Graph models**: 
   - Standard GCN/GAT (baseline)
   - CodeBERT + GNN (ablation)
   - 提案手法（full model）

### 評価指標:
- Top-K Recall (K=5, 10, 20)
- Precision@K
- AUROC, AUPRC
- **新規**: False Positive Rate@K (実用性)
- **新規**: Inference time (実用性)

---

## 実装の優先順位

1. **最優先**: エッジ特徴量活用（実装容易、効果大）
2. **高優先**: ノード特徴量統合、マルチスケール
3. **中優先**: 階層的表現、解釈可能性
4. **低優先**: 転移学習（時間がかかるが効果大）
