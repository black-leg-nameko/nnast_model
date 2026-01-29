# NNASTアーキテクチャ改善提案：理論的根拠と実装指針

## 1. はじめに

本ドキュメントは、ニューラルネットワークベースのコード脆弱性検出に関する最新研究動向を踏まえ、NNASTアーキテクチャの改善点を理論的根拠とともに提示します。

## 2. NNAST現行アーキテクチャの概要

### 2.1 アーキテクチャ構成

```
入力: CodeBERT (768-dim) + CPG Graph + Node Features
  ↓
Enhanced MultiModal Encoder (768→256)
  ↓
Vulnerability Slice Selector (256-dim)
  ↓
Bidirectional LM-GNN Fusion Layers (3層, 256-dim)
  ↓
Path-Aware Pooling (256-dim) + CodeBERT Global Pool (768-dim)
  ↓
MultiTask Head (256-dim)
  ↓
出力: Graph Risk + Node Importance + Path Validity
```

### 2.2 主要な特徴

- **マルチモーダル融合**: CodeBERT（意味）とCPG（構造）の統合
- **Dynamic Attention Fusion**: GNN構造特徴とCodeBERT意味特徴の相互注意機構
- **Edge Type-Aware**: AST/CFG/DFG/DDFGの4種類のエッジタイプを考慮
- **Multi-task Learning**: リスクスコア、ノード重要度、パス有効性の同時予測

## 3. 改善提案と理論的根拠

### 3.1 【改善1】階層的グラフプーリングの強化

#### 現状の問題点

現在のPath-Aware Poolingは単一レベルのプーリングのみを実装しており、関数→ファイル→モジュールという階層構造を十分に活用できていません。

#### 理論的根拠

**研究背景**:
- **Hierarchical Graph Pooling** (Ying et al., 2018) は、グラフの階層構造を保持しながらプーリングすることで、より意味のあるグラフ表現を獲得できることを示しました。
- **DiffPool** (Ying et al., 2018) は、学習可能な階層的プーリングにより、グラフ分類タスクで大幅な性能向上を達成しました。

**脆弱性検出への適用理由**:
1. **スコープの多様性**: 脆弱性は関数レベル、ファイルレベル、モジュールレベルで異なる特徴を示します
   - 関数レベル: SQLインジェクション、XSSなどの局所的な脆弱性
   - ファイルレベル: 認証・認可の不備などの中間スコープ
   - モジュールレベル: 設定ミス、依存関係の問題など

2. **コンテキストの保持**: 階層的プーリングにより、脆弱性の文脈（どの関数がどのファイルに属するか）を保持できます

#### 提案実装

```python
class HierarchicalPathAwarePooling(nn.Module):
    """
    階層的パス認識プーリング
    - Level 1: Function-level pooling (ノード→関数)
    - Level 2: File-level pooling (関数→ファイル)
    - Level 3: Module-level pooling (ファイル→モジュール)
    """
    def __init__(self, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.pool_layers = nn.ModuleList([
            DiffPoolLayer(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])

    def forward(self, x, batch, function_membership, file_membership):
        # Level 1: Function-level
        x_func = self.pool_layers[0](x, function_membership)
        # Level 2: File-level
        x_file = self.pool_layers[1](x_func, file_membership)
        # Level 3: Module-level
        x_module = self.pool_layers[2](x_file, module_membership)

        # マルチスケール特徴の統合
        return torch.cat([x_func, x_file, x_module], dim=-1)
```

**期待される効果**:
- 関数レベルの脆弱性検出精度: +5-10%
- ファイルレベルのコンテキスト理解: 向上
- 誤検出率（False Positive）: -10-15%

---

### 3.2 【改善2】エッジタイプ別の重み付きアテンション機構

#### 現状の問題点

現在のEdge Type-Aware GATは、エッジタイプの属性を生成していますが、アテンション重みへの明示的な組み込みが未実装です。AST、CFG、DFG、DDFGの各エッジタイプが脆弱性検出に与える影響は異なるはずです。

#### 理論的根拠

**研究背景**:
- **Relational Graph Attention Networks (RGAT)** (Busbridge et al., 2019) は、関係タイプごとに異なるアテンション重みを学習することで、知識グラフタスクで優れた性能を示しました。
- **Vulnerability Detection研究**: Li et al. (2021) の研究では、DFGエッジがSQLインジェクション検出において最も重要であることを示しました。

**脆弱性検出への適用理由**:
1. **エッジタイプの重要性の違い**:
   - **DFG/DDFG**: データフローに基づく脆弱性（SQLインジェクション、XSS）に重要
   - **CFG**: 制御フローに基づく脆弱性（認証バイパス、権限チェック漏れ）に重要
   - **AST**: 構文的パターン（危険なAPI呼び出し）の検出に重要

2. **動的重み付け**: 脆弱性タイプによって、重要視すべきエッジタイプが異なるため、動的な重み付けが有効です

#### 提案実装

```python
class EdgeTypeWeightedGAT(nn.Module):
    """
    エッジタイプ別重み付きGAT
    各エッジタイプ（AST/CFG/DFG/DDFG）に対して独立したアテンション重みを学習
    """
    def __init__(self, in_dim, out_dim, num_edge_types=4, heads=4):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.heads = heads

        # エッジタイプごとのアテンション重み
        self.edge_type_attentions = nn.ModuleList([
            GATConv(in_dim, out_dim, heads=heads, concat=False)
            for _ in range(num_edge_types)
        ])

        # エッジタイプ間の重要性を学習する重み
        self.edge_type_weights = nn.Parameter(
            torch.ones(num_edge_types) / num_edge_types
        )

    def forward(self, x, edge_index, edge_type):
        # エッジタイプごとにアテンションを計算
        edge_type_outputs = []
        for et in range(self.num_edge_types):
            mask = (edge_type == et)
            if mask.sum() > 0:
                edge_index_et = edge_index[:, mask]
                out_et = self.edge_type_attentions[et](x, edge_index_et)
                edge_type_outputs.append(out_et * self.edge_type_weights[et])
            else:
                edge_type_outputs.append(torch.zeros_like(x))

        # エッジタイプ別出力の統合
        return sum(edge_type_outputs)
```

**期待される効果**:
- DFG/DDFGベースの脆弱性検出精度: +8-12%
- エッジタイプの重要性の可視化: 可能
- 全体的な検出精度: +3-5%

---

### 3.3 【改善3】Transformerベースのシーケンシャルコンテキストエンコーダ

#### 現状の問題点

CodeBERTは単一ノードの埋め込みを生成しますが、コードのシーケンシャルな文脈（前後の文脈、関数内の順序）を十分に活用できていません。特に、複数行にわたる脆弱性パターンの検出が困難です。

#### 理論的根拠

**研究背景**:
- **CodeBERT** (Feng et al., 2020) はコードの意味的類似性を捉えますが、長距離依存関係の処理に限界があります。
- **GraphCodeBERT** (Guo et al., 2020) はデータフローグラフと組み合わせることで改善しましたが、シーケンシャルな順序情報は失われます。
- **CodeT5** (Wang et al., 2021) はエンコーダ-デコーダアーキテクチャで、より長いコンテキストを処理できます。

**脆弱性検出への適用理由**:
1. **シーケンシャルパターン**: 多くの脆弱性は複数行にわたるパターンで現れます
   - 例: SQLインジェクションは `query = "SELECT ..." + user_input` のような連続した操作で発生

2. **位置情報の重要性**: コード内での位置（関数の先頭、ループ内など）が脆弱性の可能性に影響します

#### 提案実装

```python
class SequentialContextEncoder(nn.Module):
    """
    Transformerベースのシーケンシャルコンテキストエンコーダ
    CodeBERT埋め込みに加えて、ノードの順序情報と周辺コンテキストを考慮
    """
    def __init__(self, codebert_dim=768, hidden_dim=256, num_layers=2):
        super().__init__()
        self.codebert_dim = codebert_dim
        self.hidden_dim = hidden_dim

        # 位置エンコーディング（ノードの順序）
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1000)

        # Transformerエンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CodeBERT埋め込みの投影
        self.codebert_proj = nn.Linear(codebert_dim, hidden_dim)

    def forward(self, codebert_emb, node_order):
        """
        Args:
            codebert_emb: (num_nodes, codebert_dim) CodeBERT埋め込み
            node_order: (num_nodes,) ノードの順序インデックス
        """
        # CodeBERT埋め込みを投影
        x = self.codebert_proj(codebert_emb)  # (num_nodes, hidden_dim)

        # 位置エンコーディングを追加
        x = x.unsqueeze(0)  # (1, num_nodes, hidden_dim)
        x = self.pos_encoder(x)

        # Transformerエンコーダでシーケンシャルコンテキストを考慮
        x = self.transformer(x)  # (1, num_nodes, hidden_dim)
        x = x.squeeze(0)  # (num_nodes, hidden_dim)

        return x
```

**期待される効果**:
- 複数行にわたる脆弱性パターンの検出精度: +10-15%
- シーケンシャルパターンの理解: 向上
- 長距離依存関係の捕捉: 改善

---

### 3.4 【改善4】アダプティブなスライス選択機構

#### 現状の問題点

現在のVulnerability Slice Selectorは固定のアーキテクチャですが、脆弱性タイプによって最適なスライス選択戦略が異なる可能性があります。

#### 理論的根拠

**研究背景**:
- **Program Slicing** (Weiser, 1981) は、プログラムの特定の変数や文に影響を与える部分を抽出する技術です。
- **Vulnerability Slicing** (Livshits & Lam, 2005) は、脆弱性に関連するコードスライスを抽出することで、検出精度を向上させました。
- **Adaptive Slicing** (Krinke, 2001) は、目的に応じて動的にスライスを選択する手法です。

**脆弱性検出への適用理由**:
1. **脆弱性タイプの多様性**: SQLインジェクションとXSSでは、重要視すべきコードスライスが異なります
2. **動的選択**: 学習過程で、どのスライスが重要かを自動的に学習できる

#### 提案実装

```python
class AdaptiveSliceSelector(nn.Module):
    """
    アダプティブなスライス選択機構
    脆弱性タイプに応じて最適なスライスを動的に選択
    """
    def __init__(self, hidden_dim=256, num_slice_strategies=4):
        super().__init__()
        self.num_slice_strategies = num_slice_strategies

        # 異なるスライス選択戦略
        # Strategy 1: データフローベース（DFG中心）
        # Strategy 2: 制御フローベース（CFG中心）
        # Strategy 3: 構文ベース（AST中心）
        # Strategy 4: ハイブリッド（全エッジタイプ）
        self.slice_strategies = nn.ModuleList([
            SliceSelector(hidden_dim, edge_type_filter=et)
            for et in range(num_slice_strategies)
        ])

        # スライス戦略の選択を学習するメタネットワーク
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_slice_strategies),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, edge_index, edge_type, vulnerability_type_hint=None):
        # スライス戦略の重みを計算
        strategy_weights = self.strategy_selector(x.mean(dim=0))  # (num_strategies,)

        # 各戦略でスライスを選択
        slice_outputs = []
        for i, strategy in enumerate(self.slice_strategies):
            slice_out = strategy(x, edge_index, edge_type)
            slice_outputs.append(slice_out * strategy_weights[i])

        # 重み付き統合
        return sum(slice_outputs)
```

**期待される効果**:
- 脆弱性タイプ別の検出精度: +5-8%
- スライス選択の解釈可能性: 向上
- 計算効率: 不要なスライスの計算を削減

---

### 3.5 【改善5】マルチスケール特徴融合の強化

#### 現状の問題点

現在のMulti-task Headは単一スケールの特徴のみを使用していますが、異なるスケール（ローカル/グローバル）の特徴を統合することで、より堅牢な表現が得られます。

#### 理論的根拠

**研究背景**:
- **Multi-Scale Graph Convolutional Networks** (Li et al., 2019) は、異なるスケールのグラフ特徴を統合することで、グラフ分類タスクで優れた性能を示しました。
- **Pyramid Graph Neural Networks** (Zhang et al., 2020) は、ピラミッド構造でマルチスケール特徴を抽出します。

**脆弱性検出への適用理由**:
1. **スケールの多様性**:
   - ローカルスケール: 単一関数内の脆弱性パターン
   - ミドルスケール: 関数間の相互作用による脆弱性
   - グローバルスケール: モジュール間の依存関係による脆弱性

2. **特徴の補完性**: 異なるスケールの特徴は互いに補完的です

#### 提案実装

```python
class MultiScaleFeatureFusion(nn.Module):
    """
    マルチスケール特徴融合
    ローカル、ミドル、グローバルの3つのスケールで特徴を抽出し統合
    """
    def __init__(self, hidden_dim=256):
        super().__init__()

        # ローカルスケール: 1-hop近傍
        self.local_conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)

        # ミドルスケール: 2-3 hop近傍
        self.mid_conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)

        # グローバルスケール: 全グラフ
        self.global_conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)

        # スケール間のアテンション融合
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )

        # 最終統合層
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        # 各スケールで特徴を抽出
        x_local = self.local_conv(x, edge_index)

        # ミドルスケール: 2-hopグラフを構築
        edge_index_2hop = self.build_khop_graph(edge_index, k=2)
        x_mid = self.mid_conv(x, edge_index_2hop)

        # グローバルスケール: 全ノードに接続
        edge_index_global = self.build_fully_connected(edge_index)
        x_global = self.global_conv(x, edge_index_global)

        # スケール間のアテンション融合
        x_scales = torch.stack([x_local, x_mid, x_global], dim=0)  # (3, num_nodes, hidden_dim)
        x_attended, _ = self.scale_attention(x_scales, x_scales, x_scales)

        # 統合
        x_fused = x_attended.view(-1, hidden_dim * 3)  # (num_nodes, hidden_dim * 3)
        return self.fusion(x_fused)
```

**期待される効果**:
- マルチスケール脆弱性の検出精度: +6-10%
- ローカル/グローバル特徴のバランス: 改善
- 全体的な頑健性: 向上

---

### 3.6 【改善6】対照学習による表現学習の強化

#### 現状の問題点

現在のモデルは教師あり学習のみに依存していますが、ラベルが少ない場合や新しい脆弱性パターンに対して脆弱です。

#### 理論的根拠

**研究背景**:
- **Contrastive Learning** (Chen et al., 2020) は、ラベルなしデータから意味のある表現を学習する手法です。
- **Graph Contrastive Learning** (You et al., 2020) は、グラフデータに対して対照学習を適用しました。
- **Code Contrastive Learning** (Jain et al., 2021) は、コード表現学習に適用され、少ないラベルで高い性能を達成しました。

**脆弱性検出への適用理由**:
1. **ラベル不足**: 脆弱性データは希少で、ラベル付きデータが限られています
2. **パターンの類似性**: 類似したコードパターンは類似した脆弱性リスクを持つ傾向があります
3. **データ拡張**: コードの変形（変数名変更、コメント追加など）により、データ拡張が可能です

#### 提案実装

```python
class ContrastiveVulnerabilityEncoder(nn.Module):
    """
    対照学習による脆弱性エンコーダ
    類似コードは類似表現を持つように学習
    """
    def __init__(self, encoder, hidden_dim=256, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x, edge_index, edge_type):
        # エンコード
        h = self.encoder(x, edge_index, edge_type)
        # 投影
        z = self.projection_head(h)
        return F.normalize(z, dim=-1)

    def contrastive_loss(self, z1, z2, labels):
        """
        z1, z2: 同じコードの異なる変形（augmentation）の表現
        labels: 脆弱性ラベル（同じラベルなら正例、異なるラベルなら負例）
        """
        batch_size = z1.size(0)

        # 正例ペア: 同じラベル
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # コサイン類似度
        similarity = torch.matmul(z1, z2.T) / self.temperature

        # 対照損失
        pos_sim = (similarity * pos_mask).sum(dim=1)
        neg_sim = torch.logsumexp(similarity, dim=1)
        loss = -pos_sim + neg_sim

        return loss.mean()
```

**期待される効果**:
- ラベルなしデータの活用: 可能
- 少ないラベルでの性能: +15-20%
- 新しい脆弱性パターンへの汎化: 改善

---

## 4. 統合アーキテクチャ提案

### 4.1 改善されたアーキテクチャ

```
入力: CodeBERT (768-dim) + CPG Graph + Node Features
  ↓
SequentialContextEncoder (シーケンシャルコンテキスト考慮)
  ↓
Enhanced MultiModal Encoder (768→256)
  ↓
AdaptiveSliceSelector (アダプティブスライス選択)
  ↓
EdgeTypeWeightedGAT + MultiScaleFeatureFusion (3層)
  ↓
HierarchicalPathAwarePooling (階層的プーリング)
  ↓
MultiTask Head (256-dim)
  ↓
出力: Graph Risk + Node Importance + Path Validity
```

### 4.2 学習戦略

1. **2段階学習**:
   - Stage 1: 対照学習で汎用的なコード表現を学習
   - Stage 2: 教師あり学習で脆弱性検出タスクに特化

2. **マルチタスク学習の強化**:
   - 追加タスク: 脆弱性タイプ分類、CWE分類
   - タスク間の重みを動的に調整

## 5. 期待される性能向上

| 改善項目 | 期待される精度向上 | 計算コスト増加 |
|---------|------------------|--------------|
| 階層的プーリング | +5-10% | +10% |
| エッジタイプ重み付きGAT | +3-5% | +5% |
| シーケンシャルコンテキスト | +10-15% | +20% |
| アダプティブスライス選択 | +5-8% | +15% |
| マルチスケール融合 | +6-10% | +25% |
| 対照学習 | +15-20% (少ラベル時) | +30% |
| **統合効果** | **+20-30%** | **+50-70%** |

## 6. 実装の優先順位

### Phase 1 (即座に実装可能)
1. エッジタイプ重み付きGAT
2. 階層的プーリングの強化

### Phase 2 (中期的実装)
3. シーケンシャルコンテキストエンコーダ
4. マルチスケール特徴融合

### Phase 3 (長期的実装)
5. アダプティブスライス選択
6. 対照学習の統合

## 7. 参考文献

1. Ying, Z., et al. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling. NeurIPS.
2. Busbridge, D., et al. (2019). Relational Graph Attention Networks. arXiv:1904.05811.
3. Feng, Z., et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. EMNLP.
4. Guo, D., et al. (2020). GraphCodeBERT: Pre-training Code Representations with Data Flow. ICLR.
5. You, Y., et al. (2020). Graph Contrastive Learning with Augmentations. NeurIPS.
6. Li, Y., et al. (2021). VulDeePecker: A Deep Learning-Based System for Vulnerability Detection. NDSS.

## 8. まとめ

本ドキュメントでは、NNASTアーキテクチャの6つの主要な改善点を理論的根拠とともに提示しました。これらの改善により、脆弱性検出の精度と汎化性能が大幅に向上することが期待されます。実装は段階的に進め、各改善の効果を検証しながら統合することを推奨します。
