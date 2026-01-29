"""
Graph Neural Network model for vulnerability risk scoring on CPG graphs.

Improved architecture with:
- Edge-type aware GNN layers (AST/CFG/DFG/DDFG distinction)
- Multi-modal node features (CodeBERT + structural features)
- Multi-scale feature aggregation
- Hierarchical representation learning (function-level → file-level)
- Interpretability (attention visualization, node importance)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, List
import math


class EdgeTypeWeightedGATLayer(nn.Module):
    """
    Improved Edge-type weighted GAT layer with explicit attention-based weighting.

    This implementation improves upon the basic EdgeTypeAwareGATLayer by:
    1. Using attention mechanism to dynamically weight edge types based on node context
    2. Learning edge-type importance per node rather than globally
    3. Better handling of edge type interactions

    Based on RGAT (Relational Graph Attention Networks) principles.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_edge_types: int = 4,  # AST, CFG, DFG, DDFG
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.heads = heads

        # Edge-type specific attention layers
        self.edge_type_layers = nn.ModuleList([
            GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
            for _ in range(num_edge_types)
        ])

        # Global learnable edge-type importance weights (for initialization)
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types) / num_edge_types)

        # Per-node edge-type attention mechanism
        # Query: node features, Key/Value: edge-type specific features
        self.edge_type_attention = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=False
        )

        # Edge-type embedding for attention
        self.edge_type_embedding = nn.Embedding(num_edge_types, out_dim)

        # Fusion layer to combine edge-type specific features
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),  # Original + attended
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor  # Edge type indices (num_edges,)
    ) -> torch.Tensor:
        """
        Forward pass with improved edge-type aware processing.

        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge type indices (num_edges,)

        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Process each edge type separately
        edge_type_outputs = []

        for edge_type in range(self.num_edge_types):
            # Get edges of this type
            edge_mask = (edge_attr == edge_type)

            if edge_mask.sum() == 0:
                # No edges of this type, use zero features
                edge_type_outputs.append(torch.zeros(x.size(0), self.out_dim, device=x.device))
                continue

            # Extract edges of this type
            edge_index_type = edge_index[:, edge_mask]

            # Process with type-specific layer
            x_type = self.edge_type_layers[edge_type](x, edge_index_type)
            edge_type_outputs.append(x_type)

        # Stack edge-type outputs: (num_edge_types, num_nodes, out_dim)
        edge_type_stack = torch.stack(edge_type_outputs, dim=0)

        # Global weighted combination (baseline)
        weights = F.softmax(self.edge_type_weights, dim=0)
        x_weighted = torch.sum(
            edge_type_stack * weights.view(self.num_edge_types, 1, 1),
            dim=0
        )  # (num_nodes, out_dim)

        # Per-node attention-based weighting
        # Use node features as query, edge-type features as key/value
        # Transpose for attention: (num_nodes, out_dim) -> (out_dim, num_nodes) for batch_first=False
        edge_type_stack_t = edge_type_stack.transpose(0, 1)  # (num_nodes, num_edge_types, out_dim)
        edge_type_stack_t = edge_type_stack_t.transpose(0, 1)  # (num_edge_types, num_nodes, out_dim)

        # Query: node features, Key/Value: edge-type outputs
        query = x_weighted.unsqueeze(0)  # (1, num_nodes, out_dim)
        key_value = edge_type_stack_t  # (num_edge_types, num_nodes, out_dim)

        # Attention to weight edge types per node
        x_attended, attn_weights = self.edge_type_attention(
            query, key_value, key_value
        )  # (1, num_nodes, out_dim), (num_nodes, 1, num_edge_types)
        x_attended = x_attended.squeeze(0)  # (num_nodes, out_dim)

        # Combine global weighted and attention-weighted features
        x_fused = torch.cat([x_weighted, x_attended], dim=1)  # (num_nodes, out_dim * 2)
        x_out = self.fusion(x_fused)  # (num_nodes, out_dim)

        return x_out


# Keep backward compatibility
EdgeTypeAwareGATLayer = EdgeTypeWeightedGATLayer


class CodePositionalEncoding(nn.Module):
    """
    Positional encoding for code spans (line/column information).

    This helps the model understand spatial relationships in code,
    which can be important for vulnerability patterns.
    """

    def __init__(
        self,
        d_model: int = 32,
        max_lines: int = 10000,
        max_cols: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.line_emb = nn.Embedding(max_lines, d_model // 2)
        self.col_emb = nn.Embedding(max_cols, d_model // 2)

    def forward(self, spans: torch.Tensor) -> torch.Tensor:
        """
        Encode code positions.

        Args:
            spans: (num_nodes, 4) tensor with [start_line, start_col, end_line, end_col]

        Returns:
            Positional encodings (num_nodes, d_model)
        """
        start_lines = spans[:, 0].clamp(0, self.line_emb.num_embeddings - 1)
        start_cols = spans[:, 1].clamp(0, self.col_emb.num_embeddings - 1)

        line_emb = self.line_emb(start_lines)
        col_emb = self.col_emb(start_cols)

        return torch.cat([line_emb, col_emb], dim=1)


class SequentialContextEncoder(nn.Module):
    """
    Transformer-based sequential context encoder for code.

    Enhances CodeBERT embeddings with sequential order information and
    surrounding context, which is crucial for multi-line vulnerability patterns.

    Based on Transformer encoder architecture with positional encoding.
    """

    def __init__(
        self,
        codebert_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.codebert_dim = codebert_dim
        self.hidden_dim = hidden_dim

        # Project CodeBERT embeddings to hidden dimension
        self.codebert_proj = nn.Linear(codebert_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_encoder = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(
        self,
        codebert_emb: torch.Tensor,
        node_order: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequential context.

        Args:
            codebert_emb: CodeBERT embeddings (num_nodes, codebert_dim)
            node_order: Node order indices (num_nodes,) or None (use sequential order)

        Returns:
            Context-enhanced embeddings (num_nodes, hidden_dim)
        """
        # Project CodeBERT embeddings
        x = self.codebert_proj(codebert_emb)  # (num_nodes, hidden_dim)

        # Add positional encoding
        if node_order is None:
            node_order = torch.arange(x.size(0), device=x.device)

        # Clamp to valid range
        node_order = node_order.clamp(0, self.pos_encoder.num_embeddings - 1)
        pos_emb = self.pos_encoder(node_order)  # (num_nodes, hidden_dim)
        x = x + pos_emb

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Transformer encoder expects (seq_len, batch, features)
        # Treat each node as a sequence element
        x = x.unsqueeze(1)  # (num_nodes, 1, hidden_dim)
        x = x.transpose(0, 1)  # (1, num_nodes, hidden_dim)

        # Apply transformer
        x = self.transformer(x)  # (1, num_nodes, hidden_dim)

        # Reshape back
        x = x.transpose(0, 1)  # (num_nodes, 1, hidden_dim)
        x = x.squeeze(1)  # (num_nodes, hidden_dim)

        # Output projection
        x = self.output_proj(x)

        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion for graph neural networks.

    Extracts features at different scales (local, mid, global) and fuses them
    using attention mechanism. This is crucial for vulnerability detection where
    patterns can occur at different scales.

    Based on Multi-Scale GCN (Li et al., 2019) principles.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_scales: int = 3,  # Local, Mid, Global
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Local scale: 1-hop neighbors
        self.local_conv = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Mid scale: 2-3 hop neighbors
        self.mid_conv = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Global scale: fully connected (or very high k-hop)
        self.global_conv = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Scale attention mechanism
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=False
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-scale feature fusion.

        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge attributes (optional)

        Returns:
            Fused multi-scale features (num_nodes, hidden_dim)
        """
        # Local scale: 1-hop
        x_local = self.local_conv(x, edge_index)
        x_local = F.relu(x_local)
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)

        # Mid scale: 2-hop graph
        edge_index_2hop = self._build_khop_graph(edge_index, k=2)
        x_mid = self.mid_conv(x, edge_index_2hop)
        x_mid = F.relu(x_mid)
        x_mid = F.dropout(x_mid, p=self.dropout, training=self.training)

        # Global scale: fully connected (or high k-hop)
        edge_index_global = self._build_fully_connected(edge_index, x.size(0))
        x_global = self.global_conv(x, edge_index_global)
        x_global = F.relu(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)

        # Stack scales: (num_scales, num_nodes, hidden_dim)
        x_scales = torch.stack([x_local, x_mid, x_global], dim=0)

        # Scale attention: attend across scales
        # Query: original features, Key/Value: scale features
        query = x.unsqueeze(0)  # (1, num_nodes, hidden_dim)
        x_scales_t = x_scales.transpose(0, 1)  # (num_nodes, num_scales, hidden_dim)
        x_scales_t = x_scales_t.transpose(0, 1)  # (num_scales, num_nodes, hidden_dim)

        x_attended, attn_weights = self.scale_attention(query, x_scales_t, x_scales_t)
        x_attended = x_attended.squeeze(0)  # (num_nodes, hidden_dim)

        # Concatenate all scales
        x_concat = torch.cat([x_local, x_mid, x_global], dim=-1)  # (num_nodes, hidden_dim * 3)

        # Fusion
        x_fused = self.fusion(x_concat)

        # Combine attended and fused features
        return x_fused + x_attended  # Residual connection

    def _build_khop_graph(
        self,
        edge_index: torch.Tensor,
        k: int = 2
    ) -> torch.Tensor:
        """
        Build k-hop graph by adding edges between nodes reachable in k hops.

        Args:
            edge_index: Original edge index (2, num_edges)
            k: Number of hops

        Returns:
            Extended edge index (2, num_edges_extended)
        """
        from torch_geometric.utils import to_undirected

        # Start with original edges
        edge_index_khop = edge_index.clone()

        # For k=2, add edges between nodes that share a neighbor
        if k >= 2:
            # Get adjacency list
            num_nodes = edge_index.max().item() + 1
            adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=edge_index.device)
            adj[edge_index[0], edge_index[1]] = True

            # 2-hop: A^2
            adj_2hop = torch.matmul(adj.float(), adj.float()) > 0

            # Add 2-hop edges
            new_edges = adj_2hop.nonzero(as_tuple=False).t()
            edge_index_khop = torch.cat([edge_index_khop, new_edges], dim=1)

        # Remove duplicates and self-loops
        edge_index_khop = to_undirected(edge_index_khop)

        return edge_index_khop

    def _build_fully_connected(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Build fully connected graph (or high k-hop approximation).

        Args:
            edge_index: Original edge index (for device/dtype)
            num_nodes: Number of nodes

        Returns:
            Fully connected edge index (2, num_nodes * (num_nodes - 1))
        """
        # Create fully connected graph
        nodes = torch.arange(num_nodes, device=edge_index.device)
        src = nodes.repeat(num_nodes)
        dst = nodes.repeat_interleave(num_nodes)

        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        return torch.stack([src, dst], dim=0)


class AdaptiveSliceSelector(nn.Module):
    """
    Adaptive slice selector that dynamically chooses optimal slicing strategies
    based on vulnerability type and code context.

    Different vulnerability types require different slicing strategies:
    - SQL Injection: DFG-based slicing (data flow)
    - XSS: CFG-based slicing (control flow)
    - Authentication: AST-based slicing (structural patterns)
    - Hybrid: All edge types

    Based on Adaptive Slicing (Krinke, 2001) and Vulnerability Slicing principles.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_slice_strategies: int = 4,  # DFG, CFG, AST, Hybrid
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slice_strategies = num_slice_strategies

        # Different slice selection strategies
        # Strategy 0: DFG-based (data flow)
        # Strategy 1: CFG-based (control flow)
        # Strategy 2: AST-based (structural)
        # Strategy 3: Hybrid (all edge types)
        self.slice_strategies = nn.ModuleList([
            self._create_slice_selector(hidden_dim, edge_type_filter=i)
            for i in range(num_slice_strategies)
        ])

        # Meta-network to select optimal strategy
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_slice_strategies),
            nn.Softmax(dim=-1)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def _create_slice_selector(
        self,
        hidden_dim: int,
        edge_type_filter: int
    ) -> nn.Module:
        """
        Create a slice selector for a specific edge type.

        Args:
            hidden_dim: Hidden dimension
            edge_type_filter: Edge type to focus on (0=DFG, 1=CFG, 2=AST, 3=All)

        Returns:
            Slice selector module
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        vulnerability_type_hint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adaptive slice selection.

        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge type indices (num_edges,)
            vulnerability_type_hint: Vulnerability type hint (optional)

        Returns:
            Selected slice features (num_nodes, hidden_dim)
        """
        # Compute strategy weights based on node features
        # Use mean pooling to get graph-level representation
        graph_repr = x.mean(dim=0)  # (hidden_dim,)
        strategy_weights = self.strategy_selector(graph_repr)  # (num_strategies,)

        # Apply each strategy
        slice_outputs = []
        for i, strategy in enumerate(self.slice_strategies):
            # Filter edges if needed
            if edge_attr is not None and i < 3:  # DFG, CFG, AST
                edge_mask = (edge_attr == i)
                if edge_mask.sum() > 0:
                    edge_index_filtered = edge_index[:, edge_mask]
                    # Apply GAT on filtered edges
                    from torch_geometric.nn import GATConv
                    gat = GATConv(self.hidden_dim, self.hidden_dim, heads=1, concat=False)
                    x_filtered = gat(x, edge_index_filtered)
                else:
                    x_filtered = torch.zeros_like(x)
            else:
                # Hybrid: use all edges
                x_filtered = x

            # Apply strategy
            slice_out = strategy(x_filtered)
            slice_outputs.append(slice_out * strategy_weights[i])

        # Weighted combination
        x_combined = sum(slice_outputs)

        # Final fusion
        return self.fusion(x_combined)


class ContrastiveVulnerabilityEncoder(nn.Module):
    """
    Contrastive learning wrapper for vulnerability encoder.

    Enables self-supervised learning by learning representations where
    similar code patterns have similar embeddings, even without explicit labels.

    Based on Graph Contrastive Learning (You et al., 2020) principles.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self,
        data: Batch,
        return_projection: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through encoder and projection.

        Args:
            data: Batch of graph data
            return_projection: If True, return projected features for contrastive learning

        Returns:
            Encoded features or projected features
        """
        # Encode
        h = self.encoder(data)  # (batch_size, hidden_dim)

        if return_projection:
            # Project for contrastive learning
            z = self.projection_head(h)
            # Normalize
            z = F.normalize(z, p=2, dim=-1)
            return z
        else:
            return h

    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            z1: Projected features from first augmentation (batch_size, projection_dim)
            z2: Projected features from second augmentation (batch_size, projection_dim)
            labels: Optional labels for supervised contrastive learning

        Returns:
            Contrastive loss
        """
        batch_size = z1.size(0)
        device = z1.device

        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.T) / self.temperature  # (batch_size, batch_size)

        if labels is not None:
            # Supervised contrastive learning: positive pairs have same label
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            pos_mask.fill_diagonal_(0)  # Remove self-pairs

            # Positive similarity
            pos_sim = (similarity * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

            # Negative similarity (all pairs)
            neg_sim = torch.logsumexp(similarity, dim=1)

            # Contrastive loss
            loss = -pos_sim + neg_sim
        else:
            # Self-supervised contrastive learning: positive pairs are (i, i)
            # Negative pairs are (i, j) where i != j
            labels_pos = torch.arange(batch_size, device=device)
            pos_sim = similarity[labels_pos, labels_pos]  # Diagonal elements

            # Negative similarity: log(sum(exp(sim))) excluding diagonal
            mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            similarity_masked = similarity.masked_fill(mask, float('-inf'))
            neg_sim = torch.logsumexp(similarity_masked, dim=1)

            # Contrastive loss
            loss = -pos_sim + neg_sim

        return loss.mean()

    def augment_graph(
        self,
        data: Batch,
        method: str = "edge_dropout"
    ) -> Batch:
        """
        Augment graph for contrastive learning.

        Args:
            data: Original graph batch
            method: Augmentation method ("edge_dropout", "node_mask", "feature_noise")

        Returns:
            Augmented graph batch
        """
        from torch_geometric.utils import dropout_edges

        if method == "edge_dropout":
            # Randomly drop edges
            edge_index, edge_attr = dropout_edges(
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None,
                p=0.2,
                training=self.training
            )
            data_aug = data.clone()
            data_aug.edge_index = edge_index
            if edge_attr is not None:
                data_aug.edge_attr = edge_attr
            return data_aug

        elif method == "node_mask":
            # Randomly mask node features
            data_aug = data.clone()
            mask = torch.rand(data.x.size(0), device=data.x.device) < 0.15
            data_aug.x[mask] = 0.0
            return data_aug

        elif method == "feature_noise":
            # Add Gaussian noise to features
            data_aug = data.clone()
            noise = torch.randn_like(data.x) * 0.1
            data_aug.x = data.x + noise
            return data_aug

        else:
            return data


class HierarchicalPooling(nn.Module):
    """
    Improved Hierarchical pooling with learnable DiffPool-style pooling.

    Enhanced version that:
    1. Uses learnable assignment matrices (DiffPool-style) for better clustering
    2. Supports multiple hierarchy levels (function → file → module)
    3. Learns optimal pooling strategies for vulnerability detection

    Based on DiffPool (Ying et al., 2018) principles.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.5,
        num_levels: int = 3,  # Function → File → Module
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        # Level 1: Function-level aggregation (learnable)
        self.function_assignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=-1)
        )
        self.function_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Mean + Max pooling
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Level 2: File-level aggregation (learnable)
        self.file_assignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=-1)
        )
        self.file_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Level 3: Module-level aggregation (if enabled)
        if num_levels >= 3:
            self.module_assignment = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Softmax(dim=-1)
            )
            self.module_pool = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Multi-scale feature fusion
        self.multiscale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        function_ids: Optional[torch.Tensor] = None,
        file_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Improved hierarchical pooling with learnable assignments.

        Args:
            x: Node features (num_nodes, hidden_dim)
            batch: Batch assignment (num_nodes,)
            function_ids: Function assignment for each node (num_nodes,) or None
            file_ids: File assignment for each node (num_nodes,) or None

        Returns:
            Multi-scale hierarchical representation (batch_size, hidden_dim)
        """
        level_outputs = []

        # Level 1: Function-level pooling
        if function_ids is not None:
            func_features = self._pool_by_group(x, function_ids, self.function_pool)
            level_outputs.append(func_features)
        else:
            # Fallback: treat all nodes as single function
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            func_features = self.function_pool(torch.cat([mean_pool, max_pool], dim=1))
            level_outputs.append(func_features)

        # Level 2: File-level pooling
        if file_ids is not None and len(level_outputs) > 0:
            # Use function features as input for file-level pooling
            file_features = self._pool_by_group(
                level_outputs[0].unsqueeze(0).expand(x.size(0), -1),
                file_ids,
                self.file_pool
            )
            level_outputs.append(file_features)
        else:
            # Use function features directly
            level_outputs.append(level_outputs[0])

        # Level 3: Module-level pooling (if enabled)
        if self.num_levels >= 3 and len(level_outputs) > 1:
            # For now, use file features as module features
            # In future, can add module_ids
            module_features = level_outputs[1]
            level_outputs.append(module_features)

        # Multi-scale fusion
        if len(level_outputs) > 1:
            # Concatenate all levels
            multiscale = torch.cat(level_outputs, dim=-1)  # (batch_size, hidden_dim * num_levels)
            return self.multiscale_fusion(multiscale)
        else:
            return level_outputs[0]

    def _pool_by_group(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor,
        pool_layer: nn.Module
    ) -> torch.Tensor:
        """
        Pool features by group IDs.

        Args:
            x: Features (num_nodes, hidden_dim)
            group_ids: Group assignment (num_nodes,)
            pool_layer: Pooling layer to apply

        Returns:
            Pooled features (num_groups, hidden_dim)
        """
        num_groups = group_ids.max().item() + 1
        group_features = []

        for group_id in range(num_groups):
            group_mask = (group_ids == group_id)
            if group_mask.sum() == 0:
                continue

            group_nodes = x[group_mask]

            # Mean and max pooling
            group_mean = group_nodes.mean(dim=0)
            group_max = group_nodes.max(dim=0)[0]

            # Apply pooling layer
            group_feat = pool_layer(torch.cat([group_mean, group_max], dim=0))
            group_features.append(group_feat)

        if not group_features:
            # Fallback: global pooling
            mean_pool = x.mean(dim=0)
            max_pool = x.max(dim=0)[0]
            return pool_layer(torch.cat([mean_pool, max_pool], dim=0)).unsqueeze(0)

        return torch.stack(group_features, dim=0)


class MultiModalNodeEncoder(nn.Module):
    """
    Multi-modal node encoder combining:
    - CodeBERT embeddings (semantic information)
    - Node kind embeddings (structural information)
    - Type hint embeddings (type information)

    This allows the model to leverage both semantic and structural features.
    """

    def __init__(
        self,
        codebert_dim: int = 768,
        node_kind_dim: int = 32,
        type_hint_dim: int = 32,
        out_dim: int = 256,
        num_node_kinds: int = 50,
        num_types: int = 100,
    ):
        super().__init__()

        # Embedding layers for structural features
        self.node_kind_emb = nn.Embedding(num_node_kinds, node_kind_dim)
        self.type_hint_emb = nn.Embedding(num_types, type_hint_dim)

        # Projection layers
        self.codebert_proj = nn.Linear(codebert_dim, out_dim // 2)
        self.structural_proj = nn.Linear(node_kind_dim + type_hint_dim, out_dim // 2)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(
        self,
        codebert_emb: torch.Tensor,
        node_kinds: torch.Tensor,
        type_hints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-modal node features.

        Args:
            codebert_emb: CodeBERT embeddings (num_nodes, codebert_dim)
            node_kinds: Node kind indices (num_nodes,)
            type_hints: Type hint indices (num_nodes,) or None

        Returns:
            Combined node features (num_nodes, out_dim)
        """
        # Semantic features (CodeBERT)
        x_semantic = self.codebert_proj(codebert_emb)  # (num_nodes, out_dim // 2)

        # Structural features
        node_kind_emb = self.node_kind_emb(node_kinds)  # (num_nodes, node_kind_dim)

        if type_hints is not None:
            type_emb = self.type_hint_emb(type_hints)  # (num_nodes, type_hint_dim)
            x_structural_raw = torch.cat([node_kind_emb, type_emb], dim=1)
        else:
            # If no type hints, pad with zeros
            x_structural_raw = torch.cat([
                node_kind_emb,
                torch.zeros(node_kind_emb.size(0), self.type_hint_emb.embedding_dim, device=node_kind_emb.device)
            ], dim=1)

        x_structural = self.structural_proj(x_structural_raw)  # (num_nodes, out_dim // 2)

        # Combine semantic and structural features
        x_combined = torch.cat([x_semantic, x_structural], dim=1)  # (num_nodes, out_dim)
        x_out = self.fusion(x_combined)

        return x_out


class DynamicAttentionFusionLayer(nn.Module):
    """
    Dynamic Attention Fusion Layer that combines GNN structural features
    with CodeBERT semantic features through mutual attention mechanism.

    This implements the "intermediate fusion" strategy from the design document,
    where CodeBERT embeddings are dynamically referenced during GNN message passing.
    """

    def __init__(
        self,
        hidden_dim: int,
        codebert_dim: int = 768,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        """
        Initialize Dynamic Attention Fusion Layer.

        Args:
            hidden_dim: Hidden dimension for GNN features
            codebert_dim: CodeBERT embedding dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.codebert_dim = codebert_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        # GAT layer for structural information
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout
        )

        # Projection layers for CodeBERT features
        self.codebert_proj = nn.Linear(codebert_dim, hidden_dim)

        # Mutual attention mechanism: attend to CodeBERT features based on structural context
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(codebert_dim, hidden_dim)
        self.attention_value = nn.Linear(codebert_dim, hidden_dim)

        # Fusion layer to combine structural and semantic features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        codebert_emb: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with dynamic attention fusion.

        Args:
            x: Current node features from GNN (num_nodes, hidden_dim)
            codebert_emb: CodeBERT embeddings (num_nodes, codebert_dim)
            edge_index: Edge connectivity (2, num_edges)

        Returns:
            Fused features (num_nodes, hidden_dim)
        """
        # 1. GAT for structural information
        x_struct = self.gat(x, edge_index)
        x_struct = F.relu(x_struct)
        x_struct = F.dropout(x_struct, p=self.dropout, training=self.training)

        # 2. Project CodeBERT embeddings to hidden dimension
        codebert_proj = self.codebert_proj(codebert_emb)

        # 3. Mutual attention: attend to CodeBERT features based on structural context
        # Query: structural features, Key/Value: CodeBERT features
        q = self.attention_query(x_struct)  # (num_nodes, hidden_dim)
        k = self.attention_key(codebert_emb)  # (num_nodes, hidden_dim)
        v = self.attention_value(codebert_emb)  # (num_nodes, hidden_dim)

        # Multi-head attention
        batch_size = x_struct.size(0)
        q = q.view(batch_size, self.heads, self.head_dim)  # (num_nodes, heads, head_dim)
        k = k.view(batch_size, self.heads, self.head_dim)
        v = v.view(batch_size, self.heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        x_semantic = torch.matmul(attn_weights, v)  # (num_nodes, heads, head_dim)
        x_semantic = x_semantic.view(batch_size, self.hidden_dim)  # (num_nodes, hidden_dim)

        # 4. Fusion: combine structural and semantic features
        x_fused = torch.cat([x_struct, x_semantic], dim=1)  # (num_nodes, hidden_dim * 2)
        x_out = self.fusion(x_fused)  # (num_nodes, hidden_dim)

        return x_out


class CPGTaintFlowModel(nn.Module):
    """
    Graph Neural Network model for vulnerability risk scoring on CPG graphs.

    Architecture:
    - Node embeddings: CodeBERT embeddings (768-dim)
    - GNN layers: Graph Convolutional Networks or Graph Attention Networks
    - Pooling: Mean/Max pooling for graph-level representation
    - Output: Risk score (real-valued) ∈ ℝ

    Following NNAST training strategy: No binary classification, only risk ranking.
    """

    def __init__(
        self,
        input_dim: int = 768,  # CodeBERT embedding dimension
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "GAT",  # "GCN" or "GAT" (GAT recommended for dynamic attention)
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_dynamic_attention: bool = True,  # Enable dynamic attention fusion
        use_edge_types: bool = True,  # Enable edge-type aware processing
        use_multi_modal: bool = True,  # Enable multi-modal node features
        use_multi_scale: bool = True,  # Enable multi-scale feature aggregation
        use_hierarchical: bool = True,  # Enable hierarchical pooling (function → file)
        use_positional: bool = True,  # Enable positional encoding for code spans
        use_sequential_context: bool = False,  # Phase 2.1: Enable sequential context encoder
        use_multiscale_fusion: bool = False,  # Phase 2.2: Enable multi-scale feature fusion
        use_adaptive_slice: bool = False,  # Phase 3.1: Enable adaptive slice selector
        num_edge_types: int = 4,  # Number of edge types (AST, CFG, DFG, DDFG)
        num_node_kinds: int = 50,  # Number of node kinds
        num_types: int = 100,  # Number of type hints
    ):
        """
        Initialize CPG Risk Scoring Model with improved architecture.

        Args:
            input_dim: Input node embedding dimension (CodeBERT: 768)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ("GCN" or "GAT")
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_dynamic_attention: Whether to use dynamic attention fusion mechanism
            use_edge_types: Whether to use edge-type aware processing
            use_multi_modal: Whether to use multi-modal node features
            use_multi_scale: Whether to use multi-scale feature aggregation
            num_edge_types: Number of edge types (AST, CFG, DFG, DDFG)
            num_node_kinds: Number of node kinds
            num_types: Number of type hints
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_dynamic_attention = use_dynamic_attention
        self.use_edge_types = use_edge_types
        self.use_multi_modal = use_multi_modal
        self.use_multi_scale = use_multi_scale
        self.use_hierarchical = use_hierarchical
        self.use_positional = use_positional
        self.use_sequential_context = use_sequential_context
        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_adaptive_slice = use_adaptive_slice

        # Positional encoding (if enabled)
        if use_positional:
            self.positional_encoder = CodePositionalEncoding(
                d_model=32,
                max_lines=10000,
                max_cols=200,
            )
            # Projection layer to combine positional encoding with node features
            self.pos_proj = nn.Linear(hidden_dim + 32, hidden_dim)
        else:
            self.pos_proj = None

        # Phase 2.1: Sequential context encoder (if enabled)
        if use_sequential_context:
            self.sequential_encoder = SequentialContextEncoder(
                codebert_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                num_heads=8,
                dropout=dropout
            )
            # Sequential encoder outputs hidden_dim, so we need to adjust
            seq_input_dim = hidden_dim
        else:
            self.sequential_encoder = None
            seq_input_dim = input_dim

        # Multi-modal node encoder (if enabled)
        if use_multi_modal:
            self.node_encoder = MultiModalNodeEncoder(
                codebert_dim=seq_input_dim,  # Use sequential encoder output if enabled
                node_kind_dim=32,
                type_hint_dim=32,
                out_dim=hidden_dim,
                num_node_kinds=num_node_kinds,
                num_types=num_types,
            )
        else:
            # Fallback: simple projection
            self.input_proj = nn.Linear(seq_input_dim, hidden_dim)

        # Phase 3.1: Adaptive slice selector (if enabled)
        if use_adaptive_slice:
            self.slice_selector = AdaptiveSliceSelector(
                hidden_dim=hidden_dim,
                num_slice_strategies=4,
                dropout=dropout
            )
        else:
            self.slice_selector = None

        # Phase 2.2: Multi-scale feature fusion (if enabled)
        if use_multiscale_fusion:
            self.multiscale_fusion = MultiScaleFeatureFusion(
                hidden_dim=hidden_dim,
                num_scales=3,
                heads=4,
                dropout=dropout
            )
        else:
            self.multiscale_fusion = None

        # Hierarchical pooling (if enabled) - Phase 1.2: Improved version
        if use_hierarchical:
            self.hierarchical_pool = HierarchicalPooling(
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_levels=3  # Function → File → Module
            )

        # Adjust pool_dim if using hierarchical pooling
        if use_hierarchical:
            # Hierarchical pooling outputs hidden_dim instead of hidden_dim * 2
            if use_multi_scale:
                pool_dim = hidden_dim * num_layers + input_dim
            else:
                pool_dim = hidden_dim + input_dim
            self.pool_dim = pool_dim

        # GNN layers with improved architecture
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            if use_edge_types:
                # Edge-type weighted GAT layer (Phase 1.1: Improved version)
                self.gnn_layers.append(
                    EdgeTypeWeightedGATLayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        num_edge_types=num_edge_types,
                        heads=4,
                        dropout=dropout
                    )
                )
            elif use_dynamic_attention:
                # Use dynamic attention fusion layer
                self.gnn_layers.append(
                    DynamicAttentionFusionLayer(
                        hidden_dim=hidden_dim,
                        codebert_dim=input_dim,
                        heads=4,
                        dropout=dropout
                    )
                )
            else:
                # Standard GNN layers (fallback)
                if gnn_type == "GCN":
                    self.gnn_layers.append(
                        GCNConv(
                            hidden_dim if i > 0 else hidden_dim,
                            hidden_dim
                        )
                    )
                elif gnn_type == "GAT":
                    self.gnn_layers.append(
                        GATConv(
                            hidden_dim if i > 0 else hidden_dim,
                            hidden_dim,
                            heads=4,
                            concat=False,
                            dropout=dropout
                        )
                    )
                else:
                    raise ValueError(f"Unknown GNN type: {gnn_type}")

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-scale aggregation weights (if enabled)
        if use_multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(num_layers) / num_layers)


        # Regression head: output single risk score
        self.regressor = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Single output: risk score
            nn.Sigmoid()  # Constrain output to [0, 1] range
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass with improved architecture.

        Implements:
        1. Multi-modal node encoding (CodeBERT + structural features)
        2. Edge-type aware GNN processing
        3. Multi-scale feature aggregation
        4. Late fusion with CodeBERT semantic representation

        Args:
            data: PyTorch Geometric Batch object with:
                - x: CodeBERT embeddings
                - edge_index: Edge connectivity
                - edge_attr: Edge type indices (if use_edge_types=True)
                - node_kinds: Node kind indices (if use_multi_modal=True)
                - batch: Batch assignment

        Returns:
            Risk scores tensor (batch_size,) - values in [0, 1]
        """
        x_codebert = data.x  # (num_nodes, input_dim)
        edge_index = data.edge_index
        batch = data.batch

        # Get CodeBERT embeddings (for dynamic attention fusion if needed)
        if hasattr(data, 'codebert_emb'):
            codebert_emb = data.codebert_emb
        else:
            codebert_emb = x_codebert

        # Phase 2.1: Sequential context encoding (if enabled)
        if self.use_sequential_context:
            # Get node order if available
            node_order = None
            if hasattr(data, 'node_order'):
                node_order = data.node_order
            codebert_emb = self.sequential_encoder(codebert_emb, node_order)

        # Get edge attributes (edge types)
        if self.use_edge_types:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr
            else:
                # Fallback: assume all edges are AST type (0)
                edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        else:
            edge_attr = None

        # Get node kinds (for multi-modal encoding)
        if self.use_multi_modal:
            if hasattr(data, 'node_kinds') and data.node_kinds is not None:
                node_kinds = data.node_kinds
            else:
                # Fallback: use zeros
                node_kinds = torch.zeros(x_codebert.size(0), dtype=torch.long, device=x_codebert.device)

            # Multi-modal node encoding
            x = self.node_encoder(codebert_emb, node_kinds)
        else:
            # Simple projection
            x = self.input_proj(codebert_emb)
            x = F.relu(x)

        # Phase 3.1: Adaptive slice selection (if enabled)
        if self.use_adaptive_slice:
            vulnerability_type_hint = None
            if hasattr(data, 'vulnerability_type'):
                vulnerability_type_hint = data.vulnerability_type
            x = self.slice_selector(x, edge_index, edge_attr, vulnerability_type_hint)

        # Add positional encoding (if enabled)
        if self.use_positional and self.pos_proj is not None:
            if hasattr(data, 'spans') and data.spans is not None:
                pos_emb = self.positional_encoder(data.spans)
                # Concatenate positional encoding
                x = torch.cat([x, pos_emb], dim=1)
                # Project back to hidden_dim
                x = self.pos_proj(x)
                x = F.relu(x)

        # GNN layers with improved architecture
        layer_outputs = []  # Store outputs for multi-scale aggregation

        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.use_edge_types:
                # Edge-type aware GAT layer (Phase 1.1: Improved)
                x_new = gnn_layer(x, edge_index, edge_attr)
            elif self.use_dynamic_attention:
                # Dynamic attention fusion layer
                x_new = gnn_layer(x, codebert_emb, edge_index)
            else:
                # Standard GNN layer
                x_new = gnn_layer(x, edge_index)
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            if self.batch_norms is not None:
                x_new = self.batch_norms[i](x_new)

            # Phase 2.2: Multi-scale feature fusion (if enabled, apply after each layer)
            if self.use_multiscale_fusion:
                x_new = self.multiscale_fusion(x_new, edge_index, edge_attr)

            # Residual connection (if dimensions match)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

            # Store layer output for multi-scale aggregation
            if self.use_multi_scale:
                layer_outputs.append(x)

        # Late fusion: Graph-level pooling
        if self.use_hierarchical:
            # Hierarchical pooling: function-level → file-level (Phase 1.2: Improved)
            function_ids = None
            file_ids = None
            if hasattr(data, 'function_ids') and data.function_ids is not None:
                function_ids = data.function_ids
            if hasattr(data, 'file_ids') and data.file_ids is not None:
                file_ids = data.file_ids

            if self.use_multi_scale:
                # Multi-scale hierarchical pooling
                pooled_features = []
                for layer_out in layer_outputs:
                    func_pooled = self.hierarchical_pool(layer_out, batch, function_ids, file_ids)
                    pooled_features.append(func_pooled)

                # Weighted combination
                weights = F.softmax(self.scale_weights, dim=0)
                x_gnn_pooled = sum(w * feat for w, feat in zip(weights, pooled_features))
            else:
                # Single-scale hierarchical pooling
                x_gnn_pooled = self.hierarchical_pool(x, batch, function_ids, file_ids)
        else:
            # Standard pooling
            if self.use_multi_scale:
                # Multi-scale aggregation: pool each layer's output
                pooled_features = []
                for layer_out in layer_outputs:
                    mean_pool = global_mean_pool(layer_out, batch)
                    max_pool = global_max_pool(layer_out, batch)
                    pooled_features.append(torch.cat([mean_pool, max_pool], dim=1))

                # Weighted combination of multi-scale features
                weights = F.softmax(self.scale_weights, dim=0)
                x_gnn_pooled = sum(w * feat for w, feat in zip(weights, pooled_features))
            else:
                # Single-scale: only last layer
                x_mean = global_mean_pool(x, batch)
                x_max = global_max_pool(x, batch)
                x_gnn_pooled = torch.cat([x_mean, x_max], dim=1)  # (batch_size, hidden_dim * 2)

        # CodeBERT semantic pooling (semantic features)
        codebert_mean = global_mean_pool(codebert_emb, batch)

        # Combine GNN structural and CodeBERT semantic representations
        x_pooled = torch.cat([x_gnn_pooled, codebert_mean], dim=1)  # (batch_size, pool_dim)

        # Regression: output risk score
        risk_scores = self.regressor(x_pooled)  # (batch_size, 1)

        return risk_scores.squeeze(-1)  # (batch_size,)

    def get_edge_type_importance(self) -> Optional[Dict[str, float]]:
        """
        Get learned importance weights for each edge type.
        Useful for interpretability.

        Returns:
            Dictionary mapping edge type names to importance weights, or None if not using edge types
        """
        if not self.use_edge_types or len(self.gnn_layers) == 0:
            return None

        weights = F.softmax(self.gnn_layers[0].edge_type_weights, dim=0)
        edge_type_names = ["AST", "CFG", "DFG", "DDFG"]
        return {name: weight.item() for name, weight in zip(edge_type_names, weights)}

    def forward_with_attention(
        self,
        data: Batch,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with attention weights for interpretability.

        Args:
            data: PyTorch Geometric Batch object
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (risk_scores, attention_info)
            - risk_scores: Risk scores (batch_size,)
            - attention_info: Dict with attention weights and node importance, or None
        """
        # Standard forward pass
        risk_scores = self.forward(data)

        if not return_attention:
            return risk_scores, None

        attention_info = {}

        # Get edge type importance
        edge_importance = self.get_edge_type_importance()
        if edge_importance:
            attention_info["edge_type_importance"] = edge_importance

        # Compute node importance (gradient-based or attention-based)
        # For now, use simple heuristic: nodes with high feature norm
        x_codebert = data.x
        if hasattr(data, 'codebert_emb'):
            codebert_emb = data.codebert_emb
        else:
            codebert_emb = x_codebert

        # Node importance: norm of CodeBERT embeddings (semantic importance)
        node_importance = torch.norm(codebert_emb, dim=1)  # (num_nodes,)
        attention_info["node_importance"] = node_importance.detach().cpu()

        # Batch-wise node importance
        batch = data.batch
        batch_size = batch.max().item() + 1
        attention_info["node_importance_per_graph"] = []

        for i in range(batch_size):
            batch_mask = batch == i
            graph_node_importance = node_importance[batch_mask]
            attention_info["node_importance_per_graph"].append(
                graph_node_importance.detach().cpu()
            )

        return risk_scores, attention_info

    def explain_prediction(
        self,
        data: Batch,
        top_k_nodes: int = 10
    ) -> Dict[str, Any]:
        """
        Explain model prediction by identifying important nodes and edges.

        Args:
            data: PyTorch Geometric Batch object
            top_k_nodes: Number of top important nodes to return

        Returns:
            Dictionary with explanation:
            - risk_score: Predicted risk score
            - top_nodes: Top-K important node IDs
            - edge_type_importance: Importance of each edge type
            - node_importance: Node importance scores
        """
        risk_scores, attention_info = self.forward_with_attention(data, return_attention=True)

        explanation = {
            "risk_score": risk_scores.detach().cpu().numpy().tolist(),
        }

        if attention_info:
            explanation["edge_type_importance"] = attention_info.get("edge_type_importance")

            # Get top-K nodes per graph
            batch = data.batch
            batch_size = batch.max().item() + 1
            top_nodes_per_graph = []

            if "node_importance_per_graph" in attention_info:
                for i, node_importance in enumerate(attention_info["node_importance_per_graph"]):
                    # Get node IDs for this graph
                    batch_mask = batch == i
                    node_ids = torch.where(batch_mask)[0]

                    # Get top-K nodes
                    top_k = min(top_k_nodes, len(node_ids))
                    top_indices = torch.topk(node_importance, top_k).indices
                    top_node_ids = node_ids[top_indices].cpu().numpy().tolist()

                    top_nodes_per_graph.append({
                        "graph_idx": i,
                        "top_node_ids": top_node_ids,
                        "importance_scores": node_importance[top_indices].cpu().numpy().tolist()
                    })

            explanation["top_nodes"] = top_nodes_per_graph

        return explanation

    def reset_parameters(self):
        """Reset model parameters for re-initialization."""
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class CPGNodePairModel(nn.Module):
    """
    Model for predicting taint flow between specific node pairs.

    This model takes a graph and a pair of node IDs (source, sink)
    and predicts whether there's a taint flow between them.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "GCN",
        dropout: float = 0.5,
    ):
        """
        Initialize CPG Node Pair Model.

        Args:
            input_dim: Input node embedding dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ("GCN" or "GAT")
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "GCN":
                self.gnn_layers.append(
                    GCNConv(
                        hidden_dim if i > 0 else hidden_dim,
                        hidden_dim
                    )
                )
            elif gnn_type == "GAT":
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim if i > 0 else hidden_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        dropout=dropout
                    )
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Node pair classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate source and sink embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

    def forward(
        self,
        data: Batch,
        source_ids: torch.Tensor,
        sink_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Batch object
            source_ids: Source node IDs (batch_size,)
            sink_ids: Sink node IDs (batch_size,)

        Returns:
            Logits tensor (batch_size, 2)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract source and sink node embeddings
        # Need to map global node IDs to batch-local indices
        batch_size = batch.max().item() + 1
        source_embeddings = []
        sink_embeddings = []

        for i in range(batch_size):
            batch_mask = batch == i
            batch_nodes = torch.where(batch_mask)[0]

            # Get source and sink embeddings for this graph
            # Note: This assumes source_ids[i] and sink_ids[i] are valid node IDs
            # In practice, you might need to map from global IDs to local indices
            source_idx = source_ids[i].item()
            sink_idx = sink_ids[i].item()

            # Find node in batch
            source_emb = x[batch_nodes[source_idx]] if source_idx < len(batch_nodes) else x[batch_nodes[0]]
            sink_emb = x[batch_nodes[sink_idx]] if sink_idx < len(batch_nodes) else x[batch_nodes[0]]

            source_embeddings.append(source_emb)
            sink_embeddings.append(sink_emb)

        source_embeddings = torch.stack(source_embeddings)
        sink_embeddings = torch.stack(sink_embeddings)

        # Concatenate and classify
        pair_embedding = torch.cat([source_embeddings, sink_embeddings], dim=1)
        logits = self.classifier(pair_embedding)

        return logits
