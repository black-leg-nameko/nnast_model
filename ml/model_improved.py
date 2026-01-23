"""
Improved CPG+GNN Model with Edge-type Awareness

Key improvements:
1. Edge-type aware GNN layers
2. Multi-modal node features (CodeBERT + structural)
3. Multi-scale feature aggregation
4. Hierarchical pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.data import Batch
from typing import Optional, Dict, List
import math


class EdgeTypeAwareGATLayer(nn.Module):
    """
    Edge-type aware GAT layer that processes different edge types separately.
    
    This is crucial for CPG graphs where AST/CFG/DFG edges have different semantics.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_edge_types: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.head_dim = out_dim // heads
        
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        
        # Edge-type specific attention layers
        self.edge_type_layers = nn.ModuleList([
            GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
            for _ in range(num_edge_types)
        ])
        
        # Edge-type importance weights (learnable)
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types) / num_edge_types)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * num_edge_types, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor  # Edge type indices
    ) -> torch.Tensor:
        """
        Forward pass with edge-type aware processing.
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge type indices (num_edges,)
            
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Group edges by type
        edge_type_outputs = []
        
        for edge_type in range(self.num_edge_types):
            # Get edges of this type
            edge_mask = (edge_attr == edge_type)
            if edge_mask.sum() == 0:
                # No edges of this type, use zero features
                edge_type_outputs.append(torch.zeros_like(x[:, :self.out_dim]))
                continue
            
            # Extract edges of this type
            edge_index_type = edge_index[:, edge_mask]
            
            # Process with type-specific layer
            x_type = self.edge_type_layers[edge_type](x, edge_index_type)
            edge_type_outputs.append(x_type)
        
        # Weighted combination of edge-type specific features
        weighted_outputs = []
        for i, output in enumerate(edge_type_outputs):
            weight = F.softmax(self.edge_type_weights, dim=0)[i]
            weighted_outputs.append(output * weight)
        
        # Concatenate and fuse
        x_concat = torch.cat(weighted_outputs, dim=1)  # (num_nodes, out_dim * num_edge_types)
        x_out = self.fusion(x_concat)  # (num_nodes, out_dim)
        
        return x_out


class MultiModalNodeEncoder(nn.Module):
    """
    Multi-modal node encoder combining:
    - CodeBERT embeddings (semantic)
    - Node kind embeddings (structural)
    - Type hint embeddings (type information)
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
        
        # Embedding layers
        self.node_kind_emb = nn.Embedding(num_node_kinds, node_kind_dim)
        self.type_hint_emb = nn.Embedding(num_types, type_hint_dim)
        
        # Projection layers
        self.codebert_proj = nn.Linear(codebert_dim, out_dim // 2)
        self.structural_proj = nn.Linear(node_kind_dim + type_hint_dim, out_dim // 2)
        
        # Fusion
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
            x_structural = torch.cat([node_kind_emb, type_emb], dim=1)
        else:
            x_structural = node_kind_emb
        
        x_structural = self.structural_proj(x_structural)  # (num_nodes, out_dim // 2)
        
        # Combine
        x_combined = torch.cat([x_semantic, x_structural], dim=1)  # (num_nodes, out_dim)
        x_out = self.fusion(x_combined)
        
        return x_out


class ImprovedCPGRiskModel(nn.Module):
    """
    Improved CPG Risk Scoring Model with:
    1. Edge-type awareness
    2. Multi-modal node features
    3. Multi-scale feature aggregation
    4. Hierarchical pooling
    """
    
    def __init__(
        self,
        codebert_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 4,  # AST, CFG, DFG, DDFG
        num_node_kinds: int = 50,
        num_types: int = 100,
        dropout: float = 0.5,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        
        self.codebert_dim = codebert_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_multi_scale = use_multi_scale
        
        # Multi-modal node encoder
        self.node_encoder = MultiModalNodeEncoder(
            codebert_dim=codebert_dim,
            node_kind_dim=32,
            type_hint_dim=32,
            out_dim=hidden_dim,
            num_node_kinds=num_node_kinds,
            num_types=num_types,
        )
        
        # Edge-type aware GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(
                EdgeTypeAwareGATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_edge_types=num_edge_types,
                    heads=4,
                    dropout=dropout
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Multi-scale aggregation
        if use_multi_scale:
            # Aggregate features from different layers
            self.scale_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            pool_dim = hidden_dim * 2 * num_layers + codebert_dim  # Multi-scale + CodeBERT
        else:
            pool_dim = hidden_dim * 2 + codebert_dim
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass with improved architecture.
        
        Args:
            data: PyG Batch with:
                - x: CodeBERT embeddings
                - edge_index: Edge connectivity
                - edge_attr: Edge type indices
                - node_kinds: Node kind indices
                - batch: Batch assignment
                
        Returns:
            Risk scores (batch_size,)
        """
        x_codebert = data.x  # (num_nodes, codebert_dim)
        edge_index = data.edge_index
        edge_attr = data.edge_attr  # Edge type indices
        batch = data.batch
        
        # Get node kinds (fallback to zeros if not available)
        if hasattr(data, 'node_kinds'):
            node_kinds = data.node_kinds
        else:
            node_kinds = torch.zeros(x_codebert.size(0), dtype=torch.long, device=x_codebert.device)
        
        # Multi-modal node encoding
        x = self.node_encoder(x_codebert, node_kinds)  # (num_nodes, hidden_dim)
        
        # GNN layers with multi-scale aggregation
        layer_outputs = []
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            x_new = gnn_layer(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.5, training=self.training)
            
            # Residual connection
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
            
            layer_outputs.append(x)
        
        # Multi-scale pooling
        if self.use_multi_scale:
            # Pool each layer's output
            pooled_features = []
            for layer_out in layer_outputs:
                mean_pool = global_mean_pool(layer_out, batch)
                max_pool = global_max_pool(layer_out, batch)
                pooled_features.append(torch.cat([mean_pool, max_pool], dim=1))
            
            # Weighted combination
            weights = F.softmax(self.scale_weights, dim=0)
            x_gnn_pooled = sum(w * feat for w, feat in zip(weights, pooled_features))
        else:
            # Standard pooling (last layer only)
            x_mean = global_mean_pool(layer_outputs[-1], batch)
            x_max = global_max_pool(layer_outputs[-1], batch)
            x_gnn_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # CodeBERT semantic pooling
        codebert_mean = global_mean_pool(x_codebert, batch)
        
        # Combine
        x_pooled = torch.cat([x_gnn_pooled, codebert_mean], dim=1)
        
        # Regression
        risk_scores = self.regressor(x_pooled).squeeze(-1)
        
        return risk_scores
    
    def get_edge_type_importance(self) -> Dict[str, float]:
        """
        Get learned importance weights for each edge type.
        Useful for interpretability.
        """
        weights = F.softmax(self.gnn_layers[0].edge_type_weights, dim=0)
        edge_type_names = ["AST", "CFG", "DFG", "DDFG"]
        return {name: weight.item() for name, weight in zip(edge_type_names, weights)}


# Example usage and comparison
if __name__ == "__main__":
    # Create sample data
    batch_size = 4
    num_nodes = 100
    hidden_dim = 256
    
    # Sample batch
    from torch_geometric.data import Data
    
    graphs = []
    for i in range(batch_size):
        num_nodes_graph = 20 + i * 5
        data = Data(
            x=torch.randn(num_nodes_graph, 768),  # CodeBERT embeddings
            edge_index=torch.randint(0, num_nodes_graph, (2, num_nodes_graph * 2)),
            edge_attr=torch.randint(0, 4, (num_nodes_graph * 2,)),  # Edge types
            node_kinds=torch.randint(0, 50, (num_nodes_graph,)),
            batch=torch.full((num_nodes_graph,), i, dtype=torch.long)
        )
        graphs.append(data)
    
    batch = Batch.from_data_list(graphs)
    
    # Initialize model
    model = ImprovedCPGRiskModel(
        codebert_dim=768,
        hidden_dim=256,
        num_layers=3,
        num_edge_types=4,
        num_node_kinds=50,
        num_types=100,
    )
    
    # Forward pass
    risk_scores = model(batch)
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Risk scores: {risk_scores}")
    
    # Get edge type importance
    importance = model.get_edge_type_importance()
    print(f"\nEdge type importance: {importance}")
