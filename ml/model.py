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


class EdgeTypeAwareGATLayer(nn.Module):
    """
    Edge-type aware GAT layer that processes different edge types separately.
    
    This is crucial for CPG graphs where AST/CFG/DFG/DDFG edges have different semantics.
    For example, DFG edges are critical for taint flow analysis.
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
        
        # Learnable edge-type importance weights
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types) / num_edge_types)
        
        # Fusion layer to combine edge-type specific features
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * num_edge_types, out_dim),
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
        Forward pass with edge-type aware processing.
        
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
        
        # Weighted combination of edge-type specific features
        weights = F.softmax(self.edge_type_weights, dim=0)
        weighted_outputs = [
            output * weights[i] for i, output in enumerate(edge_type_outputs)
        ]
        
        # Concatenate and fuse
        x_concat = torch.cat(weighted_outputs, dim=1)  # (num_nodes, out_dim * num_edge_types)
        x_out = self.fusion(x_concat)  # (num_nodes, out_dim)
        
        return x_out


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


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling: function-level → file-level aggregation.
    
    This allows the model to capture inter-function dependencies
    which are important for vulnerability detection.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Function-level aggregation
        self.function_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Mean + Max pooling
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # File-level aggregation (over functions)
        self.file_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        function_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hierarchical pooling: nodes → functions → file.
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            batch: Batch assignment (num_nodes,)
            function_ids: Function assignment for each node (num_nodes,) or None
            
        Returns:
            File-level representation (batch_size, hidden_dim)
        """
        if function_ids is None:
            # Fallback: treat all nodes as single function
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return self.file_pool(torch.cat([mean_pool, max_pool], dim=1))
        
        # Function-level pooling
        num_functions = function_ids.max().item() + 1
        function_features = []
        
        for func_id in range(num_functions):
            func_mask = (function_ids == func_id)
            if func_mask.sum() == 0:
                continue
            
            func_nodes = x[func_mask]
            func_batch = batch[func_mask]
            
            # Pool nodes within function
            func_mean = global_mean_pool(func_nodes, func_batch)
            func_max = global_max_pool(func_nodes, func_batch)
            func_feat = self.function_pool(torch.cat([func_mean, func_max], dim=1))
            function_features.append(func_feat)
        
        if not function_features:
            # Fallback
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return self.file_pool(torch.cat([mean_pool, max_pool], dim=1))
        
        # File-level pooling (over functions)
        func_tensor = torch.stack(function_features, dim=0)  # (num_functions, hidden_dim)
        file_mean = func_tensor.mean(dim=0)
        file_max = func_tensor.max(dim=0)[0]
        
        return self.file_pool(torch.cat([file_mean, file_max], dim=1))


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
        
        # Multi-modal node encoder (if enabled)
        if use_multi_modal:
            self.node_encoder = MultiModalNodeEncoder(
                codebert_dim=input_dim,
                node_kind_dim=32,
                type_hint_dim=32,
                out_dim=hidden_dim,
                num_node_kinds=num_node_kinds,
                num_types=num_types,
            )
        else:
            # Fallback: simple projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hierarchical pooling (if enabled)
        if use_hierarchical:
            self.hierarchical_pool = HierarchicalPooling(
                hidden_dim=hidden_dim,
                dropout=dropout
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
                # Edge-type aware GAT layer
                self.gnn_layers.append(
                    EdgeTypeAwareGATLayer(
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
            x = self.input_proj(x_codebert)
            x = F.relu(x)
        
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
                # Edge-type aware GAT layer
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
            # Hierarchical pooling: function-level → file-level
            function_ids = None
            if hasattr(data, 'function_ids') and data.function_ids is not None:
                function_ids = data.function_ids
            
            if self.use_multi_scale:
                # Multi-scale hierarchical pooling
                pooled_features = []
                for layer_out in layer_outputs:
                    func_pooled = self.hierarchical_pool(layer_out, batch, function_ids)
                    pooled_features.append(func_pooled)
                
                # Weighted combination
                weights = F.softmax(self.scale_weights, dim=0)
                x_gnn_pooled = sum(w * feat for w, feat in zip(weights, pooled_features))
            else:
                # Single-scale hierarchical pooling
                x_gnn_pooled = self.hierarchical_pool(x, batch, function_ids)
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

