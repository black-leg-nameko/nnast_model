"""
Graph Neural Network model for taint flow prediction on CPG graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import math


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
    Graph Neural Network model for predicting taint flows in CPG graphs.
    
    Architecture:
    - Node embeddings: CodeBERT embeddings (768-dim)
    - GNN layers: Graph Convolutional Networks or Graph Attention Networks
    - Pooling: Mean/Max pooling for graph-level representation
    - Output: Binary classification (taint flow exists or not)
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # CodeBERT embedding dimension
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,  # Binary classification
        gnn_type: str = "GAT",  # "GCN" or "GAT" (GAT recommended for dynamic attention)
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_dynamic_attention: bool = True,  # Enable dynamic attention fusion
    ):
        """
        Initialize CPG Taint Flow Model with dynamic attention mechanism.
        
        Args:
            input_dim: Input node embedding dimension (CodeBERT: 768)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            num_classes: Number of output classes
            gnn_type: Type of GNN ("GCN" or "GAT")
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_dynamic_attention: Whether to use dynamic attention fusion mechanism
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_dynamic_attention = use_dynamic_attention
        
        # Early fusion: Input projection (combines CodeBERT embeddings with node attributes)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers with dynamic attention fusion
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            if use_dynamic_attention:
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
        
        # Late fusion: Graph-level pooling combines GNN output and CodeBERT semantic representation
        # GNN output: hidden_dim * 2 (mean + max pooling)
        # CodeBERT semantic: input_dim (pooled CodeBERT embeddings)
        self.pool_dim = hidden_dim * 2 + input_dim  # Mean + Max pooling + CodeBERT semantic
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass with dynamic attention fusion.
        
        Implements three-stage fusion strategy:
        1. Early fusion: CodeBERT embeddings as initial node features
        2. Intermediate fusion: Dynamic attention mechanism in GNN layers
        3. Late fusion: Combine GNN graph-level representation with CodeBERT semantic representation
        
        Args:
            data: PyTorch Geometric Batch object with 'codebert_emb' attribute
            
        Returns:
            Logits tensor (batch_size, num_classes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get CodeBERT embeddings (for dynamic attention fusion)
        if hasattr(data, 'codebert_emb'):
            codebert_emb = data.codebert_emb
        else:
            # Fallback: use x as CodeBERT embeddings if not provided
            codebert_emb = x
        
        # Early fusion: Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Intermediate fusion: GNN layers with dynamic attention
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.use_dynamic_attention:
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
        
        # Late fusion: Graph-level pooling
        # 1. GNN output pooling (structural features)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_gnn_pooled = torch.cat([x_mean, x_max], dim=1)  # (batch_size, hidden_dim * 2)
        
        # 2. CodeBERT semantic pooling (semantic features)
        codebert_mean = global_mean_pool(codebert_emb, batch)
        codebert_max = global_max_pool(codebert_emb, batch)
        # Use mean pooling for CodeBERT semantic representation
        codebert_semantic = codebert_mean  # (batch_size, input_dim)
        
        # 3. Combine GNN structural and CodeBERT semantic representations
        x_pooled = torch.cat([x_gnn_pooled, codebert_semantic], dim=1)  # (batch_size, pool_dim)
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits


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

