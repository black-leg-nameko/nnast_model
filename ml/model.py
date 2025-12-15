"""
Graph Neural Network model for taint flow prediction on CPG graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple


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
        gnn_type: str = "GCN",  # "GCN" or "GAT"
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """
        Initialize CPG Taint Flow Model.
        
        Args:
            input_dim: Input node embedding dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            num_classes: Number of output classes
            gnn_type: Type of GNN ("GCN" or "GAT")
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
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
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Graph-level pooling and classification
        self.pool_dim = hidden_dim * 2  # Mean + Max pooling
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
        Forward pass.
        
        Args:
            data: PyTorch Geometric Batch object
            
        Returns:
            Logits tensor (batch_size, num_classes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_new = gnn_layer(x, edge_index)
            
            if self.batch_norms is not None:
                x_new = self.batch_norms[i](x_new)
            
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
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

