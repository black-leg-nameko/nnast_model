"""
PyTorch Dataset for CPG graphs with CodeBERT embeddings.
"""
import json
import pathlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from ir.schema import CPGGraph, CPGNode, CPGEdge
from ml.embed_codebert import CodeBERTEmbedder


class CPGGraphDataset(Dataset):
    """Dataset for CPG graphs with CodeBERT embeddings."""
    
    def __init__(
        self,
        graph_jsonl_path: str,
        taint_jsonl_path: Optional[str] = None,
        labels_jsonl_path: Optional[str] = None,
        embedder: Optional[CodeBERTEmbedder] = None,
        max_nodes: int = 1000,
        cache_embeddings: bool = True,
        skip_large_graphs: bool = False,  # Set to True to skip graphs larger than max_nodes * 2
    ):
        """
        Initialize CPG graph dataset.
        
        Args:
            graph_jsonl_path: Path to JSONL file containing CPG graphs
            taint_jsonl_path: Optional path to JSONL file containing taint records
            labels_jsonl_path: Optional path to JSONL file containing labels (for supervised learning)
            embedder: CodeBERT embedder instance (will create if None)
            max_nodes: Maximum number of nodes per graph (for padding/truncation)
            cache_embeddings: Whether to cache embeddings in memory
            skip_large_graphs: If True, skip graphs larger than max_nodes * 2 (default: False, keep all graphs)
        """
        self.graph_jsonl_path = pathlib.Path(graph_jsonl_path)
        self.taint_jsonl_path = pathlib.Path(taint_jsonl_path) if taint_jsonl_path else None
        self.labels_jsonl_path = pathlib.Path(labels_jsonl_path) if labels_jsonl_path else None
        self.max_nodes = max_nodes
        self.cache_embeddings = cache_embeddings
        self.skip_large_graphs = skip_large_graphs
        
        # Initialize embedder
        if embedder is None:
            self.embedder = CodeBERTEmbedder()
        else:
            self.embedder = embedder
        
        # Load graphs
        self.graphs = self._load_graphs()
        
        # Load taint records if provided
        self.taint_records = self._load_taint_records() if taint_jsonl_path else []
        
        # Load labels if provided
        self.labels = self._load_labels() if labels_jsonl_path else None
        
        # Cache for embeddings
        self._embedding_cache = {} if cache_embeddings else None
    
    def _load_graphs(self) -> List[Dict[str, Any]]:
        """Load CPG graphs from JSONL file."""
        graphs = []
        skipped_large = 0
        with open(self.graph_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    graph_dict = json.loads(line)
                    num_nodes = len(graph_dict.get("nodes", []))
                    
                    # Skip graphs that are too large only if skip_large_graphs is True
                    if self.skip_large_graphs and num_nodes > self.max_nodes * 2:
                        skipped_large += 1
                        continue
                    
                    graphs.append(graph_dict)
                except json.JSONDecodeError:
                    continue
        if skipped_large > 0:
            print(f"⚠️ {skipped_large}個の大きすぎるグラフをスキップしました（>{self.max_nodes * 2}ノード）")
        elif not self.skip_large_graphs:
            # Count large graphs for information
            large_count = sum(1 for g in graphs if len(g.get("nodes", [])) > self.max_nodes)
            if large_count > 0:
                print(f"ℹ️ {large_count}個の大きなグラフ（>{self.max_nodes}ノード）が含まれています。処理時に切り詰められます。")
        return graphs
    
    def _load_taint_records(self) -> List[Dict[str, Any]]:
        """Load taint records from JSONL file."""
        from ir.io import iter_taint_records
        return list(iter_taint_records(str(self.taint_jsonl_path)))
    
    def _load_labels(self) -> List[Dict[str, Any]]:
        """Load labels from JSONL file."""
        labels = []
        with open(self.labels_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label_data = json.loads(line)
                    labels.append(label_data)
                except json.JSONDecodeError:
                    continue
        return labels
    
    def _get_node_embeddings(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """Get embeddings for nodes, using cache if available."""
        if self._embedding_cache is not None:
            # Use hash of node codes as cache key
            cache_key = hash(tuple(n.get("code", "") for n in nodes))
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        embeddings = self.embedder.embed_nodes(nodes)
        
        if self._embedding_cache is not None:
            self._embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def _graph_to_pyg(self, graph_dict: Dict[str, Any]) -> Data:
        """
        Convert CPG graph dict to PyTorch Geometric Data object.
        
        Returns:
            PyG Data object with:
            - x: Node embeddings (num_nodes, 768)
            - edge_index: Edge connectivity (2, num_edges)
            - edge_attr: Edge types/kinds (num_edges,)
            - node_features: Additional node features (num_nodes, feature_dim)
        """
        nodes = graph_dict.get("nodes", [])
        edges = graph_dict.get("edges", [])
        
        # Limit nodes if needed (truncate to max_nodes, but keep the graph)
        original_num_nodes = len(nodes)
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
            # Filter edges to only include nodes within range
            node_ids = {n["id"] for n in nodes}
            edges = [e for e in edges if e["src"] in node_ids and e["dst"] in node_ids]
            # Note: Graph is truncated but not skipped - this allows learning from large graphs
            # even if we can only process a portion of them
        
        num_nodes = len(nodes)
        
        # Map original node IDs to 0-indexed consecutive IDs
        original_node_ids = [n["id"] for n in nodes]
        node_id_to_idx = {orig_id: idx for idx, orig_id in enumerate(original_node_ids)}
        
        # Get node embeddings
        node_embeddings = self._get_node_embeddings(nodes)
        
        # Build edge index with remapped node IDs
        if edges:
            edge_list = []
            for e in edges:
                src_idx = node_id_to_idx.get(e["src"])
                dst_idx = node_id_to_idx.get(e["dst"])
                if src_idx is not None and dst_idx is not None:
                    edge_list.append([src_idx, dst_idx])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Build edge attributes (edge types as fixed indices)
        # Map edge kinds to fixed indices: AST=0, CFG=1, DFG=2, DDFG=3
        # This ensures consistency across graphs and compatibility with EdgeTypeAwareGATLayer
        EDGE_TYPE_MAP = {
            "AST": 0,
            "CFG": 1,
            "DFG": 2,
            "DDFG": 3,
        }
        
        if edges and edge_index.size(1) > 0:
            edge_kinds = []
            for e in edges:
                if e["src"] in node_id_to_idx and e["dst"] in node_id_to_idx:
                    edge_kind = e.get("kind", "AST")
                    # Map to fixed index (default to AST if unknown)
                    edge_kinds.append(EDGE_TYPE_MAP.get(edge_kind, 0))
            edge_attr = torch.tensor(edge_kinds, dtype=torch.long)
        else:
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        # Additional node features (node kind, type hint, etc.)
        node_kinds = [n.get("kind", "Unknown") for n in nodes]
        unique_node_kinds = sorted(set(node_kinds))
        node_kind_to_idx = {k: i for i, k in enumerate(unique_node_kinds)}
        node_kind_features = torch.tensor(
            [node_kind_to_idx[k] for k in node_kinds],
            dtype=torch.long
        )

        # Node span information: (start_line, start_col, end_line, end_col)
        # This allows mapping model outputs back to concrete line ranges.
        node_spans_list = []
        for n in nodes:
            span = n.get("span")
            if isinstance(span, (list, tuple)) and len(span) == 4:
                sl, sc, el, ec = span
                node_spans_list.append([int(sl), int(sc), int(el), int(ec)])
            else:
                # Fallback: unknown span -> use zeros
                node_spans_list.append([0, 0, 0, 0])
        node_spans = torch.tensor(node_spans_list, dtype=torch.long) if node_spans_list else torch.empty((0, 4), dtype=torch.long)
        
        # Create PyG Data object
        # Store CodeBERT embeddings separately for dynamic attention mechanism
        codebert_emb = torch.tensor(node_embeddings, dtype=torch.float32)
        
        data = Data(
            x=codebert_emb,  # Initial node features (CodeBERT embeddings)
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            # Store CodeBERT embeddings for dynamic attention fusion
            codebert_emb=codebert_emb,  # Keep original CodeBERT embeddings
            # Store additional metadata
            node_kinds=node_kind_features,
            node_ids=torch.tensor([n["id"] for n in nodes], dtype=torch.long),
            spans=node_spans,
            file=graph_dict.get("file", ""),
        )
        
        return data
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Dict[str, Any]]:
        """
        Get a graph and its associated weak signals.
        
        Returns:
            Tuple of (graph_data, weak_signals)
            - graph_data: PyG Data object
            - weak_signals: Dict with:
                - risk_score: float ∈ [0, 1] (target for regression)
                - bandit_score: float ∈ [0, 1]
                - bandit_rules: list[str]
                - patched_flag: bool
                - synthetic_flag: bool
                - metadata: dict with additional info
        """
        graph_dict = self.graphs[idx]
        graph_data = self._graph_to_pyg(graph_dict)
        
        # Initialize weak signals with defaults
        weak_signals = {
            "risk_score": 0.0,  # Target risk score (computed from weak signals)
            "bandit_score": 0.0,
            "bandit_rules": [],
            "patched_flag": False,
            "synthetic_flag": False,
            "metadata": {},
            "project": graph_dict.get("file", "").split("/")[0] if "/" in graph_dict.get("file", "") else "unknown"
        }
        
        # Priority 1: Use labels if available (may contain weak signals)
        if self.labels is not None and idx < len(self.labels):
            label_data = self.labels[idx]
            metadata = label_data.get("metadata", {})
            
            # Extract weak signals from label data
            weak_signals["bandit_score"] = metadata.get("bandit_score", 0.0)
            weak_signals["bandit_rules"] = metadata.get("bandit_rules", [])
            weak_signals["patched_flag"] = metadata.get("patched_flag", False)
            weak_signals["synthetic_flag"] = metadata.get("synthetic_flag", False)
            weak_signals["metadata"] = metadata
            
            # Compute risk_score from weak signals
            # Priority: explicit risk_score > bandit_score > patched_flag > default
            if "risk_score" in label_data:
                weak_signals["risk_score"] = float(label_data["risk_score"])
            elif "bandit_score" in metadata:
                weak_signals["risk_score"] = float(metadata["bandit_score"])
            elif weak_signals["patched_flag"]:
                # Pre-patch samples have higher risk (if not explicitly marked as post-patch)
                if not metadata.get("is_post_patch", False):
                    weak_signals["risk_score"] = 0.8  # Pre-patch: higher risk
                else:
                    weak_signals["risk_score"] = 0.2  # Post-patch: lower risk
            elif weak_signals["synthetic_flag"]:
                weak_signals["risk_score"] = 0.9  # Synthetic vulnerabilities: high risk
            else:
                # Fallback: use bandit_score if available
                weak_signals["risk_score"] = weak_signals["bandit_score"]
        
        # Priority 2: Use taint records (semi-supervised or taint flow prediction)
        elif self.taint_records:
            file_path = graph_dict.get("file", "")
            matching_records = [
                r for r in self.taint_records
                if r.get("source", {}).get("file") == file_path
            ]
            
            if matching_records:
                # Taint records indicate potential vulnerability
                weak_signals["risk_score"] = 0.7  # Medium-high risk
                record = matching_records[0]
                
                # Map positions to node IDs
                from ir.taint_merge import map_taint_record
                
                # Reconstruct CPGGraph object
                cpg_nodes = [CPGNode(**n) for n in graph_dict["nodes"]]
                cpg_edges = [CPGEdge(**e) for e in graph_dict["edges"]]
                cpg_graph = CPGGraph(
                    file=file_path,
                    nodes=cpg_nodes,
                    edges=cpg_edges
                )
                
                src_id, sink_id, path_ids, meta = map_taint_record(cpg_graph, record)
                
                if src_id is not None and sink_id is not None:
                    weak_signals["metadata"]["taint_info"] = {
                        "source_id": src_id,
                        "sink_id": sink_id,
                        "path_ids": path_ids,
                        "meta": meta,
                    }
        
        # Extract project information for project-wise splitting
        file_path = graph_dict.get("file", "")
        if "/" in file_path:
            # Assume format: project_name/path/to/file.py
            project = file_path.split("/")[0]
            weak_signals["project"] = project
        
        return graph_data, weak_signals

