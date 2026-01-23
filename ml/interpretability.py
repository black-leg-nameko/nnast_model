"""
Interpretability utilities for CPG+GNN model.

Provides functions for visualizing model predictions and understanding
which parts of the code contribute to vulnerability detection.
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


def visualize_node_importance(
    graph_dict: Dict[str, Any],
    node_importance: torch.Tensor,
    output_path: Optional[Path] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Visualize node importance scores on the original code.
    
    Args:
        graph_dict: CPG graph dictionary
        node_importance: Node importance scores (num_nodes,)
        output_path: Optional path to save visualization
        top_k: Number of top nodes to highlight
        
    Returns:
        Dictionary with visualization data
    """
    nodes = graph_dict.get("nodes", [])
    
    # Get top-K important nodes
    top_k = min(top_k, len(nodes))
    top_indices = torch.topk(node_importance, top_k).indices.cpu().numpy()
    
    # Extract node information
    highlighted_nodes = []
    for idx in top_indices:
        if idx < len(nodes):
            node = nodes[idx]
            highlighted_nodes.append({
                "node_id": node.get("id"),
                "kind": node.get("kind"),
                "code": node.get("code", "")[:100],  # Truncate long code
                "span": node.get("span"),
                "importance": float(node_importance[idx].item())
            })
    
    visualization = {
        "file": graph_dict.get("file", ""),
        "top_nodes": highlighted_nodes,
        "total_nodes": len(nodes),
        "top_k": top_k
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(visualization, f, indent=2, ensure_ascii=False)
    
    return visualization


def create_explanation_report(
    explanation: Dict[str, Any],
    graph_dict: Dict[str, Any],
    output_path: Path
) -> str:
    """
    Create a human-readable explanation report.
    
    Args:
        explanation: Explanation dictionary from model.explain_prediction()
        graph_dict: CPG graph dictionary
        output_path: Path to save report
        
    Returns:
        Report text
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Vulnerability Detection Explanation Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Risk score
    risk_scores = explanation.get("risk_score", [])
    if risk_scores:
        report_lines.append(f"Predicted Risk Score: {risk_scores[0]:.4f}")
        report_lines.append("")
    
    # Edge type importance
    edge_importance = explanation.get("edge_type_importance")
    if edge_importance:
        report_lines.append("Edge Type Importance:")
        for edge_type, importance in sorted(edge_importance.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {edge_type}: {importance:.4f}")
        report_lines.append("")
    
    # Top nodes
    top_nodes = explanation.get("top_nodes", [])
    nodes = graph_dict.get("nodes", [])
    node_dict = {n["id"]: n for n in nodes}
    
    for graph_info in top_nodes:
        graph_idx = graph_info.get("graph_idx", 0)
        report_lines.append(f"Graph {graph_idx} - Top Important Nodes:")
        
        for node_id, importance in zip(
            graph_info.get("top_node_ids", []),
            graph_info.get("importance_scores", [])
        ):
            if node_id in node_dict:
                node = node_dict[node_id]
                code = node.get("code", "")[:80]
                report_lines.append(
                    f"  Node {node_id} ({node.get('kind')}): "
                    f"importance={importance:.4f}"
                )
                if code:
                    report_lines.append(f"    Code: {code}")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def compare_explanations(
    explanations: List[Dict[str, Any]],
    labels: List[str]
) -> Dict[str, Any]:
    """
    Compare explanations from multiple models or configurations.
    
    Args:
        explanations: List of explanation dictionaries
        labels: Labels for each explanation
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "models": labels,
        "risk_scores": [exp.get("risk_score", [0.0])[0] for exp in explanations],
        "edge_type_importance": {}
    }
    
    # Compare edge type importance
    all_edge_types = set()
    for exp in explanations:
        edge_imp = exp.get("edge_type_importance", {})
        all_edge_types.update(edge_imp.keys())
    
    for edge_type in all_edge_types:
        comparison["edge_type_importance"][edge_type] = [
            exp.get("edge_type_importance", {}).get(edge_type, 0.0)
            for exp in explanations
        ]
    
    return comparison
