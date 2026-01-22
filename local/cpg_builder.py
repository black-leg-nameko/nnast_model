#!/usr/bin/env python3
"""
CPG Builder Module

Provides unified interface for CPG construction using Rust implementation (preferred)
or Python fallback implementation.
"""
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Try to import Rust implementation
try:
    import cpg_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    # Fallback to Python implementation
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cpg.parse import parse_source
    from cpg.build_ast import ASTCPGBuilder
    from ir.schema import CPGGraph


def ast_to_json(source: str) -> str:
    """
    Convert Python AST to JSON string.
    
    Args:
        source: Python source code
        
    Returns:
        AST as JSON string
    """
    tree = ast.parse(source)
    
    def ast_to_dict(node):
        """Convert AST node to dictionary."""
        if isinstance(node, ast.AST):
            result = {
                'type': type(node).__name__,
                'lineno': getattr(node, 'lineno', None),
                'col_offset': getattr(node, 'col_offset', None),
                'end_lineno': getattr(node, 'end_lineno', None),
                'end_col_offset': getattr(node, 'end_col_offset', None),
            }
            
            # Add fields
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    result[field] = [ast_to_dict(item) for item in value]
                elif isinstance(value, ast.AST):
                    result[field] = ast_to_dict(value)
                else:
                    result[field] = value
            
            return result
        elif isinstance(node, list):
            return [ast_to_dict(item) for item in node]
        else:
            return node
    
    ast_dict = ast_to_dict(tree)
    return json.dumps(ast_dict)


def build_cpg_rust(file_path: Path, source: str) -> Dict[str, Any]:
    """
    Build CPG using Rust implementation.
    
    Args:
        file_path: Path to Python file
        source: Source code
        
    Returns:
        CPG graph as dictionary
    """
    # Convert AST to JSON
    ast_json = ast_to_json(source)
    
    # Build CPG using Rust
    graph = cpg_rust.build_cpg(str(file_path), source, ast_json)
    
    # Convert PyDict to regular dict
    # Rust実装はPyDictを返すので、辞書として扱う
    import json
    
    # PyDictをJSON経由で変換（最も確実な方法）
    if hasattr(graph, 'keys'):
        # PyDictの場合
        graph_dict = dict(graph)
    else:
        # Already a dict
        graph_dict = graph
    
    # エッジのキーを変換（Rust実装はsrc/dst、Python実装はsource/target）
    result = {
        'file': graph_dict.get('file', str(file_path)),
        'nodes': [],
        'edges': []
    }
    
    # ノードを変換
    for node in graph_dict.get('nodes', []):
        if isinstance(node, dict):
            result['nodes'].append(node)
        else:
            # PyDictの場合
            result['nodes'].append(dict(node))
    
    # エッジを変換（src/dst → source/target）
    for edge in graph_dict.get('edges', []):
        if isinstance(edge, dict):
            edge_dict = dict(edge)
        else:
            # PyDictの場合
            edge_dict = dict(edge)
        
        # キーを変換
        if 'src' in edge_dict:
            edge_dict['source'] = edge_dict.pop('src')
        if 'dst' in edge_dict:
            edge_dict['target'] = edge_dict.pop('dst')
        
        result['edges'].append(edge_dict)
    
    return result


def build_cpg_python(file_path: Path, source: str) -> Dict[str, Any]:
    """
    Build CPG using Python implementation (fallback).
    
    Args:
        file_path: Path to Python file
        source: Source code
        
    Returns:
        CPG graph as dictionary
    """
    tree = parse_source(source)
    builder = ASTCPGBuilder(str(file_path), source)
    builder.visit(tree)
    
    graph = CPGGraph(
        file=str(file_path),
        nodes=builder.nodes,
        edges=builder.edges,
    )
    
    return graph.to_dict()


def build_cpg_single_file(file_path: Path, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Build CPG for a single file.
    
    Args:
        file_path: Path to Python file
        source: Source code (if None, read from file)
        
    Returns:
        CPG graph as dictionary
    """
    if source is None:
        source = file_path.read_text(encoding="utf-8")
    
    if RUST_AVAILABLE:
        try:
            return build_cpg_rust(file_path, source)
        except Exception as e:
            print(f"Warning: Rust CPG build failed for {file_path}: {e}", file=sys.stderr)
            print("Falling back to Python implementation...", file=sys.stderr)
            return build_cpg_python(file_path, source)
    else:
        return build_cpg_python(file_path, source)


def build_cpg_from_directory(
    repo_path: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Build full CPG from directory (all Python files).
    
    Args:
        repo_path: Repository root path
        output_path: Optional output path for caching
        
    Returns:
        Merged CPG graph as dictionary
    """
    # Find all Python files
    py_files = list(repo_path.rglob("*.py"))
    
    if not py_files:
        return {"file": str(repo_path), "nodes": [], "edges": []}
    
    # Build CPG for each file
    all_nodes = []
    all_edges = []
    node_id_offset = 0
    
    for py_file in py_files:
        try:
            # Skip __pycache__ and test files if needed
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text(encoding="utf-8")
            graph = build_cpg_single_file(py_file, source)
            
            # Merge nodes and edges with offset
            file_nodes = graph.get("nodes", [])
            file_edges = graph.get("edges", [])
            
            # Adjust node IDs to avoid conflicts
            node_id_map = {}
            for node in file_nodes:
                old_id = node.get("id")
                new_id = node_id_offset + old_id if isinstance(old_id, int) else old_id
                node_id_map[old_id] = new_id
                node["id"] = new_id
                all_nodes.append(node)
            
            # Adjust edge source/target IDs
            for edge in file_edges:
                source_id = edge.get("source")
                target_id = edge.get("target")
                
                if source_id in node_id_map:
                    edge["source"] = node_id_map[source_id]
                if target_id in node_id_map:
                    edge["target"] = node_id_map[target_id]
                
                all_edges.append(edge)
            
            # Update offset for next file
            if file_nodes:
                max_id = max(
                    n.get("id") for n in file_nodes 
                    if isinstance(n.get("id"), int)
                )
                node_id_offset = max_id + 1
            
        except SyntaxError as e:
            print(f"Warning: Syntax error in {py_file}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Warning: Failed to build CPG for {py_file}: {e}", file=sys.stderr)
            continue
    
    return {
        "file": str(repo_path),
        "nodes": all_nodes,
        "edges": all_edges
    }


def build_full_cpg(
    repo_path: Path,
    output_path: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Build full CPG with caching support.
    
    Args:
        repo_path: Repository root path
        output_path: Optional output path for caching
        use_cache: Whether to use cache if available
        
    Returns:
        CPG graph as dictionary
    """
    # Check cache
    if use_cache and output_path and output_path.exists():
        try:
            with open(output_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    cached = json.loads(lines[0])
                    print(f"Using cached CPG from {output_path}")
                    return cached
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}", file=sys.stderr)
    
    # Build CPG
    if repo_path.is_file():
        # Single file
        graph = build_cpg_single_file(repo_path)
    else:
        # Directory
        graph = build_cpg_from_directory(repo_path)
    
    # Save to cache
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(json.dumps(graph) + '\n')
    
    return graph
