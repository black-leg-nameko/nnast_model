"""
Python AST to JSON converter for Rust CPG builder

This module converts Python AST nodes to JSON format that can be processed
by the Rust implementation.
"""

import ast
import json
from typing import Any, Dict, List, Optional


def ast_node_to_dict(node: ast.AST, source_lines: List[str]) -> Dict[str, Any]:
    """
    Convert Python AST node to dictionary format for JSON serialization.
    
    Args:
        node: AST node to convert
        source_lines: Source code lines for span calculation
        
    Returns:
        Dictionary representation of the AST node
    """
    # Get span information
    span = get_span(node, len(source_lines), len(source_lines[-1]) if source_lines else 0)
    
    # Get node type
    node_type = type(node).__name__
    
    # Build base dictionary
    result: Dict[str, Any] = {
        "node_type": node_type,
        "span": span,
        "symbol": None,
        "type_hint": None,
        "attrs": {},
        "children": [],
        "ctx": None,
    }
    
    # Extract symbol name for specific node types
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        result["symbol"] = node.name
    elif isinstance(node, ast.Name):
        result["symbol"] = node.id
        result["ctx"] = type(node.ctx).__name__
    elif isinstance(node, ast.arg):
        result["symbol"] = node.arg
    
    # Extract type hints
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.returns:
            try:
                result["type_hint"] = ast.unparse(node.returns)
            except Exception:
                pass
    elif isinstance(node, ast.arg):
        if node.annotation:
            try:
                result["type_hint"] = ast.unparse(node.annotation)
            except Exception:
                pass
    elif isinstance(node, ast.AnnAssign):
        if node.annotation:
            try:
                result["type_hint"] = ast.unparse(node.annotation)
            except Exception:
                pass
    
    # Recursively process children
    # For control structures, preserve field names (body, orelse, handlers, etc.)
    # so that CFG edges can be generated correctly
    for field, value in ast.iter_fields(node):
        if field in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            # Skip position fields (already in span)
            continue
        
        # Store field names for control structures in attrs
        if field in ("body", "orelse", "handlers", "finalbody", "elt", "value", "key"):
            if isinstance(value, ast.AST):
                child_dict = ast_node_to_dict(value, source_lines)
                child_dict["attrs"]["_field_name"] = field
                result["children"].append(child_dict)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_dict = ast_node_to_dict(item, source_lines)
                        child_dict["attrs"]["_field_name"] = field
                        result["children"].append(child_dict)
        else:
            # For other fields, add without field name (backward compatible)
        if isinstance(value, ast.AST):
            result["children"].append(ast_node_to_dict(value, source_lines))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    result["children"].append(ast_node_to_dict(item, source_lines))
    
    return result


def get_span(node: ast.AST, total_lines: int, last_line_len: int) -> Optional[tuple]:
    """Return (sl, sc, el, ec) span or None when node has no lineno."""
    if isinstance(node, ast.Module):
        end_col = last_line_len if total_lines > 0 else 0
        return (1, 0, max(total_lines, 1), end_col)
    
    sl = getattr(node, "lineno", None)
    sc = getattr(node, "col_offset", None)
    el = getattr(node, "end_lineno", sl)
    ec = getattr(node, "end_col_offset", sc)
    
    if sl is None:
        return None
    
    return (sl, sc or 0, el or sl, ec or 0)


def parse_and_convert_to_json(source: str) -> str:
    """
    Parse Python source code and convert AST to JSON.
    
    Args:
        source: Python source code as string
        
    Returns:
        JSON string representing the AST
    """
    tree = ast.parse(source)
    source_lines = source.splitlines()
    ast_dict = ast_node_to_dict(tree, source_lines)
    return json.dumps(ast_dict, ensure_ascii=False)


if __name__ == "__main__":
    # Test
    test_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
    json_str = parse_and_convert_to_json(test_code)
    print(json_str)
