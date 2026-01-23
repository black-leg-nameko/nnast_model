#!/usr/bin/env python3
"""
Data minimization and masking functionality

Based on the design document, minimizes data sent to the cloud and masks sensitive information.
"""
import re
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# Secret information patterns
SECRET_PATTERNS = [
    (r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', 'API_KEY'),
    (r'secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', 'SECRET_KEY'),
    (r'password["\']?\s*[:=]\s*["\']?([^\s"\']{8,})["\']?', 'PASSWORD'),
    (r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', 'TOKEN'),
    (r'bearer\s+([a-zA-Z0-9_\-\.]{20,})', 'BEARER_TOKEN'),
    (r'jwt["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{50,})["\']?', 'JWT'),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', 'PRIVATE_KEY'),
    (r'-----BEGIN\s+EC\s+PRIVATE\s+KEY-----', 'EC_PRIVATE_KEY'),
    (r'ssh-rsa\s+[A-Za-z0-9+/]+', 'SSH_KEY'),
    (r'AKIA[0-9A-Z]{16}', 'AWS_ACCESS_KEY'),
    (r'[0-9a-f]{32}', 'MD5_HASH'),  # Simple detection
]

# URL pattern
URL_PATTERN = re.compile(
    r'https?://[^\s"\']+',
    re.IGNORECASE
)

# Token-like strings
TOKEN_PATTERN = re.compile(
    r'\b[a-zA-Z0-9_\-]{32,}\b'
)


def mask_string_literal(value: str) -> str:
    """
    Mask string literal
    
    Args:
        value: String to mask
        
    Returns:
        Masked string
    """
    masked = value
    
    # Detect and remove secret information patterns
    for pattern, label in SECRET_PATTERNS:
        masked = re.sub(pattern, f'<REDACTED_{label}>', masked, flags=re.IGNORECASE)
    
    # Mask URLs
    masked = URL_PATTERN.sub('<REDACTED_URL>', masked)
    
    # Mask long token-like strings
    def mask_token(match):
        token = match.group(0)
        if len(token) >= 32:
            return f'<REDACTED_TOKEN_{len(token)}>'
        return token
    
    masked = TOKEN_PATTERN.sub(mask_token, masked)
    
    return masked


def mask_identifier(identifier: str, salt: Optional[str] = None) -> str:
    """
    Mask identifier (HMAC hashing or replacement)
    
    Args:
        identifier: Identifier (variable name, function name, etc.)
        salt: Salt for hashing (optional)
        
    Returns:
        Masked identifier
    """
    if salt:
        # HMAC hashing
        h = hmac.new(salt.encode(), identifier.encode(), hashlib.sha256)
        return f"VAR_{h.hexdigest()[:8]}"
    else:
        # Simple replacement (counter-based)
        # In actual implementation, mapping needs to be maintained
        return f"VAR_{hash(identifier) % 10000}"


def extract_subgraph(
    graph: Dict[str, Any],
    node_ids: List[int],
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Extract subgraph from CPG graph
    
    Args:
        graph: Full CPG graph
        node_ids: List of target node IDs to extract
        max_depth: Maximum search depth
        
    Returns:
        Extracted subgraph
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Mapping from node ID to index
    node_id_to_idx = {node.get("id"): idx for idx, node in enumerate(nodes)}
    
    # Set of target nodes to extract
    target_nodes = set(node_ids)
    visited_nodes = set(node_ids)
    
    # Explore adjacent nodes from edges
    for depth in range(max_depth):
        new_nodes = set()
        for edge in edges:
            src_id = edge.get("source")
            tgt_id = edge.get("target")
            
            if src_id in visited_nodes and tgt_id not in visited_nodes:
                new_nodes.add(tgt_id)
            elif tgt_id in visited_nodes and src_id not in visited_nodes:
                new_nodes.add(src_id)
        
        if not new_nodes:
            break
        
        visited_nodes.update(new_nodes)
        target_nodes.update(new_nodes)
    
    # Build subgraph
    subgraph_nodes = []
    subgraph_edges = []
    
    for node_id in target_nodes:
        if node_id in node_id_to_idx:
            node = nodes[node_id_to_idx[node_id]]
            # Mask sensitive information
            masked_node = mask_node_data(node)
            subgraph_nodes.append(masked_node)
    
    for edge in edges:
        src_id = edge.get("source")
        tgt_id = edge.get("target")
        if src_id in target_nodes and tgt_id in target_nodes:
            subgraph_edges.append(edge)
    
    return {
        "nodes": subgraph_nodes,
        "edges": subgraph_edges
    }


def mask_node_data(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask node data
    
    Args:
        node: CPG node
        
    Returns:
        Masked node
    """
    masked_node = node.copy()
    
    # Mask identifiers
    if "name" in masked_node:
        masked_node["name"] = mask_identifier(masked_node["name"])
    
    if "label" in masked_node:
        masked_node["label"] = mask_identifier(masked_node["label"])
    
    # Mask string literals
    if "value" in masked_node and isinstance(masked_node["value"], str):
        masked_node["value"] = mask_string_literal(masked_node["value"])
    
    if "code" in masked_node:
        masked_node["code"] = mask_string_literal(masked_node["code"])
    
    return masked_node


def minimize_code_snippet(
    code: str,
    line_start: int,
    line_end: int,
    context_lines: int = 5
) -> str:
    """
    Minimize code snippet (extract only necessary lines)
    
    Args:
        code: Full source code
        line_start: Start line number
        line_end: End line number
        context_lines: Number of context lines before and after
        
    Returns:
        Minimized code snippet
    """
    lines = code.split('\n')
    
    # Line numbers are 1-based
    start_idx = max(0, line_start - 1 - context_lines)
    end_idx = min(len(lines), line_end + context_lines)
    
    snippet_lines = lines[start_idx:end_idx]
    
    # Masking
    masked_lines = []
    for i, line in enumerate(snippet_lines):
        masked_line = mask_string_literal(line)
        masked_lines.append(masked_line)
    
    return '\n'.join(masked_lines)


def create_minimized_payload(
    findings: List[Dict[str, Any]],
    graph: Dict[str, Any],
    source_code: Dict[str, str],
    tenant_id: str,
    project_id: str,
    repo: str,
    job_id: str,
    max_issues: int = 5
) -> Dict[str, Any]:
    """
    Create minimized payload to send to cloud
    
    Args:
        findings: List of detected vulnerabilities
        graph: Full CPG graph
        source_code: Mapping of file path -> source code
        tenant_id: Tenant ID
        project_id: Project ID
        repo: Repository (org/repo format)
        job_id: Job ID
        max_issues: Maximum number of issues
        
    Returns:
        Minimized payload
    """
    minimized_findings = []
    
    # Process only top-K findings
    top_findings = findings[:max_issues]
    
    for finding in top_findings:
        location = finding.get("location", {})
        file_path = location.get("file", "")
        line = location.get("line", 0)
        
        # Extract subgraph
        node_ids = finding.get("node_ids", [])
        subgraph = extract_subgraph(graph, node_ids) if node_ids else {"nodes": [], "edges": []}
        
        # Extract code snippet
        code_snippet = ""
        if file_path in source_code:
            code_snippet = minimize_code_snippet(
                source_code[file_path],
                line,
                line,
                context_lines=5
            )
        
        minimized_finding = {
            "fingerprint": finding.get("fingerprint"),
            "severity": finding.get("severity", "MEDIUM"),
            "rule_id": finding.get("rule_id", "UNKNOWN"),
            "cwe": finding.get("cwe", ""),
            "location": {
                "file": file_path,
                "line": line
            },
            "subgraph": subgraph,
            "code_snippet_masked": code_snippet
        }
        
        minimized_findings.append(minimized_finding)
    
    payload = {
        "tenant_id": tenant_id,
        "project_id": project_id,
        "repo": repo,
        "job_id": job_id,
        "findings": minimized_findings,
        "limits": {
            "max_issues": max_issues
        }
    }
    
    return payload


def generate_fingerprint(
    repo: str,
    rule_id: str,
    file_path: str,
    function_name: str,
    sink_line: int
) -> str:
    """
    Generate fingerprint for Issue
    
    Args:
        repo: Repository (org/repo format)
        rule_id: Rule ID
        file_path: File path
        function_name: Function name
        sink_line: Sink line number
        
    Returns:
        Fingerprint (SHA256 hash)
    """
    fingerprint_string = f"{repo}:{rule_id}:{file_path}:{function_name}:{sink_line}"
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()
