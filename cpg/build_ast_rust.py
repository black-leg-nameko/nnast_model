"""
CPG Builder wrapper for Rust implementation

Provides the same interface as the existing ASTCPGBuilder.
"""

from typing import List, Tuple, Optional
import cpg_rust
from ir.schema import CPGNode, CPGEdge


class ASTCPGBuilderRust:
    """
    CPG Builder using Rust implementation
    
    Provides the same interface as the existing ASTCPGBuilder.
    """
    
    def __init__(self, file_path: str, source: str):
        self.file = file_path
        self.source = source
        self.nodes: List[CPGNode] = []
        self.edges: List[CPGEdge] = []
        self._built = False
    
    def visit(self, tree):
        """
        Visit AST tree (compatibility method)
        Actual processing is done in build()
        """
        # For Rust implementation, AST tree visitation is not needed
        # CPG is generated directly from source code in build()
        pass
    
    def build(self) -> Tuple[List[CPGNode], List[CPGEdge]]:
        """
        Build CPG graph
        Same interface as the existing ASTCPGBuilder
        """
        if self._built:
            return self.nodes, self.edges
        
        # Generate CPG using Rust implementation
        graph_dict = cpg_rust.build_cpg(self.file, self.source)
        
        # Convert to CPGNode and CPGEdge lists
        self.nodes = []
        self.edges = []
        
        for n in graph_dict["nodes"]:
            self.nodes.append(CPGNode(
                id=n["id"],
                kind=n["kind"],
                file=n["file"],
                span=n["span"],
                code=n.get("code"),
                symbol=n.get("symbol"),
                type_hint=n.get("type_hint"),
                flags=n.get("flags", []),
                attrs=n.get("attrs", {})
            ))
        
        for e in graph_dict["edges"]:
            self.edges.append(CPGEdge(
                src=e["src"],
                dst=e["dst"],
                kind=e["kind"],
                attrs=e.get("attrs")
            ))
        
        self._built = True
        return self.nodes, self.edges


# For backward compatibility, can be used as ASTCPGBuilder
# Can be switched via environment variables or configuration
import os

# Enable Rust implementation via environment variable
USE_RUST = os.getenv("USE_RUST_CPG", "true").lower() not in ("false", "0", "no")

if USE_RUST:
    # Use Rust implementation
    ASTCPGBuilder = ASTCPGBuilderRust
else:
    # Use existing Python implementation (default)
    from cpg.build_ast import ASTCPGBuilder
