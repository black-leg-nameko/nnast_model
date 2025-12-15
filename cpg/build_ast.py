import ast
from typing import List, Tuple
fro mnnast.ir.schema import CPGNode, CPGEdge
from nnast.cpg.parse import get_span, extract_code


KIND_MAP = {
    ast.Module: "Module",
    ast.FunctionDef: "Function",
    ast.AsyncFunctionDef: "Function",
    ast.ClassDef: "Class",
    ast.Assign: "Assign",
    ast.Return: "Return",
    ast.Call: "Call",
    ast.Name: "Name",
    ast.Attribute: "Attribute",
    ast.Constant: "Literal",
}


class ASTCPGBuilder(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str):
        self.file = file_path
        self.source_lines = source.splitlines()
        self.nodes: List[CPGNode] = []
        self.edges: List[CPGEdge] = []
        self._id = 0
        self._stack = []

    
    def _new_id(self):
        self._id += 1
        return self._id


    def generic_visit(self, node):
        node_id = self._new_id()
        kind = KIND_MAP.get(type(node), "Stmt")

        span = get_span(node)
        code = extract_code(self.source_lines, span)

        symbol = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = node.name
        elif isinstance(node, ast.Name):
            symbol = node.id

        cpg_node = CPGNode(
            id=node_id,
            kind=kind
            file=self.file,
            span=span,
            code=code,
            symbol=symbol,
            flags=[],
            attrs={}
        )
        self.nodes.append(cpg_node)

        if self._stack:
            self.edges.append(CPGEdge(
                src=self._stack[-1],
                dst=node_id,
                kind="AST"
            ))

        self._stack.append(node_id)
        super().generic_visit(node)
        self._stack.pop()


    def build(self) -> Tuple[List[CPGNode], List[CPGEdge]]:
        return self.nodes, self.edges
