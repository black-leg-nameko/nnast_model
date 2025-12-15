import ast
from typing import List, Tuple, Optional

from ir.schema import CPGNode, CPGEdge
from cpg.parse import get_span, extract_code


KIND_MAP = {
    ast.Module: "Module",
    ast.FunctionDef: "Function",
    ast.AsyncFunctionDef: "Function",
    ast.ClassDef: "Class",
    ast.Assign: "Assign",
    ast.AnnAssign: "AnnAssign",
    ast.AugAssign: "AugAssign",
    ast.Return: "Return",
    ast.Call: "Call",
    ast.Name: "Name",
    ast.Attribute: "Attribute",
    ast.Constant: "Literal",
    ast.arg: "Arg",
    ast.arguments: "Arguments",
    ast.JoinedStr: "JoinedStr",
    ast.FormattedValue: "FormattedValue",
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

    def _annotation_str(self, annotation: Optional[ast.AST]) -> Optional[str]:
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except Exception:
            return None


    def generic_visit(self, node):
        total_lines = len(self.source_lines)
        last_line_len = len(self.source_lines[-1]) if self.source_lines else 0

        span = get_span(node, total_lines=total_lines, last_line_len=last_line_len)
        if span is None:
            # Skip nodes without positions but still traverse children
            super().generic_visit(node)
            return

        node_id = self._new_id()
        kind = KIND_MAP.get(type(node), "Stmt")

        code = extract_code(self.source_lines, span)

        symbol = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = node.name
        elif isinstance(node, ast.Name):
            symbol = node.id
        elif isinstance(node, ast.arg):
            symbol = node.arg

        type_hint = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            type_hint = self._annotation_str(node.returns)
        elif isinstance(node, ast.arg):
            type_hint = self._annotation_str(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            type_hint = self._annotation_str(node.annotation)

        cpg_node = CPGNode(
            id=node_id,
            kind=kind,
            file=self.file,
            span=span,
            code=code,
            symbol=symbol,
            type_hint=type_hint,
            flags=[],
            attrs={},
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
