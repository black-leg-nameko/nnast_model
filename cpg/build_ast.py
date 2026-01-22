import ast
import os
from typing import Dict, List, Tuple, Optional

from ir.schema import CPGNode, CPGEdge
from cpg.parse import get_span, extract_code

# デフォルトでRust実装を使用（環境変数で無効化可能）
# USE_RUST_CPG=false でPython実装を明示的に使用
USE_RUST = os.getenv("USE_RUST_CPG", "true").lower() not in ("false", "0", "no")


KIND_MAP = {
    ast.Module: "Module",
    ast.FunctionDef: "Function",
    ast.AsyncFunctionDef: "Function",
    ast.ClassDef: "Class",
    ast.With: "With",
    ast.AsyncWith: "With",
    ast.Try: "Try",
    ast.ExceptHandler: "Except",
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
    ast.ListComp: "ListComp",
    ast.SetComp: "SetComp",
    ast.DictComp: "DictComp",
    ast.GeneratorExp: "Generator",
    ast.AsyncFor: "AsyncFor",
    ast.Await: "Await",
}


class ASTCPGBuilder(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str):
        self.file = file_path
        self.source_lines = source.splitlines()
        self.nodes: List[CPGNode] = []
        self.edges: List[CPGEdge] = []
        self._id = 0
        self._stack = []
        # Stack of AST nodes to keep track of parents for CFG construction
        self._ast_stack: List[ast.AST] = []
        # Map: parent AST node -> last statement node id in that block (for CFG)
        self._last_stmt_by_parent: Dict[ast.AST, int] = {}
        # Variable definition scopes for DFG (stack of {name -> defining node id})
        self._scopes: List[Dict[str, int]] = [{}]
        # Mapping from AST node to CPG node id (for CFG on control structures)
        self._node_ids: Dict[ast.AST, int] = {}
        # Stack of enclosing loops (for future extensions such as break/continue)
        self._loop_stack: List[ast.AST] = []

    
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


    def _define_symbol(self, name: str, node_id: int) -> None:
        # Register (or overwrite) definition in the current scope
        if self._scopes:
            self._scopes[-1][name] = node_id

    def _resolve_symbol(self, name: str) -> Optional[int]:
        # Look up definition from innermost to outermost scope
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def generic_visit(self, node):
        # Parent AST node (for CFG per parent-body block)
        parent_ast = self._ast_stack[-1] if self._ast_stack else None
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

        # Basic type-hint integration into attrs (DataType / ContainerType).
        attrs = {}
        if type_hint is not None:
            # Very lightweight heuristic: split container type like List[int], Sequence[str]
            container = None
            inner = None
            if "[" in type_hint and type_hint.endswith("]"):
                container, inner_part = type_hint.split("[", 1)
                container = container.strip()
                inner = inner_part[:-1].strip()  # drop trailing ]
            if container:
                attrs["DataType"] = inner
                attrs["ContainerType"] = container
            else:
                attrs["DataType"] = type_hint

        cpg_node = CPGNode(
            id=node_id,
            kind=kind,
            file=self.file,
            span=span,
            code=code,
            symbol=symbol,
            type_hint=type_hint,
            flags=[],
            attrs=attrs,
        )
        self.nodes.append(cpg_node)
        # Remember mapping from AST to CPG node id
        self._node_ids[node] = node_id

        # --- CFG: control-structure-aware edges ---
        if isinstance(node, ast.stmt) and parent_ast is not None:
            # Sequential edges within the same block
            last_stmt_id = self._last_stmt_by_parent.get(parent_ast)
            if last_stmt_id is not None:
                self.edges.append(CPGEdge(
                    src=last_stmt_id,
                    dst=node_id,
                    kind="CFG",
                ))
            self._last_stmt_by_parent[parent_ast] = node_id

            # For control structures, connect the control node to the first
            # statement of each body/branch later by remembering this node id.
            # We don't build all edges here, but we use this id as the "head"
            # when visiting children.

        # --- DFG: variables, attributes, and simple calls ---
        # Definitions:
        #   - function arguments (ast.arg)
        #   - names in Store context
        # Uses:
        #   - names in Load / Del context
        #   - attributes (obj.attr) as uses of the base object
        if isinstance(node, ast.arg) and symbol is not None:
            self._define_symbol(symbol, node_id)
        elif isinstance(node, ast.Name) and symbol is not None:
            if isinstance(node.ctx, ast.Store):
                self._define_symbol(symbol, node_id)
            else:
                def_id = self._resolve_symbol(symbol)
                if def_id is not None:
                    self.edges.append(CPGEdge(
                        src=def_id,
                        dst=node_id,
                        kind="DFG",
                    ))
        elif isinstance(node, ast.Attribute):
            # obj.attr: create DFG edge from obj definition to this attribute node
            # when obj is a simple name.
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                def_id = self._resolve_symbol(base_name)
                if def_id is not None:
                    self.edges.append(CPGEdge(
                        src=def_id,
                        dst=node_id,
                        kind="DFG",
                    ))
        elif isinstance(node, ast.Call):
            # For simple function calls like f(x, y), connect argument definitions
            # to their usage sites inside the call node.
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    def_id = self._resolve_symbol(arg.id)
                    if def_id is not None:
                        self.edges.append(CPGEdge(
                            src=def_id,
                            dst=node_id,
                            kind="DFG",
                        ))
            # For method calls like xs.count(0), also connect the base object
            # definition to the call node.
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                base_name = node.func.value.id
                def_id = self._resolve_symbol(base_name)
                if def_id is not None:
                    self.edges.append(CPGEdge(
                        src=def_id,
                        dst=node_id,
                        kind="DFG",
                    ))
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Comprehensions: connect all non-target variable uses inside the
            # comprehension to the comprehension node itself, so that the model
            # can see which external variables feed into the generated values.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and not isinstance(inner.ctx, ast.Store):
                    def_id = self._resolve_symbol(inner.id)
                    if def_id is not None:
                        self.edges.append(CPGEdge(
                            src=def_id,
                            dst=node_id,
                            kind="DFG",
                        ))

        # Enter a new scope for function bodies (local variables)
        new_scope = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if new_scope:
            self._scopes.append({})

        # Track loops for possible future handling of break/continue.
        is_loop = isinstance(node, (ast.For, ast.While, ast.AsyncFor))
        if is_loop:
            self._loop_stack.append(node)

        if self._stack:
            self.edges.append(CPGEdge(
                src=self._stack[-1],
                dst=node_id,
                kind="AST"
            ))

        self._stack.append(node_id)
        self._ast_stack.append(node)
        super().generic_visit(node)

        # --- Control-structure-aware CFG edges (after children are visited) ---
        # We only rely on very local information: CPG ids for the control node
        # itself and the first/last statements in its bodies.
        if isinstance(node, ast.If):
            head_id = node_id
            # then-branch
            if node.body:
                first_then_ast = node.body[0]
                first_then_id = self._node_ids.get(first_then_ast)
                if first_then_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_then_id,
                        kind="CFG",
                    ))
            # else-branch (including elif, which is represented as an If in orelse)
            if node.orelse:
                first_else_ast = node.orelse[0]
                first_else_id = self._node_ids.get(first_else_ast)
                if first_else_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_else_id,
                        kind="CFG",
                    ))

        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            head_id = node_id
            # Loop head -> first statement in the body
            if node.body:
                first_body_ast = node.body[0]
                first_body_id = self._node_ids.get(first_body_ast)
                if first_body_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_body_id,
                        kind="CFG",
                    ))
                # Back edge from last statement in the body to loop head
                last_body_ast = node.body[-1]
                last_body_id = self._node_ids.get(last_body_ast)
                if last_body_id is not None:
                    self.edges.append(CPGEdge(
                        src=last_body_id,
                        dst=head_id,
                        kind="CFG",
                    ))

        elif isinstance(node, ast.Try):
            head_id = node_id
            # try-body entry
            if node.body:
                first_body_ast = node.body[0]
                first_body_id = self._node_ids.get(first_body_ast)
                if first_body_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_body_id,
                        kind="CFG",
                    ))
            # except-handlers entry
            for handler in node.handlers:
                first_exc_ast = handler
                first_exc_id = self._node_ids.get(first_exc_ast)
                if first_exc_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_exc_id,
                        kind="CFG",
                    ))
            # finally/else はとりあえずエントリのみ簡易に扱う
            if node.finalbody:
                first_final_ast = node.finalbody[0]
                first_final_id = self._node_ids.get(first_final_ast)
                if first_final_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_final_id,
                        kind="CFG",
                    ))

        elif isinstance(node, (ast.With, ast.AsyncWith)):
            head_id = node_id
            # with-head -> body first statement
            if node.body:
                first_body_ast = node.body[0]
                first_body_id = self._node_ids.get(first_body_ast)
                if first_body_id is not None:
                    self.edges.append(CPGEdge(
                        src=head_id,
                        dst=first_body_id,
                        kind="CFG",
                    ))

        # Lightweight CFG inside comprehensions:
        # connect the comprehension node to its "output expression" so that
        # the model can see where generated elements come from, without
        # modeling full loop semantics.
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            head_id = node_id
            elt_ast = node.elt
            elt_id = self._node_ids.get(elt_ast)
            if elt_id is not None:
                self.edges.append(CPGEdge(
                    src=head_id,
                    dst=elt_id,
                    kind="CFG",
                ))
        elif isinstance(node, ast.DictComp):
            head_id = node_id
            # Use value expression as the "main" output of the dict comp.
            val_ast = node.value
            val_id = self._node_ids.get(val_ast)
            if val_id is not None:
                self.edges.append(CPGEdge(
                    src=head_id,
                    dst=val_id,
                    kind="CFG",
                ))

        self._ast_stack.pop()
        self._stack.pop()

        if is_loop:
            self._loop_stack.pop()

        if new_scope:
            self._scopes.pop()


    def build(self) -> Tuple[List[CPGNode], List[CPGEdge]]:
        return self.nodes, self.edges


# Rust実装のラッパー（デフォルトで有効化、環境変数で無効化可能）
if USE_RUST:
    try:
        import cpg_rust
        
        class ASTCPGBuilderRust:
            """
            Rust実装を使用するCPGビルダー
            既存のASTCPGBuilderと同じインターフェースを提供
            """
            
            def __init__(self, file_path: str, source: str):
                self.file = file_path
                self.source = source
                self.nodes: List[CPGNode] = []
                self.edges: List[CPGEdge] = []
                self._built = False
            
            def visit(self, tree):
                """
                ASTツリーを訪問（互換性のためのメソッド）
                実際の処理はbuild()で行う
                """
                # Rust実装では、ASTツリーの訪問は不要
                # build()で直接ソースコードからCPGを生成する
                pass
            
            def build(self) -> Tuple[List[CPGNode], List[CPGEdge]]:
                """
                CPGグラフを構築
                既存のASTCPGBuilderと同じインターフェース
                """
                if self._built:
                    return self.nodes, self.edges
                
                # Rust実装でCPGを生成
                graph_dict = cpg_rust.build_cpg(self.file, self.source)
                
                # CPGNodeとCPGEdgeのリストに変換
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
        
        # Rust実装を使用
        ASTCPGBuilder = ASTCPGBuilderRust
    except ImportError as e:
        # cpg_rustモジュールが利用できない場合はPython実装を使用
        import warnings
        warnings.warn(
            f"Rust実装（cpg_rust）が利用できません。Python実装を使用します。\n"
            f"エラー: {e}\n"
            f"Rust実装を使用するには: cd cpg_rust && maturin develop",
            UserWarning,
            stacklevel=2
        )
        # Python実装をそのまま使用（USE_RUST = Falseの状態）