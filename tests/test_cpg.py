import json
from dataclasses import asdict

import pytest

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder


def build_graph_from_source(src: str):
    tree = parse_source(src)
    builder = ASTCPGBuilder("test.py", src)
    builder.visit(tree)
    nodes, edges = builder.build()
    return nodes, edges


def test_function_with_type_hints_and_fstring():
    src = """import os

def greet(name: str) -> str:
    return f"Hello, {name}!"

"""
    nodes, edges = build_graph_from_source(src)

    # Take the first few nodes to keep snapshot small and stable.
    snapshot = [asdict(n) for n in nodes[:10]]
    expected = [
        {
            "id": 1,
            "kind": "Module",
            "file": "test.py",
            "span": (1, 0, 5, 0),
            "code": 'import os\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 2,
            "kind": "Stmt",
            "file": "test.py",
            "span": (1, 0, 1, 9),
            "code": "import os",
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 3,
            "kind": "Stmt",
            "file": "test.py",
            "span": (1, 7, 1, 9),
            "code": "os",
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 4,
            "kind": "Function",
            "file": "test.py",
            "span": (3, 0, 4, 28),
            "code": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"',
            "symbol": "greet",
            "type_hint": "str",
            "flags": [],
            "attrs": {},
        },
        {
            "id": 5,
            "kind": "Arg",
            "file": "test.py",
            "span": (3, 10, 3, 19),
            "code": "name: str",
            "symbol": "name",
            "type_hint": "str",
            "flags": [],
            "attrs": {},
        },
        {
            "id": 6,
            "kind": "Name",
            "file": "test.py",
            "span": (3, 16, 3, 19),
            "code": "str",
            "symbol": "str",
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 7,
            "kind": "Return",
            "file": "test.py",
            "span": (4, 4, 4, 28),
            "code": 'return f"Hello, {name}!"',
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 8,
            "kind": "JoinedStr",
            "file": "test.py",
            "span": (4, 11, 4, 28),
            "code": 'f"Hello, {name}!"',
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 9,
            "kind": "Literal",
            "file": "test.py",
            "span": (4, 13, 4, 20),
            "code": "Hello, ",
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
        {
            "id": 10,
            "kind": "FormattedValue",
            "file": "test.py",
            "span": (4, 20, 4, 26),
            "code": "{name}",
            "symbol": None,
            "type_hint": None,
            "flags": [],
            "attrs": {},
        },
    ]
    assert snapshot == expected
    assert len(edges) >= 9  # basic structural sanity


def test_multiline_block_code_extraction():
    src = """if foo:
    x = 1
    y = x + 2
"""
    nodes, _ = build_graph_from_source(src)
    # find the first Stmt spanning the block
    block = next(n for n in nodes if n.kind == "Stmt" and n.span == (1, 0, 3, 13))
    assert block.code == "if foo:\n    x = 1\n    y = x + 2"