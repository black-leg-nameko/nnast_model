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
    # 基本的な構造と型情報だけを検証する（attrsの詳細は別テストで確認）
    func = snapshot[3]
    arg = snapshot[4]

    assert func["kind"] == "Function"
    assert func["symbol"] == "greet"
    assert func["type_hint"] == "str"

    assert arg["kind"] == "Arg"
    assert arg["symbol"] == "name"
    assert arg["type_hint"] == "str"
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


def test_cfg_edges_within_block():
    src = """def f():
    x = 1
    y = 2
    z = x + y
"""
    nodes, edges = build_graph_from_source(src)

    # Helper to get CFG edge pairs
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    # Find assignment statements in the function body (three lines in order)
    assigns = sorted(
        [n for n in nodes if n.kind == "Assign"],
        key=lambda n: n.span[0],
    )
    # Expect three assignment statements corresponding to x, y, z
    assert len(assigns) == 3
    x_stmt, y_stmt, z_stmt = assigns

    assert (x_stmt.id, y_stmt.id) in cfg_edges
    assert (y_stmt.id, z_stmt.id) in cfg_edges


def test_dfg_edges_for_simple_assignments():
    src = """def f():
    x = 1
    y = x + 2
    return y
"""
    nodes, edges = build_graph_from_source(src)

    dfg_edges = {(e.src, e.dst) for e in edges if e.kind == "DFG"}

    # Definitions: x (line 2), y (line 3), function argument none
    # Uses: x (in 'x + 2'), y (in 'return y')
    x_def = next(n for n in nodes if n.kind == "Name" and n.code == "x" and n.span[0] == 2)
    x_use = next(n for n in nodes if n.kind == "Name" and n.code == "x" and n.span[0] == 3)

    y_def = next(n for n in nodes if n.kind == "Name" and n.code == "y" and n.span[0] == 3)
    y_use = next(n for n in nodes if n.kind == "Name" and n.code == "y" and n.span[0] == 4)

    assert (x_def.id, x_use.id) in dfg_edges
    assert (y_def.id, y_use.id) in dfg_edges


def test_type_hint_attrs_for_function_and_args():
    src = """from typing import List, Sequence

def f(xs: List[int], ys: Sequence[str]) -> int:
    return len(xs)
"""
    nodes, _ = build_graph_from_source(src)

    func = next(n for n in nodes if n.kind == "Function" and n.symbol == "f")
    xs_arg = next(n for n in nodes if n.kind == "Arg" and n.symbol == "xs")
    ys_arg = next(n for n in nodes if n.kind == "Arg" and n.symbol == "ys")

    # Function return type
    assert func.type_hint == "int"
    assert func.attrs.get("DataType") == "int"
    assert "ContainerType" not in func.attrs

    # xs: List[int]
    assert xs_arg.type_hint == "List[int]"
    assert xs_arg.attrs.get("ContainerType") == "List"
    assert xs_arg.attrs.get("DataType") == "int"

    # ys: Sequence[str]
    assert ys_arg.type_hint == "Sequence[str]"
    assert ys_arg.attrs.get("ContainerType") == "Sequence"
    assert ys_arg.attrs.get("DataType") == "str"


def test_dfg_for_attributes_and_calls():
    src = """def f(xs):
    n = len(xs)
    m = xs.count(0)
    return n + m
"""
    nodes, edges = build_graph_from_source(src)
    dfg_edges = {(e.src, e.dst) for e in edges if e.kind == "DFG"}

    xs_def = next(n for n in nodes if n.kind == "Arg" and n.symbol == "xs")
    len_call = next(n for n in nodes if n.kind == "Call" and "len(xs)" in n.code)
    count_call = next(n for n in nodes if n.kind == "Call" and "xs.count(0)" in n.code)

    # xs should flow into both calls via DFG edges
    assert (xs_def.id, len_call.id) in dfg_edges
    assert (xs_def.id, count_call.id) in dfg_edges


def test_cfg_for_if_then_and_fallthrough():
    src = """def f(x):
    if x:
        a = 1
    b = 2
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    if_stmt = next(n for n in nodes if n.kind == "Stmt" and "if x:" in (n.code or ""))
    a_assign = next(n for n in nodes if n.kind == "Assign" and "a = 1" in (n.code or ""))
    b_assign = next(n for n in nodes if n.kind == "Assign" and "b = 2" in (n.code or ""))

    # If should connect to then-branch head
    assert (if_stmt.id, a_assign.id) in cfg_edges
    # Sequential CFG (from if to the statement after the if) should also exist
    assert (if_stmt.id, b_assign.id) in cfg_edges


def test_cfg_for_for_loop_body_and_back_edge():
    src = """def f(xs):
    for x in xs:
        a = x
    b = 1
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    loop_stmt = next(n for n in nodes if n.kind == "Stmt" and "for x in xs" in (n.code or ""))
    a_assign = next(n for n in nodes if n.kind == "Assign" and "a = x" in (n.code or ""))

    # Loop head -> body first statement
    assert (loop_stmt.id, a_assign.id) in cfg_edges
    # Back edge from body last statement to loop head
    assert (a_assign.id, loop_stmt.id) in cfg_edges


def test_cfg_for_while_loop_body_and_back_edge():
    src = """def f(x):
    while x:
        x = x - 1
    y = 0
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    while_stmt = next(n for n in nodes if n.kind == "Stmt" and "while x:" in (n.code or ""))
    x_assign = next(n for n in nodes if n.kind == "Assign" and "x = x - 1" in (n.code or ""))

    # While head -> body first statement
    assert (while_stmt.id, x_assign.id) in cfg_edges
    # Back edge from body last statement to while head
    assert (x_assign.id, while_stmt.id) in cfg_edges


def test_cfg_for_try_except_finally_entries():
    src = """def f(x):
    try:
        a = 1 / x
    except ZeroDivisionError:
        a = 0
    finally:
        b = 1
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    try_stmt = next(n for n in nodes if n.kind == "Try")
    a_try = next(n for n in nodes if n.kind == "Assign" and "1 / x" in (n.code or ""))
    except_node = next(n for n in nodes if n.kind == "Except")
    a_except = next(n for n in nodes if n.kind == "Assign" and "a = 0" in (n.code or ""))
    b_finally = next(n for n in nodes if n.kind == "Assign" and "b = 1" in (n.code or ""))

    # Try should connect to try-body, except handler, and finally entry
    assert (try_stmt.id, a_try.id) in cfg_edges
    assert (try_stmt.id, except_node.id) in cfg_edges
    assert (try_stmt.id, b_finally.id) in cfg_edges


def test_cfg_for_with_and_async_with_entries():
    src = """import asyncio

class Dummy:
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

async def g():
    async with asyncio.Lock():
        y = 2

def f():
    with Dummy() as d:
        x = 1
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    with_stmt = next(n for n in nodes if n.kind == "With" and "with Dummy()" in (n.code or ""))
    x_assign = next(n for n in nodes if n.kind == "Assign" and "x = 1" in (n.code or ""))

    async_with_stmt = next(n for n in nodes if n.kind == "With" and "async with asyncio.Lock()" in (n.code or ""))
    y_assign = next(n for n in nodes if n.kind == "Assign" and "y = 2" in (n.code or ""))

    # with / async with heads should both connect to their body first statement
    assert (with_stmt.id, x_assign.id) in cfg_edges
    assert (async_with_stmt.id, y_assign.id) in cfg_edges


def test_comprehension_kinds_and_dfg():
    src = """def f(xs):
    ys = [x * 2 for x in xs]
    zs = {x: x + 1 for x in xs if x > 0}
"""
    nodes, edges = build_graph_from_source(src)
    dfg_edges = {(e.src, e.dst) for e in edges if e.kind == "DFG"}

    xs_def = next(n for n in nodes if n.kind == "Arg" and n.symbol == "xs")
    list_comp = next(n for n in nodes if n.kind == "ListComp")
    dict_comp = next(n for n in nodes if n.kind == "DictComp")

    # xs should flow into both comprehensions
    assert (xs_def.id, list_comp.id) in dfg_edges
    assert (xs_def.id, dict_comp.id) in dfg_edges


def test_comprehension_internal_cfg_single_edge():
    src = """def f(xs):
    ys = [x * 2 for x in xs]
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    list_comp = next(n for n in nodes if n.kind == "ListComp")
    # The output expression "x * 2" is represented as a generic Stmt node.
    elt_node = next(n for n in nodes if n.kind == "Stmt" and "x * 2" in (n.code or ""))

    assert (list_comp.id, elt_node.id) in cfg_edges


def test_async_for_cfg_and_kinds():
    src = """async def g(xs):
    async for x in xs:
        y = x
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    async_for = next(n for n in nodes if n.kind == "AsyncFor")
    y_assign = next(n for n in nodes if n.kind == "Assign" and "y = x" in (n.code or ""))

    # AsyncFor behaves like a loop: head -> body first, body last -> head
    assert (async_for.id, y_assign.id) in cfg_edges
    assert (y_assign.id, async_for.id) in cfg_edges


def test_await_kind_and_sequential_cfg():
    src = """async def g(x):
    await do(x)
    y = 1
"""
    nodes, edges = build_graph_from_source(src)
    cfg_edges = {(e.src, e.dst) for e in edges if e.kind == "CFG"}

    await_node = next(n for n in nodes if n.kind == "Await")
    await_stmt = next(n for n in nodes if n.kind == "Stmt" and "await do(x)" in (n.code or ""))
    y_assign = next(n for n in nodes if n.kind == "Assign" and "y = 1" in (n.code or ""))

    # Await expression should be present as its own node
    assert await_node is not None
    # Sequential CFG from await statement to the following statement
    assert (await_stmt.id, y_assign.id) in cfg_edges