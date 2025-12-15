import json

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder
from ir.schema import CPGGraph
from ir.taint_merge import add_ddfg_from_record


def build_graph_from_source(src: str):
    tree = parse_source(src)
    builder = ASTCPGBuilder("test.py", src)
    builder.visit(tree)
    nodes, edges = builder.build()
    return CPGGraph(file="test.py", nodes=nodes, edges=edges)


def test_add_ddfg_from_taint_record_expr_level():
    src = """def f(user):
    q = "SELECT " + user
    exec_sql(q)
"""
    graph = build_graph_from_source(src)

    # Find concrete nodes to derive realistic positions
    user_name = next(n for n in graph.nodes if n.kind == "Name" and n.code == "user")
    exec_call = next(n for n in graph.nodes if n.kind == "Call" and "exec_sql" in (n.code or ""))

    src_pos = {
        "file": graph.file,
        "line": user_name.span[0],
        "col": user_name.span[1],
    }
    sink_pos = {
        "file": graph.file,
        "line": exec_call.span[0],
        "col": exec_call.span[1],
    }

    record = {
        "source": src_pos,
        "sink": sink_pos,
        "path": [],
        "meta": {"taint_kind": "user_input", "sink_type": "sql_exec"},
    }

    add_ddfg_from_record(graph, record)

    ddfg_edges = [e for e in graph.edges if e.kind == "DDFG"]
    assert len(ddfg_edges) == 1
    edge = ddfg_edges[0]
    assert edge.src == user_name.id
    assert edge.dst == exec_call.id
    assert edge.attrs and edge.attrs.get("taint_kind") == "user_input"
    assert edge.attrs.get("sink_type") == "sql_exec"