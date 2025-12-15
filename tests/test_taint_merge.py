import json

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder
from ir.schema import CPGGraph
from ir.taint_merge import add_ddfg_from_record, normalize_taint_record


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


def test_normalize_taint_record_canonical():
    """Test that canonical format passes through unchanged."""
    canonical = {
        "source": {"file": "foo.py", "line": 10, "col": 5},
        "sink": {"file": "foo.py", "line": 20, "col": 8},
        "path": [{"file": "foo.py", "line": 15, "col": 3}],
        "meta": {"taint_kind": "user_input"},
    }
    normalized = normalize_taint_record(canonical)
    assert normalized == canonical


def test_normalize_taint_record_tained_format():
    """Test conversion from tainted-like format."""
    tainted_format = {
        "taint_source": {"filename": "foo.py", "lineno": 10, "column": 5},
        "taint_sink": {"filename": "bar.py", "lineno": 20, "column": 8},
        "trace": [
            {"path": "foo.py", "line": 15, "column": 3},
            {"path": "bar.py", "line": 18, "column": 1},
        ],
        "taint_type": "user_input",
        "sink_type": "sql_exec",
    }
    normalized = normalize_taint_record(tainted_format)
    assert normalized is not None
    assert normalized["source"] == {"file": "foo.py", "line": 10, "col": 5}
    assert normalized["sink"] == {"file": "bar.py", "line": 20, "col": 8}
    assert len(normalized["path"]) == 2
    assert normalized["path"][0] == {"file": "foo.py", "line": 15, "col": 3}
    assert normalized["meta"]["taint_kind"] == "user_input"
    assert normalized["meta"]["sink_type"] == "sql_exec"


def test_normalize_taint_record_generic_format():
    """Test conversion from generic 'from'/'to' format."""
    generic = {
        "from": {"file": "foo.py", "line": 10, "col": 5},
        "to": {"file": "foo.py", "line": 20, "col": 8},
        "via": [{"file": "foo.py", "line": 15, "col": 3}],
        "extra_field": "value",
    }
    normalized = normalize_taint_record(generic)
    assert normalized is not None
    assert normalized["source"] == {"file": "foo.py", "line": 10, "col": 5}
    assert normalized["sink"] == {"file": "foo.py", "line": 20, "col": 8}
    assert len(normalized["path"]) == 1
    assert normalized["meta"]["extra_field"] == "value"


def test_normalize_taint_record_invalid():
    """Test that invalid records return None."""
    invalid = {"some": "random", "dict": "without", "source": "or", "sink": "fields"}
    assert normalize_taint_record(invalid) is None