"""
Integration tests for DTA tool with CPG merge functionality.
"""
import json
import tempfile
from pathlib import Path

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder
from ir.schema import CPGGraph
from ir.io import iter_taint_records
from ir.taint_merge import add_ddfg_from_records
from dta.tracker import TaintTracker, taint_source, taint_sink, get_tracker


def build_graph_from_source(src: str, file_path: str = "test.py"):
    tree = parse_source(src)
    builder = ASTCPGBuilder(file_path, src)
    builder.visit(tree)
    nodes, edges = builder.build()
    return CPGGraph(file=file_path, nodes=nodes, edges=edges)


def test_dta_output_merges_with_cpg():
    """Test that DTA tool output can be merged into CPG as DDFG edges."""
    # Create a simple Python script with source and sink
    src = """def get_user_input():
    return input()

def execute_sql(query):
    pass

user_data = get_user_input()
execute_sql(user_data)
"""
    graph = build_graph_from_source(src, "test.py")

    # Simulate DTA execution and record generation
    # (In practice, this would come from running the actual script with DTA)
    user_input_node = next(
        n for n in graph.nodes if n.kind == "Name" and n.code == "user_data"
    )
    execute_sql_node = next(
        n for n in graph.nodes if n.kind == "Call" and "execute_sql" in (n.code or "")
    )

    # Create a taint record matching DTA output format
    taint_record = {
        "source": {
            "file": "test.py",
            "line": user_input_node.span[0],
            "col": user_input_node.span[1],
        },
        "sink": {
            "file": "test.py",
            "line": execute_sql_node.span[0],
            "col": execute_sql_node.span[1],
        },
        "path": [],
        "meta": {"taint_kind": "user_input", "sink_type": "sql_exec"},
    }

    # Merge into CPG
    add_ddfg_from_records(graph, [taint_record])

    # Verify DDFG edge was added
    ddfg_edges = [e for e in graph.edges if e.kind == "DDFG"]
    assert len(ddfg_edges) >= 1
    assert any(
        e.src == user_input_node.id and e.dst == execute_sql_node.id
        for e in ddfg_edges
    )


def test_dta_jsonl_integration():
    """Test reading DTA JSONL output and merging with CPG."""
    src = """def f(x):
    y = x + 1
    return y
"""
    graph = build_graph_from_source(src, "test.py")

    # Create a temporary JSONL file with taint records
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        record = {
            "source": {"file": "test.py", "line": 1, "col": 4},
            "sink": {"file": "test.py", "line": 3, "col": 4},
            "path": [],
            "meta": {"taint_kind": "user_input", "sink_type": "return"},
        }
        f.write(json.dumps(record) + "\n")
        jsonl_path = f.name

    try:
        # Read records and merge
        records = list(iter_taint_records(jsonl_path))
        add_ddfg_from_records(graph, records)

        # Verify DDFG edges were added
        ddfg_edges = [e for e in graph.edges if e.kind == "DDFG"]
        assert len(ddfg_edges) >= 1
    finally:
        Path(jsonl_path).unlink()

