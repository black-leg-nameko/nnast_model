#!/usr/bin/env python3
"""
Python実装との結果比較テスト
既存のtest_cpg.pyのテストケースをRust実装でも実行して比較
"""

import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cpg_rust
from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder

def build_graph_python(src: str):
    """Python実装でCPGを生成"""
    tree = parse_source(src)
    builder = ASTCPGBuilder("test.py", src)
    builder.visit(tree)
    nodes, edges = builder.build()
    return nodes, edges

def build_graph_rust(src: str):
    """Rust実装でCPGを生成"""
    graph = cpg_rust.build_cpg("test.py", src)
    # Python実装と同じ形式に変換
    from ir.schema import CPGNode, CPGEdge
    nodes = []
    edges = []
    
    for n in graph['nodes']:
        nodes.append(CPGNode(
            id=n['id'],
            kind=n['kind'],
            file=n['file'],
            span=n['span'],
            code=n.get('code'),
            symbol=n.get('symbol'),
            type_hint=n.get('type_hint'),
            flags=n.get('flags', []),
            attrs=n.get('attrs', {})
        ))
    
    for e in graph['edges']:
        edges.append(CPGEdge(
            src=e['src'],
            dst=e['dst'],
            kind=e['kind'],
            attrs=e.get('attrs')
        ))
    
    return nodes, edges

def compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges, test_name: str):
    """2つのグラフを比較"""
    print(f"\n=== {test_name} ===")
    
    # ノード数の比較
    print(f"Python nodes: {len(python_nodes)}, Rust nodes: {len(rust_nodes)}")
    print(f"Python edges: {len(python_edges)}, Rust edges: {len(rust_edges)}")
    
    # CFG/DFGエッジの比較
    python_cfg = [e for e in python_edges if e.kind == "CFG"]
    rust_cfg = [e for e in rust_edges if e.kind == "CFG"]
    python_dfg = [e for e in python_edges if e.kind == "DFG"]
    rust_dfg = [e for e in rust_edges if e.kind == "DFG"]
    
    print(f"Python CFG: {len(python_cfg)}, Rust CFG: {len(rust_cfg)}")
    print(f"Python DFG: {len(python_dfg)}, Rust DFG: {len(rust_dfg)}")
    
    # CFGエッジのセット比較
    python_cfg_set = {(e.src, e.dst) for e in python_cfg}
    rust_cfg_set = {(e.src, e.dst) for e in rust_cfg}
    
    missing_in_rust = python_cfg_set - rust_cfg_set
    extra_in_rust = rust_cfg_set - python_cfg_set
    
    if missing_in_rust:
        print(f"⚠️  Rust実装で欠けているCFGエッジ: {missing_in_rust}")
    if extra_in_rust:
        print(f"⚠️  Rust実装で余分なCFGエッジ: {extra_in_rust}")
    
    # DFGエッジのセット比較
    python_dfg_set = {(e.src, e.dst) for e in python_dfg}
    rust_dfg_set = {(e.src, e.dst) for e in rust_dfg}
    
    missing_dfg_in_rust = python_dfg_set - rust_dfg_set
    extra_dfg_in_rust = rust_dfg_set - python_dfg_set
    
    if missing_dfg_in_rust:
        print(f"⚠️  Rust実装で欠けているDFGエッジ: {missing_dfg_in_rust}")
    if extra_dfg_in_rust:
        print(f"⚠️  Rust実装で余分なDFGエッジ: {extra_dfg_in_rust}")
    
    # 完全一致かチェック
    cfg_match = python_cfg_set == rust_cfg_set
    dfg_match = python_dfg_set == rust_dfg_set
    
    return cfg_match and dfg_match

def test_cfg_edges_within_block():
    """test_cfg_edges_within_blockのテスト"""
    src = """def f():
    x = 1
    y = 2
    z = x + y
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges, 
                         "CFG edges within block")

def test_dfg_edges_for_simple_assignments():
    """test_dfg_edges_for_simple_assignmentsのテスト"""
    src = """def f():
    x = 1
    y = x + 2
    return y
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges,
                         "DFG edges for simple assignments")

def test_cfg_for_for_loop_body_and_back_edge():
    """test_cfg_for_for_loop_body_and_back_edgeのテスト"""
    src = """def f(xs):
    for x in xs:
        a = x
    b = 1
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges,
                         "For loop body and back edge")

def test_cfg_for_while_loop_body_and_back_edge():
    """test_cfg_for_while_loop_body_and_back_edgeのテスト"""
    src = """def f(x):
    while x:
        x = x - 1
    y = 0
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges,
                         "While loop body and back edge")

def test_cfg_for_try_except_finally_entries():
    """test_cfg_for_try_except_finally_entriesのテスト"""
    src = """def f(x):
    try:
        a = 1 / x
    except ZeroDivisionError:
        a = 0
    finally:
        b = 1
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges,
                         "Try/Except/Finally entries")

def test_comprehension_kinds_and_dfg():
    """test_comprehension_kinds_and_dfgのテスト"""
    src = """def f(xs):
    ys = [x * 2 for x in xs]
    zs = {x: x + 1 for x in xs if x > 0}
"""
    python_nodes, python_edges = build_graph_python(src)
    rust_nodes, rust_edges = build_graph_rust(src)
    
    return compare_graphs(python_nodes, python_edges, rust_nodes, rust_edges,
                         "Comprehension kinds and DFG")

if __name__ == "__main__":
    print("=" * 60)
    print("Python実装との結果比較テスト")
    print("=" * 60)
    
    results = []
    results.append(("CFG edges within block", test_cfg_edges_within_block()))
    results.append(("DFG edges for simple assignments", test_dfg_edges_for_simple_assignments()))
    results.append(("For loop body and back edge", test_cfg_for_for_loop_body_and_back_edge()))
    results.append(("While loop body and back edge", test_cfg_for_while_loop_body_and_back_edge()))
    results.append(("Try/Except/Finally entries", test_cfg_for_try_except_finally_entries()))
    results.append(("Comprehension kinds and DFG", test_comprehension_kinds_and_dfg()))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'✅ すべてのテストが一致しました！' if all_passed else '❌ 一部のテストで不一致がありました'}")
    exit(0 if all_passed else 1)
