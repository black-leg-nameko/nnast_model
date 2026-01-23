#!/usr/bin/env python3
"""
Rust実装の統合テスト
既存のPython実装と同じインターフェースで使用できることを確認
"""

import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cpg_rust
from ir.schema import CPGGraph

def test_basic_interface():
    """基本的なインターフェースのテスト"""
    print("=== Basic Interface Test ===")
    
    code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
    
    # Rust実装でCPGを生成
    graph_dict = cpg_rust.build_cpg("test.py", code)
    
    # 基本的な構造を確認
    assert "file" in graph_dict
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    
    assert graph_dict["file"] == "test.py"
    assert isinstance(graph_dict["nodes"], list)
    assert isinstance(graph_dict["edges"], list)
    
    print(f"✅ 基本インターフェース: OK")
    print(f"   Nodes: {len(graph_dict['nodes'])}")
    print(f"   Edges: {len(graph_dict['edges'])}")
    
    return True

def test_node_structure():
    """ノード構造のテスト"""
    print("\n=== Node Structure Test ===")
    
    code = """
def hello(name: str) -> str:
    x = name
    return x
"""
    
    graph_dict = cpg_rust.build_cpg("test.py", code)
    
    # ノードの構造を確認
    for node in graph_dict["nodes"][:5]:
        assert "id" in node
        assert "kind" in node
        assert "file" in node
        assert "span" in node
        assert isinstance(node["id"], int)
        assert isinstance(node["kind"], str)
        assert isinstance(node["span"], tuple)
        assert len(node["span"]) == 4
    
    print("✅ ノード構造: OK")
    return True

def test_edge_structure():
    """エッジ構造のテスト"""
    print("\n=== Edge Structure Test ===")
    
    code = """
def hello(name: str) -> str:
    x = name
    return x
"""
    
    graph_dict = cpg_rust.build_cpg("test.py", code)
    
    # エッジの構造を確認
    for edge in graph_dict["edges"][:5]:
        assert "src" in edge
        assert "dst" in edge
        assert "kind" in edge
        assert isinstance(edge["src"], int)
        assert isinstance(edge["dst"], int)
        assert isinstance(edge["kind"], str)
        assert edge["kind"] in ["AST", "CFG", "DFG"]
    
    print("✅ エッジ構造: OK")
    return True

def test_cpg_graph_conversion():
    """CPGGraphへの変換テスト"""
    print("\n=== CPGGraph Conversion Test ===")
    
    code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
    
    graph_dict = cpg_rust.build_cpg("test.py", code)
    
    # CPGGraphに変換（既存コードとの互換性）
    # これは既存のコードが期待する形式
    nodes = []
    edges = []
    
    from ir.schema import CPGNode, CPGEdge
    
    for n in graph_dict["nodes"]:
        nodes.append(CPGNode(
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
        edges.append(CPGEdge(
            src=e["src"],
            dst=e["dst"],
            kind=e["kind"],
            attrs=e.get("attrs")
        ))
    
    graph = CPGGraph(file=graph_dict["file"], nodes=nodes, edges=edges)
    
    assert graph.file == "test.py"
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    print("✅ CPGGraph変換: OK")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Rust実装の統合テスト")
    print("=" * 60)
    
    results = []
    results.append(("Basic Interface", test_basic_interface()))
    results.append(("Node Structure", test_node_structure()))
    results.append(("Edge Structure", test_edge_structure()))
    results.append(("CPGGraph Conversion", test_cpg_graph_conversion()))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'✅ すべての統合テストが成功しました！' if all_passed else '❌ 一部のテストが失敗しました'}")
    exit(0 if all_passed else 1)
