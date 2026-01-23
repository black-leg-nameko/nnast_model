#!/usr/bin/env python3
"""
パフォーマンス比較テスト
Python実装とRust実装の性能を比較
"""

import sys
import pathlib
import time
import os

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

def benchmark(code: str, name: str, iterations: int = 10):
    """ベンチマークテスト"""
    print(f"\n=== {name} ===")
    print(f"Code length: {len(code)} characters")
    print(f"Iterations: {iterations}")
    
    # Python実装のベンチマーク
    python_times = []
    for _ in range(iterations):
        start = time.time()
        python_nodes, python_edges = build_graph_python(code)
        python_times.append(time.time() - start)
    
    python_avg = sum(python_times) / len(python_times)
    python_min = min(python_times)
    python_max = max(python_times)
    
    print(f"\nPython実装:")
    print(f"  Average: {python_avg*1000:.2f}ms")
    print(f"  Min: {python_min*1000:.2f}ms")
    print(f"  Max: {python_max*1000:.2f}ms")
    print(f"  Nodes: {len(python_nodes)}, Edges: {len(python_edges)}")
    
    # Rust実装のベンチマーク
    rust_times = []
    for _ in range(iterations):
        start = time.time()
        rust_nodes, rust_edges = build_graph_rust(code)
        rust_times.append(time.time() - start)
    
    rust_avg = sum(rust_times) / len(rust_times)
    rust_min = min(rust_times)
    rust_max = max(rust_times)
    
    print(f"\nRust実装:")
    print(f"  Average: {rust_avg*1000:.2f}ms")
    print(f"  Min: {rust_min*1000:.2f}ms")
    print(f"  Max: {rust_max*1000:.2f}ms")
    print(f"  Nodes: {len(rust_nodes)}, Edges: {len(rust_edges)}")
    
    # 速度比較
    speedup = python_avg / rust_avg if rust_avg > 0 else 0
    print(f"\n速度向上: {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    print("=" * 60)
    print("パフォーマンス比較テスト")
    print("=" * 60)
    
    # 小さなコード
    small_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
    
    # 中規模のコード
    medium_code = """
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(-item)
    return result

def filter_data(data, threshold):
    filtered = []
    for x in data:
        if x > threshold:
            filtered.append(x)
    return filtered
"""
    
    # 大きなコード（複数の関数と制御構造）
    large_code = """
def complex_function(data):
    results = []
    try:
        for item in data:
            if item is None:
                continue
            processed = process_item(item)
            if processed:
                results.append(processed)
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        cleanup()
    return results

def process_item(item):
    if isinstance(item, dict):
        return {k: v * 2 for k, v in item.items()}
    elif isinstance(item, list):
        return [x * 2 for x in item if x > 0]
    else:
        return item * 2

def cleanup():
    pass
""" * 5  # 5回繰り返して大きなコードにする
    
    speedups = []
    speedups.append(("Small code", benchmark(small_code, "Small code", 100)))
    speedups.append(("Medium code", benchmark(medium_code, "Medium code", 50)))
    speedups.append(("Large code", benchmark(large_code, "Large code", 20)))
    
    print("\n" + "=" * 60)
    print("パフォーマンスサマリー")
    print("=" * 60)
    for name, speedup in speedups:
        print(f"  {name}: {speedup:.2f}x faster")
    
    avg_speedup = sum(s for _, s in speedups) / len(speedups)
    print(f"\n平均速度向上: {avg_speedup:.2f}x")
    print(f"\n{'✅ Rust実装が高速です！' if avg_speedup > 1.0 else '⚠️  パフォーマンス改善の余地があります'}")
