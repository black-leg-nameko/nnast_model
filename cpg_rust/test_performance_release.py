#!/usr/bin/env python3
"""
パフォーマンス比較テスト（リリースビルド用）

リリースビルドでのパフォーマンスを測定します。
使用方法:
    maturin build --release
    pip install target/wheels/cpg_rust-*.whl --force-reinstall
    python3 test_performance_release.py
"""

import sys
import pathlib
import os
import time
import statistics
import importlib
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def measure_time(func, *args, iterations: int = 100):
    """関数の実行時間を測定"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        "average": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "result": result,
    }

def test_code(code: str, name: str, iterations: int = 100):
    """コードのパフォーマンスをテスト"""
    print(f"\n=== {name} ===")
    print(f"Code length: {len(code)} characters")
    print(f"Iterations: {iterations}")
    
    # Python実装
    os.environ["USE_RUST_CPG"] = "false"
    # モジュールを再インポートして環境変数の変更を反映
    import cpg.build_ast
    importlib.reload(cpg.build_ast)
    from cpg.build_ast import ASTCPGBuilder as PythonBuilder
    from cpg.parse import parse_source
    
    def build_python():
        tree = parse_source(code)
        builder = PythonBuilder("test.py", code)
        builder.visit(tree)
        nodes, edges = builder.build()
        return nodes, edges
    
    python_result = measure_time(build_python, iterations=iterations)
    
    python_nodes, python_edges = python_result["result"]
    python_nodes = len(python_nodes)
    python_edges = len(python_edges)
    
    print(f"\nPython実装:")
    print(f"  Average: {python_result['average']:.2f}ms")
    print(f"  Min: {python_result['min']:.2f}ms")
    print(f"  Max: {python_result['max']:.2f}ms")
    print(f"  Median: {python_result['median']:.2f}ms")
    print(f"  StdDev: {python_result['stdev']:.2f}ms")
    print(f"  Nodes: {python_nodes}, Edges: {python_edges}")
    
    # Rust実装
    os.environ["USE_RUST_CPG"] = "true"
    # モジュールを再インポートして環境変数の変更を反映
    importlib.reload(cpg.build_ast)
    from cpg.build_ast import ASTCPGBuilder as RustBuilder
    
    def build_rust():
        builder = RustBuilder("test.py", code)
        nodes, edges = builder.build()
        return nodes, edges
    
    rust_result = measure_time(build_rust, iterations=iterations)
    
    rust_nodes, rust_edges = rust_result["result"]
    rust_nodes = len(rust_nodes)
    rust_edges = len(rust_edges)
    
    print(f"\nRust実装:")
    print(f"  Average: {rust_result['average']:.2f}ms")
    print(f"  Min: {rust_result['min']:.2f}ms")
    print(f"  Max: {rust_result['max']:.2f}ms")
    print(f"  Median: {rust_result['median']:.2f}ms")
    print(f"  StdDev: {rust_result['stdev']:.2f}ms")
    print(f"  Nodes: {rust_nodes}, Edges: {rust_edges}")
    
    # 速度比較
    speedup = python_result['average'] / rust_result['average']
    print(f"\n速度向上: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✅ Rust実装が{speedup:.2f}倍高速です！")
    else:
        print(f"⚠️  Rust実装は{1/speedup:.2f}倍遅いです")
    
    return speedup

def main():
    print("=" * 60)
    print("パフォーマンス比較テスト（リリースビルド）")
    print("=" * 60)
    
    # Small code
    small_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
    small_speedup = test_code(small_code.strip(), "Small code", iterations=100)
    
    # Medium code
    medium_code = """
def process_data(data: List[int]) -> Dict[str, int]:
    result = {}
    for item in data:
        if item > 0:
            result[str(item)] = item * 2
        else:
            result[str(item)] = 0
    return result

def main():
    data = [1, 2, -3, 4, 5]
    output = process_data(data)
    print(output)
"""
    medium_speedup = test_code(medium_code.strip(), "Medium code", iterations=50)
    
    # Large code
    large_code = """
class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
    
    def process(self, data: List[Dict]) -> List[Dict]:
        results = []
        for item in data:
            processed = self._transform(item)
            if self._validate(processed):
                results.append(processed)
        return results
    
    def _transform(self, item: Dict) -> Dict:
        result = {}
        for key, value in item.items():
            if isinstance(value, str):
                result[key] = value.upper()
            elif isinstance(value, int):
                result[key] = value * 2
            else:
                result[key] = value
        return result
    
    def _validate(self, item: Dict) -> bool:
        return len(item) > 0

def complex_function(x: int, y: int) -> Tuple[int, int]:
    if x > y:
        result1 = x * 2
        result2 = y + 10
    else:
        result1 = x + 10
        result2 = y * 2
    
    for i in range(10):
        result1 += i
        result2 -= i
    
    return result1, result2

def main():
    processor = DataProcessor({"mode": "fast"})
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    results = processor.process(data)
    print(results)
    
    a, b = complex_function(5, 3)
    print(f"Results: {a}, {b}")
"""
    large_speedup = test_code(large_code.strip(), "Large code", iterations=20)
    
    # Summary
    print("\n" + "=" * 60)
    print("パフォーマンスサマリー")
    print("=" * 60)
    print(f"  Small code: {small_speedup:.2f}x")
    print(f"  Medium code: {medium_speedup:.2f}x")
    print(f"  Large code: {large_speedup:.2f}x")
    
    avg_speedup = (small_speedup + medium_speedup + large_speedup) / 3
    print(f"\n平均速度向上: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.0:
        print(f"\n✅ Rust実装が平均{avg_speedup:.2f}倍高速です！")
    else:
        print(f"\n⚠️  パフォーマンス改善の余地があります")
        print(f"   現在: {avg_speedup:.2f}x (Python実装の{1/avg_speedup:.2f}倍)")

if __name__ == "__main__":
    main()
