#!/usr/bin/env python3
"""
包括的なCFG/DFGエッジ生成のテスト
ループ、Try文、With文、内包表記などをテスト
"""

import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cpg_rust

def test_for_loop():
    """ForループのCFGエッジテスト"""
    code = """
def process_items(items):
    result = []
    for item in items:
        result.append(item)
    return result
"""
    
    print("=== For loop test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    # 期待されるCFGエッジ
    print("\n期待されるCFGエッジ:")
    print("  Function -> Assign (result = [])")
    print("  Assign -> For")
    print("  For -> Assign (result.append)")
    print("  Assign (result.append) -> For (バックエッジ)")
    print("  For -> Return")
    
    return len(cfg_edges) > 0

def test_while_loop():
    """WhileループのCFGエッジテスト"""
    code = """
def countdown(n):
    while n > 0:
        print(n)
        n -= 1
    return n
"""
    
    print("\n=== While loop test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    # 期待されるCFGエッジ
    print("\n期待されるCFGエッジ:")
    print("  Function -> While")
    print("  While -> Call (print)")
    print("  Call -> AugAssign")
    print("  AugAssign -> While (バックエッジ)")
    print("  While -> Return")
    
    return len(cfg_edges) > 0

def test_try_except():
    """Try文のCFGエッジテスト"""
    code = """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return None
    finally:
        print("Done")
"""
    
    print("\n=== Try/Except test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    # 期待されるCFGエッジ
    print("\n期待されるCFGエッジ:")
    print("  Function -> Try")
    print("  Try -> Assign (result = a / b)")
    print("  Try -> Except")
    print("  Try -> Call (print in finally)")
    
    return len(cfg_edges) > 0

def test_with_statement():
    """With文のCFGエッジテスト"""
    code = """
def read_file(filename):
    with open(filename) as f:
        content = f.read()
        return content
"""
    
    print("\n=== With statement test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    # 期待されるCFGエッジ
    print("\n期待されるCFGエッジ:")
    print("  Function -> With")
    print("  With -> Assign (content = f.read())")
    
    return len(cfg_edges) > 0

def test_comprehension():
    """内包表記のCFG/DFGエッジテスト"""
    code = """
def process_data(items):
    x = 10
    result = [item * x for item in items]
    return result
"""
    
    print("\n=== Comprehension test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    dfg_edges = [e for e in graph['edges'] if e['kind'] == 'DFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    print(f"\nDFG edges: {len(dfg_edges)}")
    for edge in dfg_edges[:5]:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):10s} {src_node.get('symbol', 'N/A'):10s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):10s})")
    
    # 期待されるエッジ
    print("\n期待されるCFGエッジ:")
    print("  Function -> Assign (x = 10)")
    print("  Assign -> Assign (result = [...])")
    print("  ListComp -> BinOp (item * x)")
    
    print("\n期待されるDFGエッジ:")
    print("  Assign (x = 10) -> ListComp (xの使用)")
    
    return len(cfg_edges) > 0 and len(dfg_edges) > 0

def test_nested_control():
    """ネストした制御構造のテスト"""
    code = """
def process(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
        else:
            result.append(-item)
    return result
"""
    
    print("\n=== Nested control structures test ===\n")
    graph = cpg_rust.build_cpg("test.py", code)
    
    cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
    node_map = {n['id']: n for n in graph['nodes']}
    
    print(f"CFG edges: {len(cfg_edges)}")
    for edge in cfg_edges:
        src_node = node_map.get(edge['src'], {})
        dst_node = node_map.get(edge['dst'], {})
        print(f"  {edge['src']:2d} ({src_node.get('kind', '?'):15s}) -> {edge['dst']:2d} ({dst_node.get('kind', '?'):15s})")
    
    return len(cfg_edges) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("包括的なCFG/DFGエッジ生成テスト")
    print("=" * 60)
    
    results = []
    results.append(("For loop", test_for_loop()))
    results.append(("While loop", test_while_loop()))
    results.append(("Try/Except", test_try_except()))
    results.append(("With statement", test_with_statement()))
    results.append(("Comprehension", test_comprehension()))
    results.append(("Nested control", test_nested_control()))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'✅ すべてのテストが成功しました！' if all_passed else '❌ 一部のテストが失敗しました'}")
    exit(0 if all_passed else 1)
