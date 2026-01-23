#!/usr/bin/env python3
"""
実世界のコードでのテスト
実際のプロジェクトファイルでCPG生成をテスト
"""

import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cpg_rust

def test_real_file(file_path: str):
    """実際のファイルでCPG生成をテスト"""
    print(f"\n=== Testing {file_path} ===")
    
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        graph = cpg_rust.build_cpg(file_path, source)
        
        # エッジを種類別に分類
        ast_edges = [e for e in graph['edges'] if e['kind'] == 'AST']
        cfg_edges = [e for e in graph['edges'] if e['kind'] == 'CFG']
        dfg_edges = [e for e in graph['edges'] if e['kind'] == 'DFG']
        
        print(f"✅ Success!")
        print(f"   Nodes: {len(graph['nodes'])}")
        print(f"   AST edges: {len(ast_edges)}")
        print(f"   CFG edges: {len(cfg_edges)}")
        print(f"   DFG edges: {len(dfg_edges)}")
        
        # ノードの種類を集計
        node_kinds = {}
        for node in graph['nodes']:
            kind = node['kind']
            node_kinds[kind] = node_kinds.get(kind, 0) + 1
        
        print(f"\n   Node kinds (top 5):")
        for kind, count in sorted(node_kinds.items(), key=lambda x: -x[1])[:5]:
            print(f"     {kind}: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("実世界のコードでのテスト")
    print("=" * 60)
    
    # プロジェクト内の実際のファイルをテスト
    test_files = [
        "cpg/build_ast.py",
        "cli.py",
        "tests/test_cpg.py",
    ]
    
    results = []
    for file_path in test_files:
        full_path = project_root / file_path
        if full_path.exists():
            results.append((file_path, test_real_file(str(full_path))))
        else:
            print(f"\n⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'✅ すべてのテストが成功しました！' if all_passed else '❌ 一部のテストが失敗しました'}")
    exit(0 if all_passed else 1)
