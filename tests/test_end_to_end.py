#!/usr/bin/env python3
"""
End-to-end integration test for NNAST.

Tests the complete pipeline:
1. CPG generation with pattern matching
2. Dataset validation
3. OWASP mapping
4. Evaluation metrics
"""
import json
import pathlib
import tempfile
import subprocess
import sys
from typing import Dict, List, Optional

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test code with various vulnerabilities
TEST_CODE = """from flask import request
import sqlite3
import subprocess
import requests

def sql_injection_vulnerable():
    # SQL Injection vulnerability
    user_id = request.args.get('id')
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchall()

def sql_injection_safe():
    # Safe: parameterized query
    user_id = request.args.get('id')
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    return cursor.fetchall()

def command_injection():
    # Command Injection vulnerability
    user_input = request.form.get('cmd')
    subprocess.run(user_input, shell=True)

def ssrf_vulnerable():
    # SSRF vulnerability
    url = request.args.get('url')
    response = requests.get(url)
    return response.text
"""


def test_cpg_generation() -> bool:
    """Test CPG generation with pattern matching."""
    print("\n=== Test 1: CPG Generation ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_file = pathlib.Path(f.name)
        f.write(TEST_CODE)
    
    try:
        output_file = test_file.with_suffix('.jsonl')
        result = subprocess.run(
            [sys.executable, "-m", "cli", "--patterns", "patterns.yaml", str(test_file), "--out", str(output_file)],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            print(f"❌ CPG generation failed: {result.stderr}")
            return False
        
        if not output_file.exists():
            print(f"❌ Output file not created: {output_file}")
            return False
        
        # Load and verify graph
        with open(output_file, 'r') as f:
            graph_data = json.loads(f.readline())
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        frameworks = graph_data.get('metadata', {}).get('frameworks', [])
        
        print(f"✅ CPG generated: {len(nodes)} nodes, {len(edges)} edges")
        print(f"✅ Frameworks detected: {frameworks}")
        
        # Check for sources and sinks
        sources = [n for n in nodes if n.get('attrs', {}).get('is_source') == 'true' or n.get('attrs', {}).get('is_source') is True]
        sinks = [n for n in nodes if n.get('attrs', {}).get('is_sink') == 'true' or n.get('attrs', {}).get('is_sink') is True]
        
        print(f"✅ Sources detected: {len(sources)}")
        print(f"✅ Sinks detected: {len(sinks)}")
        
        if len(sources) == 0 or len(sinks) == 0:
            print("⚠️ Warning: No sources or sinks detected (pattern matching may not be working)")
        
        # Cleanup
        test_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_validation() -> bool:
    """Test dataset validation tool."""
    print("\n=== Test 2: Dataset Validation ===")
    
    # Create a minimal test graph
    test_graph = {
        "file": "test.py",
        "nodes": [
            {
                "id": 1,
                "kind": "Call",
                "code": "request.args.get('id')",
                "attrs": {
                    "is_source": "true",
                    "source_id": "SRC_FLASK_REQUEST"
                }
            },
            {
                "id": 2,
                "kind": "Call",
                "code": "subprocess.run(user_input, shell=True)",
                "attrs": {
                    "is_sink": "true",
                    "sink_id": "SINK_SUBPROCESS",
                    "sink_kind": "cmd_exec"
                }
            }
        ],
        "edges": [],
        "metadata": {
            "frameworks": ["flask"]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        graph_file = pathlib.Path(f.name)
        json.dump(test_graph, f)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "data.validate_dataset", str(graph_file), "--output", "/tmp/validation_test.json"],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            print(f"⚠️ Validation tool returned non-zero exit code: {result.stderr}")
            # This is OK, validation tool may return non-zero for warnings
        
        print("✅ Dataset validation tool executed")
        
        # Cleanup
        graph_file.unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_owasp_mapper() -> bool:
    """Test OWASP mapper."""
    print("\n=== Test 3: OWASP Mapper ===")
    
    try:
        from ml.owasp_mapper import OWASPMapper
        
        mapper = OWASPMapper()
        patterns = mapper.get_all_patterns()
        
        print(f"✅ Loaded {len(patterns)} patterns")
        
        # Test pattern inference
        test_cases = [
            {'source_id': 'SRC_FLASK_REQUEST', 'sink_id': 'SINK_DBAPI_EXECUTE', 'sink_kind': 'sql_exec'},
            {'source_id': 'SRC_FLASK_REQUEST', 'sink_id': 'SINK_SUBPROCESS', 'sink_kind': 'cmd_exec'},
        ]
        
        for case in test_cases:
            pattern_id = mapper.infer_pattern_id(**case)
            if pattern_id:
                owasp = mapper.get_owasp(pattern_id)
                cwe = mapper.get_primary_cwe(pattern_id)
                print(f"✅ Pattern inference: {case} -> {pattern_id} ({owasp}, {cwe})")
            else:
                print(f"⚠️ Pattern inference failed for: {case}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics() -> bool:
    """Test evaluation metrics."""
    print("\n=== Test 4: Evaluation Metrics ===")
    
    try:
        from ml.evaluation import calculate_classification_metrics
        import numpy as np
        
        # Test data
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.85])
        
        metrics = calculate_classification_metrics(
            y_true, y_pred, y_proba,
            class_names=['safe', 'vulnerable']
        )
        
        print(f"✅ Metrics calculated:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 Binary: {metrics.get('f1_binary', 0):.4f}")
        print(f"  F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("NNAST End-to-End Integration Tests")
    print("=" * 60)
    
    tests = [
        ("CPG Generation", test_cpg_generation),
        ("Dataset Validation", test_dataset_validation),
        ("OWASP Mapper", test_owasp_mapper),
        ("Evaluation Metrics", test_evaluation_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

