#!/usr/bin/env python3
"""
Check for missing imports in ML module files.
This script helps identify potential import errors before they occur.
"""
import ast
import pathlib
from typing import Set, Dict, List
import importlib.util


# Standard library modules that are commonly used
STANDARD_LIBS = {
    'collections', 'typing', 'json', 'pathlib', 'argparse', 'sys', 'os',
    'tempfile', 'subprocess', 'random', 'math', 'itertools', 'functools',
    'dataclasses', 'enum', 'abc', 'copy', 'datetime', 'time', 'warnings',
    'traceback', 'inspect', 'glob', 're', 'hashlib', 'base64', 'urllib',
}

# Third-party modules that should be imported
THIRD_PARTY = {
    'torch', 'numpy', 'sklearn', 'transformers', 'tqdm',
    'torch_geometric', 'torch_geometric.nn', 'torch_geometric.data',
    'torch_geometric.utils',
}


def extract_imports(file_path: pathlib.Path) -> Dict[str, Set[str]]:
    """Extract all imports from a Python file."""
    imports = {
        'standard': set(),
        'third_party': set(),
        'local': set(),
        'all': set(),
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    imports['all'].add(alias.name)
                    if module in STANDARD_LIBS:
                        imports['standard'].add(module)
                    elif module in THIRD_PARTY:
                        imports['third_party'].add(module)
                    else:
                        imports['local'].add(module)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    imports['all'].add(node.module)
                    if module in STANDARD_LIBS:
                        imports['standard'].add(module)
                    elif module in THIRD_PARTY:
                        imports['third_party'].add(module)
                    else:
                        imports['local'].add(module)
    
    except SyntaxError as e:
        print(f"⚠️ Syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"⚠️ Error parsing {file_path}: {e}")
    
    return imports


def check_common_issues(file_path: pathlib.Path) -> List[str]:
    """Check for common import issues."""
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check for Counter usage
        if 'Counter(' in content and 'from collections import Counter' not in content and 'import collections' not in content:
            issues.append("⚠️ Counter is used but not imported from collections")
        
        # Check for sklearn metrics
        if any(metric in content for metric in ['f1_score', 'precision_score', 'recall_score']):
            if 'from sklearn.metrics import' not in content and 'import sklearn' not in content:
                issues.append("⚠️ sklearn.metrics functions used but not imported")
        
        # Check for random usage
        if 'random.' in content or 'random_split' in content or 'random.shuffle' in content:
            if 'import random' not in content and 'from random import' not in content:
                if 'random_split' in content:
                    # random_split is from torch.utils.data, not random
                    pass
                else:
                    issues.append("⚠️ random module used but not imported")
        
        # Check for duplicate imports
        lines = content.split('\n')
        seen_imports = {}
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                if line.strip() in seen_imports:
                    issues.append(f"⚠️ Duplicate import at line {i}: {line.strip()[:50]}")
                seen_imports[line.strip()] = i
    
    except Exception as e:
        issues.append(f"Error checking file: {e}")
    
    return issues


def main():
    """Check all ML module files for import issues."""
    ml_dir = pathlib.Path(__file__).parent
    
    print("=" * 60)
    print("Import Check for ML Module Files")
    print("=" * 60)
    
    all_issues = []
    
    for py_file in sorted(ml_dir.glob("*.py")):
        if py_file.name == "__init__.py" or py_file.name == "check_imports.py":
            continue
        
        print(f"\n📄 {py_file.name}")
        print("-" * 60)
        
        # Extract imports
        imports = extract_imports(py_file)
        
        # Check for common issues
        issues = check_common_issues(py_file)
        
        if issues:
            all_issues.extend([(py_file.name, issue) for issue in issues])
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ No obvious import issues found")
        
        # Show imports summary
        if imports['all']:
            print(f"  Imports: {len(imports['all'])} total")
            if imports['standard']:
                print(f"    Standard: {', '.join(sorted(imports['standard']))}")
            if imports['third_party']:
                print(f"    Third-party: {', '.join(sorted(imports['third_party']))}")
            if imports['local']:
                print(f"    Local: {', '.join(sorted(imports['local']))}")
    
    print("\n" + "=" * 60)
    if all_issues:
        print(f"⚠️ Found {len(all_issues)} potential issues:")
        for file, issue in all_issues:
            print(f"  {file}: {issue}")
    else:
        print("✅ No import issues found!")
    print("=" * 60)


if __name__ == "__main__":
    main()

