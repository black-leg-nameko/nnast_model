#!/usr/bin/env python3
"""
Diagnose dataset issues that might cause poor training performance.
"""
import argparse
import json
import pathlib
from collections import Counter
from typing import Dict, List


def analyze_labels(labels_file: pathlib.Path) -> Dict:
    """Analyze label distribution in labels file."""
    labels = []
    label_metadata = []
    
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                label = data.get("label", data.get("target", None))
                if label is not None:
                    labels.append(int(label))
                    label_metadata.append(data)
            except:
                pass
    
    if not labels:
        return {"error": "No labels found"}
    
    label_counts = Counter(labels)
    total = len(labels)
    
    # Check sample types
    sample_types = Counter()
    for meta in label_metadata:
        sample_type = meta.get("metadata", {}).get("sample_type", "unknown")
        sample_types[sample_type] += 1
    
    return {
        "total": total,
        "label_distribution": dict(label_counts),
        "label_percentages": {k: 100 * v / total for k, v in label_counts.items()},
        "sample_types": dict(sample_types),
        "class_imbalance_ratio": max(label_counts.values()) / min(label_counts.values()) if len(label_counts) > 1 else float('inf'),
    }


def check_graph_label_alignment(graphs_file: pathlib.Path, labels_file: pathlib.Path) -> Dict:
    """Check if graphs and labels are aligned."""
    graph_count = 0
    label_count = 0
    
    with open(graphs_file, 'r') as f:
        for line in f:
            if line.strip():
                graph_count += 1
    
    with open(labels_file, 'r') as f:
        for line in f:
            if line.strip():
                label_count += 1
    
    return {
        "graph_count": graph_count,
        "label_count": label_count,
        "aligned": graph_count == label_count,
        "difference": abs(graph_count - label_count),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose dataset issues"
    )
    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to graphs JSONL file"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels JSONL file"
    )
    
    args = parser.parse_args()
    
    graphs_file = pathlib.Path(args.graphs)
    labels_file = pathlib.Path(args.labels)
    
    if not graphs_file.exists():
        print(f"❌ Graphs file not found: {graphs_file}")
        return
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        return
    
    print("=" * 60)
    print("Dataset Diagnosis")
    print("=" * 60)
    
    # Check alignment
    print("\n1. Graph-Label Alignment")
    print("-" * 60)
    alignment = check_graph_label_alignment(graphs_file, labels_file)
    print(f"  Graphs: {alignment['graph_count']}")
    print(f"  Labels: {alignment['label_count']}")
    if alignment['aligned']:
        print("  ✅ Graphs and labels are aligned")
    else:
        print(f"  ⚠️ MISMATCH: Difference of {alignment['difference']} samples")
    
    # Analyze labels
    print("\n2. Label Distribution")
    print("-" * 60)
    label_analysis = analyze_labels(labels_file)
    
    if "error" in label_analysis:
        print(f"  ❌ {label_analysis['error']}")
        return
    
    print(f"  Total samples: {label_analysis['total']}")
    print(f"  Label distribution:")
    for label, count in sorted(label_analysis['label_distribution'].items()):
        pct = label_analysis['label_percentages'][label]
        print(f"    Label {label}: {count} ({pct:.1f}%)")
    
    # Check class imbalance
    if label_analysis['class_imbalance_ratio'] > 2:
        print(f"\n  ⚠️ Class imbalance detected!")
        print(f"     Ratio: {label_analysis['class_imbalance_ratio']:.2f}")
        print(f"     Consider using weighted loss function")
    else:
        print(f"\n  ✅ Classes are reasonably balanced")
        print(f"     Ratio: {label_analysis['class_imbalance_ratio']:.2f}")
    
    # Check sample types
    print(f"\n  Sample types:")
    for stype, count in label_analysis['sample_types'].items():
        print(f"    {stype}: {count}")
    
    # Check if we have both classes
    if len(label_analysis['label_distribution']) < 2:
        print(f"\n  ❌ CRITICAL: Only one class found!")
        print(f"     This will cause the model to always predict the same class")
        print(f"     Check your dataset preparation")
    else:
        print(f"\n  ✅ Both classes present")
    
    # Check for all same label
    if len(set(label_analysis['label_distribution'].keys())) == 1:
        print(f"\n  ❌ CRITICAL: All samples have the same label!")
        print(f"     Model cannot learn to distinguish classes")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    issues = []
    if not alignment['aligned']:
        issues.append("Fix graph-label alignment")
    if label_analysis['class_imbalance_ratio'] > 2:
        issues.append("Use weighted loss function (already implemented)")
    if len(label_analysis['label_distribution']) < 2:
        issues.append("Add samples from the missing class")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("✅ Dataset looks good! If accuracy is still 50%, check:")
        print("   1. Model architecture")
        print("   2. Learning rate")
        print("   3. Training for more epochs")


if __name__ == "__main__":
    main()

