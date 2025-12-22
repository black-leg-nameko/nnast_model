#!/usr/bin/env python3
"""
Dataset quality validation tool for NNAST.

Validates training dataset quality according to design document v2:
- Pattern ID distribution
- Framework distribution
- Synthetic vs real data balance
- Label distribution
- CPG graph quality
"""
import argparse
import json
import pathlib
from typing import Dict, List, Optional, Set, Any
from collections import Counter, defaultdict
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class DatasetValidator:
    """Validates dataset quality for NNAST training."""
    
    def __init__(self, graphs_file: pathlib.Path, labels_file: Optional[pathlib.Path] = None):
        """
        Initialize dataset validator.
        
        Args:
            graphs_file: Path to graphs JSONL file
            labels_file: Optional path to labels JSONL file
        """
        self.graphs_file = pathlib.Path(graphs_file)
        self.labels_file = pathlib.Path(labels_file) if labels_file else None
        self.graphs: List[Dict[str, Any]] = []
        self.labels: List[Dict[str, Any]] = []
        
    def load_data(self):
        """Load graphs and labels from files."""
        print(f"Loading graphs from {self.graphs_file}...")
        with open(self.graphs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    graph = json.loads(line)
                    self.graphs.append(graph)
                except json.JSONDecodeError:
                    continue
        
        if self.labels_file and self.labels_file.exists():
            print(f"Loading labels from {self.labels_file}...")
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        label = json.loads(line)
                        self.labels.append(label)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.graphs)} graphs, {len(self.labels)} labels")
    
    def validate_graph_label_alignment(self) -> Dict[str, Any]:
        """Check if graphs and labels are aligned."""
        if not self.labels_file:
            return {"status": "skipped", "reason": "No labels file provided"}
        
        if len(self.graphs) != len(self.labels):
            return {
                "status": "error",
                "message": f"Graph count ({len(self.graphs)}) != Label count ({len(self.labels)})",
                "graphs_count": len(self.graphs),
                "labels_count": len(self.labels)
            }
        
        return {
            "status": "ok",
            "count": len(self.graphs)
        }
    
    def analyze_label_distribution(self) -> Dict[str, Any]:
        """Analyze label distribution."""
        if not self.labels:
            return {"status": "skipped", "reason": "No labels available"}
        
        label_counts = Counter()
        for label_data in self.labels:
            label = label_data.get("label", label_data.get("target", None))
            if label is not None:
                label_counts[int(label)] += 1
        
        total = sum(label_counts.values())
        if total == 0:
            return {"status": "error", "message": "No valid labels found"}
        
        percentages = {k: 100 * v / total for k, v in label_counts.items()}
        imbalance_ratio = max(label_counts.values()) / min(label_counts.values()) if len(label_counts) > 1 else float('inf')
        
        return {
            "status": "ok",
            "total": total,
            "distribution": dict(label_counts),
            "percentages": percentages,
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": imbalance_ratio < 3.0  # Threshold for balanced dataset
        }
    
    def analyze_pattern_distribution(self) -> Dict[str, Any]:
        """Analyze pattern ID distribution from labels metadata."""
        if not self.labels:
            return {"status": "skipped", "reason": "No labels available"}
        
        pattern_counts = Counter()
        pattern_by_label = defaultdict(lambda: {"vulnerable": 0, "safe": 0})
        
        for label_data in self.labels:
            metadata = label_data.get("metadata", {})
            pattern_id = metadata.get("pattern_id")
            vulnerability_type = metadata.get("vulnerability_type")
            label = label_data.get("label", 0)
            
            # Use pattern_id if available, otherwise try to infer from vulnerability_type
            pattern_key = pattern_id or vulnerability_type or "unknown"
            
            if pattern_key != "unknown":
                pattern_counts[pattern_key] += 1
                if label == 1:
                    pattern_by_label[pattern_key]["vulnerable"] += 1
                else:
                    pattern_by_label[pattern_key]["safe"] += 1
        
        return {
            "status": "ok",
            "total_patterns": len(pattern_counts),
            "pattern_distribution": dict(pattern_counts),
            "pattern_by_label": dict(pattern_by_label),
            "unique_patterns": list(pattern_counts.keys())
        }
    
    def analyze_framework_distribution(self) -> Dict[str, Any]:
        """Analyze framework distribution from CPG graphs."""
        framework_counts = Counter()
        framework_by_label = defaultdict(lambda: {"vulnerable": 0, "safe": 0})
        
        for i, graph in enumerate(self.graphs):
            # Get framework from graph metadata
            metadata = graph.get("metadata", {})
            frameworks = metadata.get("frameworks", [])
            
            if frameworks:
                framework = frameworks[0] if isinstance(frameworks, list) else frameworks
            else:
                # Try to infer from file path or code
                file_path = graph.get("file", "")
                if "flask" in file_path.lower():
                    framework = "flask"
                elif "django" in file_path.lower():
                    framework = "django"
                elif "fastapi" in file_path.lower():
                    framework = "fastapi"
                else:
                    framework = "unknown"
            
            framework_counts[framework] += 1
            
            # Get label if available
            if i < len(self.labels):
                label = self.labels[i].get("label", 0)
                if label == 1:
                    framework_by_label[framework]["vulnerable"] += 1
                else:
                    framework_by_label[framework]["safe"] += 1
        
        return {
            "status": "ok",
            "framework_distribution": dict(framework_counts),
            "framework_by_label": dict(framework_by_label),
            "unique_frameworks": list(framework_counts.keys())
        }
    
    def analyze_source_sink_distribution(self) -> Dict[str, Any]:
        """Analyze source/sink distribution from CPG graphs."""
        source_counts = Counter()
        sink_counts = Counter()
        sink_kind_counts = Counter()
        
        for graph in self.graphs:
            nodes = graph.get("nodes", [])
            for node in nodes:
                attrs = node.get("attrs", {}) or {}
                
                if attrs.get("is_source") == "true":
                    source_id = attrs.get("source_id", "unknown")
                    source_counts[source_id] += 1
                
                if attrs.get("is_sink") == "true":
                    sink_id = attrs.get("sink_id", "unknown")
                    sink_kind = attrs.get("sink_kind", "unknown")
                    sink_counts[sink_id] += 1
                    sink_kind_counts[sink_kind] += 1
        
        return {
            "status": "ok",
            "source_distribution": dict(source_counts),
            "sink_distribution": dict(sink_counts),
            "sink_kind_distribution": dict(sink_kind_counts),
            "total_sources": sum(source_counts.values()),
            "total_sinks": sum(sink_counts.values())
        }
    
    def analyze_data_source_balance(self) -> Dict[str, Any]:
        """Analyze balance between synthetic and real data."""
        if not self.labels:
            return {"status": "skipped", "reason": "No labels available"}
        
        synthetic_count = 0
        real_count = 0
        unknown_count = 0
        
        for label_data in self.labels:
            metadata = label_data.get("metadata", {})
            sample_type = metadata.get("sample_type", "")
            repo_url = metadata.get("repo_url", "")
            
            if "synthetic" in sample_type.lower() or "synthetic" in str(metadata).lower():
                synthetic_count += 1
            elif repo_url or metadata.get("cve_id") or metadata.get("commit"):
                real_count += 1
            else:
                unknown_count += 1
        
        total = synthetic_count + real_count + unknown_count
        if total == 0:
            return {"status": "error", "message": "No data found"}
        
        return {
            "status": "ok",
            "synthetic_count": synthetic_count,
            "real_count": real_count,
            "unknown_count": unknown_count,
            "total": total,
            "synthetic_percentage": 100 * synthetic_count / total if total > 0 else 0,
            "real_percentage": 100 * real_count / total if total > 0 else 0,
            "is_balanced": 0.2 <= (synthetic_count / total) <= 0.8 if total > 0 else False
        }
    
    def analyze_cpg_quality(self) -> Dict[str, Any]:
        """Analyze CPG graph quality metrics."""
        node_counts = []
        edge_counts = []
        graphs_with_sources = 0
        graphs_with_sinks = 0
        graphs_with_sanitizers = 0
        
        for graph in self.graphs:
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            
            node_counts.append(len(nodes))
            edge_counts.append(len(edges))
            
            # Check for source/sink/sanitizer attributes
            has_source = False
            has_sink = False
            has_sanitizer = False
            
            for node in nodes:
                attrs = node.get("attrs", {}) or {}
                # Check for source (support both string "true" and boolean True)
                is_source = attrs.get("is_source")
                if is_source == "true" or is_source is True:
                    has_source = True
                # Check for sink (support both string "true" and boolean True)
                is_sink = attrs.get("is_sink")
                if is_sink == "true" or is_sink is True:
                    has_sink = True
                # Check for sanitizer
                if attrs.get("sanitizer_kind"):
                    has_sanitizer = True
            
            if has_source:
                graphs_with_sources += 1
            if has_sink:
                graphs_with_sinks += 1
            if has_sanitizer:
                graphs_with_sanitizers += 1
        
        if not node_counts:
            return {"status": "error", "message": "No graphs found"}
        
        # Calculate statistics
        if HAS_NUMPY:
            node_counts_arr = np.array(node_counts)
            edge_counts_arr = np.array(edge_counts)
            node_mean = float(np.mean(node_counts_arr))
            node_median = float(np.median(node_counts_arr))
            node_std = float(np.std(node_counts_arr))
            node_min = int(np.min(node_counts_arr))
            node_max = int(np.max(node_counts_arr))
            edge_mean = float(np.mean(edge_counts_arr))
            edge_median = float(np.median(edge_counts_arr))
            edge_std = float(np.std(edge_counts_arr))
            edge_min = int(np.min(edge_counts_arr))
            edge_max = int(np.max(edge_counts_arr))
        else:
            # Fallback without numpy
            node_mean = sum(node_counts) / len(node_counts) if node_counts else 0
            node_median = sorted(node_counts)[len(node_counts)//2] if node_counts else 0
            node_std = 0.0  # Would need more complex calculation
            node_min = min(node_counts) if node_counts else 0
            node_max = max(node_counts) if node_counts else 0
            edge_mean = sum(edge_counts) / len(edge_counts) if edge_counts else 0
            edge_median = sorted(edge_counts)[len(edge_counts)//2] if edge_counts else 0
            edge_std = 0.0
            edge_min = min(edge_counts) if edge_counts else 0
            edge_max = max(edge_counts) if edge_counts else 0
        
        return {
            "status": "ok",
            "total_graphs": len(self.graphs),
            "node_statistics": {
                "mean": node_mean,
                "median": node_median,
                "std": node_std,
                "min": node_min,
                "max": node_max
            },
            "edge_statistics": {
                "mean": edge_mean,
                "median": edge_median,
                "std": edge_std,
                "min": edge_min,
                "max": edge_max
            },
            "graphs_with_sources": graphs_with_sources,
            "graphs_with_sinks": graphs_with_sinks,
            "graphs_with_sanitizers": graphs_with_sanitizers,
            "source_coverage": 100 * graphs_with_sources / len(self.graphs) if self.graphs else 0,
            "sink_coverage": 100 * graphs_with_sinks / len(self.graphs) if self.graphs else 0,
            "sanitizer_coverage": 100 * graphs_with_sanitizers / len(self.graphs) if self.graphs else 0
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("\n" + "="*60)
        print("Dataset Quality Validation")
        print("="*60)
        
        results = {
            "graphs_file": str(self.graphs_file),
            "labels_file": str(self.labels_file) if self.labels_file else None,
            "total_graphs": len(self.graphs),
            "total_labels": len(self.labels),
        }
        
        # Graph-Label alignment
        print("\n1. Checking graph-label alignment...")
        alignment = self.validate_graph_label_alignment()
        results["alignment"] = alignment
        if alignment.get("status") == "error":
            print(f"  ❌ {alignment.get('message')}")
        else:
            print(f"  ✅ {alignment.get('status', 'ok')}: {alignment.get('count', 0)} samples")
        
        # Label distribution
        print("\n2. Analyzing label distribution...")
        label_dist = self.analyze_label_distribution()
        results["label_distribution"] = label_dist
        if label_dist.get("status") == "ok":
            dist = label_dist.get("distribution", {})
            pct = label_dist.get("percentages", {})
            is_balanced = label_dist.get("is_balanced", False)
            print(f"  ✅ Total: {label_dist.get('total', 0)}")
            for label, count in dist.items():
                print(f"    Label {label}: {count} ({pct.get(label, 0):.1f}%)")
            print(f"  {'✅' if is_balanced else '⚠️'} Dataset is {'balanced' if is_balanced else 'imbalanced'} (ratio: {label_dist.get('imbalance_ratio', 0):.2f})")
        
        # Pattern distribution
        print("\n3. Analyzing pattern distribution...")
        pattern_dist = self.analyze_pattern_distribution()
        results["pattern_distribution"] = pattern_dist
        if pattern_dist.get("status") == "ok":
            patterns = pattern_dist.get("pattern_distribution", {})
            print(f"  ✅ Found {pattern_dist.get('total_patterns', 0)} unique patterns")
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {pattern}: {count}")
        
        # Framework distribution
        print("\n4. Analyzing framework distribution...")
        framework_dist = self.analyze_framework_distribution()
        results["framework_distribution"] = framework_dist
        if framework_dist.get("status") == "ok":
            frameworks = framework_dist.get("framework_distribution", {})
            print(f"  ✅ Found {len(frameworks)} frameworks")
            for fw, count in sorted(frameworks.items(), key=lambda x: x[1], reverse=True):
                print(f"    {fw}: {count}")
        
        # Source/Sink distribution
        print("\n5. Analyzing source/sink distribution...")
        source_sink_dist = self.analyze_source_sink_distribution()
        results["source_sink_distribution"] = source_sink_dist
        if source_sink_dist.get("status") == "ok":
            print(f"  ✅ Sources: {source_sink_dist.get('total_sources', 0)}, Sinks: {source_sink_dist.get('total_sinks', 0)}")
            sources = source_sink_dist.get("source_distribution", {})
            sinks = source_sink_dist.get("sink_distribution", {})
            print(f"    Top sources: {dict(list(sources.items())[:5])}")
            print(f"    Top sinks: {dict(list(sinks.items())[:5])}")
        
        # Data source balance
        print("\n6. Analyzing synthetic vs real data balance...")
        data_balance = self.analyze_data_source_balance()
        results["data_source_balance"] = data_balance
        if data_balance.get("status") == "ok":
            synth_pct = data_balance.get("synthetic_percentage", 0)
            real_pct = data_balance.get("real_percentage", 0)
            is_balanced = data_balance.get("is_balanced", False)
            print(f"  ✅ Synthetic: {data_balance.get('synthetic_count', 0)} ({synth_pct:.1f}%)")
            print(f"  ✅ Real: {data_balance.get('real_count', 0)} ({real_pct:.1f}%)")
            print(f"  {'✅' if is_balanced else '⚠️'} Data source is {'balanced' if is_balanced else 'imbalanced'}")
        
        # CPG quality
        print("\n7. Analyzing CPG graph quality...")
        cpg_quality = self.analyze_cpg_quality()
        results["cpg_quality"] = cpg_quality
        if cpg_quality.get("status") == "ok":
            node_stats = cpg_quality.get("node_statistics", {})
            edge_stats = cpg_quality.get("edge_statistics", {})
            print(f"  ✅ Nodes: mean={node_stats.get('mean', 0):.1f}, median={node_stats.get('median', 0):.1f}")
            print(f"  ✅ Edges: mean={edge_stats.get('mean', 0):.1f}, median={edge_stats.get('median', 0):.1f}")
            print(f"  ✅ Source coverage: {cpg_quality.get('source_coverage', 0):.1f}%")
            print(f"  ✅ Sink coverage: {cpg_quality.get('sink_coverage', 0):.1f}%")
            print(f"  ✅ Sanitizer coverage: {cpg_quality.get('sanitizer_coverage', 0):.1f}%")
        
        print("\n" + "="*60)
        print("Validation Complete")
        print("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset quality for NNAST training"
    )
    parser.add_argument(
        "graphs",
        type=pathlib.Path,
        help="Path to graphs JSONL file"
    )
    parser.add_argument(
        "--labels",
        type=pathlib.Path,
        help="Optional path to labels JSONL file"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional path to save validation results as JSON"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    validator = DatasetValidator(args.graphs, args.labels)
    validator.load_data()
    results = validator.validate_all()
    
    # Save results if output path specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    
    # Return exit code based on validation results
    has_errors = False
    if results.get("alignment", {}).get("status") == "error":
        has_errors = True
    
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

