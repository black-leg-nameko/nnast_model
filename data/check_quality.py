#!/usr/bin/env python3
"""
Dataset quality checker for collected vulnerability data.
"""
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict


def check_dataset_quality(dataset_dir: Path) -> Dict:
    """Check quality of collected dataset."""
    metadata_file = dataset_dir / "metadata.jsonl"
    processed_dir = dataset_dir / "processed"
    
    if not metadata_file.exists():
        return {"error": "Metadata file not found"}
    
    # Load records
    records = []
    with open(metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not records:
        return {"error": "No records found"}
    
    # Basic statistics
    stats = {
        "total_records": len(records),
        "unique_cves": len(set(r.get("cve_id") for r in records if r.get("cve_id"))),
        "unique_repos": len(set(r.get("repo_url") for r in records)),
        "unique_files": len(set(r.get("file_path") for r in records)),
    }
    
    # CVE distribution
    cve_dist = Counter(r.get("cve_id") for r in records if r.get("cve_id"))
    stats["cve_distribution"] = dict(cve_dist.most_common(10))
    
    # Vulnerability types
    vuln_types = Counter(r.get("vulnerability_type") for r in records)
    stats["vulnerability_types"] = dict(vuln_types)
    
    # Repository distribution
    repo_dist = Counter(r.get("repo_url", "").split("/")[-1] for r in records)
    stats["top_repositories"] = dict(repo_dist.most_common(5))
    
    # Code quality
    code_stats = {
        "records_with_code": 0,
        "records_without_code": 0,
        "avg_code_length_before": 0,
        "avg_code_length_after": 0,
        "avg_lines_before": 0,
        "avg_lines_after": 0,
    }
    
    code_lengths_before = []
    code_lengths_after = []
    lines_before = []
    lines_after = []
    
    for r in records:
        code_before = r.get("code_before", "")
        code_after = r.get("code_after", "")
        
        if code_before and code_after:
            code_stats["records_with_code"] += 1
            code_lengths_before.append(len(code_before))
            code_lengths_after.append(len(code_after))
            lines_before.append(len(code_before.splitlines()))
            lines_after.append(len(code_after.splitlines()))
        else:
            code_stats["records_without_code"] += 1
    
    if code_lengths_before:
        code_stats["avg_code_length_before"] = sum(code_lengths_before) // len(code_lengths_before)
        code_stats["avg_code_length_after"] = sum(code_lengths_after) // len(code_lengths_after)
        code_stats["avg_lines_before"] = sum(lines_before) // len(lines_before)
        code_stats["avg_lines_after"] = sum(lines_after) // len(lines_after)
    
    stats["code_quality"] = code_stats
    
    # CPG graph quality
    graph_stats = {
        "total_graphs": 0,
        "valid_graphs": 0,
        "avg_nodes": 0,
        "avg_edges": 0,
    }
    
    if processed_dir.exists():
        graph_files = list(processed_dir.glob("*.jsonl"))
        graph_stats["total_graphs"] = len(graph_files)
        
        total_nodes = 0
        total_edges = 0
        valid_count = 0
        
        # Sample graphs for quality check
        sample_size = min(20, len(graph_files))
        for gf in graph_files[:sample_size]:
            try:
                with open(gf) as f:
                    graph = json.loads(f.readline())
                    nodes = graph.get("nodes", [])
                    edges = graph.get("edges", [])
                    if nodes:
                        valid_count += 1
                        total_nodes += len(nodes)
                        total_edges += len(edges)
            except Exception:
                pass
        
        if valid_count > 0:
            graph_stats["valid_graphs"] = valid_count
            graph_stats["avg_nodes"] = total_nodes // valid_count
            graph_stats["avg_edges"] = total_edges // valid_count
    
    stats["graph_quality"] = graph_stats
    
    # Quality assessment
    quality_issues = []
    quality_score = 100
    
    if stats["total_records"] < 10:
        quality_issues.append("Very few records (< 10)")
        quality_score -= 30
    
    if stats["unique_cves"] < 3:
        quality_issues.append("Low CVE diversity (< 3 unique CVEs)")
        quality_score -= 20
    
    if code_stats["records_without_code"] > stats["total_records"] * 0.1:
        quality_issues.append(f"Many records without code ({code_stats['records_without_code']}/{stats['total_records']})")
        quality_score -= 15
    
    if graph_stats["valid_graphs"] == 0:
        quality_issues.append("No valid CPG graphs found")
        quality_score -= 25
    
    stats["quality_score"] = max(0, quality_score)
    stats["quality_issues"] = quality_issues
    
    return stats


def print_quality_report(stats: Dict):
    """Print quality report."""
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return
    
    print("=" * 60)
    print("Dataset Quality Report")
    print("=" * 60)
    print()
    
    print("Basic Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Unique CVE IDs: {stats['unique_cves']}")
    print(f"  Unique repositories: {stats['unique_repos']}")
    print(f"  Unique files: {stats['unique_files']}")
    print()
    
    if stats.get("cve_distribution"):
        print("CVE Distribution (Top 10):")
        for cve, count in list(stats["cve_distribution"].items())[:10]:
            print(f"  {cve}: {count} records")
        print()
    
    if stats.get("vulnerability_types"):
        print("Vulnerability Types:")
        for vtype, count in stats["vulnerability_types"].items():
            print(f"  {vtype or 'Unknown'}: {count}")
        print()
    
    if stats.get("top_repositories"):
        print("Top Repositories:")
        for repo, count in stats["top_repositories"].items():
            print(f"  {repo}: {count} records")
        print()
    
    code_quality = stats.get("code_quality", {})
    print("Code Quality:")
    print(f"  Records with code: {code_quality.get('records_with_code', 0)}/{stats['total_records']}")
    if code_quality.get("avg_code_length_before"):
        print(f"  Avg code length (before): {code_quality['avg_code_length_before']} chars")
        print(f"  Avg code length (after): {code_quality['avg_code_length_after']} chars")
        print(f"  Avg lines (before): {code_quality['avg_lines_before']}")
        print(f"  Avg lines (after): {code_quality['avg_lines_after']}")
    print()
    
    graph_quality = stats.get("graph_quality", {})
    print("CPG Graph Quality:")
    print(f"  Total graphs: {graph_quality.get('total_graphs', 0)}")
    print(f"  Valid graphs (sample): {graph_quality.get('valid_graphs', 0)}")
    if graph_quality.get("avg_nodes"):
        print(f"  Avg nodes per graph: {graph_quality['avg_nodes']}")
        print(f"  Avg edges per graph: {graph_quality['avg_edges']}")
    print()
    
    print("Quality Assessment:")
    print(f"  Quality Score: {stats.get('quality_score', 0)}/100")
    if stats.get("quality_issues"):
        print("  Issues found:")
        for issue in stats["quality_issues"]:
            print(f"    - {issue}")
    else:
        print("  ✓ No major issues found")
    print()
    
    # Recommendation
    score = stats.get("quality_score", 0)
    if score >= 80:
        print("✓ Dataset quality is good. Ready for training data preparation.")
    elif score >= 60:
        print("⚠ Dataset quality is acceptable. Consider collecting more data.")
    else:
        print("✗ Dataset quality needs improvement. Continue data collection.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dataset quality")
    parser.add_argument(
        "--dataset-dir",
        default="./dataset",
        help="Dataset directory"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    stats = check_dataset_quality(dataset_dir)
    
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_quality_report(stats)


if __name__ == "__main__":
    main()

