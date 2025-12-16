#!/usr/bin/env python3
"""
Collect diverse Python vulnerabilities focusing on specific vulnerability types.

This script uses targeted queries to collect specific types of vulnerabilities
like SQL injection, XSS, path traversal, etc.
"""
import argparse
import sys
from pathlib import Path

from data.collect_dataset import DatasetCollector, load_env_file


# Predefined query sets for different vulnerability types
VULNERABILITY_QUERIES = {
    "sql_injection": [
        "SQL injection fix",
        "SQL injection vulnerability",
        "CWE-89",
        "SQLAlchemy SQL injection",
        "Django SQL injection",
        "Flask SQL injection",
        "prevent SQL injection",
        "fix SQLi",
    ],
    "xss": [
        "XSS fix",
        "Cross-Site Scripting",
        "CWE-79",
        "XSS vulnerability",
        "prevent XSS",
        "escape XSS",
    ],
    "path_traversal": [
        "path traversal fix",
        "directory traversal",
        "CWE-22",
        "path traversal vulnerability",
        "directory traversal fix",
    ],
    "command_injection": [
        "command injection fix",
        "CWE-78",
        "command injection vulnerability",
        "os.system",
        "subprocess injection",
    ],
    "deserialization": [
        "deserialization vulnerability",
        "pickle vulnerability",
        "CWE-502",
        "unsafe deserialization",
        "yaml.load vulnerability",
    ],
    "authentication": [
        "authentication bypass",
        "CWE-287",
        "authorization bypass",
        "authentication vulnerability",
    ],
    "crypto": [
        "cryptographic weakness",
        "weak cryptography",
        "CWE-327",
        "insecure random",
        "MD5 vulnerability",
    ],
    "general": [
        "CVE-2023",
        "CVE-2024",
        "security fix",
        "vulnerability fix",
        "fix CVE",
    ],
}


def collect_diverse_vulnerabilities(
    output_dir: Path,
    vuln_types: list = None,
    limit_per_query: int = 10,
    limit_per_type: int = 30,
):
    """
    Collect diverse vulnerabilities using targeted queries.
    
    Args:
        output_dir: Output directory for dataset
        vuln_types: List of vulnerability types to collect (None = all)
        limit_per_query: Limit per individual query
        limit_per_type: Maximum records per vulnerability type
    """
    load_env_file()
    
    collector = DatasetCollector(output_dir)
    
    if vuln_types is None:
        vuln_types = list(VULNERABILITY_QUERIES.keys())
    
    total_collected = 0
    
    print("=" * 60)
    print("Diverse Vulnerability Collection")
    print("=" * 60)
    print(f"Vulnerability types: {', '.join(vuln_types)}")
    print(f"Limit per query: {limit_per_query}")
    print(f"Limit per type: {limit_per_type}")
    print()
    
    for vuln_type in vuln_types:
        if vuln_type not in VULNERABILITY_QUERIES:
            print(f"Warning: Unknown vulnerability type: {vuln_type}")
            continue
        
        queries = VULNERABILITY_QUERIES[vuln_type]
        print(f"\n{'=' * 60}")
        print(f"Collecting: {vuln_type.replace('_', ' ').title()}")
        print(f"{'=' * 60}")
        
        type_collected = 0
        
        for query in queries:
            if type_collected >= limit_per_type:
                print(f"  Reached limit for {vuln_type} ({limit_per_type} records)")
                break
            
            print(f"\n  Query: {query}")
            try:
                records = collector.collect_from_github_cve(
                    query=query,
                    language="python",
                    limit=min(limit_per_query, limit_per_type - type_collected)
                )
                
                type_collected += len(records)
                total_collected += len(records)
                
                print(f"  Collected: {len(records)} records (Total for {vuln_type}: {type_collected})")
                
                # Small delay between queries
                import time
                time.sleep(2)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\n  Total collected for {vuln_type}: {type_collected} records")
    
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print(f"Total records collected: {total_collected}")
    print(f"Dataset location: {output_dir}")
    
    return total_collected


def main():
    parser = argparse.ArgumentParser(
        description="Collect diverse Python vulnerabilities"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(VULNERABILITY_QUERIES.keys()) + ["all"],
        default=["all"],
        help="Vulnerability types to collect"
    )
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=10,
        help="Limit per individual query"
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=30,
        help="Maximum records per vulnerability type"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    vuln_types = args.types
    if "all" in vuln_types:
        vuln_types = list(VULNERABILITY_QUERIES.keys())
    
    collect_diverse_vulnerabilities(
        output_dir=output_dir,
        vuln_types=vuln_types,
        limit_per_query=args.limit_per_query,
        limit_per_type=args.limit_per_type,
    )


if __name__ == "__main__":
    main()

