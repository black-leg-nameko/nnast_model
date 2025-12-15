#!/usr/bin/env python3
"""
Collect specific types of vulnerabilities (e.g., SQL injection, XSS, etc.)
"""
import argparse
import sys
from pathlib import Path

from data.collect_dataset import load_env_file, DatasetCollector
from data.github_api import GitHubAPIClient


# Predefined queries for specific vulnerability types
VULNERABILITY_QUERIES = {
    "sql_injection": [
        "SQL injection fix",
        "SQL injection vulnerability",
        "sqli fix",
        "SQL injection CVE",
        "SQLAlchemy SQL injection",
        "SQLAlchemy injection",
        "SQLAlchemy CVE",
        "Django SQL injection",
        "Flask SQL injection",
        "parameterized query",
        "raw SQL injection",
        "execute SQL injection",
    ],
    "xss": [
        "XSS fix",
        "Cross-Site Scripting",
        "XSS vulnerability",
        "XSS CVE",
    ],
    "command_injection": [
        "command injection fix",
        "command injection vulnerability",
        "os.system",
        "subprocess injection",
    ],
    "path_traversal": [
        "path traversal fix",
        "directory traversal",
        "path traversal vulnerability",
    ],
    "deserialization": [
        "deserialization vulnerability",
        "pickle vulnerability",
        "unsafe deserialization",
    ],
    "authentication": [
        "authentication bypass",
        "auth bypass fix",
        "authentication vulnerability",
    ],
    "authorization": [
        "authorization bypass",
        "privilege escalation",
        "authorization vulnerability",
    ],
    "crypto": [
        "cryptographic weakness",
        "weak cryptography",
        "insecure random",
    ],
}


def collect_specific_vulnerabilities(
    vuln_type: str,
    output_dir: Path,
    limit_per_query: int = 10,
    total_limit: int = 50
):
    """Collect specific type of vulnerabilities."""
    if vuln_type not in VULNERABILITY_QUERIES:
        print(f"Error: Unknown vulnerability type: {vuln_type}")
        print(f"Available types: {', '.join(VULNERABILITY_QUERIES.keys())}")
        return
    
    queries = VULNERABILITY_QUERIES[vuln_type]
    collector = DatasetCollector(output_dir)
    
    print(f"Collecting {vuln_type} vulnerabilities...")
    print(f"Using {len(queries)} queries, {limit_per_query} records per query")
    print()
    
    total_collected = 0
    
    for i, query in enumerate(queries, 1):
        if total_collected >= total_limit:
            break
        
        print(f"[{i}/{len(queries)}] Query: {query}")
        remaining = total_limit - total_collected
        current_limit = min(limit_per_query, remaining)
        
        try:
            records = collector.collect_from_github_cve(
                query=query,
                language="python",
                limit=current_limit
            )
            total_collected += len(records)
            print(f"  Collected {len(records)} records (Total: {total_collected})")
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    print(f"Total collected: {total_collected} records")
    return total_collected


def main():
    parser = argparse.ArgumentParser(
        description="Collect specific types of vulnerabilities"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=list(VULNERABILITY_QUERIES.keys()),
        help="Vulnerability type to collect"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=10,
        help="Limit per query"
    )
    parser.add_argument(
        "--total-limit",
        type=int,
        default=50,
        help="Total limit across all queries"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available vulnerability types"
    )
    
    args = parser.parse_args()
    
    if args.list_types:
        print("Available vulnerability types:")
        for vtype, queries in VULNERABILITY_QUERIES.items():
            print(f"  {vtype}: {len(queries)} queries")
            print(f"    Examples: {', '.join(queries[:3])}")
        return
    
    output_dir = Path(args.output_dir)
    collect_specific_vulnerabilities(
        args.type,
        output_dir,
        args.limit_per_query,
        args.total_limit
    )


if __name__ == "__main__":
    main()

