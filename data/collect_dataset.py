#!/usr/bin/env python3
"""
Python Vulnerability Dataset Collection Pipeline

This script collects Python code with vulnerabilities from various sources:
1. GitHub repositories with CVE references
2. CVE databases (NVD, GitHub Security Advisories)
3. Security-focused Python projects

The collected data is structured for NNAST training:
- Vulnerable code (before fix)
- Fixed code (after fix)
- Vulnerability metadata (CWE, CVE ID, etc.)
- CPG graphs for both versions
"""
import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class VulnerabilityRecord:
    """Record for a vulnerability code pair."""
    cve_id: Optional[str]
    cwe_id: Optional[str]
    repo_url: str
    commit_before: str  # Vulnerable version commit hash
    commit_after: str  # Fixed version commit hash
    file_path: str
    line_range_before: Tuple[int, int]  # (start_line, end_line)
    line_range_after: Tuple[int, int]
    vulnerability_type: str  # e.g., "SQL Injection", "XSS", etc.
    description: str
    code_before: str
    code_after: str


class DatasetCollector:
    """Collects Python vulnerability datasets from various sources."""
    
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.raw_data_dir = self.output_dir / "raw"
        self.processed_data_dir = self.output_dir / "processed"
        self.metadata_file = self.output_dir / "metadata.jsonl"
        
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
    
    def collect_from_github_cve(self, query: str = "language:python CVE", limit: int = 100) -> List[Dict]:
        """
        Collect Python vulnerabilities from GitHub using search API.
        
        Note: Requires GitHub API token in GITHUB_TOKEN environment variable.
        """
        import os
        
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("Warning: GITHUB_TOKEN not set. GitHub collection will be skipped.")
            return []
        
        # Use GitHub CLI or API to search
        # This is a placeholder - actual implementation would use GitHub API
        print(f"Searching GitHub for: {query} (limit: {limit})")
        print("Note: Full GitHub API integration requires API token setup")
        
        # Placeholder: return empty list for now
        # In production, this would:
        # 1. Search GitHub for Python CVE-related commits
        # 2. Extract commit hashes, file paths, code diffs
        # 3. Parse CVE/CWE information from commit messages
        return []
    
    def collect_from_cve_database(self, cve_list: Optional[List[str]] = None) -> List[Dict]:
        """
        Collect Python vulnerabilities from CVE databases.
        
        Args:
            cve_list: Optional list of specific CVE IDs to collect
        """
        print("Collecting from CVE databases...")
        
        # Placeholder: In production, this would:
        # 1. Query NVD API or CVE database
        # 2. Filter for Python-related CVEs
        # 3. Extract GitHub links and commit information
        # 4. Download vulnerable and fixed code versions
        
        records = []
        
        if cve_list:
            for cve_id in cve_list:
                print(f"  Processing {cve_id}...")
                # Extract vulnerability information
                # This would query CVE database and extract relevant info
                pass
        
        return records
    
    def process_repository(
        self,
        repo_url: str,
        commit_before: str,
        commit_after: str,
        file_path: str
    ) -> Optional[VulnerabilityRecord]:
        """
        Process a repository to extract vulnerable and fixed code.
        
        Args:
            repo_url: GitHub repository URL
            commit_before: Commit hash of vulnerable version
            commit_after: Commit hash of fixed version
            file_path: Path to the vulnerable file
            
        Returns:
            VulnerabilityRecord if successful, None otherwise
        """
        print(f"Processing {repo_url} ({commit_before} -> {commit_after})")
        
        # Clone repository temporarily
        temp_dir = self.raw_data_dir / f"repo_{hash(repo_url)}"
        
        try:
            # Clone repository
            if not temp_dir.exists():
                subprocess.run(
                    ["git", "clone", repo_url, str(temp_dir)],
                    check=True,
                    capture_output=True
                )
            
            # Checkout vulnerable version
            subprocess.run(
                ["git", "checkout", commit_before],
                cwd=temp_dir,
                check=True,
                capture_output=True
            )
            
            # Read vulnerable code
            vulnerable_file = temp_dir / file_path
            if not vulnerable_file.exists():
                print(f"  Warning: File not found: {file_path}")
                return None
            
            code_before = vulnerable_file.read_text(encoding="utf-8")
            
            # Checkout fixed version
            subprocess.run(
                ["git", "checkout", commit_after],
                cwd=temp_dir,
                check=True,
                capture_output=True
            )
            
            # Read fixed code
            code_after = vulnerable_file.read_text(encoding="utf-8")
            
            # Extract metadata from git log
            # This is simplified - in production, would parse commit messages for CVE/CWE
            
            record = VulnerabilityRecord(
                cve_id=None,  # Would be extracted from commit message
                cwe_id=None,
                repo_url=repo_url,
                commit_before=commit_before,
                commit_after=commit_after,
                file_path=file_path,
                line_range_before=(1, len(code_before.splitlines())),
                line_range_after=(1, len(code_after.splitlines())),
                vulnerability_type="Unknown",  # Would be extracted from CVE/CWE
                description="",  # Would be extracted from commit message
                code_before=code_before,
                code_after=code_after
            )
            
            return record
            
        except subprocess.CalledProcessError as e:
            print(f"  Error processing repository: {e}")
            return None
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None
    
    def generate_cpg_graphs(self, record: VulnerabilityRecord) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Generate CPG graphs for vulnerable and fixed code.
        
        Returns:
            Tuple of (vulnerable_graph_dict, fixed_graph_dict)
        """
        import tempfile
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f_before:
            f_before.write(record.code_before)
            temp_before = f_before.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f_after:
            f_after.write(record.code_after)
            temp_after = f_after.name
        
        try:
            # Generate CPG for vulnerable code
            result_before = subprocess.run(
                [sys.executable, "-m", "cli", temp_before, "--out", "/dev/stdout"],
                capture_output=True,
                text=True
            )
            
            if result_before.returncode == 0:
                graph_before = json.loads(result_before.stdout.strip().split('\n')[0])
            else:
                print(f"  Warning: Failed to generate CPG for vulnerable code")
                graph_before = None
            
            # Generate CPG for fixed code
            result_after = subprocess.run(
                [sys.executable, "-m", "cli", temp_after, "--out", "/dev/stdout"],
                capture_output=True,
                text=True
            )
            
            if result_after.returncode == 0:
                graph_after = json.loads(result_after.stdout.strip().split('\n')[0])
            else:
                print(f"  Warning: Failed to generate CPG for fixed code")
                graph_after = None
            
            return graph_before, graph_after
            
        except Exception as e:
            print(f"  Error generating CPG graphs: {e}")
            return None, None
        finally:
            # Cleanup
            pathlib.Path(temp_before).unlink(missing_ok=True)
            pathlib.Path(temp_after).unlink(missing_ok=True)
    
    def save_record(self, record: VulnerabilityRecord, graph_before: Optional[Dict], graph_after: Optional[Dict]):
        """Save vulnerability record and CPG graphs."""
        # Save metadata
        metadata = {
            "cve_id": record.cve_id,
            "cwe_id": record.cwe_id,
            "repo_url": record.repo_url,
            "commit_before": record.commit_before,
            "commit_after": record.commit_after,
            "file_path": record.file_path,
            "vulnerability_type": record.vulnerability_type,
            "description": record.description,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        
        # Save CPG graphs
        if graph_before:
            graph_file_before = self.processed_data_dir / f"{record.commit_before}_{pathlib.Path(record.file_path).stem}_before.jsonl"
            with open(graph_file_before, "w") as f:
                f.write(json.dumps(graph_before, ensure_ascii=False) + "\n")
        
        if graph_after:
            graph_file_after = self.processed_data_dir / f"{record.commit_after}_{pathlib.Path(record.file_path).stem}_after.jsonl"
            with open(graph_file_after, "w") as f:
                f.write(json.dumps(graph_after, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Python vulnerability dataset for NNAST training"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for collected dataset"
    )
    parser.add_argument(
        "--github-query",
        default="language:python CVE",
        help="GitHub search query"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records to collect"
    )
    parser.add_argument(
        "--cve-list",
        nargs="+",
        help="Specific CVE IDs to collect"
    )
    parser.add_argument(
        "--repo",
        help="Specific repository URL to process"
    )
    parser.add_argument(
        "--commit-before",
        help="Vulnerable commit hash (requires --repo)"
    )
    parser.add_argument(
        "--commit-after",
        help="Fixed commit hash (requires --repo)"
    )
    parser.add_argument(
        "--file-path",
        help="File path in repository (requires --repo)"
    )
    
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    collector = DatasetCollector(output_dir)
    
    records_collected = 0
    
    # Process specific repository if provided
    if args.repo and args.commit_before and args.commit_after and args.file_path:
        record = collector.process_repository(
            args.repo,
            args.commit_before,
            args.commit_after,
            args.file_path
        )
        
        if record:
            graph_before, graph_after = collector.generate_cpg_graphs(record)
            collector.save_record(record, graph_before, graph_after)
            records_collected += 1
    
    # Collect from GitHub
    github_records = collector.collect_from_github_cve(args.github_query, args.limit)
    # Process GitHub records...
    
    # Collect from CVE database
    cve_records = collector.collect_from_cve_database(args.cve_list)
    # Process CVE records...
    
    print(f"\nCollected {records_collected} vulnerability record(s)")
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()

