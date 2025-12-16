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
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import time

from data.github_api import GitHubAPIClient, GitHubCommit, CodeDiff


def load_env_file(env_path: Optional[pathlib.Path] = None) -> None:
    """
    Load environment variables from .env file.
    
    Simple implementation without external dependencies.
    """
    if env_path is None:
        # Look for .env in project root
        project_root = pathlib.Path(__file__).parent.parent
        env_path = project_root / ".env"
    
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value


# Load .env file if it exists
load_env_file()


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
    
    def collect_from_github_cve(
        self,
        query: str = "CVE",
        language: str = "python",
        limit: int = 100
    ) -> List[VulnerabilityRecord]:
        """
        Collect Python vulnerabilities from GitHub using search API.
        
        Args:
            query: Search query (e.g., "CVE", "security fix", "vulnerability")
            language: Programming language filter
            limit: Maximum number of records to collect
            
        Returns:
            List of VulnerabilityRecord objects
        """
        import os
        
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("Warning: GITHUB_TOKEN not set. GitHub collection will be skipped.")
            print("Set GITHUB_TOKEN environment variable to enable GitHub API access.")
            return []
        
        print(f"Collecting Python vulnerabilities from GitHub...")
        print(f"Query: {query}, Language: {language}, Limit: {limit}")
        
        # Initialize GitHub API client
        client = GitHubAPIClient(token)
        
        # Search for CVE-related commits
        commits = client.search_commits(query=query, language=language, limit=limit)
        
        if not commits:
            print("No commits found.")
            return []
        
        records = []
        processed = 0
        
        for commit in commits:
            if processed >= limit:
                break
            
            print(f"\nProcessing commit {commit.sha[:8]} from {commit.repo_name}...")
            
            # Parse repository owner and name
            repo_parts = commit.repo_name.split("/")
            if len(repo_parts) != 2:
                print(f"  Warning: Invalid repo name format: {commit.repo_name}")
                continue
            
            owner, repo = repo_parts
            
            # Get commit details including file changes
            commit_details = client.get_commit_details(owner, repo, commit.sha)
            if not commit_details:
                print(f"  Warning: Could not get commit details")
                continue
            
            # Get parent commit (vulnerable version)
            parent_sha = client.get_parent_commit(owner, repo, commit.sha)
            if not parent_sha:
                print(f"  Warning: Could not get parent commit")
                continue
            
            # Get file changes
            diffs = client.get_commit_diff(owner, repo, commit.sha)
            if not diffs:
                print(f"  Warning: No Python file changes found")
                continue
            
            # Process each changed file
            for diff in diffs:
                if processed >= limit:
                    break
                
                file_path = diff.file_path
                print(f"  Processing file: {file_path}")
                
                # Get file content before fix (parent commit)
                code_before = client.get_file_content(owner, repo, file_path, parent_sha)
                if not code_before:
                    print(f"    Warning: Could not get file content before fix")
                    continue
                
                # Get file content after fix (current commit)
                code_after = client.get_file_content(owner, repo, file_path, commit.sha)
                if not code_after:
                    print(f"    Warning: Could not get file content after fix")
                    continue
                
                # Create vulnerability record
                record = VulnerabilityRecord(
                    cve_id=commit.cve_id,
                    cwe_id=commit.cwe_id,
                    repo_url=commit.repo_url,
                    commit_before=parent_sha,
                    commit_after=commit.sha,
                    file_path=file_path,
                    line_range_before=(1, len(code_before.splitlines())),
                    line_range_after=(1, len(code_after.splitlines())),
                    vulnerability_type=commit.vulnerability_type or "Unknown",
                    description=commit.message[:500],  # Truncate long messages
                    code_before=code_before,
                    code_after=code_after
                )
                
                records.append(record)
                processed += 1
                
                # Save record immediately
                self._save_record_immediate(record)
                
                # Small delay to respect rate limits
                time.sleep(0.5)
        
        print(f"\nCollected {len(records)} vulnerability records")
        return records
    
    def _save_record_immediate(self, record: VulnerabilityRecord):
        """Save a single record immediately (for progress tracking)."""
        # Save code files separately (they can be large)
        code_dir = self.output_dir / "code"
        code_dir.mkdir(exist_ok=True)
        
        # Save code before
        code_before_file = code_dir / f"{record.commit_before}_{pathlib.Path(record.file_path).name}_before.py"
        code_before_file.parent.mkdir(parents=True, exist_ok=True)
        code_before_file.write_text(record.code_before, encoding="utf-8")
        
        # Save code after
        code_after_file = code_dir / f"{record.commit_after}_{pathlib.Path(record.file_path).name}_after.py"
        code_after_file.parent.mkdir(parents=True, exist_ok=True)
        code_after_file.write_text(record.code_after, encoding="utf-8")
        
        # Save metadata (with file paths instead of full code)
        metadata = {
            "cve_id": record.cve_id,
            "cwe_id": record.cwe_id,
            "repo_url": record.repo_url,
            "commit_before": record.commit_before,
            "commit_after": record.commit_after,
            "file_path": record.file_path,
            "vulnerability_type": record.vulnerability_type,
            "description": record.description,
            "code_before_file": str(code_before_file.relative_to(self.output_dir)),
            "code_after_file": str(code_after_file.relative_to(self.output_dir)),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        
        # Generate and save CPG graphs
        graph_before, graph_after = self.generate_cpg_graphs(record)
        if graph_before:
            graph_file_before = self.processed_data_dir / f"{record.commit_before}_{pathlib.Path(record.file_path).stem}_before.jsonl"
            with open(graph_file_before, "w") as f:
                f.write(json.dumps(graph_before, ensure_ascii=False) + "\n")
        
        if graph_after:
            graph_file_after = self.processed_data_dir / f"{record.commit_after}_{pathlib.Path(record.file_path).stem}_after.jsonl"
            with open(graph_file_after, "w") as f:
                f.write(json.dumps(graph_after, ensure_ascii=False) + "\n")
    
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
        
        Uses GitHub API if available, otherwise falls back to git clone.
        
        Args:
            repo_url: GitHub repository URL
            commit_before: Commit hash of vulnerable version
            commit_after: Commit hash of fixed version
            file_path: Path to the vulnerable file
            
        Returns:
            VulnerabilityRecord if successful, None otherwise
        """
        import os
        
        print(f"Processing {repo_url} ({commit_before} -> {commit_after})")
        
        # Try GitHub API first
        token = os.getenv("GITHUB_TOKEN")
        if token:
            try:
                # Parse repo URL to get owner/repo
                # Format: https://github.com/owner/repo or git@github.com:owner/repo.git
                if "github.com" in repo_url:
                    parts = repo_url.replace("https://github.com/", "").replace("git@github.com:", "").replace(".git", "").split("/")
                    if len(parts) >= 2:
                        owner, repo = parts[0], parts[1]
                        
                        client = GitHubAPIClient(token)
                        
                        # Get commit messages for metadata
                        commit_details = client.get_commit_details(owner, repo, commit_after)
                        if commit_details:
                            message = commit_details.get("commit", {}).get("message", "")
                            cve_id = client._extract_cve_id(message)
                            cwe_id = client._extract_cwe_id(message)
                            vuln_type = client._extract_vulnerability_type(message)
                        else:
                            cve_id = cwe_id = vuln_type = None
                        
                        # Get file content
                        code_before = client.get_file_content(owner, repo, file_path, commit_before)
                        code_after = client.get_file_content(owner, repo, file_path, commit_after)
                        
                        if code_before and code_after:
                            record = VulnerabilityRecord(
                                cve_id=cve_id,
                                cwe_id=cwe_id,
                                repo_url=repo_url,
                                commit_before=commit_before,
                                commit_after=commit_after,
                                file_path=file_path,
                                line_range_before=(1, len(code_before.splitlines())),
                                line_range_after=(1, len(code_after.splitlines())),
                                vulnerability_type=vuln_type or "Unknown",
                                description=message[:500] if commit_details else "",
                                code_before=code_before,
                                code_after=code_after
                            )
                            return record
            except Exception as e:
                print(f"  Warning: GitHub API failed, falling back to git clone: {e}")
        
        # Fallback to git clone
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
            
            record = VulnerabilityRecord(
                cve_id=None,
                cwe_id=None,
                repo_url=repo_url,
                commit_before=commit_before,
                commit_after=commit_after,
                file_path=file_path,
                line_range_before=(1, len(code_before.splitlines())),
                line_range_after=(1, len(code_after.splitlines())),
                vulnerability_type="Unknown",
                description="",
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
        help="GitHub search query (e.g., 'CVE', 'security fix'). If not provided, GitHub collection is skipped."
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
    if args.github_query:
        github_records = collector.collect_from_github_cve(
            query=args.github_query,
            language="python",
            limit=args.limit
        )
        records_collected += len(github_records)
    
    # Collect from CVE database
    if args.cve_list:
        cve_records = collector.collect_from_cve_database(args.cve_list)
        records_collected += len(cve_records)
    
    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"Total records collected: {records_collected}")
    print(f"Dataset saved to: {output_dir}")
    print(f"  - Metadata: {collector.metadata_file}")
    print(f"  - CPG graphs: {collector.processed_data_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

