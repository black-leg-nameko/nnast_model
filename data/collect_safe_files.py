#!/usr/bin/env python3
"""
Collect safe Python files (without known vulnerabilities) from open source projects.

This script collects Python files that are:
1. From well-maintained open source projects
2. Not known to have vulnerabilities
3. Similar in structure and complexity to vulnerable files

The collected files will be used as negative samples (label: 0) in training.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional
from datetime import datetime
import random

from data.github_api import GitHubAPIClient


class SafeFileCollector:
    """Collects safe Python files from open source projects."""
    
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.raw_data_dir = self.output_dir / "raw"
        self.processed_data_dir = self.output_dir / "processed"
        self.metadata_file = self.output_dir / "metadata_safe.jsonl"
        
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
    
    def collect_from_popular_repos(
        self,
        limit: int = 100,
        min_file_size: int = 100,  # Minimum file size in bytes
        max_file_size: int = 50000,  # Maximum file size in bytes
    ) -> List[Dict]:
        """
        Collect safe files from popular Python repositories.
        
        Args:
            limit: Maximum number of files to collect
            min_file_size: Minimum file size in bytes
            max_file_size: Maximum file size in bytes
            
        Returns:
            List of safe file records
        """
        import os
        
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("Warning: GITHUB_TOKEN not set. GitHub collection will be skipped.")
            print("Set GITHUB_TOKEN environment variable to enable GitHub API access.")
            return []
        
        print(f"Collecting safe Python files from popular repositories...")
        print(f"Limit: {limit}, Size range: {min_file_size}-{max_file_size} bytes")
        
        # Initialize GitHub API client
        client = GitHubAPIClient(token)
        
        # Popular Python repositories (well-maintained, no known major vulnerabilities)
        popular_repos = [
            "python/cpython",
            "django/django",
            "flask/flask",
            "pallets/werkzeug",
            "pytest-dev/pytest",
            "psf/requests",
            "python-pillow/Pillow",
            "pydata/pandas",
            "numpy/numpy",
            "scikit-learn/scikit-learn",
        ]
        
        records = []
        collected = 0
        
        for repo_full_name in popular_repos:
            if collected >= limit:
                break
            
            print(f"\nProcessing repository: {repo_full_name}")
            
            try:
                # Get repository files
                repo_files = self._get_python_files_from_repo(
                    client, repo_full_name, min_file_size, max_file_size
                )
                
                # Sample files (avoid collecting too many from one repo)
                sample_size = min(10, len(repo_files), limit - collected)
                sampled_files = random.sample(repo_files, sample_size) if len(repo_files) > sample_size else repo_files
                
                for file_info in sampled_files:
                    if collected >= limit:
                        break
                    
                    record = self._create_safe_file_record(
                        repo_full_name, file_info
                    )
                    
                    if record:
                        records.append(record)
                        collected += 1
                        print(f"  Collected {collected}/{limit}: {file_info['path']}")
                
            except Exception as e:
                print(f"  Error processing {repo_full_name}: {e}")
                continue
        
        return records
    
    def _get_python_files_from_repo(
        self,
        client: GitHubAPIClient,
        repo_full_name: str,
        min_file_size: int,
        max_file_size: int
    ) -> List[Dict]:
        """Get Python files from a repository that match size criteria."""
        files = []
        
        try:
            # Search for Python files in the repository
            # Note: This is a simplified approach. In production, you might want
            # to use GitHub's Contents API to traverse the repository tree.
            search_query = f"repo:{repo_full_name} language:python extension:py"
            
            # Use GitHub API to search for files
            # For now, we'll use a simpler approach: clone and scan
            files = self._scan_repo_for_files(
                repo_full_name, min_file_size, max_file_size
            )
            
        except Exception as e:
            print(f"    Error getting files from {repo_full_name}: {e}")
        
        return files
    
    def _scan_repo_for_files(
        self,
        repo_full_name: str,
        min_file_size: int,
        max_file_size: int
    ) -> List[Dict]:
        """Scan a cloned repository for Python files."""
        files = []
        
        # Clone repository to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_url = f"https://github.com/{repo_full_name}.git"
            temp_path = pathlib.Path(temp_dir)
            
            try:
                # Clone repository
                subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(temp_path)],
                    check=True,
                    capture_output=True,
                    timeout=60
                )
                
                # Find Python files
                for py_file in temp_path.rglob("*.py"):
                    # Skip test files and examples (optional)
                    if any(skip in str(py_file) for skip in ["test_", "_test.py", "tests/", "examples/"]):
                        continue
                    
                    file_size = py_file.stat().st_size
                    if min_file_size <= file_size <= max_file_size:
                        # Read file content
                        try:
                            content = py_file.read_text(encoding="utf-8")
                            # Skip if file is too simple (just imports, etc.)
                            if len(content.splitlines()) < 10:
                                continue
                            
                            files.append({
                                "path": str(py_file.relative_to(temp_path)),
                                "size": file_size,
                                "content": content,
                            })
                        except UnicodeDecodeError:
                            continue
                
            except subprocess.TimeoutExpired:
                print(f"    Timeout cloning {repo_full_name}")
            except subprocess.CalledProcessError as e:
                print(f"    Error cloning {repo_full_name}: {e}")
            except Exception as e:
                print(f"    Unexpected error: {e}")
        
        return files
    
    def _create_safe_file_record(
        self,
        repo_full_name: str,
        file_info: Dict
    ) -> Optional[Dict]:
        """Create a record for a safe file."""
        record = {
            "repo_url": f"https://github.com/{repo_full_name}",
            "file_path": file_info["path"],
            "commit": "HEAD",  # Latest commit
            "label": 0,  # Safe (non-vulnerable)
            "vulnerability_type": "none",
            "cve_id": None,
            "cwe_id": None,
            "description": "Safe file from well-maintained open source project",
            "code": file_info["content"],
            "collected_at": datetime.now().isoformat(),
        }
        
        return record
    
    def generate_cpg_graph(self, code: str, output_path: pathlib.Path) -> Optional[Dict]:
        """Generate CPG graph for code."""
        import subprocess
        import tempfile
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file_path = tmp_file.name
            
            # Generate CPG
            result = subprocess.run(
                [sys.executable, "-m", "cli", tmp_file_path, "--out", str(output_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            pathlib.Path(tmp_file_path).unlink()
            
            if result.returncode == 0 and output_path.exists():
                with open(output_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            return json.loads(line)
            
        except Exception as e:
            print(f"  Warning: Failed to generate CPG: {e}")
        
        return None
    
    def save_record(self, record: Dict, graph: Optional[Dict]):
        """Save safe file record and CPG graph."""
        # Save metadata
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Save CPG graph
        if graph:
            file_stem = pathlib.Path(record["file_path"]).stem
            repo_hash = hash(record["repo_url"]) % 1000000
            graph_file = self.processed_data_dir / f"safe_{repo_hash}_{file_stem}.jsonl"
            
            with open(graph_file, "w") as f:
                f.write(json.dumps(graph, ensure_ascii=False) + "\n")
            
            record["graph_file"] = str(graph_file.relative_to(self.output_dir))
    
    def collect(
        self,
        limit: int = 100,
        generate_cpg: bool = True,
        min_file_size: int = 100,
        max_file_size: int = 50000,
    ):
        """Collect safe files."""
        print("=" * 60)
        print("Safe File Collection")
        print("=" * 60)
        
        # Collect from popular repositories
        records = self.collect_from_popular_repos(
            limit=limit,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
        )
        
        print(f"\nCollected {len(records)} safe files")
        
        # Generate CPG graphs
        if generate_cpg:
            print("\nGenerating CPG graphs...")
            for i, record in enumerate(records, 1):
                print(f"  [{i}/{len(records)}] Processing {record['file_path']}")
                
                graph = self.generate_cpg_graph(
                    record["code"],
                    self.processed_data_dir / f"temp_{i}.jsonl"
                )
                
                self.save_record(record, graph)
        
        print(f"\n✓ Collection complete!")
        print(f"  Metadata: {self.metadata_file}")
        print(f"  Processed graphs: {self.processed_data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect safe Python files for training dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for collected dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of safe files to collect"
    )
    parser.add_argument(
        "--no-cpg",
        action="store_true",
        help="Skip CPG graph generation"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum file size in bytes"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=50000,
        help="Maximum file size in bytes"
    )
    
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    collector = SafeFileCollector(output_dir)
    
    collector.collect(
        limit=args.limit,
        generate_cpg=not args.no_cpg,
        min_file_size=args.min_size,
        max_file_size=args.max_size,
    )


if __name__ == "__main__":
    main()

