#!/usr/bin/env python3
"""
Prepare training data from collected vulnerability dataset.

This script processes collected vulnerability records and prepares them
for NNAST model training by:
1. Generating CPG graphs for all code samples
2. Creating labels (vulnerable: 1, fixed: 0)
3. Splitting into train/val/test sets
4. Generating dataset statistics
"""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random


class TrainingDataPreparer:
    """Prepares training data from vulnerability dataset."""
    
    def __init__(self, dataset_dir: pathlib.Path, output_dir: pathlib.Path):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.train_graphs = self.output_dir / "train_graphs.jsonl"
        self.val_graphs = self.output_dir / "val_graphs.jsonl"
        self.test_graphs = self.output_dir / "test_graphs.jsonl"
        self.train_labels = self.output_dir / "train_labels.jsonl"
        self.val_labels = self.output_dir / "val_labels.jsonl"
        self.test_labels = self.output_dir / "test_labels.jsonl"
        self.stats_file = self.output_dir / "dataset_stats.json"
    
    def load_metadata(self) -> List[Dict]:
        """Load vulnerability metadata."""
        metadata_file = self.dataset_dir / "metadata.jsonl"
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found: {metadata_file}")
            return []
        
        records = []
        with open(metadata_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def load_safe_files_metadata(self) -> List[Dict]:
        """Load safe files metadata."""
        metadata_file = self.dataset_dir / "metadata_safe.jsonl"
        if not metadata_file.exists():
            print(f"Info: Safe files metadata not found: {metadata_file}")
            print("  Run 'python -m data.collect_safe_files' to collect safe files")
            return []
        
        records = []
        with open(metadata_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def load_cpg_graphs(self, record: Dict, version: str) -> Optional[Dict]:
        """
        Load CPG graph for a specific version.
        
        Args:
            record: Vulnerability metadata record
            version: "before" or "after"
        """
        processed_dir = self.dataset_dir / "processed"
        commit_hash = record[f"commit_{version}"]
        file_stem = pathlib.Path(record["file_path"]).stem
        graph_file = processed_dir / f"{commit_hash}_{file_stem}_{version}.jsonl"
        
        if not graph_file.exists():
            # Try to generate CPG graph on the fly from code file
            code_file_key = f"code_{version}_file"
            if code_file_key in record:
                code_file = self.dataset_dir / record[code_file_key]
                if code_file.exists():
                    return self._generate_cpg_from_file(code_file)
            return None
        
        with open(graph_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    graph = json.loads(line)
                    return graph
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _generate_cpg_from_file(self, code_file: pathlib.Path) -> Optional[Dict]:
        """Generate CPG graph from code file."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "cli", str(code_file), "--out", "/dev/stdout"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    return json.loads(lines[0])
        except Exception as e:
            print(f"  Warning: Failed to generate CPG from {code_file}: {e}")
        
        return None
    
    def _generate_cpg_on_fly(self, record: Dict, version: str) -> Optional[Dict]:
        """Generate CPG graph on the fly if not found."""
        # This would call the CPG generation CLI
        # For now, return None
        return None
    
    def load_safe_file_graph(self, record: Dict) -> Optional[Dict]:
        """Load CPG graph for a safe file."""
        # Check if graph file is specified in record
        if "graph_file" in record:
            graph_file = self.dataset_dir / record["graph_file"]
            if graph_file.exists():
                with open(graph_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                return json.loads(line)
                            except json.JSONDecodeError:
                                continue
        
        # Try to generate from code
        if "code" in record:
            return self._generate_cpg_from_code(record["code"])
        
        return None
    
    def _generate_cpg_from_code(self, code: str) -> Optional[Dict]:
        """Generate CPG graph from code string."""
        import subprocess
        import sys
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file_path = tmp_file.name
            
            result = subprocess.run(
                [sys.executable, "-m", "cli", tmp_file_path, "--out", "/dev/stdout"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            pathlib.Path(tmp_file_path).unlink()
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    return json.loads(lines[0])
        except Exception as e:
            print(f"  Warning: Failed to generate CPG from code: {e}")
        
        return None
    
    def create_training_samples(self, records: List[Dict], safe_records: List[Dict] = None) -> List[Dict]:
        """
        Create training samples from vulnerability records and safe files.
        
        Each sample contains:
        - graph: CPG graph
        - label: 1 for vulnerable, 0 for safe (fixed or originally safe)
        - metadata: CVE ID, CWE ID, vulnerability type, etc.
        
        Args:
            records: Vulnerability records (before/after pairs)
            safe_records: Safe file records (originally safe files)
        """
        samples = []
        
        # Process vulnerability records
        for record in records:
            # Load vulnerable code graph (label: 1)
            graph_before = self.load_cpg_graphs(record, "before")
            if graph_before:
                samples.append({
                    "graph": graph_before,
                    "label": 1,  # Vulnerable
                    "metadata": {
                        "cve_id": record.get("cve_id"),
                        "cwe_id": record.get("cwe_id"),
                        "vulnerability_type": record.get("vulnerability_type"),
                        "repo_url": record.get("repo_url"),
                        "commit": record.get("commit_before"),
                        "file_path": record.get("file_path"),
                        "sample_type": "vulnerable",
                    }
                })
            
            # Load fixed code graph (label: 0)
            graph_after = self.load_cpg_graphs(record, "after")
            if graph_after:
                samples.append({
                    "graph": graph_after,
                    "label": 0,  # Safe (fixed)
                    "metadata": {
                        "cve_id": record.get("cve_id"),
                        "cwe_id": record.get("cwe_id"),
                        "vulnerability_type": record.get("vulnerability_type"),
                        "repo_url": record.get("repo_url"),
                        "commit": record.get("commit_after"),
                        "file_path": record.get("file_path"),
                        "sample_type": "fixed",
                    }
                })
        
        # Process safe files (originally safe, not fixed)
        if safe_records:
            print(f"Processing {len(safe_records)} safe files...")
            for record in safe_records:
                graph = self.load_safe_file_graph(record)
                if graph:
                    samples.append({
                        "graph": graph,
                        "label": 0,  # Safe (originally safe)
                        "metadata": {
                            "cve_id": None,
                            "cwe_id": None,
                            "vulnerability_type": "none",
                            "repo_url": record.get("repo_url"),
                            "commit": record.get("commit", "HEAD"),
                            "file_path": record.get("file_path"),
                            "sample_type": "originally_safe",
                        }
                    })
        
        return samples
    
    def split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets."""
        random.seed(seed)
        random.shuffle(samples)
        
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        return train_samples, val_samples, test_samples
    
    def _balance_dataset(self, samples: List[Dict], safe_ratio: float = 0.3) -> List[Dict]:
        """
        Balance dataset to have target ratio of safe files.
        
        Args:
            samples: All training samples
            safe_ratio: Target ratio of safe files (originally_safe) to total
            
        Returns:
            Balanced samples list
        """
        # Separate samples by type
        vulnerable_samples = [s for s in samples if s["label"] == 1]
        safe_samples = [s for s in samples if s["label"] == 0]
        originally_safe = [s for s in safe_samples if s["metadata"].get("sample_type") == "originally_safe"]
        fixed_samples = [s for s in safe_samples if s["metadata"].get("sample_type") == "fixed"]
        
        print(f"  Vulnerable: {len(vulnerable_samples)}")
        print(f"  Fixed: {len(fixed_samples)}")
        print(f"  Originally safe: {len(originally_safe)}")
        
        # If no originally safe samples, return as is
        if len(originally_safe) == 0:
            print("  No originally safe samples to balance")
            balanced_samples = vulnerable_samples + fixed_samples
            random.shuffle(balanced_samples)
            return balanced_samples
        
        # Calculate target number of originally safe samples
        # We want: originally_safe / total ≈ safe_ratio
        # So: originally_safe ≈ safe_ratio * (vulnerable + fixed + originally_safe)
        # Solving: originally_safe ≈ safe_ratio * (vulnerable + fixed) / (1 - safe_ratio)
        other_samples_count = len(vulnerable_samples) + len(fixed_samples)
        
        if other_samples_count == 0:
            # Only safe samples, return as is
            balanced_samples = originally_safe
            random.shuffle(balanced_samples)
            return balanced_samples
        
        target_safe = max(1, int(safe_ratio * other_samples_count / (1 - safe_ratio)))
        
        # Sample or duplicate originally safe files to reach target
        if len(originally_safe) < target_safe:
            # Duplicate some samples
            needed = target_safe - len(originally_safe)
            if len(originally_safe) > 0:
                duplicated = random.choices(originally_safe, k=needed)
                originally_safe.extend(duplicated)
                print(f"  Duplicated {needed} safe samples to reach target ratio")
            else:
                print(f"  Warning: Cannot duplicate, no originally safe samples available")
        elif len(originally_safe) > target_safe:
            # Sample down, but keep at least 1 if we have any
            if target_safe > 0:
                originally_safe = random.sample(originally_safe, target_safe)
                print(f"  Sampled down to {target_safe} safe samples")
            else:
                # Keep at least 1 if we have any
                originally_safe = [originally_safe[0]] if originally_safe else []
                print(f"  Kept 1 safe sample (target was 0)")
        
        # Combine all samples
        balanced_samples = vulnerable_samples + fixed_samples + originally_safe
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def save_dataset(self, samples: List[Dict], graphs_file: pathlib.Path, labels_file: pathlib.Path):
        """Save dataset to JSONL files."""
        with open(graphs_file, "w") as f_graphs, open(labels_file, "w") as f_labels:
            for sample in samples:
                # Save graph
                f_graphs.write(json.dumps(sample["graph"], ensure_ascii=False) + "\n")
                
                # Save label and metadata
                label_data = {
                    "label": sample["label"],
                    "metadata": sample["metadata"]
                }
                f_labels.write(json.dumps(label_data, ensure_ascii=False) + "\n")
    
    def generate_statistics(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]) -> Dict:
        """Generate dataset statistics."""
        all_samples = train_samples + val_samples + test_samples
        
        # Count labels
        label_counts = Counter(s["label"] for s in all_samples)
        
        # Count vulnerability types
        vuln_types = Counter(
            s["metadata"].get("vulnerability_type", "Unknown")
            for s in all_samples
            if s["label"] == 1
        )
        
        # Count CWE types
        cwe_types = Counter(
            s["metadata"].get("cwe_id")
            for s in all_samples
            if s["label"] == 1 and s["metadata"].get("cwe_id")
        )
        
        stats = {
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "label_distribution": dict(label_counts),
            "vulnerability_types": dict(vuln_types),
            "cwe_distribution": dict(cwe_types),
            "vulnerable_samples": label_counts[1],
            "fixed_samples": label_counts[0],
            "class_balance": {
                "vulnerable_ratio": label_counts[1] / len(all_samples) if all_samples else 0,
                "fixed_ratio": label_counts[0] / len(all_samples) if all_samples else 0,
            }
        }
        
        return stats
    
    def prepare(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        include_safe_files: bool = True,
        safe_file_ratio: float = 0.3,  # Ratio of safe files to total samples
    ):
        """
        Prepare training data from dataset.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            include_safe_files: Whether to include safe files in dataset
            safe_file_ratio: Target ratio of safe files (originally safe) to total samples
        """
        print("Loading metadata...")
        records = self.load_metadata()
        print(f"Loaded {len(records)} vulnerability records")
        
        # Load safe files if requested
        safe_records = []
        if include_safe_files:
            safe_records = self.load_safe_files_metadata()
            print(f"Loaded {len(safe_records)} safe file records")
        
        print("Creating training samples...")
        samples = self.create_training_samples(records, safe_records)
        print(f"Created {len(samples)} training samples")
        
        # Balance dataset if needed
        if include_safe_files and safe_records:
            samples = self._balance_dataset(samples, safe_file_ratio)
            print(f"After balancing: {len(samples)} training samples")
        
        print("Splitting dataset...")
        train_samples, val_samples, test_samples = self.split_dataset(
            samples, train_ratio, val_ratio, test_ratio
        )
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        print("Saving datasets...")
        self.save_dataset(train_samples, self.train_graphs, self.train_labels)
        self.save_dataset(val_samples, self.val_graphs, self.val_labels)
        self.save_dataset(test_samples, self.test_graphs, self.test_labels)
        
        print("Generating statistics...")
        stats = self.generate_statistics(train_samples, val_samples, test_samples)
        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset preparation complete!")
        print(f"Statistics saved to: {self.stats_file}")
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Vulnerable: {stats['vulnerable_samples']} ({stats['class_balance']['vulnerable_ratio']*100:.1f}%)")
        print(f"  Fixed: {stats['fixed_samples']} ({stats['class_balance']['fixed_ratio']*100:.1f}%)")
        print(f"  Vulnerability types: {len(stats['vulnerability_types'])}")
        print(f"  CWE types: {len(stats['cwe_distribution'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from vulnerability dataset and safe files"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing collected vulnerability dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="./training_data",
        help="Output directory for prepared training data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--no-safe-files",
        action="store_true",
        help="Exclude safe files from dataset (use only vulnerable/fixed pairs)"
    )
    parser.add_argument(
        "--safe-file-ratio",
        type=float,
        default=0.3,
        help="Target ratio of originally safe files to total samples (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    preparer = TrainingDataPreparer(dataset_dir, output_dir)
    preparer.prepare(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        include_safe_files=not args.no_safe_files,
        safe_file_ratio=args.safe_file_ratio,
    )


if __name__ == "__main__":
    main()

