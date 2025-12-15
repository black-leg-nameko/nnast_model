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
            # Try to generate CPG graph on the fly
            return self._generate_cpg_on_fly(record, version)
        
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
    
    def _generate_cpg_on_fly(self, record: Dict, version: str) -> Optional[Dict]:
        """Generate CPG graph on the fly if not found."""
        # This would call the CPG generation CLI
        # For now, return None
        return None
    
    def create_training_samples(self, records: List[Dict]) -> List[Dict]:
        """
        Create training samples from vulnerability records.
        
        Each sample contains:
        - graph: CPG graph
        - label: 1 for vulnerable, 0 for fixed
        - metadata: CVE ID, CWE ID, vulnerability type, etc.
        """
        samples = []
        
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
                    }
                })
            
            # Load fixed code graph (label: 0)
            graph_after = self.load_cpg_graphs(record, "after")
            if graph_after:
                samples.append({
                    "graph": graph_after,
                    "label": 0,  # Fixed (non-vulnerable)
                    "metadata": {
                        "cve_id": record.get("cve_id"),
                        "cwe_id": record.get("cwe_id"),
                        "vulnerability_type": record.get("vulnerability_type"),
                        "repo_url": record.get("repo_url"),
                        "commit": record.get("commit_after"),
                        "file_path": record.get("file_path"),
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
    
    def prepare(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Prepare training data from dataset."""
        print("Loading metadata...")
        records = self.load_metadata()
        print(f"Loaded {len(records)} vulnerability records")
        
        print("Creating training samples...")
        samples = self.create_training_samples(records)
        print(f"Created {len(samples)} training samples")
        
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
        description="Prepare training data from vulnerability dataset"
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
    
    args = parser.parse_args()
    
    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    preparer = TrainingDataPreparer(dataset_dir, output_dir)
    preparer.prepare(args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()

