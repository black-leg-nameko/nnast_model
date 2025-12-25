#!/usr/bin/env python3
"""
Prepare training data from synthetic dataset.

This script processes synthetic dataset metadata and prepares it for NNAST model training.
"""
import argparse
import json
import pathlib
import random
from typing import Dict, List
from collections import Counter


def load_synthetic_metadata(metadata_file: pathlib.Path) -> List[Dict]:
    """Load synthetic dataset metadata."""
    records = []
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_cpg_graph(graph_path: pathlib.Path) -> Dict:
    """Load CPG graph from JSON file."""
    if not graph_path.exists():
        return None
    with open(graph_path, 'r') as f:
        return json.load(f)


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """Split dataset into train/val/test sets."""
    random.seed(seed)
    random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples


def save_jsonl(data: List[Dict], file_path: pathlib.Path):
    """Save data to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from synthetic dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing synthetic dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="./training_data",
        help="Output directory for training data"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_file = dataset_dir / "metadata.jsonl"
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return 1
    
    print("Loading metadata...")
    records = load_synthetic_metadata(metadata_file)
    print(f"Loaded {len(records)} records")
    
    # Create training samples
    print("Creating training samples...")
    samples = []
    
    for record in records:
        graph_path = pathlib.Path(record.get('graph_path', ''))
        if not graph_path or not graph_path.exists():
            print(f"  Warning: Graph not found for {record.get('file_path')}, skipping")
            continue
        
        graph = load_cpg_graph(graph_path)
        if not graph:
            print(f"  Warning: Failed to load graph from {graph_path}, skipping")
            continue
        
        sample = {
            "graph": graph,
            "label": record.get("label", 0),
            "metadata": {
                "pattern_id": record.get("pattern_id"),
                "sample_type": record.get("sample_type"),
                "framework": record.get("framework"),
                "complexity": record.get("complexity"),
                "file_path": record.get("file_path"),
            }
        }
        samples.append(sample)
    
    print(f"Created {len(samples)} training samples")
    
    # Split dataset
    print("Splitting dataset...")
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    # Save graphs
    print("Saving graphs...")
    save_jsonl([s["graph"] for s in train_samples], output_dir / "train_graphs.jsonl")
    save_jsonl([s["graph"] for s in val_samples], output_dir / "val_graphs.jsonl")
    save_jsonl([s["graph"] for s in test_samples], output_dir / "test_graphs.jsonl")
    
    # Save labels
    print("Saving labels...")
    train_labels = [{"label": s["label"], "metadata": s["metadata"]} for s in train_samples]
    val_labels = [{"label": s["label"], "metadata": s["metadata"]} for s in val_samples]
    test_labels = [{"label": s["label"], "metadata": s["metadata"]} for s in test_samples]
    
    save_jsonl(train_labels, output_dir / "train_labels.jsonl")
    save_jsonl(val_labels, output_dir / "val_labels.jsonl")
    save_jsonl(test_labels, output_dir / "test_labels.jsonl")
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "label_distribution": {
            "train": Counter(s["label"] for s in train_samples),
            "val": Counter(s["label"] for s in val_samples),
            "test": Counter(s["label"] for s in test_samples),
        },
        "pattern_distribution": {
            "train": Counter(s["metadata"]["pattern_id"] for s in train_samples),
            "val": Counter(s["metadata"]["pattern_id"] for s in val_samples),
            "test": Counter(s["metadata"]["pattern_id"] for s in test_samples),
        },
        "framework_distribution": {
            "train": Counter(s["metadata"]["framework"] for s in train_samples),
            "val": Counter(s["metadata"]["framework"] for s in val_samples),
            "test": Counter(s["metadata"]["framework"] for s in test_samples),
        },
    }
    
    # Convert Counter to dict for JSON serialization
    for key in ["label_distribution", "pattern_distribution", "framework_distribution"]:
        for split in ["train", "val", "test"]:
            stats[key][split] = dict(stats[key][split])
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Training data preparation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

