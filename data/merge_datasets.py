#!/usr/bin/env python3
"""
Merge real and synthetic datasets for training.
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict


def merge_datasets(
    real_dir: Path,
    synthetic_dir: Path,
    output_dir: Path
):
    """Merge real and synthetic datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    merged_metadata = output_dir / "metadata.jsonl"
    merged_code_dir = output_dir / "code"
    merged_processed_dir = output_dir / "processed"
    
    merged_code_dir.mkdir(exist_ok=True)
    merged_processed_dir.mkdir(exist_ok=True)
    
    # Load real dataset metadata
    real_metadata_file = real_dir / "metadata.jsonl"
    real_records = []
    if real_metadata_file.exists():
        with open(real_metadata_file) as f:
            for line in f:
                if line.strip():
                    real_records.append(json.loads(line))
    
    # Load synthetic dataset metadata
    synthetic_metadata_file = synthetic_dir / "metadata.jsonl"
    synthetic_records = []
    if synthetic_metadata_file.exists():
        with open(synthetic_metadata_file) as f:
            for line in f:
                if line.strip():
                    synthetic_records.append(json.loads(line))
    
    print(f"Real dataset: {len(real_records)} records")
    print(f"Synthetic dataset: {len(synthetic_records)} records")
    
    # Merge metadata
    merged_records = real_records + synthetic_records
    
    # Copy code files
    real_code_dir = real_dir / "code"
    synthetic_code_dir = synthetic_dir / "code"
    
    if real_code_dir.exists():
        for code_file in real_code_dir.glob("*.py"):
            shutil.copy2(code_file, merged_code_dir / code_file.name)
    
    if synthetic_code_dir.exists():
        for code_file in synthetic_code_dir.glob("*.py"):
            shutil.copy2(code_file, merged_code_dir / code_file.name)
    
    # Copy CPG graphs
    real_processed_dir = real_dir / "processed"
    synthetic_processed_dir = synthetic_dir / "processed"
    
    if real_processed_dir.exists():
        for graph_file in real_processed_dir.glob("*.jsonl"):
            shutil.copy2(graph_file, merged_processed_dir / graph_file.name)
    
    if synthetic_processed_dir.exists():
        for graph_file in synthetic_processed_dir.glob("*.jsonl"):
            shutil.copy2(graph_file, merged_processed_dir / graph_file.name)
    
    # Save merged metadata
    with open(merged_metadata, "w") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nMerged dataset: {len(merged_records)} records")
    print(f"Output directory: {output_dir}")
    
    # Show distribution
    from collections import Counter
    types = Counter(r.get("vulnerability_type") for r in merged_records)
    synthetic_count = sum(1 for r in merged_records if r.get("synthetic", False))
    
    print(f"\nSynthetic records: {synthetic_count}/{len(merged_records)} ({synthetic_count/len(merged_records)*100:.1f}%)")
    print("\nVulnerability type distribution:")
    for vtype, count in types.most_common():
        print(f"  {vtype or 'Unknown'}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge real and synthetic datasets"
    )
    parser.add_argument(
        "--real-dir",
        required=True,
        help="Real dataset directory"
    )
    parser.add_argument(
        "--synthetic-dir",
        required=True,
        help="Synthetic dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset_merged",
        help="Output directory for merged dataset"
    )
    
    args = parser.parse_args()
    
    merge_datasets(
        Path(args.real_dir),
        Path(args.synthetic_dir),
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()

