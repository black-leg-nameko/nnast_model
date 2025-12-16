#!/usr/bin/env python3
"""
Reclassify vulnerability types in existing dataset using improved classification logic.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from data.github_api import GitHubAPIClient


def reclassify_dataset(dataset_dir: Path, output_file: Optional[Path] = None):
    """Reclassify vulnerability types in existing metadata."""
    metadata_file = dataset_dir / "metadata.jsonl"
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return
    
    client = GitHubAPIClient()
    
    records = []
    with open(metadata_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Reclassifying {len(records)} records...")
    
    reclassified = 0
    for record in records:
        old_type = record.get("vulnerability_type")
        description = record.get("description", "")
        
        # Reclassify using improved logic
        new_type = client._extract_vulnerability_type(description)
        
        if new_type and new_type != old_type:
            record["vulnerability_type"] = new_type
            record["vulnerability_type_original"] = old_type  # Keep original
            reclassified += 1
    
    # Save reclassified metadata
    if output_file is None:
        output_file = dataset_dir / "metadata_reclassified.jsonl"
    
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Reclassified {reclassified} records")
    print(f"Saved to: {output_file}")
    
    # Show new distribution
    from collections import Counter
    new_types = Counter(r.get("vulnerability_type") for r in records)
    print("\nNew vulnerability type distribution:")
    for vtype, count in new_types.most_common():
        print(f"  {vtype or 'Unknown'}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reclassify vulnerability types")
    parser.add_argument(
        "--dataset-dir",
        default="./dataset",
        help="Dataset directory"
    )
    parser.add_argument(
        "--output",
        help="Output file (default: metadata_reclassified.jsonl)"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_file = Path(args.output) if args.output else None
    
    reclassify_dataset(dataset_dir, output_file)

