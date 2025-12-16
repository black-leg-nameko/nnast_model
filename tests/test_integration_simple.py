#!/usr/bin/env python3
"""
Simple integration test to verify the code works end-to-end.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import shutil


def test_collect_safe_files_help():
    """Test that collect_safe_files script can be called with --help."""
    result = subprocess.run(
        [sys.executable, "-m", "data.collect_safe_files", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Should show help message"
    assert "collect safe Python files" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_prepare_training_data_help():
    """Test that prepare_training_data script can be called with --help."""
    result = subprocess.run(
        [sys.executable, "-m", "data.prepare_training_data", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Should show help message"
    assert "prepare training data" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_prepare_training_data_with_mock_data():
    """Test prepare_training_data with minimal mock data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = pathlib.Path(temp_dir) / "dataset"
        output_dir = pathlib.Path(temp_dir) / "training_data"
        
        dataset_dir.mkdir()
        (dataset_dir / "processed").mkdir()
        
        # Create minimal vulnerability metadata
        metadata_file = dataset_dir / "metadata.jsonl"
        vuln_record = {
            "commit_before": "abc123",
            "commit_after": "def456",
            "file_path": "test.py",
            "cve_id": "CVE-2023-0001",
            "cwe_id": "CWE-79",
            "vulnerability_type": "XSS",
            "repo_url": "https://github.com/test/repo"
        }
        with open(metadata_file, "w") as f:
            f.write(json.dumps(vuln_record) + "\n")
        
        # Create minimal CPG graphs
        graph_before = {"file": "test.py", "nodes": [{"id": 1, "kind": "Function"}], "edges": []}
        graph_after = {"file": "test.py", "nodes": [{"id": 1, "kind": "Function"}], "edges": []}
        
        with open(dataset_dir / "processed" / "abc123_test_before.jsonl", "w") as f:
            f.write(json.dumps(graph_before) + "\n")
        with open(dataset_dir / "processed" / "def456_test_after.jsonl", "w") as f:
            f.write(json.dumps(graph_after) + "\n")
        
        # Run prepare_training_data
        result = subprocess.run(
            [sys.executable, "-m", "data.prepare_training_data",
             "--dataset-dir", str(dataset_dir),
             "--output-dir", str(output_dir),
             "--no-safe-files"],  # Skip safe files for this simple test
            capture_output=True,
            text=True
        )
        
        # Check that it completed successfully
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Check output files exist
        assert (output_dir / "train_graphs.jsonl").exists()
        assert (output_dir / "train_labels.jsonl").exists()
        assert (output_dir / "dataset_stats.json").exists()
        
        # Check that we have some data
        with open(output_dir / "dataset_stats.json") as f:
            stats = json.load(f)
        assert stats["total_samples"] > 0


if __name__ == "__main__":
    print("Running simple integration tests...")
    
    try:
        test_collect_safe_files_help()
        print("✓ test_collect_safe_files_help passed")
    except Exception as e:
        print(f"✗ test_collect_safe_files_help failed: {e}")
        sys.exit(1)
    
    try:
        test_prepare_training_data_help()
        print("✓ test_prepare_training_data_help passed")
    except Exception as e:
        print(f"✗ test_prepare_training_data_help failed: {e}")
        sys.exit(1)
    
    try:
        test_prepare_training_data_with_mock_data()
        print("✓ test_prepare_training_data_with_mock_data passed")
    except Exception as e:
        print(f"✗ test_prepare_training_data_with_mock_data failed: {e}")
        sys.exit(1)
    
    print("\nAll integration tests passed!")

