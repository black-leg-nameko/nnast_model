#!/usr/bin/env python3
"""
Run GNN model inference on Python files and output results as JSON.

This script loads a trained GNN model and runs inference on CPG graphs
generated from Python files, outputting vulnerability detection results.
"""
import argparse
import json
import sys
import pathlib
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ml.model import CPGTaintFlowModel
from ml.dataset import CPGGraphDataset
from ml.embed_codebert import CodeBERTEmbedder
from data.github_api import load_env_file
import subprocess
import tempfile


def collate_fn(batch):
    """Custom collate function for batching graphs and taint info."""
    graphs, taint_infos = zip(*batch)
    
    # Batch graphs using PyG's Batch
    batched_graphs = Batch.from_data_list(graphs)
    
    # Collect taint info
    batched_taint_info = list(taint_infos)
    
    return batched_graphs, batched_taint_info


def generate_cpg_from_file(file_path: Path) -> Optional[Dict]:
    """
    Generate CPG graph from Python file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        CPG graph as dict, or None on error
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        result = subprocess.run(
            [sys.executable, "-m", "cli", str(file_path), "--out", tmp_out_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            with open(tmp_out_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    graph_data = json.loads(lines[0])
                    Path(tmp_out_path).unlink(missing_ok=True)
                    return graph_data
        else:
            print(f"  Warning: CPG generation failed: {result.stderr}")
            Path(tmp_out_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"  Warning: Failed to generate CPG: {e}")
        try:
            Path(tmp_out_path).unlink(missing_ok=True)
        except:
            pass
    
    return None


def load_model(checkpoint_path: Path, device: torch.device) -> CPGTaintFlowModel:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model config from checkpoint
    # Priority: args (from training) > config > inferred from state_dict > default
    if "args" in checkpoint:
        # Get config from training args
        args = checkpoint["args"]
        config = {
            "input_dim": 768,  # CodeBERT embedding dimension
            "hidden_dim": args.get("hidden_dim", 256),
            "num_layers": args.get("num_layers", 3),
            "num_classes": args.get("num_classes", 2),
            "gnn_type": args.get("gnn_type", "GAT").upper(),
            "dropout": args.get("dropout", 0.5),
            "use_batch_norm": args.get("use_batch_norm", True),
            "use_dynamic_attention": args.get("use_dynamic_attention", True),
        }
    elif "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_state_dict" in checkpoint:
        # Infer config from state dict
        state_dict = checkpoint["model_state_dict"]
        # Try to infer input_dim from first layer
        first_layer_key = [k for k in state_dict.keys() if "input_proj" in k][0]
        input_dim = state_dict[first_layer_key].shape[1]
        config = {
            "input_dim": input_dim,
            "hidden_dim": 256,
            "num_layers": 3,
            "num_classes": 2,
            "gnn_type": "GAT",
            "dropout": 0.5,
            "use_batch_norm": True,
            "use_dynamic_attention": True,
        }
    else:
        # Default config
        config = {
            "input_dim": 768,
            "hidden_dim": 256,
            "num_layers": 3,
            "num_classes": 2,
            "gnn_type": "GAT",
            "dropout": 0.5,
            "use_batch_norm": True,
            "use_dynamic_attention": True,
        }
    
    model = CPGTaintFlowModel(**config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def run_inference(
    model: CPGTaintFlowModel,
    dataset: CPGGraphDataset,
    device: torch.device,
    batch_size: int = 32
) -> List[Dict]:
    """
    Run inference on dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to run inference on
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        List of inference results
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = []
    
    with torch.no_grad():
        for batch_graphs, batch_info in tqdm(dataloader, desc="Running inference"):
            batch_graphs = batch_graphs.to(device)
            
            # Forward pass
            logits = model(batch_graphs)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            preds = logits.argmax(dim=1)
            confidences = probs.max(dim=1)[0]
            
            # Get batch size (number of graphs in this batch)
            batch_size_actual = batch_graphs.batch.max().item() + 1
            
            # Process each graph in batch
            for i in range(batch_size_actual):
                # Mask for nodes belonging to graph i
                graph_mask = batch_graphs.batch == i

                # Get file path from graph data
                file_path = ""
                if hasattr(batch_graphs, "file"):
                    if isinstance(batch_graphs.file, (list, tuple)):
                        file_path = batch_graphs.file[i] if i < len(batch_graphs.file) else ""
                    else:
                        file_path = str(batch_graphs.file)
                
                # If file path is not available, try to get from dataset
                if not file_path and i < len(dataset):
                    graph_dict = dataset.graphs[i]
                    file_path = graph_dict.get("file", "")
                
                # Get metadata from batch_info
                metadata = {}
                if batch_info and i < len(batch_info) and batch_info[i]:
                    metadata = batch_info[i].get("metadata", {})

                # --- Node ID -> 行範囲(span) 情報の構築 ---
                node_spans: List[Dict[str, Any]] = []
                line_range: Optional[Dict[str, int]] = None
                try:
                    if hasattr(batch_graphs, "node_ids") and hasattr(batch_graphs, "spans"):
                        node_ids_tensor = batch_graphs.node_ids[graph_mask]
                        spans_tensor = batch_graphs.spans[graph_mask]
                        if node_ids_tensor.numel() > 0 and spans_tensor.numel() > 0:
                            spans_list = spans_tensor.tolist()
                            node_ids_list = node_ids_tensor.tolist()

                            # 個々のノードIDとspanの対応
                            for nid, span in zip(node_ids_list, spans_list):
                                sl, sc, el, ec = span
                                node_spans.append(
                                    {
                                        "node_id": int(nid),
                                        "span": {
                                            "start_line": int(sl),
                                            "start_col": int(sc),
                                            "end_line": int(el),
                                            "end_col": int(ec),
                                        },
                                    }
                                )

                            # グラフ全体の行範囲 (最小〜最大の行番号)
                            start_lines = [s[0] for s in spans_list]
                            end_lines = [s[2] for s in spans_list]
                            if start_lines and end_lines:
                                line_range = {
                                    "start_line": int(min(start_lines)),
                                    "end_line": int(max(end_lines)),
                                }
                except Exception:
                    # span 情報の取得に失敗した場合は、location 情報を省略
                    node_spans = []
                    line_range = None
                
                result: Dict[str, Any] = {
                    "file_path": file_path,
                    "is_vulnerable": bool(preds[i].item() == 1),
                    "confidence": confidences[i].item(),
                    "probabilities": {
                        "safe": probs[i][0].item(),
                        "vulnerable": probs[i][1].item()
                    }
                }

                # 設計書に沿った「行範囲」情報を追加（後方互換のためオプションフィールドにする）
                if line_range is not None:
                    result["line_range"] = line_range
                if node_spans:
                    result["node_spans"] = node_spans
                
                # Add metadata if available
                if metadata:
                    result["repo_url"] = metadata.get("repo_url", "")
                    result["vulnerability_type"] = metadata.get("vulnerability_type")
                    result["code"] = metadata.get("code")
                
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run GNN model inference on Python files"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input: Python file, directory, or JSONL file with CPG graphs"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model checkpoint (default: checkpoints_test_dynamic/best_model.pt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="inference_results.json",
        help="Output JSON file for inference results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified"
    )
    parser.add_argument(
        "--repo-url",
        default="",
        help="Repository URL (for metadata in results)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_env_file()
    
    # Device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Determine model path
    if args.model is None:
        # Try to find default model in checkpoints directories
        project_root = Path(__file__).parent.parent
        default_paths = [
            project_root / "checkpoints_test_dynamic" / "best_model.pt",
            project_root / "checkpoints_test" / "checkpoint_epoch_3.pt",
            project_root / "checkpoints_test" / "checkpoint_epoch_2.pt",
            project_root / "checkpoints_test" / "checkpoint_epoch_1.pt",
        ]
        
        model_path = None
        for path in default_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            print("Error: No model checkpoint found. Please specify --model path.")
            print("Searched in:")
            for path in default_paths:
                print(f"  - {path}")
            return 1
        
        print(f"Using default model: {model_path}")
    else:
        model_path = args.model
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Prepare input
    input_path = Path(args.input)
    
    # Check if input is a directory of Python files or a single file
    if input_path.is_dir():
        # Generate CPG graphs for all Python files
        print(f"Scanning directory: {input_path}")
        python_files = list(input_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        # Generate CPG graphs and save to temporary JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_jsonl = Path(tmp_file.name)
        
        graphs_generated = 0
        for py_file in tqdm(python_files, desc="Generating CPG graphs"):
            graph = generate_cpg_from_file(py_file)
            if graph:
                graph["file"] = str(py_file)
                with open(tmp_jsonl, "a") as f:
                    f.write(json.dumps(graph, ensure_ascii=False) + "\n")
                graphs_generated += 1
        
        print(f"Generated {graphs_generated} CPG graphs")
        
        if graphs_generated == 0:
            print("Error: No CPG graphs generated")
            return 1
        
        graph_jsonl_path = tmp_jsonl
        labels_jsonl_path = None
        
    elif input_path.suffix == ".jsonl":
        # Input is already a JSONL file with CPG graphs
        graph_jsonl_path = input_path
        labels_jsonl_path = None
    elif input_path.suffix == ".py":
        # Single Python file
        print(f"Processing single file: {input_path}")
        graph = generate_cpg_from_file(input_path)
        if not graph:
            print("Error: Failed to generate CPG graph")
            return 1
        
        graph["file"] = str(input_path)
        
        # Save to temporary JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_jsonl = Path(tmp_file.name)
            tmp_file.write(json.dumps(graph, ensure_ascii=False) + "\n")
        
        graph_jsonl_path = tmp_jsonl
        labels_jsonl_path = None
    else:
        print(f"Error: Unsupported input type: {input_path}")
        return 1
    
    # Initialize embedder
    print("Initializing CodeBERT embedder...")
    embedder = CodeBERTEmbedder(device=str(device))
    
    # Create dataset
    print(f"Loading dataset from {graph_jsonl_path}...")
    dataset = CPGGraphDataset(
        graph_jsonl_path=str(graph_jsonl_path),
        labels_jsonl_path=labels_jsonl_path,
        embedder=embedder,
        max_nodes=1000,
    )
    
    print(f"Loaded {len(dataset)} graphs")
    
    if len(dataset) == 0:
        print("Error: No graphs in dataset")
        return 1
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, dataset, device, args.batch_size)
    
    # Add repo_url to results if provided
    if args.repo_url:
        for result in results:
            if "repo_url" not in result or not result["repo_url"]:
                result["repo_url"] = args.repo_url
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nInference complete!")
    print(f"Results saved to: {output_path}")
    
    # Print summary
    vulnerable_count = sum(1 for r in results if r["is_vulnerable"])
    print(f"\nSummary:")
    print(f"  Total files: {len(results)}")
    print(f"  Vulnerable: {vulnerable_count}")
    print(f"  Safe: {len(results) - vulnerable_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

