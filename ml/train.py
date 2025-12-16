#!/usr/bin/env python3
"""
Training script for CPG taint flow prediction model.
"""
import argparse
import json
import pathlib
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
import torch.optim as optim
from tqdm import tqdm

from ml.model import CPGTaintFlowModel, CPGNodePairModel
from ml.dataset import CPGGraphDataset
from ml.embed_codebert import CodeBERTEmbedder
from sklearn.metrics import f1_score, precision_score, recall_score


def collate_fn(batch):
    """Custom collate function for batching graphs and taint info."""
    graphs, taint_infos = zip(*batch)
    
    # Batch graphs using PyG's Batch
    batched_graphs = Batch.from_data_list(graphs)
    
    # Collect taint info
    batched_taint_info = list(taint_infos)
    
    return batched_graphs, batched_taint_info


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_graphs, batch_info in pbar:
        batch_graphs = batch_graphs.to(device)
        
        # Extract labels from batch_info
        # Priority: explicit label > taint info > default (0)
        labels_list = []
        for info in batch_info:
            if info is None:
                labels_list.append(0)
            elif "label" in info:
                # Explicit label from labels JSONL
                labels_list.append(info["label"])
            else:
                # Taint info available (label as 1)
                labels_list.append(1)
        
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_graphs)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # Debug: Check prediction distribution (first batch only)
        if total == labels.size(0):  # First batch
            pred_dist = Counter(pred.cpu().numpy().tolist())
            label_dist = Counter(labels.cpu().numpy().tolist())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%",
                "pred_dist": dict(pred_dist),
                "label_dist": dict(label_dist),
            })
        else:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_graphs, batch_info in tqdm(dataloader, desc="Evaluating"):
            batch_graphs = batch_graphs.to(device)
            
            # Extract labels from batch_info
            labels_list = []
            for info in batch_info:
                if info is None:
                    labels_list.append(0)
                elif "label" in info:
                    labels_list.append(info["label"])
                else:
                    labels_list.append(1)
            
            labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            
            logits = model(batch_graphs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Collect predictions and labels for metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    # Calculate additional metrics
    if len(set(all_labels)) > 1:  # Only if we have both classes
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        f1 = 0.0
        precision = 0.0
        recall = 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train CPG taint flow prediction model"
    )
    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to CPG graphs JSONL file"
    )
    parser.add_argument(
        "--taint-log",
        help="Path to taint records JSONL file (optional)"
    )
    parser.add_argument(
        "--labels",
        help="Path to labels JSONL file (for supervised learning)"
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,  # Increased from 1e-4 for better learning
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers"
    )
    parser.add_argument(
        "--gnn-type",
        choices=["GCN", "GAT"],
        default="GCN",
        help="GNN type"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes per graph"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio"
    )
    parser.add_argument(
        "--device",
        help="Device to use (cuda/cpu, auto-detect if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    print("Initializing CodeBERT embedder...")
    embedder = CodeBERTEmbedder(device=str(device))
    
    # Load dataset
    print(f"Loading dataset from {args.graphs}...")
    dataset = CPGGraphDataset(
        graph_jsonl_path=args.graphs,
        taint_jsonl_path=args.taint_log,
        labels_jsonl_path=args.labels,
        embedder=embedder,
        max_nodes=args.max_nodes,
        skip_large_graphs=False,  # Keep all graphs, truncate if needed
    )
    
    print(f"Loaded {len(dataset)} graphs")
    
    # Check label distribution before splitting
    if args.labels:
        print("\nChecking label distribution in dataset...")
        label_counts = {0: 0, 1: 0}
        for i in range(min(100, len(dataset))):  # Check first 100 samples
            try:
                _, label_info = dataset[i]
                if label_info and "label" in label_info:
                    label = int(label_info["label"])
                    label_counts[label] += 1
            except:
                pass
        print(f"Label distribution (first 100 samples): {label_counts}")
        if label_counts[0] == 0 or label_counts[1] == 0:
            print("⚠️ WARNING: Only one class found in labels! Check your label file.")
    
    # Split dataset
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Check label distribution after splitting
    if args.labels:
        print("\nChecking label distribution in train/val splits...")
        train_label_counts = {0: 0, 1: 0}
        val_label_counts = {0: 0, 1: 0}
        
        for i in range(min(100, len(train_dataset))):
            try:
                _, label_info = train_dataset[i]
                if label_info and "label" in label_info:
                    label = int(label_info["label"])
                    train_label_counts[label] += 1
            except:
                pass
        
        for i in range(min(100, len(val_dataset))):
            try:
                _, label_info = val_dataset[i]
                if label_info and "label" in label_info:
                    label = int(label_info["label"])
                    val_label_counts[label] += 1
            except:
                pass
        
        print(f"Train labels (first 100): {train_label_counts}")
        print(f"Val labels (first 100): {val_label_counts}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Initialize model
    print("Initializing model...")
    model = CPGTaintFlowModel(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        gnn_type=args.gnn_type,
        dropout=0.5,
        use_dynamic_attention=True,  # Enable dynamic attention fusion
    ).to(device)
    
    # Initialize model parameters properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Calculate class weights for imbalanced datasets
    if args.labels:
        print("Calculating class weights...")
        label_counts = {0: 0, 1: 0}
        with open(args.labels, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        label_data = json.loads(line)
                        label = label_data.get("label", label_data.get("target", 0))
                        label_counts[int(label)] += 1
                    except:
                        pass
        
        total = sum(label_counts.values())
        if total > 0 and min(label_counts.values()) > 0:
            # Calculate inverse frequency weights
            weights = torch.tensor([
                total / (2 * label_counts[0]),  # Weight for class 0
                total / (2 * label_counts[1])   # Weight for class 1
            ], dtype=torch.float32, device=device)
            print(f"Class weights: {weights.cpu().numpy()}")
            print(f"Class distribution: {label_counts}")
        else:
            weights = None
            print("⚠️ Could not calculate class weights, using uniform weights")
    else:
        weights = None
    
    # Loss and optimizer
    if weights is not None:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using weighted CrossEntropyLoss with weights: {weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"LR: {current_lr:.2e}"
        )
        if 'f1' in val_metrics:
            print(
                f"  Val F1: {val_metrics['f1']:.4f}, "
                f"Precision: {val_metrics['precision']:.4f}, "
                f"Recall: {val_metrics['recall']:.4f}"
            )
        
        # Debug: Check if model is learning
        if epoch == 1:
            print(f"\n  First epoch check:")
            print(f"    Train loss: {train_metrics['loss']:.4f}")
            print(f"    Val loss: {val_metrics['loss']:.4f}")
            if train_metrics['loss'] < 0.1 or val_metrics['loss'] < 0.1:
                print(f"    ⚠️ Loss is very low - model might be overfitting or not learning properly")
            if train_metrics['loss'] > 1.0 and val_metrics['loss'] > 1.0:
                print(f"    ⚠️ Loss is very high - consider adjusting learning rate or model architecture")
        
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        if "f1" in val_metrics:
            history["val_f1"].append(val_metrics["f1"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model (based on F1 score if available, otherwise accuracy)
        best_metric = val_metrics.get("f1", val_metrics["accuracy"])
        current_best = history.get("best_val_f1", [best_val_acc])[-1] if "f1" in val_metrics and "best_val_f1" in history else best_val_acc
        
        if best_metric > current_best:
            if "f1" in val_metrics:
                if "best_val_f1" not in history:
                    history["best_val_f1"] = []
                history["best_val_f1"].append(best_metric)
                best_val_acc = best_metric  # Update to use F1
            else:
                best_val_acc = val_metrics["accuracy"]
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            metric_name = "F1" if "f1" in val_metrics else "accuracy"
            print(f"Saved best model (val_{metric_name}: {best_metric:.4f})")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    if "best_val_f1" in history:
        print(f"  Best validation F1: {history['best_val_f1'][-1]:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final diagnosis
    print("\n" + "=" * 60)
    print("Training Diagnosis")
    print("=" * 60)
    
    if len(history["train_loss"]) > 1:
        loss_improvement = history["train_loss"][0] - history["train_loss"][-1]
        acc_improvement = history["train_acc"][-1] - history["train_acc"][0]
        
        print(f"Train loss change: {loss_improvement:.4f} ({'✅ Improved' if loss_improvement > 0 else '❌ Not improving'})")
        print(f"Train accuracy change: {acc_improvement:.2f}% ({'✅ Improved' if acc_improvement > 0 else '❌ Not improving'})")
        
        if loss_improvement < 0.01:
            print("\n⚠️ WARNING: Loss barely improved. Possible issues:")
            print("   1. Learning rate too low or too high")
            print("   2. Model capacity insufficient")
            print("   3. Data quality issues")
        
        if acc_improvement < 1.0:
            print("\n⚠️ WARNING: Accuracy barely improved. Possible issues:")
            print("   1. Model not learning (check loss)")
            print("   2. Class imbalance (use weighted loss)")
            print("   3. Need more training epochs")
    
    if best_val_acc < 55.0:
        print("\n⚠️ WARNING: Validation accuracy is very low (<55%)")
        print("   This suggests the model is not learning effectively")
        print("   Recommendations:")
        print("   1. Check data quality with: python ml/diagnose_dataset.py")
        print("   2. Try different learning rates")
        print("   3. Increase model capacity (hidden_dim, num_layers)")
        print("   4. Train for more epochs")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()

