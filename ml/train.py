#!/usr/bin/env python3
"""
Training script for CPG vulnerability risk scoring model.

Following NNAST training strategy:
- Risk scoring (regression) instead of binary classification
- Weak supervision from Bandit scores, patch-diff data, and synthetic samples
- Pairwise ranking loss (recommended) or pointwise regression
- Ranking-oriented evaluation metrics (Top-K Recall, Precision@K, AUROC, AUPRC)
"""
import argparse
import json
import pathlib
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch_geometric.data import Batch
import torch.optim as optim
from tqdm import tqdm

from ml.model import CPGTaintFlowModel, CPGNodePairModel
from ml.dataset import CPGGraphDataset
from ml.embed_codebert import CodeBERTEmbedder
from ml.evaluation import (
    calculate_classification_metrics,
    format_metrics_report,
    calculate_framework_metrics
)
from sklearn.metrics import f1_score, precision_score, recall_score


def collate_fn(batch):
    """Custom collate function for batching graphs and weak signals."""
    graphs, weak_signals = zip(*batch)
    
    # Batch graphs using PyG's Batch
    batched_graphs = Batch.from_data_list(graphs)
    
    # Collect weak signals
    batched_weak_signals = list(weak_signals)
    
    return batched_graphs, batched_weak_signals


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for learning relative risk ordering.
    
    Implements margin ranking loss with constraints:
    - (pre-patch > post-patch)
    - (synthetic-vuln > original)
    - (HIGH > MEDIUM > NONE)
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize pairwise ranking loss.
        
        Args:
            margin: Margin for ranking loss
        """
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        weak_signals: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            risk_scores: Predicted risk scores (batch_size,)
            weak_signals: List of weak signal dicts
            
        Returns:
            Loss value
        """
        # Extract pairs for ranking constraints
        pairs = []
        
        for i in range(len(weak_signals)):
            for j in range(i + 1, len(weak_signals)):
                sig_i = weak_signals[i]
                sig_j = weak_signals[j]
                
                # Constraint 1: Pre-patch > Post-patch
                if sig_i.get("patched_flag") and not sig_i.get("metadata", {}).get("is_post_patch", False):
                    if sig_j.get("patched_flag") and sig_j.get("metadata", {}).get("is_post_patch", False):
                        pairs.append((i, j, 1.0))  # i should have higher score
                
                # Constraint 2: Synthetic > Original
                if sig_i.get("synthetic_flag") and not sig_j.get("synthetic_flag"):
                    pairs.append((i, j, 1.0))
                
                # Constraint 3: Higher bandit_score > Lower bandit_score
                score_i = sig_i.get("bandit_score", 0.0)
                score_j = sig_j.get("bandit_score", 0.0)
                if abs(score_i - score_j) > 0.1:  # Only if difference is significant
                    if score_i > score_j:
                        pairs.append((i, j, 1.0))
                    else:
                        pairs.append((j, i, 1.0))
        
        if not pairs:
            # Fallback to pointwise regression if no pairs found
            targets = torch.tensor(
                [sig.get("risk_score", 0.0) for sig in weak_signals],
                dtype=torch.float32,
                device=risk_scores.device
            )
            return nn.functional.mse_loss(risk_scores, targets)
        
        # Build pairs for margin ranking loss
        scores_i = torch.stack([risk_scores[i] for i, _, _ in pairs])
        scores_j = torch.stack([risk_scores[j] for _, j, _ in pairs])
        targets = torch.tensor([t for _, _, t in pairs], dtype=torch.float32, device=risk_scores.device)
        
        loss = self.ranking_loss(scores_i, scores_j, targets)
        return loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_pairwise: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        use_pairwise: If True, use pairwise ranking loss; else use pointwise regression
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_graphs, batch_weak_signals in pbar:
        batch_graphs = batch_graphs.to(device)
        
        # Extract target risk scores
        target_scores = torch.tensor(
            [sig.get("risk_score", 0.0) for sig in batch_weak_signals],
            dtype=torch.float32,
            device=device
        )
        
        # Forward pass
        optimizer.zero_grad()
        risk_scores = model(batch_graphs)  # (batch_size,)
        
        # Compute loss
        if use_pairwise and isinstance(criterion, PairwiseRankingLoss):
            loss = criterion(risk_scores, batch_weak_signals)
        else:
            # Pointwise regression (MSE or Huber)
            loss = criterion(risk_scores, target_scores)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        mse = nn.functional.mse_loss(risk_scores, target_scores).item()
        total_mse += mse
        total_samples += len(batch_weak_signals)
        
        # Debug: Check prediction distribution (first batch only)
        if total_samples == len(batch_weak_signals):  # First batch
            avg_score = risk_scores.mean().item()
            avg_target = target_scores.mean().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse:.4f}",
                "pred_avg": f"{avg_score:.3f}",
                "target_avg": f"{avg_target:.3f}",
            })
        else:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse:.4f}"
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    
    return {
        "loss": avg_loss,
        "mse": avg_mse
    }


def compute_ranking_metrics(
    risk_scores: np.ndarray,
    target_scores: np.ndarray,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute ranking-oriented evaluation metrics.
    
    Args:
        risk_scores: Predicted risk scores
        target_scores: Target risk scores (weak signals)
        k_values: List of K values for Top-K metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(risk_scores)[::-1]
    sorted_targets = target_scores[sorted_indices]
    
    # Top-K Recall: What fraction of high-risk samples are in top-K?
    # Use threshold: risk_score > 0.5 for "high risk"
    high_risk_mask = target_scores > 0.5
    num_high_risk = high_risk_mask.sum()
    
    if num_high_risk > 0:
        for k in k_values:
            top_k_indices = sorted_indices[:k]
            top_k_high_risk = high_risk_mask[top_k_indices].sum()
            recall_at_k = top_k_high_risk / num_high_risk
            metrics[f"recall@{k}"] = recall_at_k
    
    # Precision@K: What fraction of top-K are actually high-risk?
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        if k > 0:
            top_k_high_risk = high_risk_mask[top_k_indices].sum()
            precision_at_k = top_k_high_risk / k
            metrics[f"precision@{k}"] = precision_at_k
    
    # AUROC and AUPRC (treat risk_score > 0.5 as positive)
    if len(np.unique(target_scores > 0.5)) > 1:  # Both classes present
        try:
            auroc = roc_auc_score(target_scores > 0.5, risk_scores)
            metrics["auroc"] = auroc
        except ValueError:
            metrics["auroc"] = 0.0
        
        try:
            auprc = average_precision_score(target_scores > 0.5, risk_scores)
            metrics["auprc"] = auprc
        except ValueError:
            metrics["auprc"] = 0.0
    else:
        metrics["auroc"] = 0.0
        metrics["auprc"] = 0.0
    
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_pairwise: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model with ranking-oriented metrics.
    
    Args:
        use_pairwise: If True, use pairwise ranking loss; else use pointwise regression
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    all_risk_scores = []
    all_target_scores = []
    all_weak_signals = []
    
    with torch.no_grad():
        for batch_graphs, batch_weak_signals in tqdm(dataloader, desc="Evaluating"):
            batch_graphs = batch_graphs.to(device)
            
            # Extract target risk scores
            target_scores = torch.tensor(
                [sig.get("risk_score", 0.0) for sig in batch_weak_signals],
                dtype=torch.float32,
                device=device
            )
            
            risk_scores = model(batch_graphs)  # (batch_size,)
            
            # Compute loss
            if use_pairwise and isinstance(criterion, PairwiseRankingLoss):
                loss = criterion(risk_scores, batch_weak_signals)
            else:
                loss = criterion(risk_scores, target_scores)
            
            total_loss += loss.item()
            mse = nn.functional.mse_loss(risk_scores, target_scores).item()
            total_mse += mse
            
            # Collect predictions, labels, and probabilities for metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get probabilities for PR-AUC calculation
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of vulnerable class
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    
    # Calculate comprehensive metrics using evaluation module
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs) if all_probs else None
    
    # Calculate comprehensive metrics
    if len(set(all_labels)) > 1:  # Only if we have both classes
        metrics = calculate_classification_metrics(
            y_true, y_pred, y_proba,
            class_names=["safe", "vulnerable"]
        )
        
        # Keep backward compatibility with old metric names
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": metrics.get("f1_binary", 0.0),
            "precision": metrics.get("precision_binary", 0.0),
            "recall": metrics.get("recall_binary", 0.0),
            # New comprehensive metrics
            "f1_macro": metrics.get("f1_macro", 0.0),
            "precision_macro": metrics.get("precision_macro", 0.0),
            "recall_macro": metrics.get("recall_macro", 0.0),
            "f1_per_class": metrics.get("f1_per_class", {}),
            "pr_auc": metrics.get("pr_auc"),
            **metrics  # Include all other metrics
        }
    else:
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_per_class": {},
            "pr_auc": None
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
        default=0.6,
        help="Train split ratio (project-wise split)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (project-wise split)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio (project-wise split)"
    )
    parser.add_argument(
        "--loss-type",
        choices=["mse", "huber", "pairwise"],
        default="pairwise",
        help="Loss type: mse (pointwise regression), huber (robust regression), pairwise (ranking)"
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
    
    # Check weak signal distribution before splitting
    print("\nChecking weak signal distribution in dataset...")
    risk_score_stats = []
    bandit_score_stats = []
    patched_count = 0
    synthetic_count = 0
    
    for i in range(min(1000, len(dataset))):  # Check first 1000 samples
        try:
            _, weak_signals = dataset[i]
            risk_score_stats.append(weak_signals.get("risk_score", 0.0))
            bandit_score_stats.append(weak_signals.get("bandit_score", 0.0))
            if weak_signals.get("patched_flag", False):
                patched_count += 1
            if weak_signals.get("synthetic_flag", False):
                synthetic_count += 1
        except Exception as e:
            pass
    
    if risk_score_stats:
        print(f"Risk score stats (first {len(risk_score_stats)} samples):")
        print(f"  Mean: {np.mean(risk_score_stats):.3f}, Std: {np.std(risk_score_stats):.3f}")
        print(f"  Min: {np.min(risk_score_stats):.3f}, Max: {np.max(risk_score_stats):.3f}")
        print(f"Bandit score stats:")
        print(f"  Mean: {np.mean(bandit_score_stats):.3f}, Std: {np.std(bandit_score_stats):.3f}")
        print(f"Patched samples: {patched_count}/{len(risk_score_stats)}")
        print(f"Synthetic samples: {synthetic_count}/{len(risk_score_stats)}")
    
    # Project-wise split (mandatory for NNAST strategy)
    print("\nPerforming project-wise split...")
    train_dataset, val_dataset, test_dataset = project_wise_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Initialize model (risk scoring regression with improved architecture)
    print("Initializing improved risk scoring model (Phase 1 + Phase 2)...")
    model = CPGTaintFlowModel(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        dropout=0.5,
        use_dynamic_attention=False,  # Disabled when using edge-type aware layers
        use_edge_types=True,  # Enable edge-type aware processing (AST/CFG/DFG/DDFG)
        use_multi_modal=True,  # Enable multi-modal node features (CodeBERT + structural)
        use_multi_scale=True,  # Enable multi-scale feature aggregation
        use_hierarchical=True,  # Enable hierarchical pooling (function → file)
        use_positional=True,  # Enable positional encoding for code spans
        num_edge_types=4,  # AST, CFG, DFG, DDFG
        num_node_kinds=50,  # Approximate number of node kinds
        num_types=100,  # Approximate number of type hints
    ).to(device)
    
    # Print model architecture info
    print(f"Model architecture:")
    print(f"  - Edge-type aware: {model.use_edge_types}")
    print(f"  - Multi-modal nodes: {model.use_multi_modal}")
    print(f"  - Multi-scale aggregation: {model.use_multi_scale}")
    print(f"  - Hierarchical pooling: {model.use_hierarchical}")
    print(f"  - Positional encoding: {model.use_positional}")
    
    # Print edge type importance (if available)
    edge_importance = model.get_edge_type_importance()
    if edge_importance:
        print(f"  - Edge type importance: {edge_importance}")
    
    # Initialize model parameters properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function (following NNAST strategy)
    use_pairwise = args.loss_type == "pairwise"
    if args.loss_type == "pairwise":
        criterion = PairwiseRankingLoss(margin=1.0)
        print("Using Pairwise Ranking Loss (recommended)")
    elif args.loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
        print("Using Huber Loss (robust regression)")
    else:  # mse
        criterion = nn.MSELoss()
        print("Using MSE Loss (pointwise regression)")
    
    print(f"Loss type: {args.loss_type}")
    
    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_val_auroc = 0.0
    history = {
        "train_loss": [],
        "train_mse": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_macro": [],
        "val_pr_auc": [],
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            use_pairwise=use_pairwise
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, use_pairwise=use_pairwise)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train MSE: {train_metrics['mse']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val MSE: {val_metrics['mse']:.4f}, "
            f"LR: {current_lr:.2e}"
        )
        if 'f1' in val_metrics:
            print(
                f"  Val F1: {val_metrics['f1']:.4f}, "
                f"Precision: {val_metrics['precision']:.4f}, "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            if 'f1_macro' in val_metrics:
                print(
                    f"  Val Macro-F1: {val_metrics['f1_macro']:.4f}, "
                    f"PR-AUC: {val_metrics.get('pr_auc', 'N/A')}"
                )
            if 'f1_per_class' in val_metrics and val_metrics['f1_per_class']:
                print(f"  Per-class F1: {val_metrics['f1_per_class']}")
        
        # Debug: Check if model is learning
        if epoch == 1:
            print(f"\n  First epoch check:")
            print(f"    Train loss: {train_metrics['loss']:.4f}")
            print(f"    Val loss: {val_metrics['loss']:.4f}")
            if train_metrics['loss'] < 0.01 or val_metrics['loss'] < 0.01:
                print(f"    ⚠️ Loss is very low - model might be overfitting or not learning properly")
            if train_metrics['loss'] > 1.0 and val_metrics['loss'] > 1.0:
                print(f"    ⚠️ Loss is very high - consider adjusting learning rate or model architecture")
        
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        if "f1" in val_metrics:
            history["val_f1"].append(val_metrics["f1"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1_macro"].append(val_metrics.get("f1_macro", 0.0))
            pr_auc = val_metrics.get("pr_auc")
            history["val_pr_auc"].append(pr_auc if pr_auc is not None else 0.0)
        
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
        
        # Save best model (based on AUROC)
        current_auroc = val_metrics.get("auroc", 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model (val_auroc: {best_val_auroc:.4f})")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, criterion, device, use_pairwise=use_pairwise)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test AUROC: {test_metrics.get('auroc', 0.0):.4f}, AUPRC: {test_metrics.get('auprc', 0.0):.4f}")
    print(f"Test Recall@5: {test_metrics.get('recall@5', 0.0):.4f}, Recall@10: {test_metrics.get('recall@10', 0.0):.4f}")
    print(f"Test Precision@5: {test_metrics.get('precision@5', 0.0):.4f}, Precision@10: {test_metrics.get('precision@10', 0.0):.4f}")
    
    # Save training history
    history["test_metrics"] = test_metrics
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"  Best validation AUROC: {best_val_auroc:.4f}")
    print(f"  Test AUROC: {test_metrics.get('auroc', 0.0):.4f}")
    
    # Final diagnosis
    print("\n" + "=" * 60)
    print("Training Diagnosis")
    print("=" * 60)
    
    if len(history["train_loss"]) > 1:
        loss_improvement = history["train_loss"][0] - history["train_loss"][-1]
        mse_improvement = history["train_mse"][0] - history["train_mse"][-1]
        
        print(f"Train loss change: {loss_improvement:.4f} ({'✅ Improved' if loss_improvement > 0 else '❌ Not improving'})")
        print(f"Train MSE change: {mse_improvement:.4f} ({'✅ Improved' if mse_improvement > 0 else '❌ Not improving'})")
        
        if loss_improvement < 0.01:
            print("\n⚠️ WARNING: Loss barely improved. Possible issues:")
            print("   1. Learning rate too low or too high")
            print("   2. Model capacity insufficient")
            print("   3. Data quality issues")
            print("   4. Try different loss type (--loss-type pairwise/mse/huber)")
    
    if best_val_auroc < 0.6:
        print("\n⚠️ WARNING: Validation AUROC is low (<0.6)")
        print("   This suggests the model is not learning effective risk ranking")
        print("   Recommendations:")
        print("   1. Check data quality with: python ml/diagnose_dataset.py")
        print("   2. Try different learning rates")
        print("   3. Increase model capacity (hidden_dim, num_layers)")
        print("   4. Train for more epochs")
        print("   5. Verify weak signals are properly attached to dataset")
    
    print(f"\nCheckpoints saved to: {output_dir}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()

