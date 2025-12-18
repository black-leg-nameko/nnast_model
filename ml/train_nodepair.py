#!/usr/bin/env python3
"""
Training script for CPG node-pair taint flow prediction model.

This script uses DTA付きグラフ (taint_log.jsonl) を教師信号として、
CPGNodePairModel に「この(source, sink)ノードペア間にtaint flowが存在するか」を
2値分類で学習させる。

前提:
  - graphs JSONL: CPGグラフ (ir.schema.CPGGraph.to_dict() 互換)
  - taint JSONL: ir.io.iter_taint_records() で読み込める形式
"""

import argparse
import json
import pathlib
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ml.model import CPGNodePairModel
from ml.dataset import CPGGraphDataset
from ml.embed_codebert import CodeBERTEmbedder


def collate_fn_nodepair(
    batch: List[Tuple[Data, Dict[str, Any]]],
) -> Tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NodePairModel 用のカスタム collate 関数。

    各グラフについて:
      - 正例ペア: taint_info の (source_id, sink_id)
      - 負例ペア: 同じグラフ内でランダムにサンプリングした (u, v)
    を1つずつ作り、グラフを2回ずつバッチに入れる。

    戻り値:
      batched_graphs: PyG Batch
      source_idx: Tensor[num_pairs] (各グラフ内でのノードインデックス)
      sink_idx:   Tensor[num_pairs]
      labels:     Tensor[num_pairs] (1=taint flowあり, 0=なし)
    """
    graphs: List[Data] = []
    src_indices: List[int] = []
    sink_indices: List[int] = []
    labels: List[int] = []

    for graph_data, taint_info in batch:
        if not taint_info:
            continue
        src_id = taint_info.get("source_id")
        sink_id = taint_info.get("sink_id")
        if src_id is None or sink_id is None:
            continue

        # graph_data.node_ids は元のCPGノードIDを保持している
        if not hasattr(graph_data, "node_ids"):
            continue
        node_ids_tensor = graph_data.node_ids
        if node_ids_tensor.numel() == 0:
            continue

        # 元のノードID -> グラフ内インデックス (0..num_nodes-1)
        # 複数ヒットすることは通常ないはずだが、最初のものを使う
        src_idx_tensor = (node_ids_tensor == int(src_id)).nonzero(as_tuple=False)
        sink_idx_tensor = (node_ids_tensor == int(sink_id)).nonzero(as_tuple=False)
        if src_idx_tensor.numel() == 0 or sink_idx_tensor.numel() == 0:
            continue

        src_idx = int(src_idx_tensor[0].item())
        sink_idx = int(sink_idx_tensor[0].item())

        num_nodes = int(node_ids_tensor.numel())
        if num_nodes < 2:
            continue

        # --- 正例ペア ---
        graphs.append(graph_data)
        src_indices.append(src_idx)
        sink_indices.append(sink_idx)
        labels.append(1)

        # --- 負例ペア (同じグラフ内でランダムにサンプリング) ---
        # なるべく (src_idx, sink_idx) と異なるペアを選ぶ
        neg_u = src_idx
        neg_v = sink_idx
        # 単純な試行で最大10回まで違うペアを探す
        for _ in range(10):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v:
                continue
            if not (u == src_idx and v == sink_idx):
                neg_u, neg_v = u, v
                break

        graphs.append(graph_data)
        src_indices.append(int(neg_u))
        sink_indices.append(int(neg_v))
        labels.append(0)

    if not graphs:
        # バッチ内に有効なtaint_infoがなかった場合、
        # ダミーの空バッチを返す（呼び出し側でスキップさせる）
        empty_batch = Batch.from_data_list([])
        return (
            empty_batch,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    batched_graphs = Batch.from_data_list(graphs)
    source_idx_tensor = torch.tensor(src_indices, dtype=torch.long)
    sink_idx_tensor = torch.tensor(sink_indices, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return batched_graphs, source_idx_tensor, sink_idx_tensor, labels_tensor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[NodePair] Epoch {epoch}")
    for batch in pbar:
        batched_graphs, src_idx, sink_idx, labels = batch

        # 空バッチはスキップ
        if labels.numel() == 0:
            continue

        batched_graphs = batched_graphs.to(device)
        src_idx = src_idx.to(device)
        sink_idx = sink_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(batched_graphs, src_idx, sink_idx)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100.0 * correct / total if total > 0 else 0.0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[NodePair] Evaluating"):
            batched_graphs, src_idx, sink_idx, labels = batch
            if labels.numel() == 0:
                continue

            batched_graphs = batched_graphs.to(device)
            src_idx = src_idx.to(device)
            sink_idx = sink_idx.to(device)
            labels = labels.to(device)

            logits = model(batched_graphs, src_idx, sink_idx)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(
        description="Train CPG node-pair taint flow prediction model"
    )
    parser.add_argument(
        "--graphs",
        required=True,
        help="Path to CPG graphs JSONL file",
    )
    parser.add_argument(
        "--taint-log",
        required=True,
        help="Path to taint records JSONL file (DTA output)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints_nodepair",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (graphs単位ではなく、(pos+neg)ペア単位で2倍のサンプルになります)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--gnn-type",
        choices=["GCN", "GAT"],
        default="GCN",
        help="GNN type",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes per graph",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio",
    )
    parser.add_argument(
        "--device",
        help="Device to use (cuda/cpu, auto-detect if not specified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Output dir
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Embedder
    print("Initializing CodeBERT embedder...")
    embedder = CodeBERTEmbedder(device=str(device))

    # Dataset
    print(f"Loading graphs from {args.graphs} with taint log {args.taint_log} ...")
    dataset = CPGGraphDataset(
        graph_jsonl_path=args.graphs,
        taint_jsonl_path=args.taint_log,
        labels_jsonl_path=None,
        embedder=embedder,
        max_nodes=args.max_nodes,
        skip_large_graphs=False,
    )
    print(f"Loaded {len(dataset)} graphs (some may not have taint records).")

    # Train/val split
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_nodepair,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_nodepair,
        num_workers=0,
    )

    # Model
    print("Initializing CPGNodePairModel...")
    model = CPGNodePairModel(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        dropout=0.5,
    ).to(device)

    # Simple Xavier init for linear layers
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    print(
        f"NodePair model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("\nStarting node-pair training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

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
        ckpt_path = output_dir / f"checkpoint_nodepair_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)

        # Best model based on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_path = output_dir / "best_model_nodepair.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best node-pair model (val_acc={best_val_acc:.2f}%)")

    # Save history
    history_path = output_dir / "training_history_nodepair.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nNode-pair training completed.")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()


