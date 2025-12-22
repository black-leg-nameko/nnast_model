"""
Evaluation metrics module for NNAST.

Implements evaluation metrics as specified in design document v2 section 7:
- Macro-F1
- Per-class F1
- PR-AUC
- Localization accuracy (line-level)
- Framework-specific performance
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import torch

try:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        precision_recall_curve, auc,
        confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback implementations
    def f1_score(y_true, y_pred, average='binary', zero_division=0):
        """Simple F1 score calculation."""
        if average == 'binary':
            tp = sum((y_true == 1) & (y_pred == 1))
            fp = sum((y_true == 0) & (y_pred == 1))
            fn = sum((y_true == 1) & (y_pred == 0))
            if tp + fp == 0 or tp + fn == 0:
                return zero_division
            precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
            recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
            if precision + recall == 0:
                return zero_division
            return 2 * precision * recall / (precision + recall)
        return zero_division
    
    def precision_score(y_true, y_pred, average='binary', zero_division=0):
        """Simple precision calculation."""
        if average == 'binary':
            tp = sum((y_true == 1) & (y_pred == 1))
            fp = sum((y_true == 0) & (y_pred == 1))
            if tp + fp == 0:
                return zero_division
            return tp / (tp + fp)
        return zero_division
    
    def recall_score(y_true, y_pred, average='binary', zero_division=0):
        """Simple recall calculation."""
        if average == 'binary':
            tp = sum((y_true == 1) & (y_pred == 1))
            fn = sum((y_true == 1) & (y_pred == 0))
            if tp + fn == 0:
                return zero_division
            return tp / (tp + fn)
        elif average == 'macro':
            classes = np.unique(y_true)
            recalls = []
            for cls in classes:
                y_true_cls = (y_true == cls).astype(int)
                y_pred_cls = (y_pred == cls).astype(int)
                tp = sum((y_true_cls == 1) & (y_pred_cls == 1))
                fn = sum((y_true_cls == 1) & (y_pred_cls == 0))
                if tp + fn == 0:
                    recalls.append(zero_division)
                else:
                    recalls.append(tp / (tp + fn))
            return np.mean(recalls) if recalls else zero_division
        return zero_division
    
    def confusion_matrix(y_true, y_pred):
        """Simple confusion matrix calculation."""
        tn = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tp = sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])
    
    def precision_recall_curve(y_true, y_proba):
        """Simple precision-recall curve calculation."""
        # Sort by probability descending
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_proba_sorted = y_proba[sorted_indices]
        
        # Calculate precision and recall at each threshold
        thresholds = np.unique(y_proba_sorted)
        precision = []
        recall = []
        
        for threshold in thresholds:
            y_pred = (y_proba_sorted >= threshold).astype(int)
            tp = np.sum((y_true_sorted == 1) & (y_pred == 1))
            fp = np.sum((y_true_sorted == 0) & (y_pred == 1))
            fn = np.sum((y_true_sorted == 1) & (y_pred == 0))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision.append(prec)
            recall.append(rec)
        
        return np.array(precision), np.array(recall), thresholds
    
    def auc(x, y):
        """Simple AUC calculation using trapezoidal rule."""
        if len(x) < 2:
            return 0.0
        # Sort by x
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        # Trapezoidal rule
        area = 0.0
        for i in range(len(x_sorted) - 1):
            area += (x_sorted[i+1] - x_sorted[i]) * (y_sorted[i] + y_sorted[i+1]) / 2
        return area
    
    def classification_report(y_true, y_pred):
        """Simple classification report."""
        return {}


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels (binary or multi-class)
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for PR-AUC calculation)
        class_names: Optional class names (e.g., ["safe", "vulnerable"])
    
    Returns:
        Dictionary containing all metrics
    """
    if class_names is None:
        class_names = ["class_0", "class_1"]
    
    n_classes = len(np.unique(y_true))
    
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = float(np.mean(y_true == y_pred))
    
    # Binary classification metrics
    if n_classes == 2:
        metrics["f1_binary"] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        metrics["precision_binary"] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
        metrics["recall_binary"] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
        
        # Macro-F1 (average of per-class F1)
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        # Handle both array and scalar return values
        if isinstance(f1_per_class, (list, np.ndarray)):
            metrics["f1_per_class"] = {
                class_names[i] if i < len(class_names) else f"class_{i}": float(f1_per_class[i])
                for i in range(len(f1_per_class))
            }
        else:
            # Fallback for scalar (shouldn't happen but handle gracefully)
            metrics["f1_per_class"] = {
                class_names[0] if len(class_names) > 0 else "class_0": float(f1_per_class)
            }
        
        # PR-AUC
        if y_proba is not None:
            try:
                precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)
                metrics["pr_auc"] = float(pr_auc)
            except Exception as e:
                metrics["pr_auc"] = None
                metrics["pr_auc_error"] = str(e)
        else:
            metrics["pr_auc"] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
    
    # Multi-class metrics
    else:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        # Handle both array and scalar return values
        if isinstance(f1_per_class, (list, np.ndarray)):
            metrics["f1_per_class"] = {
                class_names[i] if i < len(class_names) else f"class_{i}": float(f1_per_class[i])
                for i in range(len(f1_per_class))
            }
        else:
            # Fallback for scalar (shouldn't happen but handle gracefully)
            metrics["f1_per_class"] = {
                class_names[0] if len(class_names) > 0 else "class_0": float(f1_per_class)
            }
    
    return metrics


def calculate_localization_accuracy(
    predicted_spans: List[Dict[str, Any]],
    ground_truth_spans: List[Dict[str, Any]],
    tolerance: int = 5
) -> Dict[str, float]:
    """
    Calculate line-level localization accuracy.
    
    Args:
        predicted_spans: List of predicted spans, each with 'file' and 'lines' or 'span'
        ground_truth_spans: List of ground truth spans
        tolerance: Line tolerance for matching (default: 5 lines)
    
    Returns:
        Dictionary with localization metrics
    """
    if not predicted_spans and not ground_truth_spans:
        return {
            "localization_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }
    
    if not predicted_spans:
        return {
            "localization_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    if not ground_truth_spans:
        return {
            "localization_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    def extract_lines(span: Dict[str, Any]) -> List[int]:
        """Extract line numbers from span."""
        if "lines" in span:
            return span["lines"]
        elif "span" in span:
            span_data = span["span"]
            if isinstance(span_data, (list, tuple)) and len(span_data) >= 2:
                start_line = span_data[0]
                end_line = span_data[2] if len(span_data) > 2 else span_data[0]
                return list(range(start_line, end_line + 1))
        elif "line_range" in span:
            start, end = span["line_range"]
            return list(range(start, end + 1))
        return []
    
    def lines_match(pred_lines: List[int], gt_lines: List[int], tolerance: int) -> bool:
        """Check if predicted lines match ground truth within tolerance."""
        if not pred_lines or not gt_lines:
            return False
        
        # Check if any predicted line is within tolerance of any ground truth line
        for pred_line in pred_lines:
            for gt_line in gt_lines:
                if abs(pred_line - gt_line) <= tolerance:
                    return True
        return False
    
    # Group spans by file
    pred_by_file = defaultdict(list)
    gt_by_file = defaultdict(list)
    
    for span in predicted_spans:
        file_path = span.get("file", "")
        pred_by_file[file_path].append(span)
    
    for span in ground_truth_spans:
        file_path = span.get("file", "")
        gt_by_file[file_path].append(span)
    
    # Calculate matches
    total_pred = 0
    total_gt = 0
    matched_pred = 0
    matched_gt = 0
    
    all_files = set(list(pred_by_file.keys()) + list(gt_by_file.keys()))
    
    for file_path in all_files:
        pred_spans = pred_by_file.get(file_path, [])
        gt_spans = gt_by_file.get(file_path, [])
        
        total_pred += len(pred_spans)
        total_gt += len(gt_spans)
        
        # Match predicted spans to ground truth
        matched_pred_set = set()
        matched_gt_set = set()
        
        for i, pred_span in enumerate(pred_spans):
            pred_lines = extract_lines(pred_span)
            for j, gt_span in enumerate(gt_spans):
                if j in matched_gt_set:
                    continue
                gt_lines = extract_lines(gt_span)
                if lines_match(pred_lines, gt_lines, tolerance):
                    matched_pred_set.add(i)
                    matched_gt_set.add(j)
                    break
        
        matched_pred += len(matched_pred_set)
        matched_gt += len(matched_gt_set)
    
    # Calculate metrics
    precision = matched_pred / total_pred if total_pred > 0 else 0.0
    recall = matched_gt / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Localization accuracy: average of precision and recall
    localization_accuracy = (precision + recall) / 2.0
    
    return {
        "localization_accuracy": float(localization_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_predictions": int(matched_pred),
        "total_predictions": int(total_pred),
        "matched_ground_truth": int(matched_gt),
        "total_ground_truth": int(total_gt)
    }


def calculate_framework_metrics(
    results: List[Dict[str, Any]],
    framework_key: str = "framework"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics per framework.
    
    Args:
        results: List of result dictionaries, each should have framework info
        framework_key: Key to extract framework from result dict
    
    Returns:
        Dictionary mapping framework name to metrics
    """
    framework_results = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_proba": []})
    
    for result in results:
        framework = result.get(framework_key) or result.get("metadata", {}).get("frameworks", [None])[0] if isinstance(result.get("metadata", {}).get("frameworks"), list) else None
        
        if framework is None:
            framework = "unknown"
        elif isinstance(framework, list):
            framework = framework[0] if framework else "unknown"
        
        y_true = result.get("is_vulnerable", False)
        y_pred = result.get("predicted_vulnerable", result.get("is_vulnerable", False))
        y_proba = result.get("confidence", 0.0) if y_pred else (1.0 - result.get("confidence", 0.0))
        
        framework_results[framework]["y_true"].append(int(y_true))
        framework_results[framework]["y_pred"].append(int(y_pred))
        framework_results[framework]["y_proba"].append(float(y_proba))
    
    framework_metrics = {}
    
    for framework, data in framework_results.items():
        if len(data["y_true"]) == 0:
            continue
        
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        y_proba = np.array(data["y_proba"])
        
        metrics = calculate_classification_metrics(
            y_true, y_pred, y_proba,
            class_names=["safe", "vulnerable"]
        )
        
        framework_metrics[framework] = metrics
    
    return framework_metrics


def evaluate_model_comprehensive(
    model: Any,
    dataloader: Any,
    device: torch.device,
    extract_labels_fn: Optional[callable] = None,
    extract_predictions_fn: Optional[callable] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation function that calculates all metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run on
        extract_labels_fn: Function to extract labels from batch (optional)
        extract_predictions_fn: Function to extract predictions from batch (optional)
        class_names: Optional class names
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract labels
            if extract_labels_fn:
                y_true = extract_labels_fn(batch)
            else:
                # Default: assume batch is (data, labels) tuple
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    y_true = batch[1]
                else:
                    continue
            
            # Get predictions
            if extract_predictions_fn:
                y_pred, y_proba = extract_predictions_fn(model, batch, device)
            else:
                # Default: forward pass
                if isinstance(batch, (list, tuple)):
                    data = batch[0].to(device)
                else:
                    data = batch.to(device)
                
                logits = model(data)
                y_proba = torch.softmax(logits, dim=1)
                y_pred = logits.argmax(dim=1)
            
            # Convert to numpy
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            if isinstance(y_proba, torch.Tensor):
                y_proba = y_proba.cpu().numpy()
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            if y_proba.ndim == 2:
                all_y_proba.extend(y_proba[:, 1])  # Probability of positive class
            else:
                all_y_proba.extend(y_proba)
    
    # Calculate metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_proba = np.array(all_y_proba)
    
    metrics = calculate_classification_metrics(
        y_true, y_pred, y_proba, class_names=class_names
    )
    
    return metrics


def format_metrics_report(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """
    Format metrics as a human-readable report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    
    Returns:
        Formatted string report
    """
    lines = [f"\n{'='*60}", f"{title}", f"{'='*60}"]
    
    # Basic metrics
    if "accuracy" in metrics:
        lines.append(f"\nAccuracy: {metrics['accuracy']:.4f}")
    
    # Binary classification metrics
    if "f1_binary" in metrics:
        lines.append(f"\nBinary Classification Metrics:")
        lines.append(f"  F1 Score: {metrics['f1_binary']:.4f}")
        lines.append(f"  Precision: {metrics['precision_binary']:.4f}")
        lines.append(f"  Recall: {metrics['recall_binary']:.4f}")
    
    # Macro metrics
    if "f1_macro" in metrics:
        lines.append(f"\nMacro-Averaged Metrics:")
        lines.append(f"  Macro-F1: {metrics['f1_macro']:.4f}")
        lines.append(f"  Macro-Precision: {metrics['precision_macro']:.4f}")
        lines.append(f"  Macro-Recall: {metrics['recall_macro']:.4f}")
    
    # Per-class F1
    if "f1_per_class" in metrics:
        lines.append(f"\nPer-Class F1 Scores:")
        for class_name, f1 in metrics["f1_per_class"].items():
            lines.append(f"  {class_name}: {f1:.4f}")
    
    # PR-AUC
    if "pr_auc" in metrics:
        if metrics["pr_auc"] is not None:
            lines.append(f"\nPR-AUC: {metrics['pr_auc']:.4f}")
        else:
            lines.append(f"\nPR-AUC: N/A")
    
    # Confusion matrix
    if "confusion_matrix" in metrics:
        lines.append(f"\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        lines.append(f"  TN: {metrics.get('tn', 0)}, FP: {metrics.get('fp', 0)}")
        lines.append(f"  FN: {metrics.get('fn', 0)}, TP: {metrics.get('tp', 0)}")
    
    lines.append(f"\n{'='*60}\n")
    
    return "\n".join(lines)

