"""
Weak Signal Utilities for NNAST Training Strategy

Implements weak supervision signal generation from:
- Pillar A: Bandit static analysis scores
- Pillar B: Patch-diff data (pre-patch vs post-patch)
- Pillar C: Synthetic vulnerability injection flags
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def run_bandit_on_code(source_code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run Bandit static analysis on Python source code.
    
    Args:
        source_code: Python source code string
        file_path: Optional file path (for context)
        
    Returns:
        Dict with:
            - bandit_score: float ∈ [0, 1]
            - bandit_rules: list[str] of triggered rule IDs
            - severity_counts: dict with HIGH/MEDIUM/LOW counts
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source_code)
        temp_path = f.name
    
    try:
        # Run Bandit
        result = subprocess.run(
            ['bandit', '-f', 'json', '-q', temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0 and result.stdout:
            try:
                bandit_output = json.loads(result.stdout)
            except json.JSONDecodeError:
                return {
                    "bandit_score": 0.0,
                    "bandit_rules": [],
                    "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                }
        else:
            return {
                "bandit_score": 0.0,
                "bandit_rules": [],
                "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            }
        
        # Extract results
        metrics = bandit_output.get("metrics", {})
        severity_counts = {
            "HIGH": metrics.get("SEVERITY", {}).get("HIGH", 0),
            "MEDIUM": metrics.get("SEVERITY", {}).get("MEDIUM", 0),
            "LOW": metrics.get("SEVERITY", {}).get("LOW", 0),
        }
        
        # Extract rule IDs from results
        results_list = bandit_output.get("results", [])
        bandit_rules = []
        for result_item in results_list:
            test_id = result_item.get("test_id", "")
            if test_id:
                bandit_rules.append(test_id)
        
        # Compute bandit_score according to NNAST strategy:
        # HIGH → 1.0, MEDIUM → 0.6, LOW → 0.3, NONE → 0.0
        if severity_counts["HIGH"] > 0:
            bandit_score = 1.0
        elif severity_counts["MEDIUM"] > 0:
            bandit_score = 0.6
        elif severity_counts["LOW"] > 0:
            bandit_score = 0.3
        else:
            bandit_score = 0.0
        
        return {
            "bandit_score": bandit_score,
            "bandit_rules": list(set(bandit_rules)),  # Remove duplicates
            "severity_counts": severity_counts
        }
    
    except subprocess.TimeoutExpired:
        return {
            "bandit_score": 0.0,
            "bandit_rules": [],
            "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
    except Exception as e:
        # Fallback: return default values
        return {
            "bandit_score": 0.0,
            "bandit_rules": [],
            "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except:
            pass


def compute_risk_score_from_weak_signals(
    bandit_score: float = 0.0,
    patched_flag: bool = False,
    is_post_patch: bool = False,
    synthetic_flag: bool = False,
    bandit_rules: Optional[List[str]] = None
) -> float:
    """
    Compute risk score from weak signals following NNAST strategy.
    
    Priority:
    1. Synthetic vulnerabilities → 0.9
    2. Pre-patch samples → 0.8
    3. Bandit HIGH → 1.0
    4. Bandit MEDIUM → 0.6
    5. Post-patch samples → 0.2
    6. Default → 0.0
    
    Args:
        bandit_score: Bandit score ∈ [0, 1]
        patched_flag: Whether this is from patch-diff data
        is_post_patch: Whether this is post-patch (vs pre-patch)
        synthetic_flag: Whether this is synthetic
        bandit_rules: List of triggered Bandit rule IDs
        
    Returns:
        Risk score ∈ [0, 1]
    """
    # Priority 1: Synthetic vulnerabilities
    if synthetic_flag:
        return 0.9
    
    # Priority 2: Patch-diff data
    if patched_flag:
        if is_post_patch:
            return 0.2  # Post-patch: lower risk
        else:
            return 0.8  # Pre-patch: higher risk
    
    # Priority 3: Bandit score
    if bandit_score > 0:
        return bandit_score
    
    # Default: no risk signals
    return 0.0


def attach_weak_signals_to_graph(
    graph_dict: Dict[str, Any],
    source_code: Optional[str] = None,
    patched_flag: bool = False,
    is_post_patch: bool = False,
    synthetic_flag: bool = False,
    run_bandit: bool = True
) -> Dict[str, Any]:
    """
    Attach weak signals to a CPG graph dict.
    
    Args:
        graph_dict: CPG graph dictionary
        source_code: Source code (if None, try to extract from graph)
        patched_flag: Whether this is from patch-diff data
        is_post_patch: Whether this is post-patch
        synthetic_flag: Whether this is synthetic
        run_bandit: Whether to run Bandit analysis
        
    Returns:
        Updated graph_dict with weak_signals metadata
    """
    # Extract source code if not provided
    if source_code is None:
        # Try to get from graph metadata
        source_code = graph_dict.get("source_code", "")
        if not source_code:
            # Fallback: reconstruct from nodes (simplified)
            source_code = ""
    
    # Run Bandit if requested
    bandit_result = {}
    if run_bandit and source_code:
        file_path = graph_dict.get("file", "")
        bandit_result = run_bandit_on_code(source_code, file_path)
    else:
        bandit_result = {
            "bandit_score": 0.0,
            "bandit_rules": [],
            "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
    
    # Compute risk score
    risk_score = compute_risk_score_from_weak_signals(
        bandit_score=bandit_result.get("bandit_score", 0.0),
        patched_flag=patched_flag,
        is_post_patch=is_post_patch,
        synthetic_flag=synthetic_flag,
        bandit_rules=bandit_result.get("bandit_rules", [])
    )
    
    # Attach weak signals to graph metadata
    if "metadata" not in graph_dict:
        graph_dict["metadata"] = {}
    
    graph_dict["metadata"]["weak_signals"] = {
        "risk_score": risk_score,
        "bandit_score": bandit_result.get("bandit_score", 0.0),
        "bandit_rules": bandit_result.get("bandit_rules", []),
        "patched_flag": patched_flag,
        "synthetic_flag": synthetic_flag,
        "severity_counts": bandit_result.get("severity_counts", {})
    }
    
    if patched_flag:
        graph_dict["metadata"]["weak_signals"]["is_post_patch"] = is_post_patch
    
    return graph_dict


def create_label_entry_from_weak_signals(
    graph_dict: Dict[str, Any],
    code_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a label entry (for labels.jsonl) from weak signals in graph.
    
    Args:
        graph_dict: CPG graph dictionary with weak_signals metadata
        code_id: Optional code identifier
        
    Returns:
        Label entry dict compatible with CPGGraphDataset
    """
    weak_signals = graph_dict.get("metadata", {}).get("weak_signals", {})
    
    if not weak_signals:
        # Default: no signals
        weak_signals = {
            "risk_score": 0.0,
            "bandit_score": 0.0,
            "bandit_rules": [],
            "patched_flag": False,
            "synthetic_flag": False
        }
    
    label_entry = {
        "code_id": code_id or graph_dict.get("file", ""),
        "risk_score": weak_signals.get("risk_score", 0.0),
        "metadata": {
            "bandit_score": weak_signals.get("bandit_score", 0.0),
            "bandit_rules": weak_signals.get("bandit_rules", []),
            "patched_flag": weak_signals.get("patched_flag", False),
            "synthetic_flag": weak_signals.get("synthetic_flag", False),
            "severity_counts": weak_signals.get("severity_counts", {})
        }
    }
    
    if weak_signals.get("patched_flag", False):
        label_entry["metadata"]["is_post_patch"] = weak_signals.get("is_post_patch", False)
    
    return label_entry
