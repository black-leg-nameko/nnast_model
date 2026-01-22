#!/usr/bin/env python3
"""
NNAST Local Scan

Local scan process executed within GitHub Actions.
Based on the design document, performs full CPG construction, GNN inference, data minimization, and Issue creation.
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.inference import run_inference, load_model, generate_cpg_from_file
from local.masking import create_minimized_payload, generate_fingerprint
from local.github_issue import create_issues_from_reports
from local.cpg_builder import build_full_cpg


def run_gnn_inference(
    cpg_graph: Dict[str, Any],
    model_path: Path,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Run GNN inference to rank vulnerability candidates
    
    Args:
        cpg_graph: CPG graph
        model_path: Model checkpoint path
        device: Device (cpu/cuda)
        
    Returns:
        List of detected vulnerabilities
    """
    import torch
    from ml.dataset import CPGGraphDataset
    from ml.embed_codebert import CodeBERTEmbedder
    
    device_obj = torch.device(device)
    
    # Load model
    model = load_model(model_path, device_obj)
    
    # Create dataset
    dataset = CPGGraphDataset([cpg_graph])
    
    # Run inference
    results = run_inference(model, dataset, device_obj)
    
    return results


def extract_top_k_findings(
    findings: List[Dict[str, Any]],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract top-K findings
    
    Args:
        findings: List of detection results
        k: Number to extract
        
    Returns:
        Top-K findings
    """
    # Sort by confidence
    sorted_findings = sorted(
        findings,
        key=lambda x: x.get("confidence", 0.0),
        reverse=True
    )
    
    return sorted_findings[:k]


def call_cloud_api(
    api_url: str,
    jwt_token: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call cloud API
    
    Args:
        api_url: API Gateway URL
        jwt_token: JWT token
        payload: Request payload
        
    Returns:
        Response
    """
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{api_url}/report",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()


def authenticate_with_oidc(
    api_url: str,
    github_oidc_token: str
) -> str:
    """
    Perform OIDC authentication and get JWT token
    
    Args:
        api_url: API Gateway URL
        github_oidc_token: GitHub OIDC token
        
    Returns:
        JWT token
    """
    response = requests.post(
        f"{api_url}/auth/oidc",
        json={"github_oidc_token": github_oidc_token},
        timeout=10
    )
    
    response.raise_for_status()
    data = response.json()
    return data["short_lived_jwt"]


def main():
    parser = argparse.ArgumentParser(
        description="NNAST Local Scan - Run vulnerability detection locally"
    )
    parser.add_argument(
        "repo_path",
        type=Path,
        help="Repository path to scan"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to GNN model checkpoint"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="NNAST Cloud API Gateway URL"
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token for Issue creation"
    )
    parser.add_argument(
        "--github-oidc-token",
        type=str,
        default=os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN"),
        help="GitHub OIDC token for authentication"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository (org/repo format)"
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        required=True,
        help="Tenant ID"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="Project ID"
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=os.getenv("GITHUB_SHA"),
        help="Commit SHA"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K findings to process"
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=5,
        help="Maximum number of issues to create"
    )
    parser.add_argument(
        "--cpg-cache-dir",
        type=Path,
        default=Path(".nnast_cache"),
        help="CPG cache directory"
    )
    
    args = parser.parse_args()
    
    # 1. Build full CPG
    print("[1/6] Building full CPG...")
    cpg_output = args.cpg_cache_dir / "cpg.jsonl"
    cpg_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Use Rust implementation if available, fallback to Python
    try:
        cpg_graph = build_full_cpg(args.repo_path, cpg_output, use_cache=True)
        implementation = "Rust" if hasattr(sys.modules.get('cpg_rust'), 'build_cpg') else "Python"
        print(f"  CPG built ({implementation}): {len(cpg_graph.get('nodes', []))} nodes, {len(cpg_graph.get('edges', []))} edges")
    except Exception as e:
        print(f"  Error building CPG: {e}", file=sys.stderr)
        raise
    
    # 2. GNN inference
    print("[2/6] Running GNN inference...")
    findings = run_gnn_inference(cpg_graph, args.model)
    print(f"  Found {len(findings)} potential vulnerabilities")
    
    # 3. Extract top-K
    print(f"[3/6] Extracting top-{args.top_k} findings...")
    top_findings = extract_top_k_findings(findings, args.top_k)
    print(f"  Selected {len(top_findings)} findings")
    
    # 4. Data minimization and masking
    print("[4/6] Minimizing and masking data...")
    
    # Read source code
    source_code = {}
    for py_file in args.repo_path.rglob("*.py"):
        try:
            source_code[str(py_file)] = py_file.read_text(encoding="utf-8")
        except:
            pass
    
    job_id = hashlib.sha256(f"{args.repo}:{args.commit_sha}".encode()).hexdigest()[:16]
    
    minimized_payload = create_minimized_payload(
        findings=top_findings,
        graph=cpg_graph,
        source_code=source_code,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        repo=args.repo,
        job_id=job_id,
        max_issues=args.max_issues
    )
    print(f"  Payload minimized: {len(minimized_payload['findings'])} findings")
    
    # 5. Call cloud API
    print("[5/6] Calling cloud API...")
    try:
        # OIDC authentication
        jwt_token = authenticate_with_oidc(args.api_url, args.github_oidc_token)
        
        # Report generation request
        response = call_cloud_api(args.api_url, jwt_token, minimized_payload)
        print(f"  Job queued: {response.get('job_id')}")
        
        # Wait for report generation (simple implementation)
        import time
        time.sleep(10)  # In practice, use polling or webhook
        
        # Get reports
        reports_response = requests.get(
            f"{args.api_url}/report/{job_id}",
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=30
        )
        
        if reports_response.status_code == 200:
            reports_data = reports_response.json()
            reports = reports_data.get("reports", [])
            print(f"  Reports generated: {len(reports)}")
        else:
            print(f"  Warning: Failed to get reports (status: {reports_response.status_code})")
            reports = []
            
    except Exception as e:
        print(f"  Error calling cloud API: {e}")
        reports = []
    
    # 6. Create GitHub Issues
    print("[6/6] Creating GitHub Issues...")
    if reports and args.github_token:
        try:
            issues = create_issues_from_reports(
                reports=reports,
                github_token=args.github_token,
                repo=args.repo,
                commit_sha=args.commit_sha,
                max_issues=args.max_issues
            )
            print(f"  Created/updated {len(issues)} issues")
        except Exception as e:
            print(f"  Error creating issues: {e}")
    else:
        print("  Skipped (no reports or no GitHub token)")
    
    print("\n[OK] Scan completed successfully")


if __name__ == "__main__":
    main()
