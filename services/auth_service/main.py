#!/usr/bin/env python3
"""
NNAST Auth Service

Service that verifies GitHub Actions OIDC tokens and issues short-lived JWTs.
"""
import os
import json
import time
import hmac
import hashlib
import base64
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import boto3
from flask import Flask, request, jsonify
from jose import jwt, jwk
from jose.utils import base64url_decode
import requests

app = Flask(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JWT_SECRET = os.getenv("JWT_SECRET", "")

# AWS clients
dynamodb = boto3.resource("dynamodb")
audit_logs_table = dynamodb.Table(f"{ENVIRONMENT}-nnast-audit-logs")

# GitHub OIDC configuration
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_OIDC_AUDIENCE = "nnast-cloud"
JWT_EXPIRATION_MINUTES = 15


def get_github_oidc_jwks() -> Dict[str, Any]:
    """Get GitHub OIDC JWKS"""
    jwks_url = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks.json"
    response = requests.get(jwks_url, timeout=10)
    response.raise_for_status()
    return response.json()


def verify_github_oidc_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify GitHub OIDC token
    
    Returns:
        On success: Decoded token payload
        On failure: None
    """
    try:
        # Get header (unverified)
        unverified_header = jwt.get_unverified_header(token)
        
        # Get JWKS
        jwks = get_github_oidc_jwks()
        
        # Get key ID
        kid = unverified_header.get("kid")
        if not kid:
            return None
        
        # Find corresponding key
        key = None
        for jwk_key in jwks.get("keys", []):
            if jwk_key.get("kid") == kid:
                key = jwk.construct(jwk_key)
                break
        
        if not key:
            return None
        
        # Verify token
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=GITHUB_OIDC_AUDIENCE,
            issuer=GITHUB_OIDC_ISSUER
        )
        
        return payload
        
    except Exception as e:
        app.logger.error(f"OIDC token verification failed: {e}")
        return None


def get_project_by_repo(github_org: str, github_repo: str) -> Optional[Dict[str, Any]]:
    """Get project information from repository"""
    try:
        projects_table = dynamodb.Table(f"{ENVIRONMENT}-nnast-projects")
        
        response = projects_table.query(
            IndexName="repo-index",
            KeyConditionExpression="github_org = :org AND github_repo = :repo",
            ExpressionAttributeValues={
                ":org": github_org,
                ":repo": github_repo
            }
        )
        
        items = response.get("Items", [])
        if items:
            return items[0]
        return None
        
    except Exception as e:
        app.logger.error(f"Failed to get project: {e}")
        return None


def check_subject_allowed(subject: str, project: Dict[str, Any]) -> bool:
    """Check if OIDC subject is allowed"""
    allowed_patterns = project.get("allowed_subject_patterns", [])
    
    if not allowed_patterns:
        # Default: allow repo:{org}/{repo}:*
        return True
    
    for pattern in allowed_patterns:
        # Simple wildcard matching
        pattern_regex = pattern.replace("*", ".*")
        import re
        if re.match(pattern_regex, subject):
            return True
    
    return False


def generate_jwt(tenant_id: str, project_id: str, repo: str, subject: str) -> str:
    """Generate short-lived JWT"""
    now = datetime.utcnow()
    exp = now + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    payload = {
        "iss": "nnast-cloud",
        "sub": subject,
        "aud": "nnast-api",
        "exp": int(exp.timestamp()),
        "iat": int(now.timestamp()),
        "tenant_id": tenant_id,
        "project_id": project_id,
        "repo": repo,
        "scopes": ["report:generate"]
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def log_audit_event(
    tenant_id: str,
    actor: str,
    action: str,
    result: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log audit event"""
    try:
        log_id = f"{int(time.time() * 1000)}-{hashlib.md5(actor.encode()).hexdigest()[:8]}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        item = {
            "log_id": log_id,
            "tenant_id": tenant_id,
            "actor": actor,
            "action": action,
            "result": result,
            "timestamp": timestamp,
            "ttl": int((datetime.utcnow() + timedelta(days=365)).timestamp())
        }
        
        if metadata:
            item["metadata"] = json.dumps(metadata)
        
        audit_logs_table.put_item(Item=item)
        
    except Exception as e:
        app.logger.error(f"Failed to log audit event: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({"status": "healthy"}), 200


@app.route("/auth/oidc", methods=["POST"])
def auth_oidc():
    """
    Verify GitHub OIDC token and issue short-lived JWT
    
    Request:
        {
            "github_oidc_token": "..."
        }
    
    Response:
        {
            "short_lived_jwt": "..."
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request body"}), 400
        
        github_token = data.get("github_oidc_token")
        if not github_token:
            return jsonify({"error": "github_oidc_token is required"}), 400
        
        # Verify OIDC token
        payload = verify_github_oidc_token(github_token)
        if not payload:
            return jsonify({"error": "Invalid OIDC token"}), 401
        
        # Get subject
        subject = payload.get("sub", "")
        if not subject:
            return jsonify({"error": "Invalid token subject"}), 401
        
        # Extract repository information from subject
        # Format: repo:{org}/{repo}:ref:refs/heads/{branch}
        parts = subject.split(":")
        if len(parts) < 2 or parts[0] != "repo":
            return jsonify({"error": "Invalid subject format"}), 401
        
        repo_path = parts[1]  # {org}/{repo}
        repo_parts = repo_path.split("/")
        if len(repo_parts) != 2:
            return jsonify({"error": "Invalid repo format"}), 401
        
        github_org = repo_parts[0]
        github_repo = repo_parts[1]
        
        # Get project
        project = get_project_by_repo(github_org, github_repo)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        # Check if subject is allowed
        if not check_subject_allowed(subject, project):
            log_audit_event(
                tenant_id=project.get("tenant_id", "unknown"),
                actor=subject,
                action="auth",
                result="denied",
                metadata={"reason": "subject_not_allowed"}
            )
            return jsonify({"error": "Subject not allowed"}), 403
        
        tenant_id = project.get("tenant_id")
        project_id = project.get("project_id")
        
        # Generate JWT
        short_jwt = generate_jwt(tenant_id, project_id, repo_path, subject)
        
        # Audit log
        log_audit_event(
            tenant_id=tenant_id,
            actor=subject,
            action="auth",
            result="success",
            metadata={"project_id": project_id, "repo": repo_path}
        )
        
        return jsonify({
            "short_lived_jwt": short_jwt
        }), 200
        
    except Exception as e:
        app.logger.error(f"Auth error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET environment variable is required")
    
    app.run(host="0.0.0.0", port=8080, debug=(ENVIRONMENT == "dev"))
