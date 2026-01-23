#!/usr/bin/env python3
"""
NNAST Report Service

Service that receives report generation jobs from SQS and generates Markdown reports using Bedrock LLM.
"""
import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import boto3
from flask import Flask, request, jsonify

app = Flask(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
REPORT_QUEUE_URL = os.getenv("REPORT_QUEUE_URL", "")

# AWS clients
sqs = boto3.client("sqs")
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "ap-northeast-1"))
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

jobs_table = dynamodb.Table(f"{ENVIRONMENT}-nnast-jobs")
reports_bucket = f"{ENVIRONMENT}-nnast-reports-{boto3.client('sts').get_caller_identity()['Account']}"


def generate_report_markdown(finding: Dict[str, Any]) -> str:
    """
    Generate report Markdown using Bedrock LLM
    
    Args:
        finding: Vulnerability detection information
        
    Returns:
        Report in Markdown format
    """
    severity = finding.get("severity", "MEDIUM")
    rule_id = finding.get("rule_id", "UNKNOWN")
    cwe = finding.get("cwe", "")
    location = finding.get("location", {})
    subgraph = finding.get("subgraph", {})
    code_snippet = finding.get("code_snippet_masked", "")
    
    # Build prompt for Bedrock
    prompt = f"""You are a security expert. Based on the following vulnerability detection information, generate a diagnostic report in Markdown format for developers.

## Detection Information
- **Severity**: {severity}
- **Rule ID**: {rule_id}
- **CWE**: {cwe}
- **Location**: {location.get('file', 'unknown')}:{location.get('line', 'unknown')}

## Code Snippet
```python
{code_snippet}
```

## Subgraph Information
{json.dumps(subgraph, indent=2, ensure_ascii=False)}

## Report Requirements
1. **Summary**: Briefly explain the type and impact of the vulnerability
2. **Detailed Explanation**: Explain why this code is vulnerable and what attacks are possible
3. **Impact Scope**: The impact of this vulnerability (confidential information leakage, service disruption, etc.)
4. **Fix Method**: Fix proposal including specific fix code examples
5. **Reference Information**: Links to related CWE and best practices

Please output in Markdown format. Format code blocks appropriately.
"""
    
    try:
        # Call Bedrock API
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        markdown = response_body["content"][0]["text"]
        
        return markdown
        
    except Exception as e:
        app.logger.error(f"Bedrock API error: {e}")
        # Fallback: Generate simple report
        return f"""# Vulnerability Detection Report

## Summary
- **Severity**: {severity}
- **Rule ID**: {rule_id}
- **CWE**: {cwe}
- **Location**: {location.get('file', 'unknown')}:{location.get('line', 'unknown')}

## Code Snippet
```python
{code_snippet}
```

## Note
Failed to generate detailed diagnostic report. Please review manually.
"""
    except Exception as e:
        app.logger.error(f"Failed to generate report: {e}")
        return None


def process_report_job(job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process report generation job
    
    Args:
        job_data: Job data
        
    Returns:
        List of generated reports
    """
    job_id = job_data.get("job_id", str(uuid.uuid4()))
    tenant_id = job_data.get("tenant_id")
    project_id = job_data.get("project_id")
    repo = job_data.get("repo")
    findings = job_data.get("findings", [])
    limits = job_data.get("limits", {})
    max_issues = limits.get("max_issues", 5)
    
    # Update job status
    try:
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #status = :status, started_at = :started_at",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": "running",
                ":started_at": datetime.utcnow().isoformat() + "Z"
            }
        )
    except Exception as e:
        app.logger.error(f"Failed to update job status: {e}")
    
    reports = []
    
    # Process only top-K findings
    findings_to_process = findings[:max_issues]
    
    for finding in findings_to_process:
        try:
            fingerprint = finding.get("fingerprint")
            
            # Generate report
            markdown = generate_report_markdown(finding)
            if not markdown:
                continue
            
            # Save to S3
            s3_key = f"reports/{job_id}/{fingerprint}.md"
            s3.put_object(
                Bucket=reports_bucket,
                Key=s3_key,
                Body=markdown.encode("utf-8"),
                ContentType="text/markdown"
            )
            
            # Build report information
            report = {
                "fingerprint": fingerprint,
                "title": f"[NNAST][{finding.get('severity', 'MEDIUM')}][{finding.get('cwe', 'UNKNOWN')}] {finding.get('rule_id', 'UNKNOWN')} in {finding.get('location', {}).get('file', 'unknown')}",
                "labels": [
                    "security",
                    "nnast",
                    f"severity:{finding.get('severity', 'medium').lower()}",
                    f"cwe:{finding.get('cwe', 'unknown').lower()}"
                ],
                "markdown": markdown,
                "s3_key": s3_key
            }
            
            reports.append(report)
            
        except Exception as e:
            app.logger.error(f"Failed to process finding: {e}", exc_info=True)
            continue
    
    # Update job status to completed
    try:
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #status = :status, finished_at = :finished_at, report_count = :count",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": "done",
                ":finished_at": datetime.utcnow().isoformat() + "Z",
                ":count": len(reports)
            }
        )
    except Exception as e:
        app.logger.error(f"Failed to update job status: {e}")
    
    return reports


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({"status": "healthy"}), 200


@app.route("/report", methods=["POST"])
def create_report():
    """
    Create report generation job (enqueue to SQS)
    
    Request:
        {
            "tenant_id": "...",
            "project_id": "...",
            "repo": "...",
            "job_id": "...",
            "findings": [...],
            "limits": {"max_issues": 5}
        }
    
    Response:
        {
            "job_id": "...",
            "status": "queued"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request body"}), 400
        
        # JWT verification (simple implementation, implement proper verification for production)
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        
        job_id = data.get("job_id", str(uuid.uuid4()))
        tenant_id = data.get("tenant_id")
        project_id = data.get("project_id")
        repo = data.get("repo")
        findings = data.get("findings", [])
        
        if not all([tenant_id, project_id, repo]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Record job in DynamoDB
        try:
            jobs_table.put_item(
                Item={
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "repo": repo,
                    "status": "queued",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "findings_count": len(findings),
                    "ttl": int((datetime.utcnow() + timedelta(days=30)).timestamp())
                }
            )
        except Exception as e:
            app.logger.error(f"Failed to create job: {e}")
            return jsonify({"error": "Failed to create job"}), 500
        
        # Send message to SQS
        try:
            sqs.send_message(
                QueueUrl=REPORT_QUEUE_URL,
                MessageBody=json.dumps(data),
                MessageAttributes={
                    "job_id": {
                        "StringValue": job_id,
                        "DataType": "String"
                    },
                    "tenant_id": {
                        "StringValue": tenant_id,
                        "DataType": "String"
                    }
                }
            )
        except Exception as e:
            app.logger.error(f"Failed to send message to SQS: {e}")
            return jsonify({"error": "Failed to queue job"}), 500
        
        return jsonify({
            "job_id": job_id,
            "status": "queued"
        }), 202
        
    except Exception as e:
        app.logger.error(f"Report creation error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/report/<job_id>", methods=["GET"])
def get_report(job_id: str):
    """
    Get report generation results
    
    Response:
        {
            "job_id": "...",
            "status": "done",
            "reports": [...]
        }
    """
    try:
        # Get job information
        response = jobs_table.get_item(Key={"job_id": job_id})
        if "Item" not in response:
            return jsonify({"error": "Job not found"}), 404
        
        job = response["Item"]
        status = job.get("status")
        
        if status != "done":
            return jsonify({
                "job_id": job_id,
                "status": status
            }), 200
        
        # Get reports from S3
        reports = []
        # TODO: Implement getting reports from S3 and returning them
        
        return jsonify({
            "job_id": job_id,
            "status": status,
            "reports": reports
        }), 200
        
    except Exception as e:
        app.logger.error(f"Get report error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def sqs_worker():
    """Worker that receives and processes messages from SQS"""
    while True:
        try:
            # Receive message
            response = sqs.receive_message(
                QueueUrl=REPORT_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # Long polling
                MessageAttributeNames=["All"]
            )
            
            messages = response.get("Messages", [])
            if not messages:
                continue
            
            for message in messages:
                try:
                    # Parse message body
                    job_data = json.loads(message["Body"])
                    
                    # Generate reports
                    reports = process_report_job(job_data)
                    
                    # Delete message
                    sqs.delete_message(
                        QueueUrl=REPORT_QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"]
                    )
                    
                    app.logger.info(f"Processed job {job_data.get('job_id')}: {len(reports)} reports generated")
                    
                except Exception as e:
                    app.logger.error(f"Failed to process message: {e}", exc_info=True)
                    # Delete message even on error (will be sent to DLQ)
                    try:
                        sqs.delete_message(
                            QueueUrl=REPORT_QUEUE_URL,
                            ReceiptHandle=message["ReceiptHandle"]
                        )
                    except:
                        pass
                        
        except Exception as e:
            app.logger.error(f"SQS worker error: {e}", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    import threading
    
    # Start SQS worker in separate thread
    worker_thread = threading.Thread(target=sqs_worker, daemon=True)
    worker_thread.start()
    
    app.run(host="0.0.0.0", port=8080, debug=(ENVIRONMENT == "dev"))
