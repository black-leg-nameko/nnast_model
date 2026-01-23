# NNAST Terraform Infrastructure

This directory contains Terraform code for building AWS infrastructure for the NNAST system.

## Architecture

Based on the design document (`nnast_system_design.md`), the following AWS resources are built:

- **VPC**: Network configuration (public/private subnets, NAT Gateway)
- **KMS**: Encryption keys (for DynamoDB, S3, SQS, CloudTrail)
- **DynamoDB**: Data store (Tenant, Project, Job, AuditLog tables)
- **S3**: Storage (report artifacts, CloudTrail logs)
- **SQS**: Message queue (report generation jobs, DLQ)
- **ECS**: Container execution environment (Auth Service, Report Worker)
- **API Gateway**: HTTP API (OIDC authentication, report generation endpoints)
- **IAM**: Roles and policies (GitHub Actions OIDC, ECS tasks)
- **CloudWatch**: Logs and alarms
- **CloudTrail**: Audit logs

## Setup

### 1. Prerequisites

- Terraform >= 1.0
- AWS CLI configured
- Appropriate AWS permissions

### 2. Prepare configuration file

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and modify settings according to your environment.

### 3. Initialize Terraform

```bash
cd infra/terraform
terraform init
```

### 4. Review plan

```bash
terraform plan
```

### 5. Apply

```bash
terraform apply
```

## Module Structure

- `modules/vpc`: VPC and network resources
- `modules/kms`: KMS keys
- `modules/dynamodb`: DynamoDB tables
- `modules/s3`: S3 buckets
- `modules/sqs`: SQS queues
- `modules/ecs`: ECS cluster and services
- `modules/api_gateway`: API Gateway
- `modules/iam`: IAM roles and policies
- `modules/cloudwatch`: CloudWatch logs and alarms
- `modules/cloudtrail`: CloudTrail configuration

## Important Outputs

After applying, the following output values are available:

- `api_gateway_url`: API Gateway endpoint URL
- `oidc_provider_arn`: OIDC provider ARN for GitHub Actions
- `github_actions_role_arn`: IAM role ARN for GitHub Actions

## Notes

1. **KMS Key Deletion Protection**: KMS keys have a 30-day deletion protection period
2. **S3 Bucket Names**: Must be globally unique (includes account ID)
3. **ECS Task Definitions**: Container images must be pushed to ECR beforehand
4. **Secrets Manager**: JWT secrets are auto-generated, but manual configuration is recommended for production

## Next Steps

After infrastructure setup:

1. Push container images to ECR
2. Configure OIDC authentication in GitHub Actions workflows
3. Register tenants/projects in the management interface
4. Execute NNAST from GitHub Actions
