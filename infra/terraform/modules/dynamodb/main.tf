# Tenant table
resource "aws_dynamodb_table" "tenants" {
  name           = "${var.environment}-nnast-tenants"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "tenant_id"

  attribute {
    name = "tenant_id"
    type = "S"
  }

  server_side_encryption {
    enabled     = true
    kms_key_id  = var.kms_key_id
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name = "${var.environment}-nnast-tenants"
  }
}

# Project table
resource "aws_dynamodb_table" "projects" {
  name           = "${var.environment}-nnast-projects"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "project_id"

  attribute {
    name = "project_id"
    type = "S"
  }

  global_secondary_index {
    name     = "tenant-id-index"
    hash_key = "tenant_id"
  }

  global_secondary_index {
    name     = "repo-index"
    hash_key = "github_org"
    range_key = "github_repo"
  }

  attribute {
    name = "tenant_id"
    type = "S"
  }

  attribute {
    name = "github_org"
    type = "S"
  }

  attribute {
    name = "github_repo"
    type = "S"
  }

  server_side_encryption {
    enabled     = true
    kms_key_id  = var.kms_key_id
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name = "${var.environment}-nnast-projects"
  }
}

# Job table
resource "aws_dynamodb_table" "jobs" {
  name           = "${var.environment}-nnast-jobs"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  global_secondary_index {
    name     = "tenant-id-index"
    hash_key = "tenant_id"
  }

  global_secondary_index {
    name     = "project-id-index"
    hash_key = "project_id"
  }

  global_secondary_index {
    name            = "status-created-index"
    hash_key        = "status"
    range_key       = "created_at"
  }

  attribute {
    name = "tenant_id"
    type = "S"
  }

  attribute {
    name = "project_id"
    type = "S"
  }

  attribute {
    name = "status"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_id  = var.kms_key_id
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name = "${var.environment}-nnast-jobs"
  }
}

# AuditLog table
resource "aws_dynamodb_table" "audit_logs" {
  name           = "${var.environment}-nnast-audit-logs"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "log_id"

  attribute {
    name = "log_id"
    type = "S"
  }

  global_secondary_index {
    name     = "tenant-timestamp-index"
    hash_key = "tenant_id"
    range_key = "timestamp"
  }

  global_secondary_index {
    name     = "actor-timestamp-index"
    hash_key = "actor"
    range_key = "timestamp"
  }

  attribute {
    name = "tenant_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "actor"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_id  = var.kms_key_id
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name = "${var.environment}-nnast-audit-logs"
  }
}

output "table_names" {
  value = {
    tenants   = aws_dynamodb_table.tenants.name
    projects  = aws_dynamodb_table.projects.name
    jobs      = aws_dynamodb_table.jobs.name
    audit_logs = aws_dynamodb_table.audit_logs.name
  }
}

output "table_arns" {
  value = [
    aws_dynamodb_table.tenants.arn,
    aws_dynamodb_table.projects.arn,
    aws_dynamodb_table.jobs.arn,
    aws_dynamodb_table.audit_logs.arn
  ]
}
