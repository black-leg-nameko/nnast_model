# S3 bucket for report artifacts
resource "aws_s3_bucket" "reports" {
  bucket = "${var.environment}-nnast-reports-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.environment}-nnast-reports"
  }
}

resource "aws_s3_bucket_versioning" "reports" {
  bucket = aws_s3_bucket.reports.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "reports" {
  bucket = aws_s3_bucket.reports.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_id
    }
  }
}

resource "aws_s3_bucket_public_access_block" "reports" {
  bucket = aws_s3_bucket.reports.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "reports" {
  bucket = aws_s3_bucket.reports.id

  rule {
    id     = "delete-old-reports"
    status = "Enabled"

    expiration {
      days = 90
    }
  }
}

# S3 bucket for CloudTrail logs
resource "aws_s3_bucket" "cloudtrail" {
  bucket = "${var.environment}-nnast-cloudtrail-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.environment}-nnast-cloudtrail"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_id
    }
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  rule {
    id     = "delete-old-logs"
    status = "Enabled"

    expiration {
      days = 365
    }
  }
}

data "aws_caller_identity" "current" {}

output "bucket_names" {
  value = {
    reports    = aws_s3_bucket.reports.id
    cloudtrail = aws_s3_bucket.cloudtrail.id
  }
}

output "bucket_arns" {
  value = [
    aws_s3_bucket.reports.arn,
    aws_s3_bucket.cloudtrail.arn
  ]
}

output "cloudtrail_bucket_id" {
  value = aws_s3_bucket.cloudtrail.id
}
