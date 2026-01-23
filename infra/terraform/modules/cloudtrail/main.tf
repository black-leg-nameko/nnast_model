# CloudTrail for audit logging
resource "aws_cloudtrail" "main" {
  name           = "${var.environment}-nnast-cloudtrail"
  s3_bucket_name = var.s3_bucket_id

  include_global_service_events = true
  is_multi_region_trail         = true
  enable_logging                = true
  enable_log_file_validation    = true

  kms_key_id = var.kms_key_id

  event_selector {
    read_write_type                 = "All"
    include_management_events        = true
    exclude_management_event_sources = []

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::${var.s3_bucket_id}/*"]
    }

    data_resource {
      type   = "AWS::DynamoDB::Table"
      values = ["arn:aws:dynamodb:*:*:table/*"]
    }
  }

  tags = {
    Name = "${var.environment}-nnast-cloudtrail"
  }

  depends_on = [aws_s3_bucket_policy.cloudtrail]
}

# S3 Bucket Policy for CloudTrail
resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = var.s3_bucket_id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = "arn:aws:s3:::${var.s3_bucket_id}"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/${var.environment}-nnast-cloudtrail"
          }
        }
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "arn:aws:s3:::${var.s3_bucket_id}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/${var.environment}-nnast-cloudtrail"
            "s3:x-amz-acl"  = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

output "cloudtrail_arn" {
  value = aws_cloudtrail.main.arn
}
