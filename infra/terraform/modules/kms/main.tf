# KMS Key for DynamoDB encryption
resource "aws_kms_key" "dynamodb" {
  description             = "KMS key for DynamoDB encryption - ${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name = "${var.environment}-nnast-dynamodb-key"
  }
}

resource "aws_kms_alias" "dynamodb" {
  name          = "alias/${var.environment}-nnast-dynamodb"
  target_key_id = aws_kms_key.dynamodb.key_id
}

# KMS Key for S3 encryption
resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption - ${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name = "${var.environment}-nnast-s3-key"
  }
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${var.environment}-nnast-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# KMS Key for SQS encryption
resource "aws_kms_key" "sqs" {
  description             = "KMS key for SQS encryption - ${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name = "${var.environment}-nnast-sqs-key"
  }
}

resource "aws_kms_alias" "sqs" {
  name          = "alias/${var.environment}-nnast-sqs"
  target_key_id = aws_kms_key.sqs.key_id
}

# KMS Key for CloudTrail encryption
resource "aws_kms_key" "cloudtrail" {
  description             = "KMS key for CloudTrail encryption - ${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow CloudTrail to encrypt logs"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "cloudtrail.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = {
    Name = "${var.environment}-nnast-cloudtrail-key"
  }
}

resource "aws_kms_alias" "cloudtrail" {
  name          = "alias/${var.environment}-nnast-cloudtrail"
  target_key_id = aws_kms_key.cloudtrail.key_id
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

output "dynamodb_key_id" {
  value = aws_kms_key.dynamodb.key_id
}

output "s3_key_id" {
  value = aws_kms_key.s3.key_id
}

output "sqs_key_id" {
  value = aws_kms_key.sqs.key_id
}

output "cloudtrail_key_id" {
  value = aws_kms_key.cloudtrail.key_id
}

output "all_key_arns" {
  value = [
    aws_kms_key.dynamodb.arn,
    aws_kms_key.s3.arn,
    aws_kms_key.sqs.arn,
    aws_kms_key.cloudtrail.arn
  ]
}

output "key_ids" {
  value = {
    dynamodb   = aws_kms_key.dynamodb.key_id
    s3         = aws_kms_key.s3.key_id
    sqs        = aws_kms_key.sqs.key_id
    cloudtrail = aws_kms_key.cloudtrail.key_id
  }
}
