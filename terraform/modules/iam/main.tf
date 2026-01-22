# OIDC Provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url = var.github_oidc_issuer

  client_id_list = [
    var.github_oidc_audience
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
  ]

  tags = {
    Name = "${var.environment}-nnast-github-oidc"
  }
}

# IAM Role for GitHub Actions
resource "aws_iam_role" "github_actions" {
  name = "${var.environment}-nnast-github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.github_oidc_issuer, "https://", "")}:aud" = var.github_oidc_audience
          }
          StringLike = {
            "${replace(var.github_oidc_issuer, "https://", "")}:sub" = "repo:*"
          }
        }
      }
    ]
  })

  tags = {
    Name = "${var.environment}-nnast-github-actions-role"
  }
}

# Policy for GitHub Actions to call API Gateway
resource "aws_iam_role_policy" "github_actions_api" {
  name = "${var.environment}-nnast-github-actions-api-policy"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "execute-api:Invoke"
        ]
        Resource = [
          "arn:aws:execute-api:${var.region}:${var.account_id}:*/*/POST/auth/oidc",
          "arn:aws:execute-api:${var.region}:${var.account_id}:*/*/POST/report"
        ]
      }
    ]
  })
}

# IAM Policy for ECS Task Execution Role
resource "aws_iam_role_policy" "task_execution" {
  name = "${var.environment}-nnast-task-execution-policy"
  role = var.ecs_task_execution_role

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = var.kms_key_arns
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account_id}:log-group:/ecs/${var.environment}-nnast:*"
      }
    ]
  })
}

# IAM Policy for ECS Task Role (Auth Service)
resource "aws_iam_role_policy" "task_auth" {
  name = "${var.environment}-nnast-task-auth-policy"
  role = var.ecs_task_role

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = var.dynamodb_table_arns
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arns
      }
    ]
  })
}

# IAM Policy for ECS Task Role (Report Service)
resource "aws_iam_role_policy" "task_report" {
  name = "${var.environment}-nnast-task-report-policy"
  role = var.ecs_task_role

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = var.report_queue_arn
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = var.dynamodb_table_arns
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = [
          for arn in var.s3_bucket_arns : "${arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = "arn:aws:bedrock:${var.region}::foundation-model/${var.bedrock_model_id}"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arns
      }
    ]
  })
}

variable "github_oidc_issuer" {
  description = "GitHub OIDC issuer URL"
  type        = string
  default     = "https://token.actions.githubusercontent.com"
}

variable "github_oidc_audience" {
  description = "GitHub OIDC audience"
  type        = string
  default     = "nnast-cloud"
}

output "oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}
