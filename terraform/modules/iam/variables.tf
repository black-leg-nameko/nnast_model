variable "environment" {
  description = "Environment name"
  type        = string
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "report_queue_arn" {
  description = "SQS Report Queue ARN"
  type        = string
}

variable "dynamodb_table_arns" {
  description = "DynamoDB table ARNs"
  type        = list(string)
}

variable "s3_bucket_arns" {
  description = "S3 bucket ARNs"
  type        = list(string)
}

variable "kms_key_arns" {
  description = "KMS key ARNs"
  type        = list(string)
}

variable "bedrock_model_id" {
  description = "Bedrock model ID"
  type        = string
}

variable "ecs_task_execution_role" {
  description = "ECS task execution role ARN"
  type        = string
}

variable "ecs_task_role" {
  description = "ECS task role ARN"
  type        = string
}
