variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "Public subnet IDs"
  type        = list(string)
}

variable "report_queue_url" {
  description = "SQS Report Queue URL"
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

variable "task_cpu" {
  description = "CPU units for ECS tasks"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory (MB) for ECS tasks"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}
