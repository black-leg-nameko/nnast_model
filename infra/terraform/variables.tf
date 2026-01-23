variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-northeast-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "bedrock_model_id" {
  description = "Bedrock model ID for LLM inference (e.g., anthropic.claude-3-sonnet-20240229-v1:0)"
  type        = string
  default     = "anthropic.claude-3-sonnet-20240229-v1:0"
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

variable "top_k_findings" {
  description = "Maximum number of findings to process per PR"
  type        = number
  default     = 5
}

variable "max_issues_per_pr" {
  description = "Maximum number of issues to create per PR"
  type        = number
  default     = 5
}

variable "ecs_task_cpu" {
  description = "CPU units for ECS tasks"
  type        = number
  default     = 512
}

variable "ecs_task_memory" {
  description = "Memory (MB) for ECS tasks"
  type        = number
  default     = 1024
}

variable "ecs_desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}

variable "ecs_max_count" {
  description = "Maximum number of ECS tasks for auto-scaling"
  type        = number
  default     = 10
}

variable "enable_cloudtrail" {
  description = "Enable CloudTrail for audit logging"
  type        = bool
  default     = true
}
