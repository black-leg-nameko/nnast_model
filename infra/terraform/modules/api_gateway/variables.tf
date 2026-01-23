variable "environment" {
  description = "Environment name"
  type        = string
}

variable "auth_service_invoke_url" {
  description = "Auth service invoke URL"
  type        = string
}

variable "report_service_invoke_url" {
  description = "Report service invoke URL"
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

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
}
