variable "environment" {
  description = "Environment name"
  type        = string
}

variable "s3_bucket_id" {
  description = "S3 bucket ID for CloudTrail logs"
  type        = string
}

variable "kms_key_id" {
  description = "KMS key ID for CloudTrail encryption"
  type        = string
}
