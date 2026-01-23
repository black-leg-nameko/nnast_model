output "api_gateway_url" {
  description = "API Gateway endpoint URL"
  value       = module.api_gateway.api_url
}

output "api_gateway_id" {
  description = "API Gateway ID"
  value       = module.api_gateway.api_id
}

output "report_queue_url" {
  description = "SQS Report Queue URL"
  value       = module.sqs.report_queue_url
}

output "report_queue_arn" {
  description = "SQS Report Queue ARN"
  value       = module.sqs.report_queue_arn
}

output "dynamodb_table_names" {
  description = "DynamoDB table names"
  value       = module.dynamodb.table_names
}

output "s3_bucket_names" {
  description = "S3 bucket names"
  value       = module.s3.bucket_names
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_service_names" {
  description = "ECS service names"
  value       = module.ecs.service_names
}

output "kms_key_ids" {
  description = "KMS key IDs"
  value       = module.kms.key_ids
  sensitive   = false
}

output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value       = module.cloudwatch.log_group_names
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for GitHub Actions"
  value       = module.iam.oidc_provider_arn
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions (for reference)"
  value       = module.iam.github_actions_role_arn
}
