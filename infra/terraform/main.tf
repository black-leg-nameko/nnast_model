terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
  
  backend "s3" {
    # Configure backend in terraform.tfvars or via environment variables
    # bucket = "nnast-terraform-state"
    # key    = "nnast/terraform.tfstate"
    # region = "ap-northeast-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "NNAST"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

# KMS for encryption
module "kms" {
  source = "./modules/kms"
  
  environment = var.environment
}

# DynamoDB Tables
module "dynamodb" {
  source = "./modules/dynamodb"
  
  environment = var.environment
  kms_key_id  = module.kms.dynamodb_key_id
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"
  
  environment = var.environment
  kms_key_id  = module.kms.s3_key_id
}

# SQS Queues
module "sqs" {
  source = "./modules/sqs"
  
  environment = var.environment
  kms_key_id  = module.kms.sqs_key_id
}

# ECS Cluster and Services
module "ecs" {
  source = "./modules/ecs"
  
  environment           = var.environment
  vpc_id                = module.vpc.vpc_id
  private_subnet_ids    = module.vpc.private_subnet_ids
  public_subnet_ids     = module.vpc.public_subnet_ids
  report_queue_url      = module.sqs.report_queue_url
  report_queue_arn      = module.sqs.report_queue_arn
  dynamodb_table_arns   = module.dynamodb.table_arns
  s3_bucket_arns        = module.s3.bucket_arns
  kms_key_arns          = module.kms.all_key_arns
  bedrock_model_id      = var.bedrock_model_id
}

# API Gateway
module "api_gateway" {
  source = "./modules/api_gateway"
  
  environment              = var.environment
  auth_service_invoke_url  = module.ecs.auth_service_invoke_url
  report_service_invoke_url = module.ecs.report_service_invoke_url
  vpc_id                   = module.vpc.vpc_id
  vpc_cidr                 = var.vpc_cidr
  private_subnet_ids       = module.vpc.private_subnet_ids
}

# CloudWatch Logs
module "cloudwatch" {
  source = "./modules/cloudwatch"
  
  environment        = var.environment
  report_queue_name  = module.sqs.report_queue_name
  ecs_cluster_name   = module.ecs.cluster_name
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"
  
  environment              = var.environment
  account_id               = data.aws_caller_identity.current.account_id
  region                   = data.aws_region.current.name
  report_queue_arn         = module.sqs.report_queue_arn
  dynamodb_table_arns      = module.dynamodb.table_arns
  s3_bucket_arns           = module.s3.bucket_arns
  kms_key_arns             = module.kms.all_key_arns
  bedrock_model_id         = var.bedrock_model_id
  ecs_task_execution_role  = module.ecs.task_execution_role_arn
  ecs_task_role           = module.ecs.task_role_arn
}

# CloudTrail for audit logging
module "cloudtrail" {
  source = "./modules/cloudtrail"
  
  environment = var.environment
  s3_bucket_id = module.s3.cloudtrail_bucket_id
  kms_key_id   = module.kms.cloudtrail_key_id
}
