# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.environment}-nnast"
  retention_in_days = 30

  tags = {
    Name = "${var.environment}-nnast-api-gateway-logs"
  }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.environment}-nnast"
  retention_in_days = 30

  tags = {
    Name = "${var.environment}-nnast-ecs-logs"
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "sqs_queue_depth" {
  name          = "${var.environment}-nnast-sqs-queue-depth"
  description   = "Alert when SQS queue depth is high"
  alarm_actions = []  # Add SNS topic ARN for notifications

  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 2
  threshold           = 100
  comparison_operator = "GreaterThanThreshold"

  dimensions = {
    QueueName = var.report_queue_name
  }

  tags = {
    Name = "${var.environment}-nnast-sqs-queue-depth-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_task_failures" {
  name          = "${var.environment}-nnast-ecs-task-failures"
  description   = "Alert when ECS tasks are failing"
  alarm_actions = []  # Add SNS topic ARN for notifications

  metric_name         = "StoppedTaskCount"
  namespace           = "AWS/ECS"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"

  dimensions = {
    ClusterName = var.ecs_cluster_name
  }

  tags = {
    Name = "${var.environment}-nnast-ecs-task-failures-alarm"
  }
}

variable "report_queue_name" {
  description = "SQS Report Queue name"
  type        = string
  default     = ""
}

variable "ecs_cluster_name" {
  description = "ECS cluster name"
  type        = string
  default     = ""
}

output "log_group_names" {
  value = [
    aws_cloudwatch_log_group.api_gateway.name,
    aws_cloudwatch_log_group.ecs.name
  ]
}
