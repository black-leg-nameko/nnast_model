# SQS Queue for report generation jobs
resource "aws_sqs_queue" "report" {
  name                       = "${var.environment}-nnast-report-queue"
  visibility_timeout_seconds = 300  # 5 minutes (should be > task timeout)
  message_retention_seconds   = 86400  # 24 hours
  receive_wait_time_seconds   = 20  # Long polling

  kms_master_key_id                 = var.kms_key_id
  kms_data_key_reuse_period_seconds = 300

  tags = {
    Name = "${var.environment}-nnast-report-queue"
  }
}

# Dead Letter Queue for failed jobs
resource "aws_sqs_queue" "report_dlq" {
  name                      = "${var.environment}-nnast-report-dlq"
  message_retention_seconds  = 1209600  # 14 days

  kms_master_key_id                 = var.kms_key_id
  kms_data_key_reuse_period_seconds = 300

  tags = {
    Name = "${var.environment}-nnast-report-dlq"
  }
}

resource "aws_sqs_queue_redrive_policy" "report" {
  queue_url = aws_sqs_queue.report.id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.report_dlq.arn
    maxReceiveCount     = 3
  })
}

output "report_queue_url" {
  value = aws_sqs_queue.report.id
}

output "report_queue_arn" {
  value = aws_sqs_queue.report.arn
}

output "report_dlq_url" {
  value = aws_sqs_queue.report_dlq.id
}

output "report_dlq_arn" {
  value = aws_sqs_queue.report_dlq.arn
}

output "report_queue_name" {
  value = aws_sqs_queue.report.name
}
