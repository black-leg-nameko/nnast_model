# API Gateway
resource "aws_apigatewayv2_api" "main" {
  name          = "${var.environment}-nnast-api"
  protocol_type = "HTTP"
  description   = "NNAST API Gateway"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
    max_age      = 300
  }

  tags = {
    Name = "${var.environment}-nnast-api"
  }
}

# VPC Link for private ECS services
resource "aws_apigatewayv2_vpc_link" "main" {
  name               = "${var.environment}-nnast-vpc-link"
  security_group_ids = [aws_security_group.vpc_link.id]
  subnet_ids         = var.private_subnet_ids

  tags = {
    Name = "${var.environment}-nnast-vpc-link"
  }
}

# Security Group for VPC Link
resource "aws_security_group" "vpc_link" {
  name        = "${var.environment}-nnast-vpc-link-sg"
  description = "Security group for API Gateway VPC Link"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-nnast-vpc-link-sg"
  }
}

# Integration - Auth Service
resource "aws_apigatewayv2_integration" "auth_service" {
  api_id = aws_apigatewayv2_api.main.id

  integration_type   = "HTTP_PROXY"
  integration_method = "POST"
  integration_uri     = "${var.auth_service_invoke_url}/auth/oidc"
  connection_type     = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.main.id
}

# Integration - Report Service
resource "aws_apigatewayv2_integration" "report_service" {
  api_id = aws_apigatewayv2_api.main.id

  integration_type   = "HTTP_PROXY"
  integration_method  = "POST"
  integration_uri     = "${var.report_service_invoke_url}/report"
  connection_type     = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.main.id
}

# Route - /auth/oidc
resource "aws_apigatewayv2_route" "auth_oidc" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /auth/oidc"

  target = "integrations/${aws_apigatewayv2_integration.auth_service.id}"
}

# Route - /report
resource "aws_apigatewayv2_route" "report" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /report"

  target = "integrations/${aws_apigatewayv2_integration.report_service.id}"
}

# Stage
resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  tags = {
    Name = "${var.environment}-nnast-api-stage"
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.environment}-nnast"
  retention_in_days = 30

  tags = {
    Name = "${var.environment}-nnast-api-gateway-logs"
  }
}


output "api_id" {
  value = aws_apigatewayv2_api.main.id
}

output "api_url" {
  value = aws_apigatewayv2_api.main.api_endpoint
}
