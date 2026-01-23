# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.environment}-nnast-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.environment}-nnast-cluster"
  }
}

# CloudWatch Log Group for ECS
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.environment}-nnast"
  retention_in_days = 30

  kms_key_id = var.kms_key_arns[0]  # Use first KMS key for logs

  tags = {
    Name = "${var.environment}-nnast-ecs-logs"
  }
}

# ECS Task Execution Role
resource "aws_iam_role" "task_execution" {
  name = "${var.environment}-nnast-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.environment}-nnast-task-execution-role"
  }
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role
resource "aws_iam_role" "task" {
  name = "${var.environment}-nnast-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.environment}-nnast-task-role"
  }
}

# ECS Task Definition - Auth Service
resource "aws_ecs_task_definition" "auth_service" {
  family                   = "${var.environment}-nnast-auth-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name  = "auth-service"
      image = "${var.auth_service_image}:${var.auth_service_tag}"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        }
      ]
      
      secrets = [
        {
          name      = "JWT_SECRET"
          valueFrom = aws_secretsmanager_secret.jwt_secret.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "auth-service"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name = "${var.environment}-nnast-auth-service"
  }
}

# ECS Task Definition - Report Service
resource "aws_ecs_task_definition" "report_service" {
  family                   = "${var.environment}-nnast-report-service"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name  = "report-service"
      image = "${var.report_service_image}:${var.report_service_tag}"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        },
        {
          name  = "BEDROCK_MODEL_ID"
          value = var.bedrock_model_id
        },
        {
          name  = "REPORT_QUEUE_URL"
          value = var.report_queue_url
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "report-service"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name = "${var.environment}-nnast-report-service"
  }
}

# Secrets Manager for JWT secret
resource "aws_secretsmanager_secret" "jwt_secret" {
  name        = "${var.environment}-nnast-jwt-secret"
  description = "JWT secret for NNAST auth service"

  kms_key_id = var.kms_key_arns[0]

  tags = {
    Name = "${var.environment}-nnast-jwt-secret"
  }
}

resource "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id = aws_secretsmanager_secret.jwt_secret.id
  secret_string = jsonencode({
    secret = random_password.jwt_secret.result
  })
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# Add provider requirement
# Note: random provider should be declared in root main.tf

# ECS Service - Auth Service
resource "aws_ecs_service" "auth_service" {
  name            = "${var.environment}-nnast-auth-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.auth_service.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.auth_service.arn
    container_name   = "auth-service"
    container_port   = 8080
  }

  depends_on = [
    aws_lb_listener.auth_service
  ]

  tags = {
    Name = "${var.environment}-nnast-auth-service"
  }
}

# ECS Service - Report Worker (consumes from SQS)
resource "aws_ecs_service" "report_worker" {
  name            = "${var.environment}-nnast-report-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.report_service.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  tags = {
    Name = "${var.environment}-nnast-report-worker"
  }
}

# Security Group for ECS
resource "aws_security_group" "ecs" {
  name        = "${var.environment}-nnast-ecs-sg"
  description = "Security group for ECS tasks"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-nnast-ecs-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.environment}-nnast-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = false

  tags = {
    Name = "${var.environment}-nnast-alb"
  }
}

# Security Group for ALB
resource "aws_security_group" "alb" {
  name        = "${var.environment}-nnast-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-nnast-alb-sg"
  }
}

# Target Group - Auth Service
resource "aws_lb_target_group" "auth_service" {
  name        = "${var.environment}-nnast-auth-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }

  tags = {
    Name = "${var.environment}-nnast-auth-tg"
  }
}

# ALB Listener - Auth Service
resource "aws_lb_listener" "auth_service" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.auth_service.arn
  }
}

data "aws_region" "current" {}

# Variables for container images (to be set via terraform.tfvars or environment)
variable "auth_service_image" {
  description = "Docker image for auth service (Go implementation)"
  type        = string
  default     = "nnast/auth-service"
}

variable "auth_service_tag" {
  description = "Docker image tag for auth service"
  type        = string
  default     = "latest"
}

variable "report_service_image" {
  description = "Docker image for report service"
  type        = string
  default     = "nnast/report-service"
}

variable "report_service_tag" {
  description = "Docker image tag for report service"
  type        = string
  default     = "latest"
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_names" {
  value = {
    auth_service  = aws_ecs_service.auth_service.name
    report_worker = aws_ecs_service.report_worker.name
  }
}

output "task_execution_role_arn" {
  value = aws_iam_role.task_execution.arn
}

output "task_role_arn" {
  value = aws_iam_role.task.arn
}

output "auth_service_invoke_url" {
  value = "http://${aws_lb.main.dns_name}"
}

output "report_service_invoke_url" {
  value = "http://${aws_lb.main.dns_name}"  # Same ALB, different path
}
