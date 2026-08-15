provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      Owner       = var.owner
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_vpc" "default" { default = true }

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# -------------------------------------------------------
# S3 BUCKETS
# -------------------------------------------------------

resource "aws_s3_bucket" "dvc_storage" {
  bucket        = "${var.project_name}-dvc-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
  tags          = { Name = "${var.project_name}-dvc-storage" }
}

resource "aws_s3_bucket_versioning" "dvc_storage" {
  bucket = aws_s3_bucket.dvc_storage.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_storage" {
  bucket = aws_s3_bucket.dvc_storage.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_storage" {
  bucket                  = aws_s3_bucket.dvc_storage.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "${var.project_name}-mlflow-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
  tags          = { Name = "${var.project_name}-mlflow-artifacts" }
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "evidently_reports" {
  bucket        = "${var.project_name}-evidently-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
  tags          = { Name = "${var.project_name}-evidently-reports" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "evidently_reports" {
  bucket = aws_s3_bucket.evidently_reports.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "evidently_reports" {
  bucket                  = aws_s3_bucket.evidently_reports.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -------------------------------------------------------
# RDS POSTGRESQL — MLflow backend store
# -------------------------------------------------------

resource "random_password" "mlflow_db" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-mlflow-db-subnet"
  subnet_ids = data.aws_subnets.default.ids
  tags       = { Name = "${var.project_name}-mlflow-db-subnet" }
}

resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "MLflow RDS security group"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
    description = "PostgreSQL from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-rds-sg" }
}

resource "aws_db_instance" "mlflow" {
  identifier        = "${var.project_name}-mlflow-db"
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  storage_encrypted = true

  db_name  = var.mlflow_db_name
  username = var.mlflow_db_username
  password = random_password.mlflow_db.result

  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  skip_final_snapshot     = true
  deletion_protection     = false
  publicly_accessible     = false

  tags = { Name = "${var.project_name}-mlflow-db" }
}

resource "aws_secretsmanager_secret" "mlflow_db" {
  name                    = "${var.project_name}/mlflow/db-credentials"
  recovery_window_in_days = 0
  tags                    = { Name = "${var.project_name}-mlflow-db-secret" }
}

resource "aws_secretsmanager_secret_version" "mlflow_db" {
  secret_id = aws_secretsmanager_secret.mlflow_db.id
  secret_string = jsonencode({
    username = var.mlflow_db_username
    password = random_password.mlflow_db.result
    host     = aws_db_instance.mlflow.address
    port     = 5432
    dbname   = var.mlflow_db_name
  })
}

# -------------------------------------------------------
# IAM ROLE — MLOps EC2 instances
# -------------------------------------------------------

resource "aws_iam_role" "mlops_runner" {
  name = "${var.project_name}-runner-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "mlops_runner" {
  name = "${var.project_name}-runner-policy"
  role = aws_iam_role.mlops_runner.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3MLOpsAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
          "s3:ListBucket", "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.dvc_storage.arn,
          "${aws_s3_bucket.dvc_storage.arn}/*",
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
          aws_s3_bucket.evidently_reports.arn,
          "${aws_s3_bucket.evidently_reports.arn}/*"
        ]
      },
      {
        Sid      = "SecretsManagerAccess"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.mlflow_db.arn]
      },
      {
        Sid      = "SNSPublish"
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = [aws_sns_topic.drift_alerts.arn]
      },
      {
        Sid    = "SSMAccess"
        Effect = "Allow"
        Action = ["ssm:GetParameter", "ssm:PutParameter", "ssm:GetParameters"]
        Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/mlops/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.mlops_runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "mlops_runner" {
  name = "${var.project_name}-runner-profile"
  role = aws_iam_role.mlops_runner.name
}

# -------------------------------------------------------
# SNS — Drift detection alerts
# -------------------------------------------------------

resource "aws_sns_topic" "drift_alerts" {
  name = "${var.project_name}-drift-alerts"
  tags = { Name = "${var.project_name}-drift-alerts" }
}

resource "aws_sns_topic_subscription" "drift_email" {
  topic_arn = aws_sns_topic.drift_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# -------------------------------------------------------
# SSM PARAMETERS
# -------------------------------------------------------

resource "aws_ssm_parameter" "dvc_bucket" {
  name  = "/mlops/dvc-bucket"
  type  = "String"
  value = aws_s3_bucket.dvc_storage.bucket
  tags  = { Name = "${var.project_name}-dvc-bucket-param" }
}

resource "aws_ssm_parameter" "mlflow_bucket" {
  name  = "/mlops/mlflow-bucket"
  type  = "String"
  value = aws_s3_bucket.mlflow_artifacts.bucket
  tags  = { Name = "${var.project_name}-mlflow-bucket-param" }
}

resource "aws_ssm_parameter" "evidently_bucket" {
  name  = "/mlops/evidently-bucket"
  type  = "String"
  value = aws_s3_bucket.evidently_reports.bucket
  tags  = { Name = "${var.project_name}-evidently-bucket-param" }
}

resource "aws_ssm_parameter" "sns_topic_arn" {
  name  = "/mlops/drift-sns-arn"
  type  = "String"
  value = aws_sns_topic.drift_alerts.arn
  tags  = { Name = "${var.project_name}-sns-arn-param" }
}
