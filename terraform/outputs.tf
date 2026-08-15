output "dvc_bucket" {
  description = "S3 bucket for DVC data versioning"
  value       = aws_s3_bucket.dvc_storage.bucket
}

output "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "evidently_bucket" {
  description = "S3 bucket for Evidently reports"
  value       = aws_s3_bucket.evidently_reports.bucket
}

output "mlflow_db_endpoint" {
  description = "RDS endpoint for MLflow backend"
  value       = aws_db_instance.mlflow.address
}

output "mlflow_db_secret_arn" {
  description = "Secrets Manager ARN for MLflow DB credentials"
  value       = aws_secretsmanager_secret.mlflow_db.arn
}

output "mlops_instance_profile" {
  description = "Instance profile for MLOps EC2 instances"
  value       = aws_iam_instance_profile.mlops_runner.name
}

output "drift_alerts_topic_arn" {
  description = "SNS topic ARN for drift alerts"
  value       = aws_sns_topic.drift_alerts.arn
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI using RDS backend"
  value       = "postgresql://${var.mlflow_db_username}:PASSWORD@${aws_db_instance.mlflow.address}:5432/${var.mlflow_db_name}"
  sensitive   = true
}

output "destroy_command" {
  description = "Command to destroy all resources"
  value       = "terraform destroy --auto-approve"
}
