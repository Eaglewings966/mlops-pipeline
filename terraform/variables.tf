variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "mlops-pipeline"
}

variable "environment" {
  description = "Environment"
  type        = string
  default     = "dev"
}

variable "owner" {
  description = "Owner tag"
  type        = string
  default     = "emmanuel-ubani"
}

variable "mlflow_db_username" {
  description = "MLflow backend database username"
  type        = string
  default     = "mlflowuser"
}

variable "mlflow_db_name" {
  description = "MLflow backend database name"
  type        = string
  default     = "mlflowdb"
}

variable "alert_email" {
  description = "Email for drift alerts"
  type        = string
  default     = "devopsalert3@gmail.com"
}

variable "github_repo" {
  description = "GitHub repo in owner/name format, used to scope the OIDC trust policy"
  type        = string
  default     = "Eaglewings966/mlops-pipeline"
}
