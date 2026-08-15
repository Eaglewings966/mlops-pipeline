"""
Central configuration for the MLOps pipeline.
All settings read from environment variables or AWS SSM.
"""

import os
import json
import boto3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLOpsConfig:
    """Complete MLOps pipeline configuration."""

    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )
    dvc_bucket: str = field(
        default_factory=lambda: os.environ.get("DVC_BUCKET", "")
    )
    mlflow_bucket: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_BUCKET", "")
    )
    evidently_bucket: str = field(
        default_factory=lambda: os.environ.get("EVIDENTLY_BUCKET", "")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    )
    mlflow_model_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_MODEL_NAME", "fraud-detection-xgboost")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "data"))
    )
    raw_data_path: Path = field(
        default_factory=lambda: Path("data/raw/transactions.csv")
    )
    processed_data_path: Path = field(
        default_factory=lambda: Path("data/processed/transactions_processed.csv")
    )
    reference_data_path: Path = field(
        default_factory=lambda: Path("data/reference/reference_data.csv")
    )
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_optuna_trials: int = 50
    model_stage: str = field(
        default_factory=lambda: os.environ.get("MODEL_STAGE", "Production")
    )
    serving_host: str = "0.0.0.0"
    serving_port: int = 8000
    drift_threshold: float = 0.05
    psi_threshold: float = 0.2
    sns_topic_arn: str = field(
        default_factory=lambda: os.environ.get("SNS_TOPIC_ARN", "")
    )
    feature_columns: list = field(default_factory=lambda: [
        "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2",
        "P_emaildomain", "R_emaildomain",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
        "C9", "C10", "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
        "D9", "D10", "D11", "D12", "D13", "D14", "D15",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "V1", "V2", "V3", "V4", "V5", "V6",
        "TransactionHour", "TransactionDay",
        "amt_per_card", "transaction_count_per_card"
    ])
    target_column: str = "isFraud"


def get_mlflow_db_credentials() -> dict:
    """
    Fetch MLflow database credentials from AWS Secrets Manager.
    Falls back to environment variables for local development.
    """
    secret_name = os.environ.get(
        "MLFLOW_DB_SECRET_NAME",
        "mlops-pipeline/mlflow/db-credentials"
    )
    try:
        client = boto3.client(
            "secretsmanager",
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception:
        return {
            "username": os.environ.get("DB_USERNAME", "mlflowuser"),
            "password": os.environ.get("DB_PASSWORD", ""),
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "dbname": os.environ.get("DB_NAME", "mlflowdb")
        }


def build_mlflow_tracking_uri() -> str:
    """Build MLflow tracking URI from RDS credentials."""
    creds = get_mlflow_db_credentials()
    return (
        f"postgresql://{creds['username']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['dbname']}"
    )


def get_config() -> MLOpsConfig:
    """Return a populated MLOpsConfig instance."""
    return MLOpsConfig()
