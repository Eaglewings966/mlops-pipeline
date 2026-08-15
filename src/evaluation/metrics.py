"""
Model evaluation metrics — standalone module tracked by DVC.
Computes and persists the full evaluation report for a registered MLflow model.
"""

import json
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
)
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    data_path: str,
    output_path: str = "data/processed/evaluation_metrics.json",
) -> Dict[str, Any]:
    """
    Load the Production model from MLflow and evaluate it on the processed dataset.

    Args:
        data_path: Path to processed CSV
        output_path: Path to write evaluation metrics JSON

    Returns:
        Dictionary of evaluation metrics
    """
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    logger.info("Loading Production model from MLflow registry...")
    model_uri = f"models:/{config.mlflow_model_name}/{config.model_stage}"
    model = mlflow.xgboost.load_model(model_uri)

    df = pd.read_csv(data_path)
    feature_cols = [c for c in config.feature_columns if c in df.columns]
    X = df[feature_cols].values
    y = df[config.target_column].values

    # Use last 20% as held-out test set (same split as trainer)
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    metrics = {
        "average_precision": float(average_precision_score(y_test, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "test_rows": len(y_test),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
    }

    logger.info(
        f"Evaluation — AUPRC={metrics['average_precision']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
