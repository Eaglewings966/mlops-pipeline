"""
Predictor utilities for the serving layer.
Wraps model loading and feature preparation for use outside the API context.
"""

import mlflow
import mlflow.xgboost
import numpy as np
from typing import Dict, Any
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_production_model():
    """Load the Production model from MLflow Model Registry."""
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    model_uri = f"models:/{config.mlflow_model_name}/{config.model_stage}"
    logger.info(f"Loading model from {model_uri}")
    return mlflow.xgboost.load_model(model_uri)


def predict_single(model, features: Dict[str, Any], feature_columns: list) -> float:
    """
    Run inference for a single transaction feature dict.

    Args:
        model: Loaded XGBoost model
        features: Dict of feature name → value
        feature_columns: Ordered list of feature names

    Returns:
        Fraud probability (float between 0 and 1)
    """
    vector = []
    for col in feature_columns:
        value = features.get(col, 0.0)
        if value is None:
            value = 0.0
        if isinstance(value, str):
            value = float(hash(value) % 1000)
        vector.append(float(value))

    X = np.array([vector])
    return float(model.predict_proba(X)[0, 1])
