"""
SHAP-based model explainability — standalone module tracked by DVC.
Generates and saves SHAP summary and waterfall plots for the Production model.
"""

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_shap_plots(
    data_path: str,
    output_dir: str = "evidently_reports",
    n_samples: int = 1000,
) -> None:
    """
    Generate SHAP summary plot for the Production model.

    Args:
        data_path: Path to processed CSV
        output_dir: Directory to save plots
        n_samples: Number of samples to compute SHAP values on
    """
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    logger.info("Loading Production model for SHAP analysis...")
    model_uri = f"models:/{config.mlflow_model_name}/{config.model_stage}"
    model = mlflow.xgboost.load_model(model_uri)

    df = pd.read_csv(data_path)
    feature_cols = [c for c in config.feature_columns if c in df.columns]
    X = df[feature_cols].values[:n_samples]

    logger.info(f"Computing SHAP values for {n_samples} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP summary plot saved: {output_dir}/shap_summary.png")
