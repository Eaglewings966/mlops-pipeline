"""
XGBoost model trainer with full MLflow experiment tracking.
Handles class imbalance with SMOTE oversampling.
"""

import json
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.config import get_config
from src.training.hyperparameter_tuning import run_hyperparameter_tuning
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load training parameters from params.yaml."""
    with open(params_path) as f:
        return yaml.safe_load(f)


def train_fraud_model(
    data_path: str,
    params_path: str = "params.yaml",
    run_tuning: bool = True
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], str]:
    """
    Train the XGBoost fraud detection model with full MLflow tracking.

    Steps:
    1. Load processed data and split into train/val/test
    2. Handle class imbalance with SMOTE
    3. Run Optuna hyperparameter tuning (optional)
    4. Train final model with best parameters
    5. Evaluate on test set and generate SHAP plots
    6. Log everything to MLflow and register model

    Args:
        data_path: Path to processed CSV
        params_path: Path to params.yaml
        run_tuning: Whether to run Optuna tuning

    Returns:
        Tuple of (trained model, metrics dict, MLflow run ID)
    """
    config = get_config()
    params = load_params(params_path)

    logger.info("Loading processed data...")
    df = pd.read_csv(data_path)
    feature_cols = [c for c in config.feature_columns if c in df.columns]
    X = df[feature_cols].values
    y = df[config.target_column].values

    logger.info(
        f"Dataset: {len(df):,} rows, {len(feature_cols)} features, "
        f"fraud rate: {y.mean():.3%}"
    )

    data_params = params.get("data", {})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_params.get("test_size", 0.2),
        random_state=data_params.get("random_state", 42),
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=data_params.get("validation_size", 0.1),
        random_state=data_params.get("random_state", 42),
        stratify=y_train
    )

    logger.info(f"Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    logger.info("Applying SMOTE oversampling to training set...")
    smote = SMOTE(
        random_state=data_params.get("random_state", 42),
        sampling_strategy=0.1
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(
        f"After SMOTE: {len(X_train_res):,} samples, "
        f"fraud rate: {y_train_res.mean():.3%}"
    )

    # ----------------------------------------------------------------
    # Core training — runs regardless of MLflow availability
    # ----------------------------------------------------------------
    n_trials = params.get("training", {}).get("n_optuna_trials", 50)
    if run_tuning:
        logger.info(f"Running hyperparameter tuning ({n_trials} trials)...")
        best_params, _ = run_hyperparameter_tuning(
            X_train_res, y_train_res, params,
            n_trials=n_trials, mlflow_run_id=None
        )
    else:
        logger.info("Using default XGBoost parameters...")
        best_params = params.get("xgboost_defaults", {})

    final_params = {
        **best_params,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": data_params.get("random_state", 42),
        "n_jobs": -1
    }

    logger.info("Training final model...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "average_precision": float(average_precision_score(y_test, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "test_fraud_count": int(y_test.sum()),
        "test_total": len(y_test)
    }

    logger.info(
        f"Test metrics: AUPRC={metrics['average_precision']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}"
    )

    # Save metrics locally regardless of MLflow
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model locally
    Path("models/registry").mkdir(parents=True, exist_ok=True)
    model.save_model("models/registry/model.json")

    # ----------------------------------------------------------------
    # MLflow logging — optional, won't fail the pipeline if unreachable
    # ----------------------------------------------------------------
    run_id = "local-no-mlflow"
    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

        with mlflow.start_run(run_name="fraud-detection-xgboost") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")

            mlflow.log_params({
                "dataset_rows": len(df),
                "feature_count": len(feature_cols),
                "train_rows": len(X_train_res),
                "test_rows": len(X_test),
                "train_fraud_rate": float(y_train_res.mean()),
                "test_fraud_rate": float(y_test.mean()),
                "smote_applied": True,
                "tuning_method": "optuna_tpe_hyperband" if run_tuning else "defaults",
            })
            for section, section_params in params.items():
                if isinstance(section_params, dict):
                    for k, v in section_params.items():
                        if not isinstance(v, (dict, list)):
                            mlflow.log_param(f"{section}.{k}", v)

            mlflow.log_metrics(metrics)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar(im)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix — Fraud Detection")
            plt.tight_layout()
            plt.savefig("/tmp/confusion_matrix.png", dpi=100)
            plt.close()
            mlflow.log_artifact("/tmp/confusion_matrix.png")

            # SHAP
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test[:1000])
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_cols, show=False)
                plt.tight_layout()
                plt.savefig("/tmp/shap_summary.png", dpi=100, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact("/tmp/shap_summary.png")
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")

            from mlflow.models.signature import infer_signature
            mlflow.xgboost.log_model(
                model,
                "fraud-detection-model",
                registered_model_name=config.mlflow_model_name,
                signature=infer_signature(X_test[:100], model.predict_proba(X_test[:100])),
                input_example=X_test[:5]
            )

            # Promote to Production
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(config.mlflow_model_name, stages=["None"])
            if versions:
                latest_version = versions[0].version
                client.transition_model_version_stage(
                    name=config.mlflow_model_name,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"Model v{latest_version} promoted to Production stage")

    except Exception as e:
        logger.warning(f"MLflow logging failed (server unreachable?): {e}")
        logger.warning("Training artifacts saved locally — pipeline continues.")

    return model, metrics, run_id
