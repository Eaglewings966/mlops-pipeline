"""
Hyperparameter tuning with Optuna for XGBoost fraud detection model.
Uses Tree-structured Parzen Estimator (TPE) with Hyperband pruning.
All trials are logged to MLflow automatically.
"""

import optuna
import mlflow
import xgboost as xgb
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from src.config import get_config
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any]
) -> float:
    """
    Optuna objective function for XGBoost hyperparameter optimization.
    Maximizes average precision score using stratified 5-fold CV.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        params: Base parameters from params.yaml

    Returns:
        Mean average precision score across CV folds
    """
    hp = params.get("hyperparameter_space", {})

    model_params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            hp.get("n_estimators", {}).get("low", 100),
            hp.get("n_estimators", {}).get("high", 1000),
            step=hp.get("n_estimators", {}).get("step", 100)
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            hp.get("max_depth", {}).get("low", 3),
            hp.get("max_depth", {}).get("high", 10)
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            hp.get("learning_rate", {}).get("low", 0.01),
            hp.get("learning_rate", {}).get("high", 0.3),
            log=hp.get("learning_rate", {}).get("log", True)
        ),
        "subsample": trial.suggest_float(
            "subsample",
            hp.get("subsample", {}).get("low", 0.5),
            hp.get("subsample", {}).get("high", 1.0)
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            hp.get("colsample_bytree", {}).get("low", 0.5),
            hp.get("colsample_bytree", {}).get("high", 1.0)
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            hp.get("min_child_weight", {}).get("low", 1),
            hp.get("min_child_weight", {}).get("high", 10)
        ),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            hp.get("scale_pos_weight", {}).get("low", 1),
            hp.get("scale_pos_weight", {}).get("high", 20)
        ),
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1
    }

    config = get_config()
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state
    )

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train[train_idx], y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            early_stopping_rounds=50,
            verbose=False
        )
        score = average_precision_score(
            y_train[val_idx],
            model.predict_proba(X_train[val_idx])[:, 1]
        )
        cv_scores.append(score)
        trial.report(np.mean(cv_scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(cv_scores))


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any],
    n_trials: int = 50,
    mlflow_run_id: str = None
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Parameters from params.yaml
        n_trials: Number of optimization trials
        mlflow_run_id: MLflow run ID to log study results to

    Returns:
        Tuple of (best parameters dict, Optuna study object)
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")

    study = optuna.create_study(
        direction=params.get("optuna", {}).get("direction", "maximize"),
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, params),
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info(
        f"Tuning complete. Best average precision: {study.best_value:.4f}"
    )
    logger.info(f"Best parameters: {study.best_params}")

    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id, nested=True):
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_average_precision", study.best_value)
            mlflow.log_metric("n_trials_completed", len(study.trials))

    return study.best_params, study
