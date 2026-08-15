"""Tests for the training module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


def make_processed_dataset(n_rows=5000, fraud_rate=0.035):
    np.random.seed(42)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    return pd.DataFrame({
        "TransactionAmt": np.concatenate([
            np.random.exponential(100, n_legit),
            np.random.exponential(800, n_fraud),
        ]),
        "card1": np.random.randint(1000, 9999, n_rows).astype(float),
        "C1": np.random.randint(0, 10, n_rows).astype(float),
        "V1": np.random.randn(n_rows),
        "TransactionHour": np.random.randint(0, 24, n_rows),
        "TransactionDay": np.random.randint(0, 7, n_rows),
        "amt_per_card": np.random.exponential(100, n_rows),
        "transaction_count_per_card": np.random.randint(1, 50, n_rows),
        "isFraud": np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int),
    }).sample(frac=1, random_state=42).reset_index(drop=True)


def test_load_params(tmp_path):
    from src.training.trainer import load_params

    params_content = "training:\n  n_optuna_trials: 10\ndata:\n  test_size: 0.2\n"
    params_path = str(tmp_path / "params.yaml")
    with open(params_path, "w") as f:
        f.write(params_content)

    params = load_params(params_path)
    assert params["training"]["n_optuna_trials"] == 10
    assert params["data"]["test_size"] == 0.2


def test_hyperparameter_tuning_returns_best_params(tmp_path):
    from src.training.hyperparameter_tuning import run_hyperparameter_tuning

    df = make_processed_dataset()
    X = df[["TransactionAmt", "C1", "V1"]].values
    y = df["isFraud"].values

    params = {
        "hyperparameter_space": {
            "n_estimators": {"low": 50, "high": 100, "step": 50},
            "max_depth": {"low": 3, "high": 4},
            "learning_rate": {"low": 0.1, "high": 0.2, "log": False},
            "subsample": {"low": 0.8, "high": 1.0},
            "colsample_bytree": {"low": 0.8, "high": 1.0},
            "min_child_weight": {"low": 1, "high": 2},
            "scale_pos_weight": {"low": 5, "high": 10},
        },
        "optuna": {"direction": "maximize"},
    }

    best_params, study = run_hyperparameter_tuning(X, y, params, n_trials=2)

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params
    assert len(study.trials) == 2
