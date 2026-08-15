"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def make_dataset(n_rows=15000, fraud_rate=0.02, neg_amounts=0):
    np.random.seed(42)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    amounts = np.concatenate([
        np.random.exponential(100, n_legit),
        np.random.exponential(500, n_fraud),
    ])
    if neg_amounts:
        amounts[:neg_amounts] = -50.0
    return pd.DataFrame({
        "TransactionAmt": amounts,
        "isFraud": np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int),
        "card1": np.random.randint(1000, 9999, n_rows).astype(float),
        "C1": np.random.randint(0, 10, n_rows).astype(float),
        "V1": np.random.randn(n_rows),
    })


def test_validate_passes_on_good_data(tmp_path):
    from src.data.validation import validate_fraud_data

    df = make_dataset()
    data_path = str(tmp_path / "data.csv")
    output_path = str(tmp_path / "report.json")
    df.to_csv(data_path, index=False)

    result = validate_fraud_data(data_path, output_path)

    assert result["passed"] is True
    assert result["total_rows"] > 0
    assert Path(output_path).exists()


def test_validate_fails_on_negative_amounts(tmp_path):
    from src.data.validation import validate_fraud_data

    df = make_dataset(neg_amounts=101)
    data_path = str(tmp_path / "bad.csv")
    output_path = str(tmp_path / "report.json")
    df.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Data validation failed"):
        validate_fraud_data(data_path, output_path)


def test_validate_fails_on_high_fraud_rate(tmp_path):
    from src.data.validation import validate_fraud_data

    df = make_dataset(fraud_rate=0.5)
    data_path = str(tmp_path / "high_fraud.csv")
    output_path = str(tmp_path / "report.json")
    df.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Data validation failed"):
        validate_fraud_data(data_path, output_path)


def test_validation_report_contains_all_checks(tmp_path):
    from src.data.validation import validate_fraud_data

    df = make_dataset()
    data_path = str(tmp_path / "data.csv")
    output_path = str(tmp_path / "report.json")
    df.to_csv(data_path, index=False)

    result = validate_fraud_data(data_path, output_path)
    check_names = [c["name"] for c in result["checks"]]

    assert "target_column_is_binary" in check_names
    assert "transaction_amount_is_positive" in check_names
    assert "fraud_rate_within_expected_range" in check_names
    assert "no_completely_empty_rows" in check_names
    assert "minimum_row_count" in check_names
