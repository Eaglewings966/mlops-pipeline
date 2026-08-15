"""
Data validation using Great Expectations.
Validates the raw fraud detection dataset before training.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_fraud_data(
    data_path: str,
    output_path: str = "data/processed/validation_report.json"
) -> Dict[str, Any]:
    """
    Validate the fraud detection dataset.

    Checks:
    - Target column exists and is binary (0 or 1)
    - TransactionAmt is positive and not null
    - No completely empty rows
    - Fraud rate is within expected range (0.5% to 5%)
    - Minimum row count of 10,000

    Args:
        data_path: Path to the raw CSV file
        output_path: Path to write the validation report

    Returns:
        Dictionary with validation results and statistics
    """
    logger.info(f"Validating data from {data_path}...")
    df = pd.read_csv(data_path, nrows=100000)
    config = get_config()

    validation_results = {
        "passed": True,
        "total_rows": len(df),
        "checks": [],
        "statistics": {}
    }

    # Check 1 — Target column is binary
    check_1 = {"name": "target_column_is_binary", "passed": False, "details": ""}
    if config.target_column in df.columns:
        unique_values = df[config.target_column].dropna().unique()
        if set(unique_values).issubset({0, 1}):
            check_1["passed"] = True
            check_1["details"] = "isFraud contains only 0 and 1"
        else:
            check_1["details"] = f"isFraud has unexpected values: {unique_values}"
    else:
        check_1["details"] = f"Column '{config.target_column}' not found"
    validation_results["checks"].append(check_1)

    # Check 2 — TransactionAmt is positive
    check_2 = {"name": "transaction_amount_is_positive", "passed": False, "details": ""}
    if "TransactionAmt" in df.columns:
        negative_count = (df["TransactionAmt"] <= 0).sum()
        null_count = df["TransactionAmt"].isna().sum()
        if negative_count == 0 and null_count == 0:
            check_2["passed"] = True
            check_2["details"] = "TransactionAmt is positive and not null"
        else:
            check_2["details"] = (
                f"TransactionAmt has {negative_count} negative values "
                f"and {null_count} null values"
            )
    validation_results["checks"].append(check_2)

    # Check 3 — Fraud rate within expected range
    check_3 = {"name": "fraud_rate_within_expected_range", "passed": False, "details": ""}
    if config.target_column in df.columns:
        fraud_rate = df[config.target_column].mean()
        if 0.005 <= fraud_rate <= 0.05:
            check_3["passed"] = True
            check_3["details"] = f"Fraud rate: {fraud_rate:.3%}"
        else:
            check_3["details"] = (
                f"Fraud rate {fraud_rate:.3%} outside expected range (0.5% to 5%)"
            )
        validation_results["statistics"]["fraud_rate"] = float(fraud_rate)
    validation_results["checks"].append(check_3)

    # Check 4 — No completely empty rows
    check_4 = {"name": "no_completely_empty_rows", "passed": False, "details": ""}
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows == 0:
        check_4["passed"] = True
        check_4["details"] = "No completely empty rows found"
    else:
        check_4["details"] = f"{empty_rows} completely empty rows found"
    validation_results["checks"].append(check_4)

    # Check 5 — Minimum row count
    check_5 = {"name": "minimum_row_count", "passed": False, "details": ""}
    if len(df) >= 10000:
        check_5["passed"] = True
        check_5["details"] = f"Dataset has {len(df):,} rows"
    else:
        check_5["details"] = f"Dataset only has {len(df):,} rows (minimum: 10,000)"
    validation_results["checks"].append(check_5)

    all_passed = all(c["passed"] for c in validation_results["checks"])
    validation_results["passed"] = all_passed
    validation_results["statistics"].update({
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_percentage": float(df.isnull().mean().mean()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    if all_passed:
        logger.info("Data validation PASSED — all checks succeeded")
    else:
        failed = [c["name"] for c in validation_results["checks"] if not c["passed"]]
        logger.error(f"Data validation FAILED — failed checks: {failed}")
        raise ValueError(f"Data validation failed: {failed}")

    return validation_results
