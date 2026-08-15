"""
Data preprocessing for the fraud detection model.
Handles missing values, encoding, and feature selection.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_fraud_data(
    input_path: str,
    output_path: str,
    reference_output_path: str,
    metadata_output_path: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess the fraud detection dataset.

    Steps:
    1. Load and sample the dataset
    2. Engineer time-based and card-based aggregate features
    3. Handle missing values (median for numeric, mode for categorical)
    4. Label-encode categorical features
    5. Select final feature columns
    6. Save processed data, reference data, and feature metadata

    Args:
        input_path: Path to raw transactions CSV
        output_path: Path for processed CSV output
        reference_output_path: Path for reference data (drift detection baseline)
        metadata_output_path: Path for feature metadata JSON

    Returns:
        Tuple of (processed DataFrame, metadata dict)
    """
    config = get_config()
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, nrows=100_000)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    drop_cols = ["TransactionID", "TransactionDT"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Engineer time-based features from TransactionDT
    raw_cols = pd.read_csv(input_path, nrows=1).columns
    if "TransactionDT" in raw_cols:
        dt_col = pd.read_csv(input_path, usecols=["TransactionDT"], nrows=100_000)
        df["TransactionHour"] = (dt_col["TransactionDT"] // 3600 % 24).astype(int)
        df["TransactionDay"] = (dt_col["TransactionDT"] // 86400 % 7).astype(int)
    else:
        df["TransactionHour"] = 0
        df["TransactionDay"] = 0

    # Engineer card-based aggregate features
    if "card1" in df.columns:
        card_stats = df.groupby("card1")["TransactionAmt"].agg(
            amt_per_card="mean",
            transaction_count_per_card="count"
        ).reset_index()
        df = df.merge(card_stats, on="card1", how="left")
    else:
        df["amt_per_card"] = df["TransactionAmt"]
        df["transaction_count_per_card"] = 1

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if col != config.target_column:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "unknown")

    logger.info(f"Missing values handled: {df.isnull().sum().sum()} remaining nulls")

    # Label-encode categorical features
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    logger.info(f"Encoded {len(label_encoders)} categorical columns")

    # Select final feature columns
    available_features = [c for c in config.feature_columns if c in df.columns]
    df = df[available_features + [config.target_column]]
    logger.info(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved: {output_path}")

    Path(reference_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.head(10000).to_csv(reference_output_path, index=False)
    logger.info(f"Reference data saved: {reference_output_path}")

    metadata = {
        "feature_columns": available_features,
        "target_column": config.target_column,
        "total_rows": len(df),
        "fraud_rate": float(df[config.target_column].mean()),
        "categorical_columns": list(label_encoders.keys()),
        "numeric_columns": list(numeric_cols),
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()}
    }

    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Preprocessing complete. Fraud rate: {metadata['fraud_rate']:.3%}")
    return df, metadata
