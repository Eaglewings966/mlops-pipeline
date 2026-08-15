"""
Feature engineering utilities.
Extracted here so DVC can track this file as a dependency of the preprocess stage.
The actual logic lives in preprocessing.py — this module exposes named helpers
that can be imported independently for serving-time feature construction.
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """Add TransactionHour and TransactionDay from a TransactionDT seconds column."""
    df["TransactionHour"] = (dt_series // 3600 % 24).astype(int)
    df["TransactionDay"] = (dt_series // 86400 % 7).astype(int)
    return df


def add_card_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-card mean amount and transaction count features."""
    card_stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(amt_per_card="mean", transaction_count_per_card="count")
        .reset_index()
    )
    return df.merge(card_stats, on="card1", how="left")
