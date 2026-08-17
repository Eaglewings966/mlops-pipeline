#!/usr/bin/env python3
"""
Generate minimal synthetic fraud transaction data for CI testing.
Creates a small CSV compatible with the IEEE-CIS Fraud Detection schema.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_rows=1000, output_path="data/raw/transactions.csv"):
    """Generate synthetic transactions with minimal required columns."""
    np.random.seed(42)
    
    # Core transaction features
    data = {
        "TransactionID": range(1, n_rows + 1),
        "TransactionDT": np.random.randint(0, 1000000, n_rows),
        "TransactionAmt": np.random.lognormal(4, 1.5, n_rows),
        "ProductCD": np.random.choice(["W", "C", "H", "R", "S"], n_rows),
        "card1": np.random.randint(1000, 20000, n_rows),
        "card2": np.random.choice([100, 150, 200, 250, 300, 350, 400, 450, 500], n_rows),
        "card3": np.random.choice([150, 185], n_rows),
        "card4": np.random.choice(["visa", "mastercard", "american express", "discover"], n_rows),
        "card5": np.random.choice([102, 117, 120, 122, 123, 124, 125, 126, 205, 224, 226], n_rows),
        "card6": np.random.choice(["credit", "debit"], n_rows),
        "addr1": np.random.randint(100, 600, n_rows),
        "addr2": np.random.randint(10, 100, n_rows),
        "dist1": np.random.uniform(0, 10000, n_rows),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n_rows),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com", None], n_rows),
        # Device info
        "DeviceType": np.random.choice(["desktop", "mobile"], n_rows),
        "DeviceInfo": np.random.choice(["Windows", "iOS", "Android", "MacOS"], n_rows),
        # Target: 10% fraud rate
        "isFraud": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Add some M columns (transaction metadata)
    for i in range(1, 10):
        df[f"M{i}"] = np.random.choice(["T", "F", np.nan], n_rows, p=[0.3, 0.3, 0.4])
    
    # Add some V columns (Vesta engineered features)
    for i in range(1, 20):
        df[f"V{i}"] = np.random.randn(n_rows)
    
    # Add C columns (counts/aggregates)
    for i in range(1, 15):
        df[f"C{i}"] = np.random.randint(0, 100, n_rows)
    
    # Add D columns (timedeltas)
    for i in range(1, 16):
        df[f"D{i}"] = np.random.uniform(0, 1000, n_rows)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Generated {len(df):,} rows → {output_path}")
    print(f"  Fraud rate: {df['isFraud'].mean():.1%}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    generate_sample_data()
