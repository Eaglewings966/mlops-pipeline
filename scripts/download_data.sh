#!/bin/bash
# Download IEEE-CIS Fraud Detection dataset from Kaggle
# Requires: pip install kaggle and ~/.kaggle/kaggle.json configured
set -euo pipefail

DATA_DIR="data/raw"
mkdir -p ${DATA_DIR}

echo "================================================"
echo "Downloading IEEE-CIS Fraud Detection Dataset"
echo "================================================"

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "Kaggle credentials not found at ~/.kaggle/kaggle.json"
  echo "1. Go to https://www.kaggle.com/account"
  echo "2. Click 'Create New API Token'"
  echo "3. Save kaggle.json to ~/.kaggle/ and run: chmod 600 ~/.kaggle/kaggle.json"
  echo ""
  echo "Alternative: place train_transaction.csv in data/raw/ manually"
  exit 1
fi

pip3.11 install kaggle --quiet

echo "Downloading dataset (~500MB)..."
kaggle competitions download -c ieee-fraud-detection -p ${DATA_DIR}

echo "Extracting..."
unzip -o ${DATA_DIR}/ieee-fraud-detection.zip -d ${DATA_DIR}

if [ -f "${DATA_DIR}/train_transaction.csv" ]; then
  cp ${DATA_DIR}/train_transaction.csv ${DATA_DIR}/transactions.csv
  echo "Dataset ready: ${DATA_DIR}/transactions.csv"
  echo "Rows: $(wc -l < ${DATA_DIR}/transactions.csv)"
else
  echo "ERROR: train_transaction.csv not found"
  ls -la ${DATA_DIR}/
  exit 1
fi

echo "================================================"
echo "Download complete: $(du -sh ${DATA_DIR}/transactions.csv | cut -f1)"
echo "================================================"
