#!/bin/bash
# Run the full MLOps training pipeline
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"

echo "================================================"
echo "MLOps Pipeline — Fraud Detection"
echo "================================================"

# Load config from SSM
export DVC_BUCKET=$(aws ssm get-parameter --name /mlops/dvc-bucket --region ${AWS_REGION} --query Parameter.Value --output text)
export MLFLOW_BUCKET=$(aws ssm get-parameter --name /mlops/mlflow-bucket --region ${AWS_REGION} --query Parameter.Value --output text)
export EVIDENTLY_BUCKET=$(aws ssm get-parameter --name /mlops/evidently-bucket --region ${AWS_REGION} --query Parameter.Value --output text)
export SNS_TOPIC_ARN=$(aws ssm get-parameter --name /mlops/drift-sns-arn --region ${AWS_REGION} --query Parameter.Value --output text)
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="fraud-detection"
export MLFLOW_MODEL_NAME="fraud-detection-xgboost"

echo "DVC Bucket:      ${DVC_BUCKET}"
echo "MLflow Bucket:   ${MLFLOW_BUCKET}"
echo "MLflow URI:      ${MLFLOW_TRACKING_URI}"
echo ""

# Run via DVC (tracks lineage) or directly
if command -v dvc &> /dev/null && [ -f dvc.yaml ]; then
  echo "Running pipeline via DVC..."
  dvc repro
else
  echo "Running pipeline directly..."
  python3.11 -m src.pipelines.training_pipeline validate
  python3.11 -m src.pipelines.training_pipeline preprocess
  python3.11 -m src.pipelines.training_pipeline train --no-tuning
  python3.11 -m src.pipelines.training_pipeline monitor
fi

echo ""
echo "================================================"
echo "Pipeline complete"
echo "================================================"
