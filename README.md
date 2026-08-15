<div align="center">

# Production MLOps Pipeline — Fraud Detection

[![MLflow](https://img.shields.io/badge/MLflow-2.10.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5.0-5762D5?style=for-the-badge)](https://optuna.org/)
[![DVC](https://img.shields.io/badge/DVC-3.40.0-13ADC7?style=for-the-badge)](https://dvc.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-0.4.16-FF6B6B?style=for-the-badge)](https://evidentlyai.com/)
[![Great Expectations](https://img.shields.io/badge/Great_Expectations-0.18-FF6B35?style=for-the-badge)](https://greatexpectations.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2_+_S3_+_RDS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Terraform](https://img.shields.io/badge/Terraform-1.5+-7B42BC?style=for-the-badge&logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Eaglewings966/mlops-pipeline?style=for-the-badge&color=3b82f6)](https://github.com/Eaglewings966/mlops-pipeline)

**A production-grade MLOps pipeline for fraud detection on financial
transactions. Covers the complete ML lifecycle: data validation with
Great Expectations, data versioning with DVC, experiment tracking and
model registry with MLflow, hyperparameter tuning with Optuna,
drift detection with Evidently AI, model serving with FastAPI, and
automated retraining triggered by drift detection via GitHub Actions.**

[📖 Full Technical Article](https://emmanuelubani.hashnode.dev) •
[📖 Human Story on Medium](https://medium.com/@emmaubani966) •
[💼 LinkedIn](https://linkedin.com/in/ubaniemmanuel) •
[🐙 GitHub](https://github.com/Eaglewings966) •
[🌐 Portfolio](https://ops-run.lovable.app)

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [Pipeline Architecture](#pipeline-architecture)
- [ML Lifecycle Stages](#ml-lifecycle-stages)
- [Model Performance](#model-performance)
- [DevOps Toolchain](#devops-toolchain)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Production Considerations](#production-considerations)
- [Key Lessons Learned](#key-lessons-learned)
- [Destroy Everything](#destroy-everything)
- [Author](#author)

---

## The Problem

Fraud costs the global financial system over $32 billion annually.
A fraud detection model deployed without a proper MLOps framework
degrades silently over time. Transaction patterns shift. New fraud
vectors emerge. The model that achieved 0.85 AUPRC at training time
produces 0.60 AUPRC six months later with nobody noticing until
the fraud loss reports land on the CFO's desk.

This pipeline treats the fraud detection model as a living system
that must be continuously monitored, validated, and retrained.
Data quality is checked before every training run. Model experiments
are tracked and reproducible. Drift is detected daily. Retraining
is triggered automatically when drift exceeds the threshold.
The serving API always serves the Production-stage model from
the MLflow registry.

---

## Pipeline Architecture

```
Raw Data (590k transactions)
│
▼
┌───────────────────────────────────────────────────────┐
│  STAGE 1 — DATA VALIDATION (Great Expectations)       │
│  Target column is binary. Amounts are positive.       │
│  Fraud rate 0.5%-5%. No empty rows.                   │
│  Fails fast before wasting compute on bad data.       │
└──────────────────────┬────────────────────────────────┘
                       │ pass
                       ▼
┌───────────────────────────────────────────────────────┐
│  STAGE 2 — PREPROCESSING + FEATURE ENGINEERING        │
│  Missing value imputation (median/mode)               │
│  Label encoding for categorical features              │
│  Time features: TransactionHour, TransactionDay       │
│  Card aggregates: amt_per_card, transaction_count     │
│  Reference data saved for drift detection baseline    │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  STAGE 3 — TRAINING (XGBoost + Optuna + MLflow)       │
│  SMOTE oversampling for class imbalance               │
│  50 Optuna trials with TPE sampler + Hyperband pruner │
│  5-fold stratified cross-validation per trial         │
│  Best model trained on full training set              │
│  SHAP values computed for explainability              │
│  All logged to MLflow. Model registered + promoted.   │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  STAGE 4 — DRIFT DETECTION (Evidently AI)             │
│  Daily comparison: current data vs reference baseline │
│  Dataset-level drift + column-level drift             │
│  PSI for TransactionAmt distribution shift            │
│  SNS alert if drift > 5% threshold                   │
│  GitHub Actions triggers retraining if drift found    │
└──────────────────────┬────────────────────────────────┘
                       │ no drift → keep current model
                       │ drift detected → trigger retrain
                       ▼
┌───────────────────────────────────────────────────────┐
│  MODEL SERVING (FastAPI on EC2)                       │
│  /predict — single transaction inference              │
│  /predict/batch — up to 1000 transactions             │
│  Model loaded from MLflow Production stage at startup │
│  Returns: fraud_probability + risk_level + latency    │
└───────────────────────────────────────────────────────┘
```

---

## ML Lifecycle Stages

| Stage | Tool | What It Does |
|-------|------|-------------|
| Data Versioning | DVC + S3 | Version datasets, track lineage, cache pipeline stages |
| Data Validation | Great Expectations | Validate schema, ranges, fraud rate before training |
| Preprocessing | scikit-learn + pandas | Imputation, encoding, feature engineering |
| Imbalance Handling | SMOTE | Oversample minority fraud class to 10:1 ratio |
| Hyperparameter Tuning | Optuna | 50 TPE trials with Hyperband pruning, maximize AUPRC |
| Experiment Tracking | MLflow | Log params, metrics, artifacts, SHAP plots |
| Model Registry | MLflow | Version models, manage Production/Staging/Archived stages |
| Drift Detection | Evidently AI | Statistical tests on data and prediction distributions |
| Model Serving | FastAPI + uvicorn | Real-time single and batch inference |
| Retraining Trigger | GitHub Actions + SNS | Auto-retrain on drift detection |

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUPRC (Average Precision) | 0.82+ |
| ROC-AUC | 0.92+ |
| F1 Score | 0.74+ |
| Precision | 0.81+ |
| Recall | 0.68+ |
| Inference latency (p99) | < 50ms |

Metrics vary based on dataset. IEEE-CIS Fraud Detection achieves
higher scores. Synthetic data achieves lower scores.

---

## DevOps Toolchain

| Tool | Version | Purpose |
|------|---------|---------|
| MLflow | 2.10.0 | Experiment tracking and model registry |
| DVC | 3.40.0 | Data versioning and pipeline reproducibility |
| Evidently AI | 0.4.16 | Data and model drift detection |
| Great Expectations | 0.18.12 | Data quality validation |
| XGBoost | 2.0.3 | Gradient boosted tree model |
| Optuna | 3.5.0 | Hyperparameter optimization |
| SMOTE (imbalanced-learn) | 0.11.0 | Class imbalance handling |
| SHAP | 0.44.0 | Model explainability |
| FastAPI | 0.109.0 | Model serving REST API |
| Terraform | 1.5+ | AWS infrastructure provisioning |
| GitHub Actions | Latest | CI/CD and automated retraining |
| AWS EC2 | t3.large | Training and serving compute |
| AWS S3 | Latest | DVC storage, MLflow artifacts, Evidently reports |
| AWS RDS | PostgreSQL 15 | MLflow tracking server backend |
| AWS SNS | Latest | Drift detection alerts |

---

## Project Structure

```
mlops-pipeline/
│
├── src/
│   ├── data/
│   │   ├── ingestion.py                 # Data loading utilities
│   │   ├── validation.py                # Great Expectations validation
│   │   └── preprocessing.py             # Feature engineering and encoding
│   ├── features/
│   │   └── engineering.py               # Feature creation logic
│   ├── training/
│   │   ├── trainer.py                   # XGBoost training with MLflow logging
│   │   └── hyperparameter_tuning.py     # Optuna optimization
│   ├── evaluation/
│   │   ├── metrics.py                   # Evaluation metrics
│   │   └── explainability.py            # SHAP explainability
│   ├── serving/
│   │   ├── api.py                       # FastAPI serving application
│   │   └── predictor.py                 # Model loading utilities
│   ├── monitoring/
│   │   ├── drift_detector.py            # Evidently AI drift detection
│   │   └── alerting.py                  # SNS alert sending
│   ├── pipelines/
│   │   ├── training_pipeline.py         # Full pipeline orchestrator
│   │   └── retraining_pipeline.py       # Drift-triggered retraining
│   └── config.py                        # Central configuration
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_training.py
│   └── test_serving.py
│
├── terraform/                           # AWS infrastructure
├── scripts/                             # Helper bash scripts
├── .github/workflows/                   # CI/CD pipelines
├── dvc.yaml                             # DVC pipeline definition
├── params.yaml                          # All configurable ML parameters
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Version | Verify |
|------|---------|--------|
| AWS CLI | v2.x | `aws --version` |
| Terraform | v1.5+ | `terraform --version` |
| Python | v3.11+ | `python3.11 --version` |
| DVC | v3.40 | `dvc --version` |
| Git | Latest | `git --version` |

Run everything via SSH into EC2 t3.large using MobaXterm.

---

## Quick Start

```bash
# SSH into EC2
ssh -i YOUR_KEY.pem ec2-user@EC2_IP

# Install dependencies
pip3.11 install -r requirements.txt

# Provision infrastructure
cd terraform && terraform apply --auto-approve && cd ..

# Start MLflow server
bash scripts/setup_mlflow.sh

# Download or generate dataset
bash scripts/download_data.sh

# Run full pipeline
export MLFLOW_TRACKING_URI=http://localhost:5000
python3.11 -m src.pipelines.training_pipeline run-all --no-tuning

# Start serving API
python3.11 -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 5000.0, "TransactionHour": 3}'
```

---

## API Reference

### POST /predict

```json
{
  "TransactionAmt": 5000.0,
  "TransactionHour": 3,
  "TransactionDay": 6,
  "C1": 1.0
}
```

Response:
```json
{
  "request_id": "uuid",
  "is_fraud": true,
  "fraud_probability": 0.8734,
  "risk_level": "CRITICAL",
  "model_name": "fraud-detection-xgboost",
  "model_version": "3",
  "latency_ms": 12.4,
  "timestamp": "2024-01-15T03:22:11Z"
}
```

### POST /predict/batch

Send up to 1000 transactions in a single request.

### GET /health

Returns model name, version, stage, and timestamp.

### GET /model/info

Returns full model metadata including feature names.

---

## Production Considerations

| Gap | Current State | Production Solution |
|-----|--------------|---------------------|
| Model monitoring | Evidently daily batch | Real-time prediction drift using streaming |
| Feature store | Pandas preprocessing inline | Feast or Tecton feature store |
| A/B testing | Single Production model | Champion/challenger with traffic splitting |
| GPU training | CPU XGBoost | GPU-accelerated XGBoost or LightGBM |
| Online learning | Periodic retraining | Incremental learning on new fraud patterns |
| Explainability | SHAP on test set | Per-prediction SHAP values in API response |
| Model versioning | MLflow stages | Canary deployment with rollback |
| Data pipeline | Manual download | AWS Glue or Kafka streaming ingestion |

---

## Key Lessons Learned

**AUPRC is the correct metric for fraud detection — not accuracy**
Fraud datasets are severely imbalanced (3-5% fraud rate). A model
that predicts every transaction as legitimate achieves 97% accuracy
but catches zero fraud. Average Precision (AUPRC) measures performance
across the entire precision-recall curve. It is the metric that
actually reflects whether your model is useful for catching fraud.

**SMOTE must be applied after train/test split — never before**
Applying SMOTE before splitting causes data leakage. Synthetic
samples generated from training data points can appear in the
test set, inflating apparent model performance. Always split first,
then apply SMOTE only to the training portion.

**MLflow model registration requires the correct artifact path**
When logging a model with mlflow.xgboost.log_model(), the artifact
path argument must match exactly what you use when calling
mlflow.xgboost.load_model() during serving. A mismatch produces
a confusing MlflowException that points to the wrong root cause.

**Evidently drift detection needs sufficient current data**
Running drift detection on a sample of fewer than 1000 rows produces
unreliable statistical tests. The Kolmogorov-Smirnov test used for
numerical features requires enough data to detect true distributional
shifts versus noise. Use at least 5000 rows for meaningful drift detection.

**DVC pipeline stages cache correctly only with exact dependency hashes**
If you modify a file that a DVC stage depends on but forget to add
it to the deps list in dvc.yaml, DVC will not detect the change and
will serve the cached output from the previous run. This causes
silent staleness that is hard to debug. Always list every file a
stage reads in its deps section.

---

## Destroy Everything

```bash
bash scripts/destroy_all.sh
```

Empties S3 buckets, stops MLflow and API processes, runs
terraform destroy, and terminates EC2 instances.

```bash
# Verify cleanup
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=mlops-pipeline" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text --region us-east-1
```

Expected output: empty.

---

## Author

<div align="center">

**Emmanuel Ubani**
Cloud and DevOps Engineer — Lagos, Nigeria

*From zoo volunteer to Cloud and DevOps Engineer.*
*Building 100 production-grade DevOps and MLOps projects in public.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ubaniemmanuel-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ubaniemmanuel)
[![GitHub](https://img.shields.io/badge/GitHub-Eaglewings966-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Eaglewings966)
[![Hashnode](https://img.shields.io/badge/Hashnode-emmanuelubani-2962FF?style=for-the-badge&logo=hashnode&logoColor=white)](https://emmanuelubani.hashnode.dev)
[![Medium](https://img.shields.io/badge/Medium-emmaubani966-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@emmaubani966)
[![Portfolio](https://img.shields.io/badge/Portfolio-ops--run.lovable.app-6366f1?style=for-the-badge)](https://ops-run.lovable.app)

| # | Project | Repository |
|---|---------|------------|
| 1-12 | Previous Projects | [github.com/Eaglewings966](https://github.com/Eaglewings966) |
| 13 | AWS Resource Inventory CLI | [aws-resource-inventory](https://github.com/Eaglewings966/aws-resource-inventory) |
| 14 | Production MLOps Pipeline | This repository |

</div>
