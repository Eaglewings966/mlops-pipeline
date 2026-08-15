"""
FastAPI model serving API for the fraud detection model.
Loads the Production model from MLflow Model Registry.
"""

import time
import uuid
import mlflow
import mlflow.xgboost
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

model = None
model_version = None
feature_columns = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model from MLflow at startup."""
    global model, model_version, feature_columns

    logger.info("Loading fraud detection model from MLflow registry...")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    try:
        model_uri = f"models:/{config.mlflow_model_name}/{config.model_stage}"
        model = mlflow.xgboost.load_model(model_uri)
        feature_columns = config.feature_columns

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(
            config.mlflow_model_name, stages=[config.model_stage]
        )
        if versions:
            model_version = versions[0].version

        logger.info(
            f"Model loaded: {config.mlflow_model_name} "
            f"v{model_version} [{config.model_stage}]"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    yield
    logger.info("Shutting down fraud detection API")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production fraud detection model serving API. "
        "Built with MLflow, XGBoost, and FastAPI. "
        "Project 14 — MLOps Pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class TransactionRequest(BaseModel):
    """Request schema for a single transaction prediction."""
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Optional[str] = Field(default="W")
    card1: Optional[float] = Field(default=0.0)
    card2: Optional[float] = Field(default=0.0)
    card4: Optional[str] = Field(default="visa")
    addr1: Optional[float] = Field(default=0.0)
    addr2: Optional[float] = Field(default=87.0)
    C1: Optional[float] = Field(default=1.0)
    C2: Optional[float] = Field(default=1.0)
    D1: Optional[float] = Field(default=0.0)
    V1: Optional[float] = Field(default=0.0)
    V2: Optional[float] = Field(default=0.0)
    TransactionHour: Optional[int] = Field(default=12, ge=0, le=23)
    TransactionDay: Optional[int] = Field(default=3, ge=0, le=6)
    amt_per_card: Optional[float] = Field(default=None)
    transaction_count_per_card: Optional[int] = Field(default=1)

    @validator("TransactionAmt")
    def validate_amount(cls, v):
        if v > 1_000_000:
            raise ValueError("TransactionAmt exceeds maximum ($1M)")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    request_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_name: str
    model_version: str
    latency_ms: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    request_id: str
    predictions: List[Dict[str, Any]]
    total_transactions: int
    flagged_as_fraud: int
    avg_latency_ms: float
    timestamp: str


def prepare_features(transaction: TransactionRequest) -> np.ndarray:
    """Convert a TransactionRequest into a feature vector."""
    transaction_dict = transaction.dict()
    if transaction_dict.get("amt_per_card") is None:
        transaction_dict["amt_per_card"] = transaction_dict["TransactionAmt"]

    feature_vector = []
    for col in feature_columns:
        value = transaction_dict.get(col, 0.0)
        if value is None:
            value = 0.0
        if isinstance(value, str):
            value = float(hash(value) % 1000)
        feature_vector.append(float(value))

    return np.array([feature_vector])


def get_risk_level(probability: float) -> str:
    """Convert fraud probability to a human-readable risk level."""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.5:
        return "HIGH"
    elif probability >= 0.2:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": config.mlflow_model_name,
        "model_version": model_version,
        "model_stage": config.model_stage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Return information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": config.mlflow_model_name,
        "model_version": model_version,
        "model_stage": config.model_stage,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "mlflow_tracking_uri": config.mlflow_tracking_uri
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    """Predict whether a single transaction is fraudulent."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        X = prepare_features(request)
        fraud_probability = float(model.predict_proba(X)[0, 1])
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction: id={request_id}, amt={request.TransactionAmt:.2f}, "
            f"prob={fraud_probability:.4f}, latency={latency_ms:.1f}ms"
        )

        return PredictionResponse(
            request_id=request_id,
            is_fraud=fraud_probability >= 0.5,
            fraud_probability=round(fraud_probability, 4),
            risk_level=get_risk_level(fraud_probability),
            model_name=config.mlflow_model_name,
            model_version=str(model_version or "unknown"),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud probability for a batch of up to 1000 transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        X_batch = np.vstack([prepare_features(t) for t in request.transactions])
        probabilities = model.predict_proba(X_batch)[:, 1]

        predictions = [
            {
                "index": i,
                "TransactionAmt": t.TransactionAmt,
                "is_fraud": bool(prob >= 0.5),
                "fraud_probability": round(float(prob), 4),
                "risk_level": get_risk_level(float(prob))
            }
            for i, (t, prob) in enumerate(zip(request.transactions, probabilities))
        ]

        latency_ms = (time.time() - start_time) * 1000
        flagged = sum(1 for p in predictions if p["is_fraud"])

        logger.info(
            f"Batch: id={request_id}, n={len(predictions)}, "
            f"flagged={flagged}, latency={latency_ms:.1f}ms"
        )

        return BatchPredictionResponse(
            request_id=request_id,
            predictions=predictions,
            total_transactions=len(predictions),
            flagged_as_fraud=flagged,
            avg_latency_ms=round(latency_ms / len(predictions), 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
