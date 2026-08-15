"""
Model and data drift detection using Evidently AI.
Compares current production data against the reference dataset.
Triggers retraining via SNS when drift is detected.
"""

import json
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_drift(
    reference_data_path: str,
    current_data_path: str,
    output_path: str = "evidently_reports/drift_report.html",
    metrics_output_path: str = "evidently_reports/drift_metrics.json"
) -> Dict[str, Any]:
    """
    Run Evidently AI drift detection comparing reference vs current data.

    Args:
        reference_data_path: Path to reference dataset (training baseline)
        current_data_path: Path to current production data
        output_path: Path for HTML drift report
        metrics_output_path: Path for drift metrics JSON

    Returns:
        Dictionary with drift detection results
    """
    config = get_config()

    logger.info("Loading reference and current datasets...")
    reference_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)

    if len(current_df) > 10000:
        current_df = current_df.sample(n=10000, random_state=42)

    logger.info(
        f"Reference: {len(reference_df):,} rows | Current: {len(current_df):,} rows"
    )

    feature_cols = [
        c for c in config.feature_columns
        if c in reference_df.columns and c in current_df.columns
    ]

    column_mapping = ColumnMapping(
        target=config.target_column,
        numerical_features=[
            c for c in feature_cols
            if reference_df[c].dtype in [np.float64, np.int64]
        ],
        categorical_features=[
            c for c in feature_cols
            if reference_df[c].dtype == object
        ]
    )

    metrics_list = [DatasetDriftMetric(), DataDriftTable()]
    for col in ["TransactionAmt", "C1", "V1"]:
        if col in feature_cols:
            metrics_list.append(ColumnDriftMetric(column_name=col))

    logger.info("Running Evidently drift analysis...")
    report = Report(metrics=metrics_list)

    shared_cols = [c for c in feature_cols + [config.target_column] if c in current_df.columns]
    report.run(
        reference_data=reference_df[feature_cols + [config.target_column]],
        current_data=current_df[shared_cols],
        column_mapping=column_mapping
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info(f"Drift report saved: {output_path}")

    drift_results = _extract_drift_metrics(report.as_dict(), config)

    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(drift_results, f, indent=2)

    _upload_to_s3(output_path, metrics_output_path, config)

    if drift_results["drift_detected"]:
        logger.warning(
            f"DATA DRIFT DETECTED! "
            f"Share of drifted columns: {drift_results['share_drifted_columns']:.1%}"
        )
        _send_drift_alert(drift_results, config)
    else:
        logger.info(
            f"No significant drift detected. "
            f"Share of drifted columns: {drift_results['share_drifted_columns']:.1%}"
        )

    return drift_results


def _extract_drift_metrics(report_dict: Dict, config) -> Dict[str, Any]:
    """Extract key drift metrics from the Evidently report dictionary."""
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": False,
        "share_drifted_columns": 0.0,
        "dataset_drift": False,
        "column_drift_results": {},
        "drift_threshold": config.drift_threshold
    }

    try:
        for metric_result in report_dict.get("metrics", []):
            metric_id = metric_result.get("metric", "")
            result = metric_result.get("result", {})

            if "DatasetDriftMetric" in metric_id:
                metrics["share_drifted_columns"] = float(
                    result.get("share_drifted_columns", 0)
                )
                metrics["dataset_drift"] = result.get("dataset_drift", False)
                metrics["number_drifted_columns"] = result.get(
                    "number_of_drifted_columns", 0
                )
            elif "ColumnDriftMetric" in metric_id:
                col_name = result.get("column_name", "unknown")
                metrics["column_drift_results"][col_name] = {
                    "drift_score": result.get("drift_score", 0),
                    "drift_detected": result.get("drift_detected", False),
                    "stattest": result.get("stattest", "unknown")
                }

        metrics["drift_detected"] = (
            metrics["share_drifted_columns"] > config.drift_threshold
        )
    except Exception as e:
        logger.error(f"Failed to extract drift metrics: {e}")

    return metrics


def _upload_to_s3(html_path: str, metrics_path: str, config) -> None:
    """Upload drift reports to S3."""
    if not config.evidently_bucket:
        logger.warning("No Evidently bucket configured — skipping S3 upload")
        return

    s3 = boto3.client("s3", region_name=config.aws_region)
    timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d/%H%M%S")

    try:
        for file_path in [html_path, metrics_path]:
            if Path(file_path).exists():
                key = f"drift-reports/{timestamp}/{Path(file_path).name}"
                s3.upload_file(
                    file_path, config.evidently_bucket, key,
                    ExtraArgs={"ServerSideEncryption": "aws:kms"}
                )
                logger.info(f"Uploaded to s3://{config.evidently_bucket}/{key}")
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")


def _send_drift_alert(drift_results: Dict[str, Any], config) -> None:
    """Send SNS alert when drift is detected."""
    if not config.sns_topic_arn:
        logger.warning("No SNS topic configured — skipping alert")
        return

    sns = boto3.client("sns", region_name=config.aws_region)
    message = (
        f"DATA DRIFT DETECTED — Fraud Detection Model\n\n"
        f"Timestamp: {drift_results['timestamp']}\n"
        f"Share of drifted columns: {drift_results['share_drifted_columns']:.1%}\n"
        f"Drift threshold: {drift_results['drift_threshold']:.1%}\n"
        f"Number of drifted columns: {drift_results.get('number_drifted_columns', 'N/A')}\n\n"
        f"Recommended action: Trigger model retraining pipeline\n\n"
        f"Column-level drift:\n"
    )
    for col, col_drift in drift_results.get("column_drift_results", {}).items():
        if col_drift.get("drift_detected"):
            message += (
                f"  - {col}: drift_score={col_drift['drift_score']:.4f} "
                f"({col_drift['stattest']})\n"
            )

    try:
        sns.publish(
            TopicArn=config.sns_topic_arn,
            Subject="🚨 DRIFT ALERT — Fraud Detection Model Requires Retraining",
            Message=message
        )
        logger.info("Drift alert sent via SNS")
    except Exception as e:
        logger.error(f"Failed to send SNS alert: {e}")
