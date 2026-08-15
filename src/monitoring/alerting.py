"""
Alerting utilities for the monitoring layer.
Wraps SNS publishing with structured alert payloads.
"""

import json
import boto3
from datetime import datetime, timezone
from typing import Dict, Any
from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def send_alert(subject: str, message: str) -> bool:
    """
    Publish an alert to the configured SNS topic.

    Args:
        subject: Email subject line
        message: Alert body text

    Returns:
        True if published successfully, False otherwise
    """
    config = get_config()
    if not config.sns_topic_arn:
        logger.warning("SNS_TOPIC_ARN not set — skipping alert")
        return False

    try:
        sns = boto3.client("sns", region_name=config.aws_region)
        sns.publish(TopicArn=config.sns_topic_arn, Subject=subject, Message=message)
        logger.info(f"Alert sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        return False


def send_model_deployed_alert(model_name: str, version: str, metrics: Dict[str, Any]) -> bool:
    """Send deployment notification with key metrics."""
    message = (
        f"Model Deployed: {model_name} v{version}\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Metrics:\n"
        f"  AUPRC:   {metrics.get('average_precision', 'N/A'):.4f}\n"
        f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}\n"
        f"  F1:      {metrics.get('f1_score', 'N/A'):.4f}\n\n"
        f"Project: mlops-pipeline\n"
        f"GitHub: github.com/Eaglewings966/mlops-pipeline"
    )
    return send_alert(f"✅ Model Deployed — {model_name} v{version}", message)
