"""
Automated retraining pipeline.
Triggered by drift detection — pulls latest data, retrains, and promotes the new model.
"""

import typer
from rich.console import Console
from rich import print as rprint

app = typer.Typer(name="retraining-pipeline")
console = Console()


@app.command()
def retrain(
    data_path: str = typer.Option("data/processed/transactions_processed.csv"),
    params_path: str = typer.Option("params.yaml"),
    tuning: bool = typer.Option(False, "--tuning/--no-tuning"),
):
    """
    Retrain the fraud detection model after drift is detected.
    Skips Optuna tuning by default for faster turnaround.
    """
    from src.training.trainer import train_fraud_model
    from src.monitoring.alerting import send_model_deployed_alert
    from src.config import get_config

    config = get_config()
    console.rule("[bold yellow]Automated Retraining Pipeline")

    rprint("[yellow]Drift detected — starting retraining...[/yellow]")
    model, metrics, run_id = train_fraud_model(data_path, params_path, run_tuning=tuning)

    rprint("[green]✓ Retraining complete[/green]")
    rprint(f"  MLflow run ID: {run_id}")
    rprint(f"  AUPRC={metrics['average_precision']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

    send_model_deployed_alert(config.mlflow_model_name, run_id[:8], metrics)
    rprint("[green]✓ Deployment alert sent[/green]")


if __name__ == "__main__":
    app()
