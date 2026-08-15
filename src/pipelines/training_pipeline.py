"""
Training pipeline orchestrator.
Runs all stages: validate → preprocess → train → evaluate → monitor.
Each stage can be called individually or the full pipeline can run at once.
"""

import sys
import typer
from rich.console import Console
from rich import print as rprint

app = typer.Typer(name="training-pipeline")
console = Console()


@app.command()
def validate(
    data_path: str = typer.Option("data/raw/transactions.csv")
):
    """Stage 1: Validate raw data."""
    from src.data.validation import validate_fraud_data

    console.rule("[bold green]Stage 1: Data Validation")
    results = validate_fraud_data(data_path)

    for check in results["checks"]:
        status = "[green]✓[/green]" if check["passed"] else "[red]✗[/red]"
        rprint(f"  {status} {check['name']}: {check['details']}")

    if not results["passed"]:
        sys.exit(1)


@app.command()
def preprocess(
    input_path: str = typer.Option("data/raw/transactions.csv"),
    output_path: str = typer.Option("data/processed/transactions_processed.csv"),
    reference_output: str = typer.Option("data/reference/reference_data.csv"),
    metadata_output: str = typer.Option("data/processed/feature_metadata.json"),
):
    """Stage 2: Preprocess data and engineer features."""
    from src.data.preprocessing import preprocess_fraud_data

    console.rule("[bold green]Stage 2: Data Preprocessing")
    df, metadata = preprocess_fraud_data(
        input_path, output_path, reference_output, metadata_output
    )
    rprint(f"[green]✓ Preprocessing complete[/green]")
    rprint(f"  Rows: {len(df):,} | Features: {len(metadata['feature_columns'])} | Fraud rate: {metadata['fraud_rate']:.3%}")


@app.command()
def train(
    data_path: str = typer.Option("data/processed/transactions_processed.csv"),
    params_path: str = typer.Option("params.yaml"),
):
    """Stage 3: Train XGBoost model (no Optuna tuning)."""
    from src.training.trainer import train_fraud_model

    console.rule("[bold green]Stage 3: Model Training")
    model, metrics, run_id = train_fraud_model(data_path, params_path, run_tuning=False)
    rprint("[green]✓ Training complete[/green]")
    rprint(f"  MLflow run ID: {run_id}")
    rprint(f"  AUPRC={metrics['average_precision']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_score']:.4f}")


@app.command()
def evaluate(
    data_path: str = typer.Option("data/processed/transactions_processed.csv"),
):
    """Stage 4: Evaluate model and persist evaluation metrics."""
    from src.evaluation.metrics import evaluate_model

    console.rule("[bold green]Stage 4: Model Evaluation")
    metrics = evaluate_model(data_path)
    rprint("[green]✓ Evaluation complete[/green]")
    rprint(f"  AUPRC={metrics['average_precision']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_score']:.4f}")


@app.command()
def monitor(
    reference_path: str = typer.Option("data/reference/reference_data.csv"),
    current_path: str = typer.Option("data/processed/transactions_processed.csv"),
):
    """Stage 5: Run drift detection with Evidently AI."""
    from src.monitoring.drift_detector import detect_drift

    console.rule("[bold green]Stage 5: Drift Detection")
    results = detect_drift(reference_path, current_path)

    if results["drift_detected"]:
        rprint(f"[red]⚠ DRIFT DETECTED — {results['share_drifted_columns']:.1%} of columns drifted[/red]")
        rprint("[yellow]SNS alert sent. Retraining recommended.[/yellow]")
    else:
        rprint(
            f"[green]✓ No drift detected — "
            f"{results['share_drifted_columns']:.1%} drifted "
            f"(threshold: {results['drift_threshold']:.1%})[/green]"
        )


@app.command()
def run_all(
    data_path: str = typer.Option("data/raw/transactions.csv"),
):
    """Run the complete MLOps pipeline end-to-end (no Optuna tuning)."""
    from src.data.validation import validate_fraud_data
    from src.data.preprocessing import preprocess_fraud_data
    from src.training.trainer import train_fraud_model
    from src.monitoring.drift_detector import detect_drift

    console.rule("[bold green]Full MLOps Pipeline")

    validate_fraud_data(data_path)

    df, metadata = preprocess_fraud_data(
        data_path,
        "data/processed/transactions_processed.csv",
        "data/reference/reference_data.csv",
        "data/processed/feature_metadata.json",
    )

    model, metrics, run_id = train_fraud_model(
        "data/processed/transactions_processed.csv",
        run_tuning=False,
    )

    detect_drift(
        "data/reference/reference_data.csv",
        "data/processed/transactions_processed.csv",
    )

    console.rule("[bold green]Pipeline Complete")
    rprint("[green]All stages completed successfully[/green]")
    rprint(f"  MLflow run ID: {run_id}")
    rprint(f"  AUPRC={metrics['average_precision']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    app()
