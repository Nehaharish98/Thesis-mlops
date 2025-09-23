# src/ml/mlflow_utils.py
import mlflow
import time
import os

def start_run(exp_name: str, run_name: str = None, tags: dict | None = None):
    """Start an MLflow run inside a given experiment."""
    mlflow.set_experiment(exp_name)   # creates if it doesn't exist
    run = mlflow.start_run(run_name=run_name)
    if tags:
        for k, v in tags.items():
            mlflow.set_tag(k, v)
    mlflow.set_tag("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    sha = os.getenv("GIT_COMMIT", "")
    if sha:
        mlflow.set_tag("git_commit", sha)
    return run

def log_params(d: dict):
    """Log parameters to MLflow."""
    for k, v in d.items():
        mlflow.log_param(k, v)

def log_metrics(d: dict, step: int | None = None):
    """Log metrics to MLflow."""
    for k, v in d.items():
        mlflow.log_metric(k, float(v), step=step)

def log_artifact(path: str):
    """Log an artifact (file) to MLflow."""
    mlflow.log_artifact(path)

def end_run():
    """End the current MLflow run."""
    mlflow.end_run()
