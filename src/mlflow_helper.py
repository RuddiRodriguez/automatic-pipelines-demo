"""Record experiments locally without starting a background server."""
from pathlib import Path

import mlflow


class MLFlowLogger:
    def __init__(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(directory.as_uri())

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

    def log_artifacts(self, paths):
        for path in paths:
            mlflow.log_artifact(str(path))
