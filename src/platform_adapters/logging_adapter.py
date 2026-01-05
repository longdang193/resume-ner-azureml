"""Logging adapters for different platforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LoggingAdapter(ABC):
    """Abstract interface for platform-specific logging."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to platform-specific logging system.

        Args:
            metrics: Dictionary of metric names to values.
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to platform-specific logging system.

        Args:
            params: Dictionary of parameter names to values.
        """
        pass


class AzureMLLoggingAdapter(LoggingAdapter):
    """Logging adapter for Azure ML jobs."""

    def __init__(self):
        """Initialize Azure ML logging adapter."""
        self._azureml_run = None
        self._try_init_azureml()

    def _try_init_azureml(self) -> None:
        """Try to initialize Azure ML run context."""
        try:
            from azureml.core import Run
            self._azureml_run = Run.get_context()
        except Exception:
            # Azure ML not available (e.g., running locally)
            self._azureml_run = None

    def _to_float_or_none(self, v):
        """Convert value to float if possible, otherwise return None."""
        # numpy scalars
        try:
            import numpy as np
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
        except (ImportError, Exception):
            pass

        # bool -> int -> float
        if isinstance(v, bool):
            return float(int(v))

        # int/float
        if isinstance(v, (int, float)):
            return float(v)

        # numeric string
        if isinstance(v, str):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        # Skip nested dicts, lists, etc.
        return None

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to both MLflow and Azure ML native logging."""
        import mlflow
        import os
        
        # Check if we should use client API (refit mode without active run)
        use_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
        active_run = mlflow.active_run()
        
        # MLflow only accepts numeric values
        for k, v in metrics.items():
            val = self._to_float_or_none(v)
            if val is not None:
                if use_run_id and not active_run:
                    # Use client API when logging to existing run without active context
                    client = mlflow.tracking.MlflowClient()
                    client.log_metric(use_run_id, k, val)
                else:
                    # Use active run context (normal case)
                    mlflow.log_metric(k, val)

        # Azure ML native logging accepts strings too
        if self._azureml_run is not None:
            for k, v in metrics.items():
                val = self._to_float_or_none(v)
                if val is not None:
                    self._azureml_run.log(k, val)
                elif isinstance(v, str):
                    # Azure ML can handle string metrics
                    self._azureml_run.log(k, v)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to both MLflow and Azure ML native logging."""
        import mlflow
        import os
        
        # Check if we should use client API (refit mode without active run)
        use_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
        active_run = mlflow.active_run()
        
        if use_run_id and not active_run:
            # Use client API when logging to existing run without active context
            client = mlflow.tracking.MlflowClient()
            for k, v in params.items():
                client.log_param(use_run_id, k, str(v))
        else:
            # Use active run context (normal case)
            mlflow.log_params(params)

        # Azure ML native logging doesn't have a direct params equivalent,
        # but we can log them as metrics with a prefix if needed
        if self._azureml_run is not None:
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    self._azureml_run.log(f"param_{k}", v)


class LocalLoggingAdapter(LoggingAdapter):
    """Logging adapter for local execution."""

    def _to_float_or_none(self, v):
        """Convert value to float if possible, otherwise return None."""
        # numpy scalars
        try:
            import numpy as np
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
        except (ImportError, Exception):
            pass

        # bool -> int -> float
        if isinstance(v, bool):
            return float(int(v))

        # int/float
        if isinstance(v, (int, float)):
            return float(v)

        # numeric string
        if isinstance(v, str):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        # Skip nested dicts, lists, etc.
        return None

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow only."""
        import mlflow
        import os
        
        # Check if we should use client API (refit mode without active run)
        use_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
        active_run = mlflow.active_run()
        
        # MLflow only accepts numeric values
        for k, v in metrics.items():
            val = self._to_float_or_none(v)
            if val is not None:
                if use_run_id and not active_run:
                    # Use client API when logging to existing run without active context
                    client = mlflow.tracking.MlflowClient()
                    client.log_metric(use_run_id, k, val)
                else:
                    # Use active run context (normal case)
                    mlflow.log_metric(k, val)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow only."""
        import mlflow
        import os
        
        # Check if we should use client API (refit mode without active run)
        use_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
        active_run = mlflow.active_run()
        
        if use_run_id and not active_run:
            # Use client API when logging to existing run without active context
            client = mlflow.tracking.MlflowClient()
            for k, v in params.items():
                client.log_param(use_run_id, k, str(v))
        else:
            # Use active run context (normal case)
            mlflow.log_params(params)
