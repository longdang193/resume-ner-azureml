"""MLflow context management for different platforms."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class MLflowContextManager(ABC):
    """Abstract interface for MLflow context management."""

    @abstractmethod
    def get_context(self) -> AbstractContextManager[Any]:
        """
        Get the MLflow context manager for this platform.

        Returns:
            Context manager that handles MLflow run lifecycle.
        """
        pass


class AzureMLMLflowContextManager(MLflowContextManager):
    """MLflow context manager for Azure ML jobs.

    Azure ML automatically creates an MLflow run context for each job.
    We should NOT call mlflow.start_run() when running in Azure ML, as it creates
    a nested/separate run, causing metrics to be logged to the wrong run.
    """

    def get_context(self) -> AbstractContextManager[Any]:
        """Return a no-op context manager (Azure ML handles MLflow automatically)."""
        from contextlib import nullcontext
        return nullcontext()


class LocalMLflowContextManager(MLflowContextManager):
    """MLflow context manager for local execution."""

    def get_context(self) -> AbstractContextManager[Any]:
        """Return MLflow start_run context manager for local execution."""
        import mlflow
        import os

        # Set experiment from environment variable if provided
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        return mlflow.start_run()
