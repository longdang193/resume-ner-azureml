"""Shared MLflow fixtures for tests."""

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

import mlflow
import pytest


@pytest.fixture
def mock_mlflow_tracking(monkeypatch, tmp_path):
    """Mock MLflow to use local file-based tracking.
    
    Sets up a local file-based MLflow tracking URI and mocks the setup function
    to use it. Also mocks Azure ML client creation.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
        tmp_path: Pytest temporary directory fixture
        
    Returns:
        Tracking URI string (file://...)
    """
    # Set local tracking URI
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow_tracking_dir.mkdir()
    tracking_uri = f"file://{mlflow_tracking_dir}"
    
    # Mock setup_mlflow_from_config to use local tracking
    def mock_setup_mlflow_from_config(experiment_name, config_dir=None):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return tracking_uri
    
    monkeypatch.setattr(
        "shared.mlflow_setup.setup_mlflow_from_config",
        mock_setup_mlflow_from_config
    )
    
    # Mock Azure ML client creation if attempted
    mock_ml_client = Mock()
    monkeypatch.setattr(
        "azure.ai.ml.MLClient",
        lambda **kwargs: mock_ml_client,
        raising=False
    )
    
    return tracking_uri


def mock_mlflow_client():
    """Create a mock MLflow client with common operations.
    
    Returns:
        Tuple of (mock_client, mock_parent_run)
    """
    mock_parent_run = Mock()
    mock_parent_run.info.run_id = "hpo_parent_123"
    mock_parent_run.info.experiment_id = "exp_123"
    mock_parent_run.info.status = "RUNNING"
    
    def get_run_side_effect(run_id):
        if run_id == "hpo_parent_123" or isinstance(run_id, str):
            # Set up tags with string values (not Mock objects)
            mock_parent_run.data.tags = {
                "code.study_key_hash": "a" * 64,
                "code.study_family_hash": "b" * 64,
            }
            return mock_parent_run
        return mock_parent_run
    
    mock_client = Mock()
    mock_client.get_run.side_effect = get_run_side_effect
    mock_client.create_run = Mock()
    mock_client.set_tag = Mock()
    mock_client.log_metric = Mock()
    mock_client.log_param = Mock()
    mock_client.set_terminated = Mock()
    
    return mock_client, mock_parent_run


def mock_mlflow_run(
    run_id: str = "test_run_id_123",
    tags: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, str]] = None,
) -> Mock:
    """Create a mock MLflow run with specified attributes.
    
    Args:
        run_id: Run ID
        tags: Dictionary of tags
        metrics: Dictionary of metrics
        params: Dictionary of parameters
        
    Returns:
        Mock MLflow run object
    """
    run = Mock()
    run.info.run_id = run_id
    run.info.experiment_id = "test_experiment_id"
    run.info.status = "FINISHED"
    run.info.start_time = 1234567890
    
    run.data.tags = tags or {}
    run.data.metrics = metrics or {}
    run.data.params = params or {}
    
    return run






