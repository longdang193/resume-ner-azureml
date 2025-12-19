"""Tests for best configuration selection from HPO sweep jobs."""

import pytest
from unittest.mock import MagicMock, patch
import sys
from orchestration.jobs.selection import (
    _configure_mlflow,
    _get_metrics_from_mlflow,
    _get_params_from_mlflow,
    get_best_trial_from_sweep,
)


class TestConfigureMlflow:
    """Tests for _configure_mlflow function."""

    @patch("orchestration.jobs.selection.mlflow")
    def test_configure_mlflow_sets_tracking_uri(self, mock_mlflow):
        """Test that MLflow tracking URI is set from workspace."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspaces.get.return_value = mock_workspace
        mock_ml_client.workspace_name = "test-workspace"
        
        _configure_mlflow(mock_ml_client)
        
        mock_ml_client.workspaces.get.assert_called_once_with(name="test-workspace")
        mock_mlflow.set_tracking_uri.assert_called_once_with("azureml://workspace/experiments")


class TestGetMetricsFromMlflow:
    """Tests for _get_metrics_from_mlflow function."""

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_metrics_success(self, mock_mlflow):
        """Test successfully fetching metrics from MLflow."""
        mock_run = MagicMock()
        mock_run.data.metrics = {"f1": 0.95, "precision": 0.92, "recall": 0.93}
        mock_mlflow.get_run.return_value = mock_run
        
        result = _get_metrics_from_mlflow("run-123")
        
        assert result == {"f1": 0.95, "precision": 0.92, "recall": 0.93}
        mock_mlflow.get_run.assert_called_once_with("run-123")

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_metrics_empty(self, mock_mlflow):
        """Test fetching metrics when run has no metrics."""
        mock_run = MagicMock()
        mock_run.data.metrics = None
        mock_mlflow.get_run.return_value = mock_run
        
        result = _get_metrics_from_mlflow("run-123")
        
        assert result == {}

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_metrics_error_handling(self, mock_mlflow):
        """Test that errors are handled gracefully."""
        mock_mlflow.get_run.side_effect = Exception("MLflow error")
        
        result = _get_metrics_from_mlflow("run-123")
        
        assert result == {}


class TestGetParamsFromMlflow:
    """Tests for _get_params_from_mlflow function."""

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_params_success(self, mock_mlflow):
        """Test successfully fetching params from MLflow."""
        mock_run = MagicMock()
        mock_run.data.params = {"learning_rate": "3e-5", "dropout": "0.2"}
        mock_mlflow.get_run.return_value = mock_run
        
        result = _get_params_from_mlflow("run-123")
        
        assert result == {"learning_rate": "3e-5", "dropout": "0.2"}
        mock_mlflow.get_run.assert_called_once_with("run-123")

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_params_empty(self, mock_mlflow):
        """Test fetching params when run has no params."""
        mock_run = MagicMock()
        mock_run.data.params = None
        mock_mlflow.get_run.return_value = mock_run
        
        result = _get_params_from_mlflow("run-123")
        
        assert result == {}

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_params_error_handling(self, mock_mlflow):
        """Test that errors are handled gracefully."""
        mock_mlflow.get_run.side_effect = Exception("MLflow error")
        
        result = _get_params_from_mlflow("run-123")
        
        assert result == {}


class TestGetBestTrialFromSweep:
    """Tests for get_best_trial_from_sweep function."""

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_best_trial_maximize(self, mock_mlflow):
        """Test getting best trial when goal is maximize."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspaces.get.return_value = mock_workspace
        mock_ml_client.workspace_name = "test-workspace"
        
        mock_sweep_job = MagicMock()
        mock_sweep_job.name = "sweep-job-123"
        
        mock_trial1 = MagicMock()
        mock_trial1.status = "Completed"
        mock_trial1.name = "trial-1"
        mock_trial2 = MagicMock()
        mock_trial2.status = "Completed"
        mock_trial2.name = "trial-2"
        mock_trial3 = MagicMock()
        mock_trial3.status = "Failed"
        mock_trial3.name = "trial-3"
        
        mock_ml_client.jobs.list.return_value = [mock_trial1, mock_trial2, mock_trial3]
        
        mock_run1 = MagicMock()
        mock_run1.data.metrics = {"f1": 0.90}
        mock_run2 = MagicMock()
        mock_run2.data.metrics = {"f1": 0.95}
        mock_mlflow.get_run.side_effect = [mock_run1, mock_run2]
        
        trial, value = get_best_trial_from_sweep(
            mock_ml_client, mock_sweep_job, "f1", "maximize"
        )
        
        assert trial == mock_trial2
        assert value == 0.95

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_best_trial_minimize(self, mock_mlflow):
        """Test getting best trial when goal is minimize."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspaces.get.return_value = mock_workspace
        mock_ml_client.workspace_name = "test-workspace"
        
        mock_sweep_job = MagicMock()
        mock_sweep_job.name = "sweep-job-123"
        
        mock_trial1 = MagicMock()
        mock_trial1.status = "Completed"
        mock_trial1.name = "trial-1"
        mock_trial2 = MagicMock()
        mock_trial2.status = "Completed"
        mock_trial2.name = "trial-2"
        
        mock_ml_client.jobs.list.return_value = [mock_trial1, mock_trial2]
        
        mock_run1 = MagicMock()
        mock_run1.data.metrics = {"loss": 0.5}
        mock_run2 = MagicMock()
        mock_run2.data.metrics = {"loss": 0.3}
        mock_mlflow.get_run.side_effect = [mock_run1, mock_run2]
        
        trial, value = get_best_trial_from_sweep(
            mock_ml_client, mock_sweep_job, "loss", "minimize"
        )
        
        assert trial == mock_trial2
        assert value == 0.3

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_best_trial_no_completed_trials(self, mock_mlflow):
        """Test when no trials are completed."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspaces.get.return_value = mock_workspace
        mock_ml_client.workspace_name = "test-workspace"
        
        mock_sweep_job = MagicMock()
        mock_sweep_job.name = "sweep-job-123"
        
        mock_trial1 = MagicMock()
        mock_trial1.status = "Failed"
        mock_trial2 = MagicMock()
        mock_trial2.status = "Canceled"
        
        mock_ml_client.jobs.list.return_value = [mock_trial1, mock_trial2]
        
        trial, value = get_best_trial_from_sweep(
            mock_ml_client, mock_sweep_job, "f1", "maximize"
        )
        
        assert trial is None
        assert value is None

    @patch("orchestration.jobs.selection.mlflow")
    def test_get_best_trial_missing_metric(self, mock_mlflow):
        """Test when trials don't have the objective metric."""
        mock_ml_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.mlflow_tracking_uri = "azureml://workspace/experiments"
        mock_ml_client.workspaces.get.return_value = mock_workspace
        mock_ml_client.workspace_name = "test-workspace"
        
        mock_sweep_job = MagicMock()
        mock_sweep_job.name = "sweep-job-123"
        
        mock_trial1 = MagicMock()
        mock_trial1.status = "Completed"
        mock_trial1.name = "trial-1"
        
        mock_ml_client.jobs.list.return_value = [mock_trial1]
        
        mock_run1 = MagicMock()
        mock_run1.data.metrics = {"precision": 0.92}  # Missing "f1"
        mock_mlflow.get_run.return_value = mock_run1
        
        trial, value = get_best_trial_from_sweep(
            mock_ml_client, mock_sweep_job, "f1", "maximize"
        )
        
        assert trial is None
        assert value is None

