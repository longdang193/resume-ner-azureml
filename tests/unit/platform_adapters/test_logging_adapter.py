"""Tests for logging adapters."""

import pytest
import sys
from unittest.mock import MagicMock, patch
from platform_adapters.logging_adapter import (
    AzureMLLoggingAdapter,
    LocalLoggingAdapter,
)


class TestAzureMLLoggingAdapter:
    """Tests for AzureMLLoggingAdapter."""

    @patch("azureml.core.Run")
    def test_log_metrics_with_azureml(self, mock_run):
        """Test logging metrics with Azure ML context available."""
        mock_mlflow = MagicMock()
        mock_azureml_run = MagicMock()
        mock_run.get_context.return_value = mock_azureml_run
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = AzureMLLoggingAdapter()
            
            metrics = {"macro-f1": 0.85, "loss": 0.3}
            adapter.log_metrics(metrics)
            
            # Should log to both MLflow and Azure ML
            assert mock_mlflow.log_metric.call_count == 2
            assert mock_azureml_run.log.call_count == 2

    @patch("azureml.core.Run")
    def test_log_metrics_without_azureml(self, mock_run):
        """Test logging metrics without Azure ML context."""
        mock_mlflow = MagicMock()
        mock_run.get_context.side_effect = Exception("Not in Azure ML")
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = AzureMLLoggingAdapter()
            
            metrics = {"macro-f1": 0.85}
            adapter.log_metrics(metrics)
            
            # Should only log to MLflow
            assert mock_mlflow.log_metric.call_count == 1

    @patch("azureml.core.Run")
    def test_log_params_with_azureml(self, mock_run):
        """Test logging parameters with Azure ML context."""
        mock_mlflow = MagicMock()
        mock_azureml_run = MagicMock()
        mock_run.get_context.return_value = mock_azureml_run
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = AzureMLLoggingAdapter()
            
            params = {"learning_rate": 2e-5, "batch_size": 16}
            adapter.log_params(params)
            
            # Should log to MLflow
            mock_mlflow.log_params.assert_called_once_with(params)
            # Should also log to Azure ML with param_ prefix
            assert mock_azureml_run.log.call_count == 2

    @patch("azureml.core.Run")
    def test_log_params_without_azureml(self, mock_run):
        """Test logging parameters without Azure ML context."""
        mock_mlflow = MagicMock()
        mock_run.get_context.side_effect = Exception("Not in Azure ML")
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = AzureMLLoggingAdapter()
            
            params = {"learning_rate": 2e-5}
            adapter.log_params(params)
            
            # Should only log to MLflow
            mock_mlflow.log_params.assert_called_once_with(params)


class TestLocalLoggingAdapter:
    """Tests for LocalLoggingAdapter."""

    def test_log_metrics(self):
        """Test logging metrics locally."""
        mock_mlflow = MagicMock()
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = LocalLoggingAdapter()
            
            metrics = {"macro-f1": 0.85, "loss": 0.3}
            adapter.log_metrics(metrics)
            
            # Should log to MLflow only
            assert mock_mlflow.log_metric.call_count == 2

    def test_log_params(self):
        """Test logging parameters locally."""
        mock_mlflow = MagicMock()
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            adapter = LocalLoggingAdapter()
            
            params = {"learning_rate": 2e-5, "batch_size": 16}
            adapter.log_params(params)
            
            # Should log to MLflow only
            mock_mlflow.log_params.assert_called_once_with(params)

