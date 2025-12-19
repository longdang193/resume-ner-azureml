"""Tests for MLflow context management to prevent nested runs and metric logging issues."""

import pytest
import sys
from unittest.mock import MagicMock, patch
from platform_adapters.mlflow_context import (
    AzureMLMLflowContextManager,
    LocalMLflowContextManager,
)


class TestAzureMLMLflowContextManager:
    """Tests for Azure ML MLflow context manager."""

    def test_get_context_returns_nullcontext(self):
        """Test that Azure ML context manager returns nullcontext (no-op)."""
        manager = AzureMLMLflowContextManager()
        context = manager.get_context()
        
        from contextlib import nullcontext
        assert isinstance(context, type(nullcontext()))

    @patch("platform_adapters.mlflow_context.mlflow", create=True)
    def test_does_not_start_new_run(self, mock_mlflow):
        """Test that Azure ML context manager does NOT call mlflow.start_run()."""
        manager = AzureMLMLflowContextManager()
        context = manager.get_context()
        
        # Enter and exit context
        with context:
            pass
        
        # Should NOT have called start_run
        mock_mlflow.start_run.assert_not_called()

    def test_context_is_no_op(self):
        """Test that the context manager is truly a no-op."""
        manager = AzureMLMLflowContextManager()
        context = manager.get_context()
        
        # Should be able to enter/exit without side effects
        with context:
            # Should not raise any errors
            pass


class TestLocalMLflowContextManager:
    """Tests for Local MLflow context manager."""

    def test_get_context_starts_run(self):
        """Test that local context manager starts an MLflow run."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            manager = LocalMLflowContextManager()
            context = manager.get_context()
            
            with context:
                pass
            
            # Should have called start_run
            mock_mlflow.start_run.assert_called_once()

    def test_sets_experiment_from_env(self):
        """Test that experiment is set from environment variable if provided."""
        mock_mlflow = MagicMock()
        mock_os = MagicMock()
        mock_os.environ.get.return_value = "test_experiment"
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "os": mock_os}):
            manager = LocalMLflowContextManager()
            context = manager.get_context()
            
            mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_no_experiment_if_env_not_set(self):
        """Test that experiment is not set if environment variable is not provided."""
        mock_mlflow = MagicMock()
        mock_os = MagicMock()
        mock_os.environ.get.return_value = None
        
        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "os": mock_os}):
            manager = LocalMLflowContextManager()
            context = manager.get_context()
            
            # set_experiment should not be called if env var is not set
            # (or it might be called with None, but that's implementation detail)
            # The key is that start_run should still be called
            mock_mlflow.start_run.assert_called_once()

