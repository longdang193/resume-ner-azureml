"""Tests for metric logging utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

from training.logging import log_metrics


class TestLogMetrics:
    """Tests for log_metrics function."""

    def test_log_metrics_creates_file(self, temp_dir):
        """Test that metrics are written to file."""
        output_dir = temp_dir / "outputs"
        metrics = {"f1": 0.95, "precision": 0.92, "recall": 0.93}
        
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        metrics_file = output_dir / "metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == metrics

    def test_log_metrics_creates_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        output_dir = temp_dir / "new" / "outputs"
        metrics = {"f1": 0.95}
        
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        assert output_dir.exists()
        assert (output_dir / "metrics.json").exists()

    def test_log_metrics_calls_adapter(self, temp_dir):
        """Test that logging adapter is called with metrics."""
        output_dir = temp_dir / "outputs"
        metrics = {"f1": 0.95, "precision": 0.92}
        
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        mock_adapter.log_metrics.assert_called_once_with(metrics)

    @patch("platform_adapters.get_platform_adapter")
    def test_log_metrics_creates_default_adapter(self, mock_get_adapter, temp_dir):
        """Test that default adapter is created when none provided."""
        output_dir = temp_dir / "outputs"
        metrics = {"f1": 0.95}
        
        mock_platform_adapter = MagicMock()
        mock_logging_adapter = MagicMock()
        mock_platform_adapter.get_logging_adapter.return_value = mock_logging_adapter
        mock_get_adapter.return_value = mock_platform_adapter
        
        log_metrics(output_dir, metrics, logging_adapter=None)
        
        mock_get_adapter.assert_called_once()
        mock_platform_adapter.get_logging_adapter.assert_called_once()
        mock_logging_adapter.log_metrics.assert_called_once_with(metrics)

    def test_log_metrics_with_empty_dict(self, temp_dir):
        """Test logging empty metrics dictionary."""
        output_dir = temp_dir / "outputs"
        metrics = {}
        
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == {}

    def test_log_metrics_with_nested_values(self, temp_dir):
        """Test logging metrics with nested structure."""
        output_dir = temp_dir / "outputs"
        metrics = {
            "overall": {"f1": 0.95, "precision": 0.92},
            "per_label": {"PERSON": 0.98, "ORG": 0.91}
        }
        
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == metrics

    def test_log_metrics_overwrites_existing_file(self, temp_dir):
        """Test that existing metrics file is overwritten."""
        output_dir = temp_dir / "outputs"
        output_dir.mkdir()
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text('{"old": "data"}')
        
        metrics = {"new": "data"}
        mock_adapter = MagicMock()
        
        log_metrics(output_dir, metrics, logging_adapter=mock_adapter)
        
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == metrics
        assert "old" not in saved_metrics

