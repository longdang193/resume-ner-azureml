"""Integration tests for tracking.enabled configuration options.

Tests that tracking.benchmark.enabled, tracking.training.enabled, and
tracking.conversion.enabled actually control MLflow run creation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import nullcontext
import mlflow

from infrastructure.tracking.mlflow.trackers.benchmark_tracker import MLflowBenchmarkTracker
from infrastructure.tracking.mlflow.trackers.training_tracker import MLflowTrainingTracker
from infrastructure.tracking.mlflow.trackers.conversion_tracker import MLflowConversionTracker


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with mlflow.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


class TestBenchmarkTrackingEnabled:
    """Test tracking.benchmark.enabled option."""

    def test_benchmark_tracking_enabled_creates_run(self, tmp_config_dir, tmp_path):
        """Test that benchmark tracking creates run when enabled=true."""
        # Create mlflow.yaml with benchmark enabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: true
    log_artifacts: true
""")
        
        output_dir = tmp_path / "outputs" / "benchmarking"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowBenchmarkTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_benchmark_run(
                    run_name="test_benchmark",
                    backbone="distilbert",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is not None
                    assert handle.run_id == "test_run_id"
            
            # Should have created a run
            assert mock_start_run.called

    def test_benchmark_tracking_disabled_skips_run(self, tmp_config_dir, tmp_path):
        """Test that benchmark tracking skips run creation when enabled=false."""
        # Create mlflow.yaml with benchmark disabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: false
    log_artifacts: true
""")
        
        output_dir = tmp_path / "outputs" / "benchmarking"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowBenchmarkTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_benchmark_run(
                    run_name="test_benchmark",
                    backbone="distilbert",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is None
            
            # Should NOT have created a run
            assert not mock_start_run.called


class TestTrainingTrackingEnabled:
    """Test tracking.training.enabled option."""

    def test_training_tracking_enabled_creates_run(self, tmp_config_dir, tmp_path):
        """Test that training tracking creates run when enabled=true."""
        # Create mlflow.yaml with training enabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: true
""")
        
        output_dir = tmp_path / "outputs" / "final_training"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowTrainingTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_training_run(
                    run_name="test_training",
                    backbone="distilbert",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is not None
                    assert handle.run_id == "test_run_id"
            
            # Should have created a run
            assert mock_start_run.called

    def test_training_tracking_disabled_skips_run(self, tmp_config_dir, tmp_path):
        """Test that training tracking skips run creation when enabled=false."""
        # Create mlflow.yaml with training disabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: false
    log_checkpoint: true
    log_metrics_json: true
""")
        
        output_dir = tmp_path / "outputs" / "final_training"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowTrainingTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_training_run(
                    run_name="test_training",
                    backbone="distilbert",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is None
            
            # Should NOT have created a run
            assert not mock_start_run.called


class TestConversionTrackingEnabled:
    """Test tracking.conversion.enabled option."""

    def test_conversion_tracking_enabled_creates_run(self, tmp_config_dir, tmp_path):
        """Test that conversion tracking creates run when enabled=true."""
        # Create mlflow.yaml with conversion enabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: true
""")
        
        output_dir = tmp_path / "outputs" / "conversion"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowConversionTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_conversion_run(
                    run_name="test_conversion",
                    conversion_type="onnx",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is not None
                    assert handle.run_id == "test_run_id"
            
            # Should have created a run
            assert mock_start_run.called

    def test_conversion_tracking_disabled_skips_run(self, tmp_config_dir, tmp_path):
        """Test that conversion tracking skips run creation when enabled=false."""
        # Create mlflow.yaml with conversion disabled
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: false
    log_onnx_model: true
    log_conversion_log: true
""")
        
        output_dir = tmp_path / "outputs" / "conversion"
        output_dir.mkdir(parents=True)
        
        tracker = MLflowConversionTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with tracker.start_conversion_run(
                    run_name="test_conversion",
                    conversion_type="onnx",
                    output_dir=output_dir,
                ) as handle:
                    assert handle is None
            
            # Should NOT have created a run
            assert not mock_start_run.called


class TestLogArtifactsOptions:
    """Test tracking.*.log_* options control artifact logging."""

    def test_benchmark_log_artifacts_disabled_skips_logging(self, tmp_config_dir, tmp_path):
        """Test that benchmark artifact logging is skipped when log_artifacts=false."""
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: true
    log_artifacts: false
""")
        
        output_dir = tmp_path / "outputs" / "benchmarking"
        output_dir.mkdir(parents=True)
        benchmark_json = output_dir / "benchmark.json"
        benchmark_json.write_text('{"test": "data"}')
        
        tracker = MLflowBenchmarkTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with tracker.start_benchmark_run(
                        run_name="test_benchmark",
                        backbone="distilbert",
                        output_dir=output_dir,
                    ):
                        tracker.log_benchmark_results(
                            batch_sizes=[1, 8],
                            iterations=100,
                            warmup_iterations=10,
                            max_length=512,
                            device="cpu",
                            benchmark_json_path=benchmark_json,
                            benchmark_data={"batch_1": {"mean_ms": 10.5}, "batch_8": {"mean_ms": 50.2}},
                        )
                    
                    # Should NOT have logged artifact
                    assert not mock_log_artifact.called

    def test_training_log_checkpoint_disabled_skips_logging(self, tmp_config_dir, tmp_path):
        """Test that training checkpoint logging is skipped when log_checkpoint=false."""
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: false
    log_metrics_json: true
""")
        
        output_dir = tmp_path / "outputs" / "final_training"
        output_dir.mkdir(parents=True)
        checkpoint_dir = output_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        
        tracker = MLflowTrainingTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with patch("mlflow.log_artifacts") as mock_log_artifacts:
                    with tracker.start_training_run(
                        run_name="test_training",
                        backbone="distilbert",
                        output_dir=output_dir,
                    ):
                        tracker.log_training_artifacts(checkpoint_dir=checkpoint_dir)
                    
                    # Should NOT have logged checkpoint
                    assert not mock_log_artifacts.called

    def test_training_log_metrics_json_disabled_skips_logging(self, tmp_config_dir, tmp_path):
        """Test that training metrics.json logging is skipped when log_metrics_json=false."""
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: false
""")
        
        output_dir = tmp_path / "outputs" / "final_training"
        output_dir.mkdir(parents=True)
        checkpoint_dir = output_dir / "checkpoint"
        checkpoint_dir.mkdir()
        metrics_json = output_dir / "metrics.json"
        metrics_json.write_text('{"macro-f1": 0.9}')
        
        tracker = MLflowTrainingTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with tracker.start_training_run(
                        run_name="test_training",
                        backbone="distilbert",
                        output_dir=output_dir,
                    ):
                        tracker.log_training_artifacts(
                            checkpoint_dir=checkpoint_dir,
                            metrics_json_path=metrics_json,
                        )
                    
                    # Should have logged checkpoint but NOT metrics.json
                    # Check that log_artifact was not called (metrics.json uses log_artifact, checkpoint uses log_artifacts)
                    artifact_calls = [call for call in mock_log_artifact.call_args_list if "metrics.json" in str(call)]
                    assert len(artifact_calls) == 0

    def test_conversion_log_onnx_model_disabled_skips_logging(self, tmp_config_dir, tmp_path):
        """Test that conversion ONNX model logging is skipped when log_onnx_model=false."""
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: false
    log_conversion_log: true
""")
        
        output_dir = tmp_path / "outputs" / "conversion"
        output_dir.mkdir(parents=True)
        onnx_model = output_dir / "model.onnx"
        onnx_model.write_bytes(b"fake onnx model")
        
        tracker = MLflowConversionTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with tracker.start_conversion_run(
                        run_name="test_conversion",
                        conversion_type="onnx",
                        output_dir=output_dir,
                    ):
                        tracker.log_conversion_results(
                            conversion_success=True,
                            onnx_model_path=onnx_model,
                        )
                    
                    # Should NOT have logged ONNX model
                    onnx_calls = [call for call in mock_log_artifact.call_args_list if "model.onnx" in str(call) or "onnx" in str(call).lower()]
                    assert len(onnx_calls) == 0

    def test_conversion_log_conversion_log_disabled_skips_logging(self, tmp_config_dir, tmp_path):
        """Test that conversion log logging is skipped when log_conversion_log=false."""
        mlflow_yaml = tmp_config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: false
""")
        
        output_dir = tmp_path / "outputs" / "conversion"
        output_dir.mkdir(parents=True)
        conversion_log = output_dir / "conversion_log.txt"
        conversion_log.write_text("conversion log content")
        
        tracker = MLflowConversionTracker("test-experiment")
        
        with patch("mlflow.start_run") as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_run.info.experiment_id = "test_experiment_id"
            mock_run.info.artifact_uri = "test_artifact_uri"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            
            with patch("mlflow.get_tracking_uri", return_value="file:///tmp/mlflow"):
                with patch("mlflow.log_artifact") as mock_log_artifact:
                    with tracker.start_conversion_run(
                        run_name="test_conversion",
                        conversion_type="onnx",
                        output_dir=output_dir,
                    ):
                        tracker.log_conversion_results(
                            conversion_success=True,
                            onnx_model_path=None,
                            conversion_log_path=conversion_log,
                        )
                    
                    # Should NOT have logged conversion log
                    # Check all calls to see if conversion_log was logged
                    all_calls = [str(call) for call in mock_log_artifact.call_args_list]
                    log_calls = [call for call in all_calls if "conversion_log" in call.lower()]
                    # Also check for "conversion_log.txt" in artifact_path
                    artifact_path_calls = [call for call in mock_log_artifact.call_args_list if len(call[0]) > 1 and "conversion_log" in str(call[0][1]).lower()]
                    assert len(log_calls) == 0 and len(artifact_path_calls) == 0, f"Found {len(log_calls)} log_calls and {len(artifact_path_calls)} artifact_path_calls. All calls: {all_calls}"

