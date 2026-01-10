"""Unit tests for trial_id extraction in benchmarking."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from benchmarking.utils import run_benchmarking


def _create_mock_tracker():
    """Helper to create a mock MLflow tracker for tests."""
    mock_tracker = Mock()
    mock_run = Mock()
    mock_run.run_id = "test_run_id"
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__ = Mock(return_value=mock_run)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_tracker.start_benchmark_run = Mock(return_value=mock_context_manager)
    return mock_tracker


def _create_mock_output_file(output_path):
    """Helper to create a mock benchmark output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('{"batch_size_1": {"mean_latency_ms": 5.0}}')


class TestTrialIdExtraction:
    """Test trial_id extraction logic in run_benchmarking."""

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_from_parameter_old_format(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test that trial_id parameter is used when provided (old format: trial_1_20251231_161745)."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "1_20251231_161745"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call with trial_id parameter (old format)
        trial_id = "trial_1_20251231_161745"
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=trial_id,
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == trial_id

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_from_parameter_new_format(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test that trial_id parameter is used when provided (new format: trial-25d03eeb)."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial-25d03eeb"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file (required for MLflow tracking to trigger)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"batch_size_1": {"mean_latency_ms": 5.0}}')
        
        # Need to provide tracker to enable MLflow tracking (which calls create_naming_context)
        mock_tracker = Mock()
        mock_run = Mock()
        mock_run.run_id = "test_run_id"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = Mock(return_value=mock_run)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracker.start_benchmark_run = Mock(return_value=mock_context_manager)
        
        # Call with trial_id parameter (new format)
        trial_id = "trial-25d03eeb"
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,  # Provide tracker to enable MLflow tracking
            trial_id=trial_id,
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == trial_id

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_extraction_from_path_old_format(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_test_data_file,
    ):
        """Test that trial_id is extracted from checkpoint path (old format: trial_1_20251231_161745)."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Create checkpoint path with old format trial directory
        checkpoint_dir = tmp_path / "hpo" / "local" / "distilbert" / "study-abc123" / "trial_1_20251231_161745" / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial_1_20251231_161745"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call without trial_id parameter, should extract from path
        success = run_benchmarking(
            checkpoint_dir=checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=None,  # Not provided
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with extracted trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == "trial_1_20251231_161745"

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_extraction_from_path_new_format(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_test_data_file,
    ):
        """Test that trial_id is extracted from checkpoint path (new format: trial-25d03eeb)."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Create checkpoint path with new format trial directory
        checkpoint_dir = tmp_path / "hpo" / "local" / "distilbert" / "study-abc123" / "trial-25d03eeb" / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial-25d03eeb"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call without trial_id parameter, should extract from path
        success = run_benchmarking(
            checkpoint_dir=checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=None,  # Not provided
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with extracted trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == "trial-25d03eeb"

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_extraction_from_refit_checkpoint_path(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_test_data_file,
    ):
        """Test that trial_id is extracted from refit checkpoint path (trial-25d03eeb/refit/checkpoint)."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Create checkpoint path with refit structure
        checkpoint_dir = tmp_path / "hpo" / "local" / "distilbert" / "study-abc123" / "trial-25d03eeb" / "refit" / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial-25d03eeb"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call without trial_id parameter, should extract from path (walking up to find trial-25d03eeb)
        success = run_benchmarking(
            checkpoint_dir=checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=None,  # Not provided
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with extracted trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == "trial-25d03eeb"

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_fallback_to_trial_key_hash(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test that trial_id falls back to trial_key_hash when not provided and path extraction fails."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial-25d03eeb"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call without trial_id parameter and with checkpoint that doesn't have trial in path
        # Should fallback to constructing from trial_key_hash
        trial_key_hash = "25d03eebe00267cc"
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,  # No trial in path
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=None,  # Not provided
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash=trial_key_hash,
        )
        
        # Verify create_naming_context was called with constructed trial_id
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        # Should construct as trial-{first_8_chars_of_hash}
        assert call_kwargs["trial_id"] == f"trial-{trial_key_hash[:8]}"

    @patch("benchmarking.utils.subprocess.run")
    @patch("infrastructure.naming.create_naming_context")
    @patch("common.shared.platform_detection.detect_platform")
    def test_trial_id_parameter_overrides_path_extraction(
        self,
        mock_detect_platform,
        mock_create_context,
        mock_subprocess,
        tmp_path,
        mock_test_data_file,
    ):
        """Test that trial_id parameter takes precedence over path extraction."""
        project_root = tmp_path
        output_path = tmp_path / "benchmark.json"
        
        # Create checkpoint path with trial directory
        checkpoint_dir = tmp_path / "hpo" / "local" / "distilbert" / "study-abc123" / "trial-25d03eeb" / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Mock naming context
        mock_context = Mock()
        mock_context.trial_id = "trial-custom123"
        mock_create_context.return_value = mock_context
        mock_detect_platform.return_value = "local"
        
        # Create output file and tracker for MLflow tracking
        _create_mock_output_file(output_path)
        mock_tracker = _create_mock_tracker()
        
        # Call with trial_id parameter, should use parameter not path
        custom_trial_id = "trial-custom123"
        success = run_benchmarking(
            checkpoint_dir=checkpoint_dir,  # Has trial-25d03eeb in path
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
            tracker=mock_tracker,
            trial_id=custom_trial_id,  # Provided, should override path
            benchmark_source="hpo_trial",
            study_key_hash="abc123",
            trial_key_hash="def456",
        )
        
        # Verify create_naming_context was called with parameter trial_id, not path
        assert mock_create_context.called
        call_kwargs = mock_create_context.call_args.kwargs
        assert call_kwargs["trial_id"] == custom_trial_id
        assert call_kwargs["trial_id"] != "trial-25d03eeb"

