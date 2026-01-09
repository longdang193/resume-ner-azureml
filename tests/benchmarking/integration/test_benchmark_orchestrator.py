"""Component tests for benchmark orchestrator."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from benchmarking.orchestrator import benchmark_best_trials


class TestBenchmarkOrchestratorConfigUsage:
    """Test that benchmark_best_trials() uses config options correctly."""

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_config_batch_sizes(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        sample_benchmark_config,
    ):
        """Test that benchmark_batch_sizes from config is passed to run_benchmarking."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract batch_sizes from config
        batch_sizes = sample_benchmark_config["benchmarking"]["batch_sizes"]
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with batch_sizes from config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=sample_benchmark_config,
            benchmark_batch_sizes=batch_sizes,
        )
        
        # Verify run_benchmarking was called with correct batch_sizes
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["batch_sizes"] == batch_sizes

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_config_iterations(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        sample_benchmark_config,
    ):
        """Test that benchmark_iterations from config is passed to run_benchmarking."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract iterations from config
        iterations = sample_benchmark_config["benchmarking"]["iterations"]
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with iterations from config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=sample_benchmark_config,
            benchmark_iterations=iterations,
        )
        
        # Verify run_benchmarking was called with correct iterations
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["iterations"] == iterations

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_config_warmup(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        sample_benchmark_config,
    ):
        """Test that benchmark_warmup from config is passed to run_benchmarking."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract warmup_iterations from config
        warmup = sample_benchmark_config["benchmarking"]["warmup_iterations"]
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with warmup from config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=sample_benchmark_config,
            benchmark_warmup=warmup,
        )
        
        # Verify run_benchmarking was called with correct warmup_iterations
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["warmup_iterations"] == warmup

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_config_max_length(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        sample_benchmark_config,
    ):
        """Test that benchmark_max_length from config is passed to run_benchmarking."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract max_length from config
        max_length = sample_benchmark_config["benchmarking"]["max_length"]
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with max_length from config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=sample_benchmark_config,
            benchmark_max_length=max_length,
        )
        
        # Verify run_benchmarking was called with correct max_length
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["max_length"] == max_length

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_config_device(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        custom_benchmark_config,
    ):
        """Test that benchmark_device from config is passed to run_benchmarking."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract device from config
        device = custom_benchmark_config["benchmarking"]["device"]
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with device from config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=custom_benchmark_config,
            benchmark_device=device,
        )
        
        # Verify run_benchmarking was called with correct device
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["device"] == device

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_output_filename(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        sample_benchmark_config,
    ):
        """Test that output.filename from config is used for output path."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract filename from config
        config_filename = sample_benchmark_config["output"]["filename"]
        
        # Setup mocks
        output_dir = root_dir / "benchmarking" / "test"
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = output_dir
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call benchmark_best_trials
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=sample_benchmark_config,
        )
        
        # Verify run_benchmarking was called with output_path using config filename
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        output_path = call_args.kwargs["output_path"]
        assert output_path.name == config_filename

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_uses_custom_output_filename(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        custom_benchmark_config,
    ):
        """Test that custom output.filename from config is used for output path."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract custom filename from config
        custom_filename = custom_benchmark_config["output"]["filename"]
        assert custom_filename == "custom_benchmark.json"
        
        # Setup mocks
        output_dir = root_dir / "benchmarking" / "test"
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = output_dir
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call benchmark_best_trials with custom config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=custom_benchmark_config,
        )
        
        # Verify run_benchmarking was called with output_path using custom filename
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        output_path = call_args.kwargs["output_path"]
        assert output_path.name == custom_filename
        assert output_path.name == "custom_benchmark.json"

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_all_config_options_together(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
        custom_benchmark_config,
    ):
        """Test that all config options are used together correctly."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Extract all options from config
        batch_sizes = custom_benchmark_config["benchmarking"]["batch_sizes"]
        iterations = custom_benchmark_config["benchmarking"]["iterations"]
        warmup = custom_benchmark_config["benchmarking"]["warmup_iterations"]
        max_length = custom_benchmark_config["benchmarking"]["max_length"]
        device = custom_benchmark_config["benchmarking"]["device"]
        filename = custom_benchmark_config["output"]["filename"]
        
        # Setup mocks
        output_dir = root_dir / "benchmarking" / "test"
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = output_dir
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Call with all config options
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=custom_benchmark_config,
            benchmark_batch_sizes=batch_sizes,
            benchmark_iterations=iterations,
            benchmark_warmup=warmup,
            benchmark_max_length=max_length,
            benchmark_device=device,
        )
        
        # Verify all options were passed correctly
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["batch_sizes"] == batch_sizes
        assert call_args.kwargs["iterations"] == iterations
        assert call_args.kwargs["warmup_iterations"] == warmup
        assert call_args.kwargs["max_length"] == max_length
        assert call_args.kwargs["device"] == device
        # Verify output filename from config is used
        assert call_args.kwargs["output_path"].name == filename

    def test_benchmark_best_trials_defaults_when_config_missing(
        self,
        tmp_path,
        mock_best_trials,
        mock_test_data_file,
        mock_data_config,
        mock_hpo_config,
    ):
        """Test that defaults are used when benchmark_config is None."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Call without benchmark_config
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=None,
            # Don't pass config options - should use defaults
        )
        
        # Function should handle None config gracefully
        # (defaults are in function signature)
        assert isinstance(result, dict)

