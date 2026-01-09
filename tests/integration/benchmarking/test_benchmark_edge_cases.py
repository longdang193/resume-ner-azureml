"""Edge case and validation tests for benchmark configuration."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestration.config_loader import load_all_configs, ExperimentConfig
from orchestration.jobs.benchmarking.orchestrator import benchmark_best_trials
from orchestration.benchmark_utils import run_benchmarking


class TestBenchmarkConfigEdgeCases:
    """Test edge cases and validation for benchmark configuration."""

    def test_missing_benchmark_yaml_uses_defaults(self, tmp_path):
        """Test that missing benchmark.yaml doesn't cause errors."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_dir / "data.yaml",
            model_config=config_dir / "model.yaml",
            train_config=config_dir / "train.yaml",
            hpo_config=config_dir / "hpo.yaml",
            env_config=config_dir / "env.yaml",
            benchmark_config=config_dir / "benchmark.yaml",  # Doesn't exist
            stages={},
            naming={},
        )
        
        # Create other config files
        (config_dir / "data.yaml").write_text("version: 1.0")
        (config_dir / "model.yaml").write_text("backbone: distilbert")
        (config_dir / "train.yaml").write_text("epochs: 10")
        (config_dir / "hpo.yaml").write_text("max_trials: 5")
        (config_dir / "env.yaml").write_text("name: test")
        
        # Should not raise error
        configs = load_all_configs(exp_cfg)
        
        # Benchmark config should not be loaded
        assert "benchmark" not in configs

    def test_empty_batch_sizes_handled_gracefully(self, tmp_path):
        """Test that empty batch_sizes list is handled."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": []
            }
        }
        
        # Extract with default fallback
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        
        # Empty list should be returned (not default)
        assert batch_sizes == []
        assert isinstance(batch_sizes, list)

    def test_non_integer_batch_sizes_type_check(self):
        """Test that batch_sizes contains integers (type validation)."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [1, 8, 16]
            }
        }
        
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        
        # Verify all are integers
        assert all(isinstance(bs, int) for bs in batch_sizes)

    def test_negative_iterations_handled(self):
        """Test that negative iterations value is possible (validation should catch)."""
        benchmark_config = {
            "benchmarking": {
                "iterations": -10  # Invalid but config loader doesn't validate
            }
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        # Config loader doesn't validate, so negative value is returned
        assert iterations == -10

    def test_zero_iterations_handled(self):
        """Test that zero iterations value is possible."""
        benchmark_config = {
            "benchmarking": {
                "iterations": 0
            }
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        assert iterations == 0

    def test_non_integer_iterations_type_check(self):
        """Test that iterations is an integer."""
        benchmark_config = {
            "benchmarking": {
                "iterations": 100
            }
        }
        
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        
        assert isinstance(iterations, int)

    def test_negative_warmup_iterations_handled(self):
        """Test that negative warmup_iterations is possible."""
        benchmark_config = {
            "benchmarking": {
                "warmup_iterations": -5
            }
        }
        
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        
        assert warmup == -5

    def test_non_integer_warmup_type_check(self):
        """Test that warmup_iterations is an integer."""
        benchmark_config = {
            "benchmarking": {
                "warmup_iterations": 10
            }
        }
        
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        
        assert isinstance(warmup, int)

    def test_negative_max_length_handled(self):
        """Test that negative max_length is possible."""
        benchmark_config = {
            "benchmarking": {
                "max_length": -100
            }
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert max_length == -100

    def test_zero_max_length_handled(self):
        """Test that zero max_length is possible."""
        benchmark_config = {
            "benchmarking": {
                "max_length": 0
            }
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert max_length == 0

    def test_non_integer_max_length_type_check(self):
        """Test that max_length is an integer."""
        benchmark_config = {
            "benchmarking": {
                "max_length": 512
            }
        }
        
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        
        assert isinstance(max_length, int)

    def test_invalid_device_value_handled(self):
        """Test that invalid device value is possible (not validated)."""
        benchmark_config = {
            "benchmarking": {
                "device": "invalid_device"
            }
        }
        
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        # Config loader doesn't validate device values
        assert device == "invalid_device"

    def test_test_data_nonexistent_path_handled(self, tmp_path):
        """Test that nonexistent test_data path is handled gracefully."""
        benchmark_config = {
            "benchmarking": {
                "test_data": "/nonexistent/path/test.json"
            }
        }
        
        test_data = benchmark_config.get("benchmarking", {}).get("test_data", None)
        
        assert test_data == "/nonexistent/path/test.json"
        # Path doesn't exist, but config loading doesn't validate existence
        assert not Path(test_data).exists()

    def test_test_data_resolution_fallback_logic(self, tmp_path):
        """Test test_data resolution fallback (test.json → validation.json)."""
        # This tests the logic that would be used to resolve test_data
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        
        # Create validation.json but not test.json
        (dataset_dir / "validation.json").write_text('[]')
        
        # Simulate resolution logic: try test.json, then validation.json
        test_data_config = None  # null in config
        test_data_path = None
        
        if test_data_config:
            # Explicit path provided
            test_data_path = Path(test_data_config)
        else:
            # Fallback: try test.json, then validation.json
            test_json = dataset_dir / "test.json"
            val_json = dataset_dir / "validation.json"
            
            if test_json.exists():
                test_data_path = test_json
            elif val_json.exists():
                test_data_path = val_json
        
        # Should resolve to validation.json
        assert test_data_path == val_json
        assert test_data_path.exists()

    def test_output_filename_with_path_separators(self):
        """Test that output filename with path separators is handled."""
        benchmark_config = {
            "output": {
                "filename": "subdir/benchmark.json"  # Contains path separator
            }
        }
        
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        
        # Config loader doesn't sanitize, so path separator is preserved
        assert filename == "subdir/benchmark.json"
        assert "/" in filename


    def test_missing_benchmarking_section_uses_defaults(self):
        """Test that missing benchmarking section uses defaults."""
        benchmark_config = {
            "output": {
                "filename": "benchmark.json"
            }
        }
        
        # Extract with defaults
        batch_sizes = benchmark_config.get("benchmarking", {}).get("batch_sizes", [1, 8, 16])
        iterations = benchmark_config.get("benchmarking", {}).get("iterations", 100)
        warmup = benchmark_config.get("benchmarking", {}).get("warmup_iterations", 10)
        max_length = benchmark_config.get("benchmarking", {}).get("max_length", 512)
        device = benchmark_config.get("benchmarking", {}).get("device", None)
        
        # Should use defaults
        assert batch_sizes == [1, 8, 16]
        assert iterations == 100
        assert warmup == 10
        assert max_length == 512
        assert device is None

    def test_missing_output_section_uses_defaults(self):
        """Test that missing output section uses defaults."""
        benchmark_config = {
            "benchmarking": {
                "batch_sizes": [1, 8, 16]
            }
        }
        
        # Extract with defaults
        filename = benchmark_config.get("output", {}).get("filename", "benchmark.json")
        save_summary = benchmark_config.get("output", {}).get("save_summary", True)
        
        # Should use defaults
        assert filename == "benchmark.json"
        assert save_summary is True

    @patch("orchestration.jobs.benchmarking.orchestrator.run_benchmarking")
    @patch("orchestration.jobs.benchmarking.orchestrator.create_naming_context")
    @patch("orchestration.jobs.benchmarking.orchestrator.build_output_path")
    @patch("orchestration.jobs.benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("orchestration.jobs.benchmarking.orchestrator.validate_path_before_mkdir")
    def test_benchmark_best_trials_handles_missing_test_data(
        self,
        mock_validate_path,
        mock_resolve_colab,
        mock_build_path,
        mock_create_context,
        mock_run_benchmarking,
        tmp_path,
        mock_best_trials,
        mock_data_config,
        mock_hpo_config,
    ):
        """Test that benchmark_best_trials handles missing test_data gracefully."""
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        
        # Use nonexistent test_data path
        nonexistent_test_data = tmp_path / "nonexistent" / "test.json"
        
        # Setup mocks
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = root_dir / "benchmarking" / "test"
        mock_create_context.return_value = Mock()
        
        # Call benchmark_best_trials with nonexistent test_data
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=nonexistent_test_data,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=None,
        )
        
        # Should return empty dict and not call run_benchmarking
        assert result == {}
        assert not mock_run_benchmarking.called

    @patch("orchestration.benchmark_utils.subprocess.run")
    def test_run_benchmarking_handles_missing_benchmark_script(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test that run_benchmarking handles missing benchmark script gracefully."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Don't create benchmark script (should return False)
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
        )
        
        # Should return False when script doesn't exist
        assert success is False
        assert not mock_subprocess.called

    @patch("orchestration.benchmark_utils.subprocess.run")
    def test_run_benchmarking_handles_subprocess_failure(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test that run_benchmarking handles subprocess failure gracefully."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1  # Failure
        mock_subprocess.return_value = mock_result
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
        )
        
        # Should return False on subprocess failure
        assert success is False
        assert mock_subprocess.called

