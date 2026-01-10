"""Component tests for benchmark utils."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from benchmarking.utils import run_benchmarking


class TestBenchmarkUtilsConfigUsage:
    """Test that run_benchmarking() uses config options correctly."""

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_batch_sizes(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that batch_sizes from config are passed to subprocess."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Extract batch_sizes from config
        batch_sizes = sample_benchmark_config["benchmarking"]["batch_sizes"]
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=batch_sizes,
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
        )
        
        # Verify subprocess was called with correct batch_sizes
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        # Find --batch-sizes in args and verify values
        batch_sizes_idx = call_args.index("--batch-sizes")
        batch_size_values = call_args[batch_sizes_idx + 1:]
        # Values continue until next flag
        next_flag_idx = len(batch_size_values)
        for i, arg in enumerate(batch_size_values):
            if arg.startswith("--"):
                next_flag_idx = i
                break
        
        actual_batch_sizes = [int(arg) for arg in batch_size_values[:next_flag_idx]]
        assert actual_batch_sizes == batch_sizes

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_iterations(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that iterations from config is passed to subprocess."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Extract iterations from config
        iterations = sample_benchmark_config["benchmarking"]["iterations"]
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=iterations,
            warmup_iterations=10,
            max_length=512,
            device=None,
            project_root=project_root,
        )
        
        # Verify subprocess was called with correct iterations
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        iterations_idx = call_args.index("--iterations")
        iterations_value = int(call_args[iterations_idx + 1])
        assert iterations_value == iterations

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_warmup_iterations(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that warmup_iterations from config is passed to subprocess."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Extract warmup_iterations from config
        warmup = sample_benchmark_config["benchmarking"]["warmup_iterations"]
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=warmup,
            max_length=512,
            device=None,
            project_root=project_root,
        )
        
        # Verify subprocess was called with correct warmup
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        warmup_idx = call_args.index("--warmup")
        warmup_value = int(call_args[warmup_idx + 1])
        assert warmup_value == warmup

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_max_length(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that max_length from config is passed to subprocess."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Extract max_length from config
        max_length = sample_benchmark_config["benchmarking"]["max_length"]
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=10,
            max_length=max_length,
            device=None,
            project_root=project_root,
        )
        
        # Verify subprocess was called with correct max_length
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        max_length_idx = call_args.index("--max-length")
        max_length_value = int(call_args[max_length_idx + 1])
        assert max_length_value == max_length

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_device_when_provided(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        custom_benchmark_config,
    ):
        """Test that device from config is passed to subprocess when not null."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Extract device from config
        device = custom_benchmark_config["benchmarking"]["device"]
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=device,
            project_root=project_root,
        )
        
        # Verify subprocess was called with --device flag
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        assert "--device" in call_args
        device_idx = call_args.index("--device")
        device_value = call_args[device_idx + 1]
        assert device_value == device

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_skips_device_when_null(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that --device flag is not included when device is None."""
        output_path = tmp_path / "benchmark.json"
        project_root = tmp_path
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking with device=None
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=[1, 8],
            iterations=100,
            warmup_iterations=10,
            max_length=512,
            device=None,  # null in config
            project_root=project_root,
        )
        
        # Verify subprocess was called without --device flag
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        assert "--device" not in call_args

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_uses_output_path(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        sample_benchmark_config,
    ):
        """Test that output_path uses filename from config."""
        project_root = tmp_path
        
        # Extract filename from config
        filename = sample_benchmark_config["output"]["filename"]
        output_path = tmp_path / filename
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
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
        
        # Verify subprocess was called with correct output path
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        output_idx = call_args.index("--output")
        output_value = call_args[output_idx + 1]
        assert Path(output_value).name == filename

    @patch("benchmarking.utils.subprocess.run")
    def test_run_benchmarking_all_config_options_together(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
        custom_benchmark_config,
    ):
        """Test that all config options are passed correctly to subprocess."""
        project_root = tmp_path
        
        # Extract all options from config
        batch_sizes = custom_benchmark_config["benchmarking"]["batch_sizes"]
        iterations = custom_benchmark_config["benchmarking"]["iterations"]
        warmup = custom_benchmark_config["benchmarking"]["warmup_iterations"]
        max_length = custom_benchmark_config["benchmarking"]["max_length"]
        device = custom_benchmark_config["benchmarking"]["device"]
        filename = custom_benchmark_config["output"]["filename"]
        output_path = tmp_path / filename
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "benchmarks" / "benchmark_inference.py"
        benchmark_script.parent.mkdir(parents=True)
        benchmark_script.write_text("# mock script")
        
        # Call run_benchmarking with all config options
        success = run_benchmarking(
            checkpoint_dir=mock_checkpoint_dir,
            test_data_path=mock_test_data_file,
            output_path=output_path,
            batch_sizes=batch_sizes,
            iterations=iterations,
            warmup_iterations=warmup,
            max_length=max_length,
            device=device,
            project_root=project_root,
        )
        
        # Verify all options were passed correctly
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        
        # Verify batch_sizes
        batch_sizes_idx = call_args.index("--batch-sizes")
        batch_size_values = []
        i = batch_sizes_idx + 1
        while i < len(call_args) and not call_args[i].startswith("--"):
            batch_size_values.append(int(call_args[i]))
            i += 1
        assert batch_size_values == batch_sizes
        
        # Verify iterations
        iterations_idx = call_args.index("--iterations")
        assert int(call_args[iterations_idx + 1]) == iterations
        
        # Verify warmup
        warmup_idx = call_args.index("--warmup")
        assert int(call_args[warmup_idx + 1]) == warmup
        
        # Verify max_length
        max_length_idx = call_args.index("--max-length")
        assert int(call_args[max_length_idx + 1]) == max_length
        
        # Verify device
        assert "--device" in call_args
        device_idx = call_args.index("--device")
        assert call_args[device_idx + 1] == device
        
        # Verify output
        output_idx = call_args.index("--output")
        assert Path(call_args[output_idx + 1]).name == filename

