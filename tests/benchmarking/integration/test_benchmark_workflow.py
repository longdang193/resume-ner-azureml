"""Integration tests for end-to-end benchmarking workflow."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from infrastructure.config.loader import load_experiment_config, load_all_configs
from benchmarking.orchestrator import benchmark_best_trials
from benchmarking.utils import run_benchmarking


class TestBenchmarkWorkflow:
    """Test end-to-end benchmarking workflow with config loading."""

    @patch("benchmarking.orchestrator.run_benchmarking")
    @patch("benchmarking.orchestrator.create_naming_context")
    @patch("benchmarking.orchestrator.build_output_path")
    @patch("benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("benchmarking.orchestrator.validate_path_before_mkdir")
    def test_workflow_loads_config_and_uses_all_options(
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
    ):
        """Test full workflow: load config, extract options, run benchmarking."""
        # Setup config directory structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create benchmark.yaml
        benchmark_yaml = config_dir / "benchmark.yaml"
        benchmark_yaml.write_text("""
benchmarking:
  batch_sizes: [1, 8, 16]
  iterations: 100
  warmup_iterations: 10
  max_length: 512
  device: null
  test_data: null
output:
  filename: "benchmark.json"
""")
        
        # Create other required config files
        (config_dir / "data.yaml").write_text("version: 1.0")
        (config_dir / "model.yaml").write_text("backbone: distilbert")
        (config_dir / "train.yaml").write_text("epochs: 10")
        (config_dir / "hpo.yaml").write_text("max_trials: 5")
        (config_dir / "env.yaml").write_text("name: test")
        (config_dir / "experiment").mkdir()
        (config_dir / "experiment" / "test.yaml").write_text("""
data_config: data.yaml
model_config: model.yaml
train_config: train.yaml
hpo_config: hpo.yaml
env_config: env.yaml
benchmark_config: benchmark.yaml
""")
        
        # Load configs
        exp_cfg = load_experiment_config(config_dir, "test")
        configs = load_all_configs(exp_cfg)
        
        # Verify benchmark config was loaded
        assert "benchmark" in configs
        benchmark_config = configs["benchmark"]
        
        # Extract options from config
        batch_sizes = benchmark_config["benchmarking"]["batch_sizes"]
        iterations = benchmark_config["benchmarking"]["iterations"]
        warmup = benchmark_config["benchmarking"]["warmup_iterations"]
        max_length = benchmark_config["benchmarking"]["max_length"]
        device = benchmark_config["benchmarking"]["device"]
        filename = benchmark_config["output"]["filename"]
        
        # Setup mocks for benchmark_best_trials
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        output_dir = root_dir / "benchmarking" / "test"
        
        # Ensure checkpoint exists
        checkpoint_dir = Path(mock_best_trials["distilbert"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Create benchmark script
        benchmark_script = root_dir / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = output_dir
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Run benchmarking with config options
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=benchmark_config,
            benchmark_batch_sizes=batch_sizes,
            benchmark_iterations=iterations,
            benchmark_warmup=warmup,
            benchmark_max_length=max_length,
            benchmark_device=device,
        )
        
        # Verify run_benchmarking was called with all config options
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        assert call_args.kwargs["batch_sizes"] == batch_sizes
        assert call_args.kwargs["iterations"] == iterations
        assert call_args.kwargs["warmup_iterations"] == warmup
        assert call_args.kwargs["max_length"] == max_length
        assert call_args.kwargs["device"] == device
        assert call_args.kwargs["output_path"].name == filename

    @patch("benchmarking.utils.subprocess.run")
    def test_workflow_custom_config_values(
        self,
        mock_subprocess,
        tmp_path,
        mock_checkpoint_dir,
        mock_test_data_file,
    ):
        """Test workflow with custom config values."""
        # Setup config with custom values
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        benchmark_yaml = config_dir / "benchmark.yaml"
        benchmark_yaml.write_text("""
benchmarking:
  batch_sizes: [2, 4, 32]
  iterations: 200
  warmup_iterations: 20
  max_length: 256
  device: "cuda"
  test_data: null
output:
  filename: "custom_benchmark.json"
""")
        
        # Load config
        from common.shared.yaml_utils import load_yaml
        benchmark_config = load_yaml(benchmark_yaml)
        
        # Extract options
        batch_sizes = benchmark_config["benchmarking"]["batch_sizes"]
        iterations = benchmark_config["benchmarking"]["iterations"]
        warmup = benchmark_config["benchmarking"]["warmup_iterations"]
        max_length = benchmark_config["benchmarking"]["max_length"]
        device = benchmark_config["benchmarking"]["device"]
        filename = benchmark_config["output"]["filename"]
        
        # Setup mocks
        project_root = tmp_path
        output_path = tmp_path / filename
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create mock benchmark script
        benchmark_script = project_root / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        # Run benchmarking with custom config
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
        
        # Verify subprocess received custom values
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
        
        # Verify output filename
        output_idx = call_args.index("--output")
        assert Path(call_args[output_idx + 1]).name == filename

    @patch("benchmarking.orchestrator.run_benchmarking")
    @patch("benchmarking.orchestrator.create_naming_context")
    @patch("benchmarking.orchestrator.build_output_path")
    @patch("benchmarking.orchestrator.resolve_output_path_for_colab")
    @patch("benchmarking.orchestrator.validate_path_before_mkdir")
    def test_workflow_defaults_when_config_missing(
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
    ):
        """Test workflow uses defaults when benchmark.yaml is missing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create experiment config without benchmark.yaml
        (config_dir / "experiment").mkdir()
        (config_dir / "experiment" / "test.yaml").write_text("""
data_config: data.yaml
model_config: model.yaml
train_config: train.yaml
hpo_config: hpo.yaml
env_config: env.yaml
""")
        
        # Create other required config files
        (config_dir / "data.yaml").write_text("version: 1.0")
        (config_dir / "model.yaml").write_text("backbone: distilbert")
        (config_dir / "train.yaml").write_text("epochs: 10")
        (config_dir / "hpo.yaml").write_text("max_trials: 5")
        (config_dir / "env.yaml").write_text("name: test")
        
        # Load configs (benchmark.yaml doesn't exist)
        exp_cfg = load_experiment_config(config_dir, "test")
        configs = load_all_configs(exp_cfg)
        
        # Benchmark config should not be loaded
        assert "benchmark" not in configs
        
        # Setup mocks
        root_dir = tmp_path / "outputs"
        root_dir.mkdir()
        output_dir = root_dir / "benchmarking" / "test"
        
        # Ensure checkpoint exists
        checkpoint_dir = Path(mock_best_trials["distilbert"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
        
        # Create benchmark script
        benchmark_script = root_dir / "src" / "benchmarking" / "cli.py"
        benchmark_script.parent.mkdir(parents=True, exist_ok=True)
        benchmark_script.write_text("# mock script")
        
        mock_validate_path.side_effect = lambda p, **kwargs: p
        mock_resolve_colab.side_effect = lambda p: p
        mock_build_path.return_value = output_dir
        mock_create_context.return_value = Mock()
        mock_run_benchmarking.return_value = True
        
        # Run benchmarking without config (uses function defaults)
        result = benchmark_best_trials(
            best_trials=mock_best_trials,
            test_data_path=mock_test_data_file,
            root_dir=root_dir,
            environment="local",
            data_config=mock_data_config,
            hpo_config=mock_hpo_config,
            benchmark_config=None,  # No config
            # Use function defaults
        )
        
        # Verify run_benchmarking was called with defaults
        assert mock_run_benchmarking.called
        call_args = mock_run_benchmarking.call_args
        # Defaults: batch_sizes=[1, 8, 16], iterations=100, warmup=10, max_length=512, device=None
        assert call_args.kwargs["batch_sizes"] == [1, 8, 16]
        assert call_args.kwargs["iterations"] == 100
        assert call_args.kwargs["warmup_iterations"] == 10
        assert call_args.kwargs["max_length"] == 512
        assert call_args.kwargs["device"] is None

