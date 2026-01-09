"""Unit tests for benchmark.yaml config loading."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from config.loader import (
    load_experiment_config,
    load_all_configs,
    ExperimentConfig,
)


class TestBenchmarkConfigLoading:
    """Test loading benchmark.yaml configuration."""

    def test_load_experiment_config_includes_benchmark(self, tmp_path):
        """Test that load_experiment_config includes benchmark_config path."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        experiment_dir = config_root / "experiment"
        experiment_dir.mkdir()
        
        experiment_yaml = experiment_dir / "test.yaml"
        experiment_yaml.write_text("""
data_config: data/data.yaml
model_config: model/model.yaml
train_config: train/train.yaml
hpo_config: hpo/smoke.yaml
env_config: env/env.yaml
benchmark_config: benchmark.yaml
""")
        
        exp_cfg = load_experiment_config(config_root, "test")
        
        # Should resolve benchmark_config path
        assert exp_cfg.benchmark_config == config_root / "benchmark.yaml"

    def test_load_experiment_config_defaults_to_benchmark_yaml(self, tmp_path):
        """Test that benchmark_config defaults to benchmark.yaml if not specified."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        experiment_dir = config_root / "experiment"
        experiment_dir.mkdir()
        
        experiment_yaml = experiment_dir / "test.yaml"
        experiment_yaml.write_text("""
data_config: data/data.yaml
model_config: model/model.yaml
train_config: train/train.yaml
hpo_config: hpo/smoke.yaml
env_config: env/env.yaml
""")
        
        exp_cfg = load_experiment_config(config_root, "test")
        
        # Should default to benchmark.yaml
        assert exp_cfg.benchmark_config == config_root / "benchmark.yaml"

    def test_load_all_configs_loads_benchmark_if_exists(self, tmp_path):
        """Test that load_all_configs loads benchmark.yaml if file exists."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        # Create benchmark.yaml
        benchmark_yaml = config_root / "benchmark.yaml"
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
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data.yaml",
            model_config=config_root / "model.yaml",
            train_config=config_root / "train.yaml",
            hpo_config=config_root / "hpo.yaml",
            env_config=config_root / "env.yaml",
            benchmark_config=benchmark_yaml,
            stages={},
            naming={},
        )
        
        # Create other config files (required for load_all_configs)
        (config_root / "data.yaml").write_text("version: 1.0")
        (config_root / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env.yaml").write_text("name: test")
        
        configs = load_all_configs(exp_cfg)
        
        # Should include benchmark config
        assert "benchmark" in configs
        assert "benchmarking" in configs["benchmark"]
        assert "output" in configs["benchmark"]
        assert configs["benchmark"]["benchmarking"]["batch_sizes"] == [1, 8, 16]
        assert configs["benchmark"]["benchmarking"]["iterations"] == 100
        assert configs["benchmark"]["benchmarking"]["warmup_iterations"] == 10
        assert configs["benchmark"]["benchmarking"]["max_length"] == 512
        assert configs["benchmark"]["benchmarking"]["device"] is None
        assert configs["benchmark"]["benchmarking"]["test_data"] is None
        assert configs["benchmark"]["output"]["filename"] == "benchmark.json"

    def test_load_all_configs_skips_benchmark_if_not_exists(self, tmp_path):
        """Test that load_all_configs skips benchmark.yaml if file doesn't exist."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data.yaml",
            model_config=config_root / "model.yaml",
            train_config=config_root / "train.yaml",
            hpo_config=config_root / "hpo.yaml",
            env_config=config_root / "env.yaml",
            benchmark_config=config_root / "benchmark.yaml",  # File doesn't exist
            stages={},
            naming={},
        )
        
        # Create other config files
        (config_root / "data.yaml").write_text("version: 1.0")
        (config_root / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env.yaml").write_text("name: test")
        
        configs = load_all_configs(exp_cfg)
        
        # Should not include benchmark config
        assert "benchmark" not in configs

    def test_load_benchmark_config_structure(self, tmp_path):
        """Test that loaded benchmark config has correct structure."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        benchmark_yaml = config_root / "benchmark.yaml"
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
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data.yaml",
            model_config=config_root / "model.yaml",
            train_config=config_root / "train.yaml",
            hpo_config=config_root / "hpo.yaml",
            env_config=config_root / "env.yaml",
            benchmark_config=benchmark_yaml,
            stages={},
            naming={},
        )
        
        # Create other config files
        (config_root / "data.yaml").write_text("version: 1.0")
        (config_root / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env.yaml").write_text("name: test")
        
        configs = load_all_configs(exp_cfg)
        
        benchmark = configs["benchmark"]
        
        # Verify structure
        assert isinstance(benchmark, dict)
        assert "benchmarking" in benchmark
        assert "output" in benchmark
        
        # Verify benchmarking section
        benchmarking = benchmark["benchmarking"]
        assert isinstance(benchmarking, dict)
        assert "batch_sizes" in benchmarking
        assert "iterations" in benchmarking
        assert "warmup_iterations" in benchmarking
        assert "max_length" in benchmarking
        assert "device" in benchmarking
        assert "test_data" in benchmarking
        
        # Verify output section
        output = benchmark["output"]
        assert isinstance(output, dict)
        assert "filename" in output

    def test_load_benchmark_config_with_custom_values(self, tmp_path):
        """Test loading benchmark config with custom values."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        benchmark_yaml = config_root / "benchmark.yaml"
        benchmark_yaml.write_text("""
benchmarking:
  batch_sizes: [2, 4, 32]
  iterations: 200
  warmup_iterations: 20
  max_length: 256
  device: "cuda"
  test_data: "/custom/path/test.json"
output:
  filename: "custom_benchmark.json"
""")
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data.yaml",
            model_config=config_root / "model.yaml",
            train_config=config_root / "train.yaml",
            hpo_config=config_root / "hpo.yaml",
            env_config=config_root / "env.yaml",
            benchmark_config=benchmark_yaml,
            stages={},
            naming={},
        )
        
        # Create other config files
        (config_root / "data.yaml").write_text("version: 1.0")
        (config_root / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env.yaml").write_text("name: test")
        
        configs = load_all_configs(exp_cfg)
        
        benchmark = configs["benchmark"]
        
        # Verify custom values are loaded correctly
        assert benchmark["benchmarking"]["batch_sizes"] == [2, 4, 32]
        assert benchmark["benchmarking"]["iterations"] == 200
        assert benchmark["benchmarking"]["warmup_iterations"] == 20
        assert benchmark["benchmarking"]["max_length"] == 256
        assert benchmark["benchmarking"]["device"] == "cuda"
        assert benchmark["benchmarking"]["test_data"] == "/custom/path/test.json"
        assert benchmark["output"]["filename"] == "custom_benchmark.json"

