"""Unit tests for config loading and hashing."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from infrastructure.config.loader import (
    compute_config_hash,
    compute_config_hashes,
    CONFIG_HASH_LENGTH,
    ExperimentConfig,
    load_experiment_config,
    load_all_configs,
    create_config_metadata,
    snapshot_configs,
    validate_config_immutability,
)


class TestConfigHashComputation:
    """Test config hash computation functions."""

    def test_compute_config_hash_length(self):
        """Test that hash length = CONFIG_HASH_LENGTH (16)."""
        config = {"key": "value", "number": 42}
        hash_value = compute_config_hash(config)
        
        assert len(hash_value) == CONFIG_HASH_LENGTH
        assert len(hash_value) == 16

    def test_compute_config_hash_deterministic(self):
        """Test that same config produces same hash."""
        config = {"key": "value", "number": 42}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        assert hash1 == hash2

    def test_compute_config_hash_different_configs(self):
        """Test that different configs produce different hashes."""
        config1 = {"key": "value", "number": 42}
        config2 = {"key": "different", "number": 42}
        config3 = {"key": "value", "number": 43}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)
        
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_compute_config_hash_no_randomness(self):
        """Test that hash is deterministic (no randomness)."""
        config = {"key": "value"}
        
        # Run multiple times - should always produce same hash
        hashes = [compute_config_hash(config) for _ in range(10)]
        
        assert len(set(hashes)) == 1  # All hashes should be identical

    def test_compute_config_hash_order_independent(self):
        """Test that hash is independent of key order (sort_keys=True)."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}
        config3 = {"b": 2, "c": 3, "a": 1}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)
        
        # All should produce same hash due to sort_keys=True
        assert hash1 == hash2 == hash3

    def test_compute_config_hashes_all_domains(self):
        """Test computing hashes for all domain configs."""
        configs = {
            "data": {"version": "1.0", "path": "/data"},
            "model": {"backbone": "distilbert"},
            "train": {"epochs": 10},
            "hpo": {"max_trials": 5},
            "env": {"name": "test"},
        }
        
        hashes = compute_config_hashes(configs)
        
        assert len(hashes) == len(configs)
        assert "data" in hashes
        assert "model" in hashes
        assert "train" in hashes
        assert "hpo" in hashes
        assert "env" in hashes
        
        # All hashes should be correct length
        for hash_value in hashes.values():
            assert len(hash_value) == CONFIG_HASH_LENGTH

    def test_compute_config_hashes_deterministic(self):
        """Test that compute_config_hashes is deterministic."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
        }
        
        hashes1 = compute_config_hashes(configs)
        hashes2 = compute_config_hashes(configs)
        
        assert hashes1 == hashes2


class TestConfigLoading:
    """Test config loading functions."""

    def test_load_experiment_config(self, tmp_path):
        """Test loading experiment config from YAML."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        experiment_dir = config_root / "experiment"
        experiment_dir.mkdir()
        
        experiment_yaml = experiment_dir / "test_experiment.yaml"
        experiment_yaml.write_text("""
experiment_name: test_experiment
data_config: data/data.yaml
model_config: model/model.yaml
train_config: train/train.yaml
hpo_config: hpo/smoke.yaml
env_config: env/env.yaml
benchmark_config: benchmark/benchmark.yaml
stages:
  hpo: true
naming:
  project_name: resume-ner
""")
        
        exp_cfg = load_experiment_config(config_root, "test_experiment")
        
        assert exp_cfg.name == "test_experiment"
        assert exp_cfg.data_config == config_root / "data" / "data.yaml"
        assert exp_cfg.model_config == config_root / "model" / "model.yaml"
        assert exp_cfg.train_config == config_root / "train" / "train.yaml"
        assert exp_cfg.hpo_config == config_root / "hpo" / "smoke.yaml"
        assert exp_cfg.env_config == config_root / "env" / "env.yaml"
        assert exp_cfg.benchmark_config == config_root / "benchmark" / "benchmark.yaml"
        assert exp_cfg.stages == {"hpo": True}
        assert exp_cfg.naming == {"project_name": "resume-ner"}

    def test_load_experiment_config_defaults(self, tmp_path):
        """Test loading experiment config with default benchmark config."""
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
        
        # Should use experiment name if experiment_name not in YAML
        assert exp_cfg.name == "test"
        # Should default to benchmark.yaml if not specified
        assert exp_cfg.benchmark_config == config_root / "benchmark.yaml"
        # Should default to empty dicts
        assert exp_cfg.stages == {}
        assert exp_cfg.naming == {}

    def test_load_all_configs(self, tmp_path):
        """Test loading all domain configs."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        # Create config files
        (config_root / "data").mkdir()
        (config_root / "model").mkdir()
        (config_root / "train").mkdir()
        (config_root / "hpo").mkdir()
        (config_root / "env").mkdir()
        
        (config_root / "data" / "data.yaml").write_text("version: 1.0")
        (config_root / "model" / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train" / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo" / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env" / "env.yaml").write_text("name: test")
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data" / "data.yaml",
            model_config=config_root / "model" / "model.yaml",
            train_config=config_root / "train" / "train.yaml",
            hpo_config=config_root / "hpo" / "hpo.yaml",
            env_config=config_root / "env" / "env.yaml",
            benchmark_config=config_root / "benchmark.yaml",
            stages={},
            naming={},
        )
        
        configs = load_all_configs(exp_cfg)
        
        assert "data" in configs
        assert configs["data"]["version"] == 1.0
        assert "model" in configs
        assert configs["model"]["backbone"] == "distilbert"
        assert "train" in configs
        assert configs["train"]["epochs"] == 10
        assert "hpo" in configs
        assert configs["hpo"]["max_trials"] == 5
        assert "env" in configs
        assert configs["env"]["name"] == "test"
        # benchmark not loaded if file doesn't exist
        assert "benchmark" not in configs

    def test_load_all_configs_with_benchmark(self, tmp_path):
        """Test loading all configs including benchmark if file exists."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        # Create config files including benchmark
        (config_root / "data").mkdir()
        (config_root / "model").mkdir()
        (config_root / "train").mkdir()
        (config_root / "hpo").mkdir()
        (config_root / "env").mkdir()
        
        (config_root / "data" / "data.yaml").write_text("version: 1.0")
        (config_root / "model" / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train" / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo" / "hpo.yaml").write_text("max_trials: 5")
        (config_root / "env" / "env.yaml").write_text("name: test")
        (config_root / "benchmark.yaml").write_text("enabled: true")
        
        exp_cfg = ExperimentConfig(
            name="test",
            data_config=config_root / "data" / "data.yaml",
            model_config=config_root / "model" / "model.yaml",
            train_config=config_root / "train" / "train.yaml",
            hpo_config=config_root / "hpo" / "hpo.yaml",
            env_config=config_root / "env" / "env.yaml",
            benchmark_config=config_root / "benchmark.yaml",
            stages={},
            naming={},
        )
        
        configs = load_all_configs(exp_cfg)
        
        assert "benchmark" in configs
        assert configs["benchmark"]["enabled"] is True


class TestConfigMetadata:
    """Test config metadata creation."""

    def test_create_config_metadata(self):
        """Test creating config metadata dictionary."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
            "train": {"epochs": 10},
            "hpo": {"max_trials": 5},
            "env": {"name": "test"},
        }
        
        config_hashes = compute_config_hashes(configs)
        metadata = create_config_metadata(configs, config_hashes)
        
        assert "data_config_hash" in metadata
        assert metadata["data_config_hash"] == config_hashes["data"]
        assert "model_config_hash" in metadata
        assert metadata["model_config_hash"] == config_hashes["model"]
        assert "train_config_hash" in metadata
        assert metadata["train_config_hash"] == config_hashes["train"]
        assert "hpo_config_hash" in metadata
        assert metadata["hpo_config_hash"] == config_hashes["hpo"]
        assert "env_config_hash" in metadata
        assert metadata["env_config_hash"] == config_hashes["env"]
        assert "data_version" in metadata
        assert metadata["data_version"] == "1.0"
        assert "model_backbone" in metadata
        assert metadata["model_backbone"] == "distilbert"


class TestConfigImmutability:
    """Test config immutability validation."""

    def test_snapshot_configs(self):
        """Test creating config snapshots."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
        }
        
        snapshots = snapshot_configs(configs)
        
        assert "data" in snapshots
        assert "model" in snapshots
        # Snapshots should be JSON strings
        assert isinstance(snapshots["data"], str)
        assert isinstance(snapshots["model"], str)
        # Should be valid JSON
        assert json.loads(snapshots["data"]) == configs["data"]
        assert json.loads(snapshots["model"]) == configs["model"]

    def test_validate_config_immutability_unchanged(self):
        """Test validation passes when configs are unchanged."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
        }
        
        snapshots = snapshot_configs(configs)
        
        # Should not raise
        validate_config_immutability(configs, snapshots)

    def test_validate_config_immutability_changed(self):
        """Test validation raises when configs are mutated."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
        }
        
        snapshots = snapshot_configs(configs)
        
        # Mutate config
        configs["data"]["version"] = "2.0"
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Config 'data' was mutated at runtime"):
            validate_config_immutability(configs, snapshots)

    def test_validate_config_immutability_multiple_changes(self):
        """Test validation detects multiple config mutations."""
        configs = {
            "data": {"version": "1.0"},
            "model": {"backbone": "distilbert"},
            "train": {"epochs": 10},
        }
        
        snapshots = snapshot_configs(configs)
        
        # Mutate multiple configs
        configs["data"]["version"] = "2.0"
        configs["model"]["backbone"] = "roberta"
        
        # Should raise for first mutation detected
        with pytest.raises(ValueError, match="was mutated at runtime"):
            validate_config_immutability(configs, snapshots)

