"""Tests for configuration loading utilities."""

import pytest
from pathlib import Path
from orchestration.config_loader import (
    load_experiment_config,
    load_all_configs,
    compute_config_hash,
    compute_config_hashes,
    create_config_metadata,
    snapshot_configs,
    validate_config_immutability,
    ExperimentConfig,
)


class TestLoadExperimentConfig:
    """Tests for load_experiment_config function."""

    def test_valid_config(self, mock_configs):
        """Test loading a valid experiment configuration."""
        config_root = mock_configs["root"]
        exp_cfg = load_experiment_config(config_root, "test_experiment")

        assert isinstance(exp_cfg, ExperimentConfig)
        assert exp_cfg.name == "test_experiment"
        assert exp_cfg.data_config.exists()
        assert exp_cfg.model_config.exists()
        assert exp_cfg.train_config.exists()
        assert exp_cfg.hpo_config.exists()
        assert exp_cfg.env_config.exists()

    def test_missing_experiment_file(self, mock_configs):
        """Test that missing experiment file raises FileNotFoundError."""
        config_root = mock_configs["root"]

        with pytest.raises(FileNotFoundError):
            load_experiment_config(config_root, "nonexistent_experiment")

    def test_invalid_yaml_structure(self, temp_dir):
        """Test that invalid YAML structure raises an error."""
        config_root = temp_dir / "config"
        config_root.mkdir()
        (config_root / "experiment").mkdir()

        invalid_file = config_root / "experiment" / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            load_experiment_config(config_root, "invalid")

    def test_experiment_name_override(self, mock_configs):
        """Test that experiment_name in YAML overrides the filename."""
        config_root = mock_configs["root"]

        # Modify the experiment config to have a different name
        import yaml
        exp_file = mock_configs["experiment"]
        with open(exp_file, "r") as f:
            exp_data = yaml.safe_load(f)
        exp_data["experiment_name"] = "custom_name"
        with open(exp_file, "w") as f:
            yaml.dump(exp_data, f)

        exp_cfg = load_experiment_config(config_root, "test_experiment")
        assert exp_cfg.name == "custom_name"

    def test_default_experiment_name(self, mock_configs):
        """Test that filename is used when experiment_name is not in YAML."""
        config_root = mock_configs["root"]

        # Remove experiment_name from config
        import yaml
        exp_file = mock_configs["experiment"]
        with open(exp_file, "r") as f:
            exp_data = yaml.safe_load(f)
        exp_data.pop("experiment_name", None)
        with open(exp_file, "w") as f:
            yaml.dump(exp_data, f)

        exp_cfg = load_experiment_config(config_root, "test_experiment")
        assert exp_cfg.name == "test_experiment"


class TestLoadAllConfigs:
    """Tests for load_all_configs function."""

    def test_all_configs_loaded(self, mock_configs):
        """Test that all configs are loaded correctly."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)

        assert "data" in configs
        assert "model" in configs
        assert "train" in configs
        assert "hpo" in configs
        assert "env" in configs

    def test_config_structure(self, mock_configs):
        """Test that loaded configs have expected structure."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)

        assert "version" in configs["data"] or "schema" in configs["data"]
        assert "backbone" in configs["model"]
        assert "training" in configs["train"]
        assert "search_space" in configs["hpo"]


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_deterministic_hashing(self):
        """Test that same config produces same hash."""
        config1 = {"key1": "value1", "key2": 123}
        config2 = {"key1": "value1", "key2": 123}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        config1 = {"key1": "value1"}
        config2 = {"key1": "value2"}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2

    def test_hash_length(self):
        """Test that hash has expected length."""
        config = {"key": "value"}
        hash_str = compute_config_hash(config)

        assert len(hash_str) == 16  # CONFIG_HASH_LENGTH

    def test_order_independent(self):
        """Test that hash is independent of key order."""
        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key2": "value2", "key1": "value1"}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2  # Should be same due to sort_keys=True

    def test_nested_configs(self):
        """Test hashing with nested dictionaries."""
        config1 = {"outer": {"inner": "value"}}
        config2 = {"outer": {"inner": "value"}}
        config3 = {"outer": {"inner": "different"}}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)

        assert hash1 == hash2
        assert hash1 != hash3


class TestComputeConfigHashes:
    """Tests for compute_config_hashes function."""

    def test_all_configs_hashed(self, mock_configs):
        """Test that hashes are computed for all configs."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        hashes = compute_config_hashes(configs)

        assert "data" in hashes
        assert "model" in hashes
        assert "train" in hashes
        assert "hpo" in hashes
        assert "env" in hashes

    def test_hash_values_are_strings(self, mock_configs):
        """Test that all hash values are strings."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        hashes = compute_config_hashes(configs)

        for hash_value in hashes.values():
            assert isinstance(hash_value, str)
            assert len(hash_value) == 16


class TestCreateConfigMetadata:
    """Tests for create_config_metadata function."""

    def test_metadata_structure(self, mock_configs):
        """Test that metadata has expected structure."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        hashes = compute_config_hashes(configs)
        metadata = create_config_metadata(configs, hashes)

        assert "data_config_hash" in metadata
        assert "model_config_hash" in metadata
        assert "train_config_hash" in metadata
        assert "hpo_config_hash" in metadata
        assert "env_config_hash" in metadata
        assert "data_version" in metadata
        assert "model_backbone" in metadata

    def test_metadata_values_are_strings(self, mock_configs):
        """Test that all metadata values are strings."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        hashes = compute_config_hashes(configs)
        metadata = create_config_metadata(configs, hashes)

        for value in metadata.values():
            assert isinstance(value, str)


class TestSnapshotAndValidateConfigs:
    """Tests for snapshot_configs and validate_config_immutability functions."""

    def test_snapshot_configs(self, mock_configs):
        """Test that configs are snapshotted correctly."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        snapshots = snapshot_configs(configs)

        assert "data" in snapshots
        assert "model" in snapshots
        assert "train" in snapshots
        assert "hpo" in snapshots
        assert "env" in snapshots

        # All snapshots should be JSON strings
        for snapshot in snapshots.values():
            assert isinstance(snapshot, str)
            import json
            json.loads(snapshot)  # Should be valid JSON

    def test_validate_unchanged_configs(self, mock_configs):
        """Test that unchanged configs pass validation."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        snapshots = snapshot_configs(configs)

        # Should not raise
        validate_config_immutability(configs, snapshots)

    def test_validate_mutated_configs(self, mock_configs):
        """Test that mutated configs raise ValueError."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        snapshots = snapshot_configs(configs)

        # Mutate a config
        configs["data"]["new_key"] = "new_value"

        with pytest.raises(ValueError, match="was mutated"):
            validate_config_immutability(configs, snapshots)

    def test_validate_multiple_mutations(self, mock_configs):
        """Test that multiple config mutations are detected."""
        exp_cfg = load_experiment_config(
            mock_configs["root"], "test_experiment")
        configs = load_all_configs(exp_cfg)
        snapshots = snapshot_configs(configs)

        # Mutate multiple configs
        configs["data"]["key1"] = "value1"
        configs["model"]["key2"] = "value2"

        with pytest.raises(ValueError, match="was mutated"):
            validate_config_immutability(configs, snapshots)
