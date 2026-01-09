"""Unit tests for artifact_acquisition.yaml config loading."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from orchestration.config_loader import (
    load_experiment_config,
    load_all_configs,
    ExperimentConfig,
)
from shared.yaml_utils import load_yaml


class TestArtifactAcquisitionConfigLoading:
    """Test loading artifact_acquisition.yaml configuration."""

    def test_load_yaml_loads_artifact_acquisition_config(self, tmp_path):
        """Test that load_yaml can load artifact_acquisition.yaml."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "local"
  - "drive"
  - "mlflow"
local:
  match_strategy: "tags"
  require_exact_match: true
  validate: true
drive:
  enabled: true
  folder_path: "resume-ner-checkpoints"
  validate: true
mlflow:
  enabled: true
  validate: true
  download_timeout: 300
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        assert isinstance(config, dict)
        assert "priority" in config
        assert "local" in config
        assert "drive" in config
        assert "mlflow" in config

    def test_artifact_acquisition_config_structure(self, tmp_path):
        """Test that artifact_acquisition.yaml has correct structure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "local"
  - "drive"
  - "mlflow"
local:
  match_strategy: "tags"
  require_exact_match: true
  validate: true
drive:
  enabled: true
  folder_path: "resume-ner-checkpoints"
  validate: true
mlflow:
  enabled: true
  validate: true
  download_timeout: 300
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Verify priority section
        assert isinstance(config["priority"], list)
        assert len(config["priority"]) == 3
        assert "local" in config["priority"]
        assert "drive" in config["priority"]
        assert "mlflow" in config["priority"]
        
        # Verify local section
        assert isinstance(config["local"], dict)
        assert "match_strategy" in config["local"]
        assert "require_exact_match" in config["local"]
        assert "validate" in config["local"]
        
        # Verify drive section
        assert isinstance(config["drive"], dict)
        assert "enabled" in config["drive"]
        assert "folder_path" in config["drive"]
        assert "validate" in config["drive"]
        
        # Verify mlflow section
        assert isinstance(config["mlflow"], dict)
        assert "enabled" in config["mlflow"]
        assert "validate" in config["mlflow"]
        assert "download_timeout" in config["mlflow"]

    def test_artifact_acquisition_config_default_values(self, tmp_path):
        """Test that config has expected default values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "local"
  - "drive"
  - "mlflow"
local:
  match_strategy: "tags"
  require_exact_match: true
  validate: true
drive:
  enabled: true
  folder_path: "resume-ner-checkpoints"
  validate: true
mlflow:
  enabled: true
  validate: true
  download_timeout: 300
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Verify default values match actual config file
        assert config["priority"] == ["local", "drive", "mlflow"]
        assert config["local"]["match_strategy"] == "tags"
        assert config["local"]["require_exact_match"] is True
        assert config["local"]["validate"] is True
        assert config["drive"]["enabled"] is True
        assert config["drive"]["folder_path"] == "resume-ner-checkpoints"
        assert config["drive"]["validate"] is True
        assert config["mlflow"]["enabled"] is True
        assert config["mlflow"]["validate"] is True
        assert config["mlflow"]["download_timeout"] == 300

    def test_artifact_acquisition_config_custom_values(self, tmp_path):
        """Test loading config with custom values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "mlflow"
  - "local"
local:
  match_strategy: "metadata_run_id"
  require_exact_match: false
  validate: false
drive:
  enabled: false
  folder_path: "custom-checkpoints"
  validate: false
mlflow:
  enabled: true
  validate: false
  download_timeout: 600
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Verify custom values
        assert config["priority"] == ["mlflow", "local"]
        assert config["local"]["match_strategy"] == "metadata_run_id"
        assert config["local"]["require_exact_match"] is False
        assert config["local"]["validate"] is False
        assert config["drive"]["enabled"] is False
        assert config["drive"]["folder_path"] == "custom-checkpoints"
        assert config["drive"]["validate"] is False
        assert config["mlflow"]["enabled"] is True
        assert config["mlflow"]["validate"] is False
        assert config["mlflow"]["download_timeout"] == 600

    def test_artifact_acquisition_config_missing_sections(self, tmp_path):
        """Test that missing sections are handled gracefully."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "local"
local:
  validate: true
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Should load successfully even with missing sections
        assert "priority" in config
        assert "local" in config
        # drive and mlflow sections are missing, but that's OK for this test
        # (actual usage would need them, but config loader doesn't validate)

    def test_artifact_acquisition_config_types(self, tmp_path):
        """Test that config values have correct types."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        artifact_acquisition_yaml.write_text("""
priority:
  - "local"
  - "drive"
  - "mlflow"
local:
  match_strategy: "tags"
  require_exact_match: true
  validate: true
drive:
  enabled: true
  folder_path: "resume-ner-checkpoints"
  validate: true
mlflow:
  enabled: true
  validate: true
  download_timeout: 300
""")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Verify types
        assert isinstance(config["priority"], list)
        assert all(isinstance(item, str) for item in config["priority"])
        assert isinstance(config["local"]["match_strategy"], str)
        assert isinstance(config["local"]["require_exact_match"], bool)
        assert isinstance(config["local"]["validate"], bool)
        assert isinstance(config["drive"]["enabled"], bool)
        assert isinstance(config["drive"]["folder_path"], str)
        assert isinstance(config["drive"]["validate"], bool)
        assert isinstance(config["mlflow"]["enabled"], bool)
        assert isinstance(config["mlflow"]["validate"], bool)
        assert isinstance(config["mlflow"]["download_timeout"], int)

    def test_load_actual_artifact_acquisition_yaml(self):
        """Test loading the actual artifact_acquisition.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        artifact_acquisition_yaml = config_dir / "artifact_acquisition.yaml"
        
        if not artifact_acquisition_yaml.exists():
            pytest.skip("artifact_acquisition.yaml not found in config directory")
        
        config = load_yaml(artifact_acquisition_yaml)
        
        # Verify it loads successfully
        assert isinstance(config, dict)
        assert "priority" in config
        assert "local" in config
        assert "drive" in config
        assert "mlflow" in config
        
        # Verify structure matches expected
        assert isinstance(config["priority"], list)
        assert isinstance(config["local"], dict)
        assert isinstance(config["drive"], dict)
        assert isinstance(config["mlflow"], dict)

