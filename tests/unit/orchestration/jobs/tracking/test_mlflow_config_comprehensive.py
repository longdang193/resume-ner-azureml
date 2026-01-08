"""Comprehensive unit tests for MLflow configuration."""

import yaml
from pathlib import Path
import pytest
import os

from orchestration.jobs.tracking.config.loader import (
    load_mlflow_config,
    get_naming_config,
    get_auto_increment_config,
    get_run_finder_config,
)
from orchestration.naming import build_mlflow_experiment_name


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


class TestAzureMLConfiguration:
    """Test Azure ML configuration (5.1)."""

    def test_azure_ml_enabled(self, config_dir):
        """Test azure_ml.enabled: true/false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
azure_ml:
  enabled: true
  workspace_name: "resume-ner-ws"
""")
        
        config = load_mlflow_config(config_dir)
        assert config["azure_ml"]["enabled"] is True
        assert config["azure_ml"]["workspace_name"] == "resume-ner-ws"
    
    def test_azure_ml_disabled(self, config_dir):
        """Test azure_ml.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
azure_ml:
  enabled: false
  workspace_name: "resume-ner-ws"
""")
        
        config = load_mlflow_config(config_dir)
        assert config["azure_ml"]["enabled"] is False


class TestExperimentNaming:
    """Test experiment naming (5.3)."""

    def test_build_mlflow_experiment_name_format(self):
        """Test build_mlflow_experiment_name function format."""
        experiment_name = build_mlflow_experiment_name(
            "resume_ner_baseline", "hpo", "distilbert"
        )
        
        assert experiment_name == "resume_ner_baseline-hpo-distilbert"
    
    def test_build_mlflow_experiment_name_all_stages(self):
        """Test all stages: hpo, benchmarking, final_training, conversion."""
        stages = ["hpo", "benchmarking", "final_training", "conversion"]
        
        for stage in stages:
            experiment_name = build_mlflow_experiment_name(
                "resume_ner_baseline", stage, "distilbert"
            )
            assert stage in experiment_name
            assert "distilbert" in experiment_name
    
    def test_build_mlflow_experiment_name_special_characters(self):
        """Test handles special characters in experiment_name and backbone."""
        experiment_name = build_mlflow_experiment_name(
            "resume_ner/baseline", "hpo", "distilbert-base"
        )
        # Should handle special characters appropriately
        assert "resume_ner" in experiment_name or "baseline" in experiment_name


class TestStageSpecificTrackingConfiguration:
    """Test stage-specific tracking configuration (5.4)."""

    def test_tracking_benchmark_enabled(self, config_dir):
        """Test tracking.benchmark.enabled: true/false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: true
    log_artifacts: true
""")
        
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["benchmark"]["enabled"] is True
        assert config["tracking"]["benchmark"]["log_artifacts"] is True
    
    def test_tracking_training_config(self, config_dir):
        """Test tracking.training config."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: true
""")
        
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["training"]["enabled"] is True
        assert config["tracking"]["training"]["log_checkpoint"] is True
        assert config["tracking"]["training"]["log_metrics_json"] is True
    
    def test_tracking_conversion_config(self, config_dir):
        """Test tracking.conversion config."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: true
""")
        
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["conversion"]["enabled"] is True
        assert config["tracking"]["conversion"]["log_onnx_model"] is True
        assert config["tracking"]["conversion"]["log_conversion_log"] is True


class TestNamingConfigurationDetails:
    """Test naming configuration details (5.5)."""

    def test_naming_project_name(self, config_dir):
        """Test naming.project_name."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  project_name: "resume-ner"
""")
        
        naming_config = get_naming_config(config_dir)
        assert naming_config["project_name"] == "resume-ner"
    
    def test_naming_project_name_default(self, config_dir):
        """Test naming.project_name default."""
        # Don't create mlflow.yaml
        naming_config = get_naming_config(config_dir)
        assert naming_config["project_name"] == "resume-ner"  # Default
    
    def test_naming_tags_config(self, config_dir):
        """Test naming.tags config."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  tags:
    max_length: 250
    sanitize: true
""")
        
        naming_config = get_naming_config(config_dir)
        assert naming_config["tags"]["max_length"] == 250
        assert naming_config["tags"]["sanitize"] is True
    
    def test_naming_run_name_config(self, config_dir):
        """Test naming.run_name config."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    max_length: 100
    shorten_fingerprints: true
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
        
        naming_config = get_naming_config(config_dir)
        assert naming_config["run_name"]["max_length"] == 100
        assert naming_config["run_name"]["shorten_fingerprints"] is True
        
        auto_inc_config = get_auto_increment_config(config_dir, "hpo")
        assert auto_inc_config["enabled"] is True
        assert auto_inc_config["processes"]["hpo"] is True
        assert auto_inc_config["format"] == "{base}.{version}"


class TestIndexCacheConfiguration:
    """Test index cache configuration (5.6)."""

    def test_index_enabled(self, config_dir):
        """Test index.enabled: true/false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: true
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        
        config = load_mlflow_config(config_dir)
        assert config["index"]["enabled"] is True
        assert config["index"]["max_entries"] == 1000
        assert config["index"]["file_name"] == "mlflow_index.json"
    
    def test_index_disabled(self, config_dir):
        """Test index.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: false
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        
        config = load_mlflow_config(config_dir)
        assert config["index"]["enabled"] is False


class TestRunFinderConfiguration:
    """Test run finder configuration (5.7)."""

    def test_run_finder_strict_mode_default_true(self, config_dir):
        """Test run_finder.strict_mode_default: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
run_finder:
  strict_mode_default: true
""")
        
        run_finder_config = get_run_finder_config(config_dir)
        assert run_finder_config["strict_mode_default"] is True
    
    def test_run_finder_strict_mode_default_false(self, config_dir):
        """Test run_finder.strict_mode_default: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
run_finder:
  strict_mode_default: false
""")
        
        run_finder_config = get_run_finder_config(config_dir)
        assert run_finder_config["strict_mode_default"] is False
    
    def test_run_finder_default(self, config_dir):
        """Test run_finder default (strict_mode_default: true)."""
        # Don't create mlflow.yaml
        run_finder_config = get_run_finder_config(config_dir)
        assert run_finder_config["strict_mode_default"] is True  # Default

