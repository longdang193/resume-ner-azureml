"""Explicit tests for all mlflow.yaml configuration options (lines 1-92).

This test file ensures every single config option in mlflow.yaml is explicitly tested.
"""

import yaml
from pathlib import Path
import pytest

from orchestration.jobs.tracking.config.loader import (
    load_mlflow_config,
    get_naming_config,
    get_auto_increment_config,
    get_run_finder_config,
    get_index_config,
)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


class TestAzureMLConfiguration:
    """Test azure_ml section (lines 7-13)."""

    def test_azure_ml_enabled(self, config_dir):
        """Test azure_ml.enabled: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
azure_ml:
  enabled: true
  workspace_name: "resume-ner-ws"
""")
        config = load_mlflow_config(config_dir)
        assert config["azure_ml"]["enabled"] is True

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

    def test_azure_ml_workspace_name(self, config_dir):
        """Test azure_ml.workspace_name."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
azure_ml:
  enabled: true
  workspace_name: "resume-ner-ws"
""")
        config = load_mlflow_config(config_dir)
        assert config["azure_ml"]["workspace_name"] == "resume-ner-ws"


class TestTrackingConfiguration:
    """Test tracking section (lines 32-48)."""

    def test_tracking_benchmark_enabled(self, config_dir):
        """Test tracking.benchmark.enabled: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: true
    log_artifacts: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["benchmark"]["enabled"] is True

    def test_tracking_benchmark_disabled(self, config_dir):
        """Test tracking.benchmark.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: false
    log_artifacts: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["benchmark"]["enabled"] is False

    def test_tracking_benchmark_log_artifacts(self, config_dir):
        """Test tracking.benchmark.log_artifacts: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  benchmark:
    enabled: true
    log_artifacts: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["benchmark"]["log_artifacts"] is True

    def test_tracking_training_enabled(self, config_dir):
        """Test tracking.training.enabled: true."""
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

    def test_tracking_training_disabled(self, config_dir):
        """Test tracking.training.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: false
    log_checkpoint: true
    log_metrics_json: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["training"]["enabled"] is False

    def test_tracking_training_log_checkpoint(self, config_dir):
        """Test tracking.training.log_checkpoint: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["training"]["log_checkpoint"] is True

    def test_tracking_training_log_metrics_json(self, config_dir):
        """Test tracking.training.log_metrics_json: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["training"]["log_metrics_json"] is True

    def test_tracking_conversion_enabled(self, config_dir):
        """Test tracking.conversion.enabled: true."""
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

    def test_tracking_conversion_disabled(self, config_dir):
        """Test tracking.conversion.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: false
    log_onnx_model: true
    log_conversion_log: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["conversion"]["enabled"] is False

    def test_tracking_conversion_log_onnx_model(self, config_dir):
        """Test tracking.conversion.log_onnx_model: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["conversion"]["log_onnx_model"] is True

    def test_tracking_conversion_log_conversion_log(self, config_dir):
        """Test tracking.conversion.log_conversion_log: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
tracking:
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: true
""")
        config = load_mlflow_config(config_dir)
        assert config["tracking"]["conversion"]["log_conversion_log"] is True


class TestNamingConfiguration:
    """Test naming section (lines 52-76)."""

    def test_naming_project_name(self, config_dir):
        """Test naming.project_name: 'resume-ner'."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  project_name: "resume-ner"
""")
        naming_config = get_naming_config(config_dir)
        assert naming_config["project_name"] == "resume-ner"

    def test_naming_tags_max_length(self, config_dir):
        """Test naming.tags.max_length: 250."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  tags:
    max_length: 250
    sanitize: true
""")
        naming_config = get_naming_config(config_dir)
        assert naming_config["tags"]["max_length"] == 250

    def test_naming_tags_sanitize(self, config_dir):
        """Test naming.tags.sanitize: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  tags:
    max_length: 250
    sanitize: true
""")
        naming_config = get_naming_config(config_dir)
        assert naming_config["tags"]["sanitize"] is True

    def test_naming_run_name_max_length(self, config_dir):
        """Test naming.run_name.max_length: 100."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    max_length: 100
    shorten_fingerprints: true
""")
        naming_config = get_naming_config(config_dir)
        assert naming_config["run_name"]["max_length"] == 100

    def test_naming_run_name_shorten_fingerprints(self, config_dir):
        """Test naming.run_name.shorten_fingerprints: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    max_length: 100
    shorten_fingerprints: true
""")
        naming_config = get_naming_config(config_dir)
        assert naming_config["run_name"]["shorten_fingerprints"] is True

    def test_naming_run_name_auto_increment_enabled(self, config_dir):
        """Test naming.run_name.auto_increment.enabled: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
        auto_inc_config = get_auto_increment_config(config_dir, "hpo")
        assert auto_inc_config["enabled"] is True

    def test_naming_run_name_auto_increment_processes_hpo(self, config_dir):
        """Test naming.run_name.auto_increment.processes.hpo: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
        auto_inc_config = get_auto_increment_config(config_dir, "hpo")
        assert auto_inc_config["processes"]["hpo"] is True

    def test_naming_run_name_auto_increment_processes_benchmarking(self, config_dir):
        """Test naming.run_name.auto_increment.processes.benchmarking: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
        auto_inc_config = get_auto_increment_config(config_dir, "benchmarking")
        assert auto_inc_config["processes"]["benchmarking"] is True

    def test_naming_run_name_auto_increment_format(self, config_dir):
        """Test naming.run_name.auto_increment.format: '{base}.{version}'."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
        auto_inc_config = get_auto_increment_config(config_dir, "hpo")
        assert auto_inc_config["format"] == "{base}.{version}"


class TestIndexConfiguration:
    """Test index section (lines 80-83)."""

    def test_index_enabled(self, config_dir):
        """Test index.enabled: true."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: true
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        index_config = get_index_config(config_dir)
        assert index_config["enabled"] is True

    def test_index_disabled(self, config_dir):
        """Test index.enabled: false."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: false
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        index_config = get_index_config(config_dir)
        assert index_config["enabled"] is False

    def test_index_max_entries(self, config_dir):
        """Test index.max_entries: 1000."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: true
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        index_config = get_index_config(config_dir)
        assert index_config["max_entries"] == 1000

    def test_index_file_name(self, config_dir):
        """Test index.file_name: 'mlflow_index.json'."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
index:
  enabled: true
  max_entries: 1000
  file_name: "mlflow_index.json"
""")
        index_config = get_index_config(config_dir)
        assert index_config["file_name"] == "mlflow_index.json"


class TestRunFinderConfiguration:
    """Test run_finder section (lines 87-90)."""

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

