"""Tests to ensure complete coverage of paths.yaml configuration (lines 1-431)."""

import json
import yaml
from pathlib import Path
import pytest

from orchestration.paths import (
    load_paths_config,
    resolve_output_path,
    get_cache_file_path,
    get_timestamped_cache_filename,
    get_cache_strategy_config,
    get_drive_backup_base,
    get_drive_backup_path,
)
from core.normalize import normalize_for_path


class TestBaseDirectories:
    """Test base directory configuration (lines 34-40)."""

    def test_base_outputs(self, tmp_path):
        """Test base.outputs."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["outputs"] == "outputs"

    def test_base_notebooks(self, tmp_path):
        """Test base.notebooks."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  notebooks: "notebooks"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["notebooks"] == "notebooks"

    def test_base_config(self, tmp_path):
        """Test base.config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  config: "config"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["config"] == "config"

    def test_base_src(self, tmp_path):
        """Test base.src."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  src: "src"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["src"] == "src"

    def test_base_tests(self, tmp_path):
        """Test base.tests."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  tests: "tests"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["tests"] == "tests"

    def test_base_mlruns(self, tmp_path):
        """Test base.mlruns."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  mlruns: "mlruns"
""")
        config = load_paths_config(config_dir)
        assert config["base"]["mlruns"] == "mlruns"


class TestOutputsSubdirectories:
    """Test outputs subdirectories (lines 74-101)."""

    def test_outputs_hpo_tests(self, tmp_path):
        """Test outputs.hpo_tests."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  hpo_tests: "hpo_tests"
""")
        config = load_paths_config(config_dir)
        assert config["outputs"]["hpo_tests"] == "hpo_tests"
        
        path = resolve_output_path(tmp_path, config_dir, "hpo_tests")
        assert path == tmp_path / "outputs" / "hpo_tests"

    def test_outputs_dry_run(self, tmp_path):
        """Test outputs.dry_run."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  dry_run: "dry_run"
""")
        config = load_paths_config(config_dir)
        assert config["outputs"]["dry_run"] == "dry_run"
        
        path = resolve_output_path(tmp_path, config_dir, "dry_run")
        assert path == tmp_path / "outputs" / "dry_run"

    def test_outputs_e2e_test(self, tmp_path):
        """Test outputs.e2e_test."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  e2e_test: "e2e_test"
""")
        config = load_paths_config(config_dir)
        assert config["outputs"]["e2e_test"] == "e2e_test"
        
        path = resolve_output_path(tmp_path, config_dir, "e2e_test")
        assert path == tmp_path / "outputs" / "e2e_test"

    def test_outputs_pytest_logs(self, tmp_path):
        """Test outputs.pytest_logs."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  pytest_logs: "pytest_logs"
""")
        config = load_paths_config(config_dir)
        assert config["outputs"]["pytest_logs"] == "pytest_logs"
        
        path = resolve_output_path(tmp_path, config_dir, "pytest_logs")
        assert path == tmp_path / "outputs" / "pytest_logs"


class TestFilesConfiguration:
    """Test files configuration (lines 140-165)."""

    def test_files_metrics(self, tmp_path):
        """Test files.metrics."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
files:
  metrics: "metrics.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["metrics"] == "metrics.json"

    def test_files_benchmark(self, tmp_path):
        """Test files.benchmark."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
files:
  benchmark: "benchmark.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["benchmark"] == "benchmark.json"

    def test_files_checkpoint_dir(self, tmp_path):
        """Test files.checkpoint_dir."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
files:
  checkpoint_dir: "checkpoint"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["checkpoint_dir"] == "checkpoint"

    def test_files_cache_best_config_latest(self, tmp_path):
        """Test files.cache.best_config_latest."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
files:
  cache:
    best_config_latest: "latest_best_configuration.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["best_config_latest"] == "latest_best_configuration.json"
        
        # Note: get_cache_file_path uses cache_strategies.latest.filename first,
        # then falls back to files.cache.best_config_latest
        # Since cache_strategies is not set, it will use the default fallback
        # This test verifies the config option exists and is loaded correctly

    def test_files_cache_best_config_index(self, tmp_path):
        """Test files.cache.best_config_index."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
files:
  cache:
    best_config_index: "index.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["best_config_index"] == "index.json"
        
        # Note: get_cache_file_path uses cache_strategies.index.filename first
        # This test verifies the config option exists
        path = get_cache_file_path(
            tmp_path, config_dir, "best_configurations", file_type="index"
        )
        # Path should exist (may use default or config value)
        assert path.name.endswith(".json")

    def test_files_cache_final_training_latest(self, tmp_path):
        """Test files.cache.final_training_latest."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  final_training: "final_training"
files:
  cache:
    final_training_latest: "latest_final_training_cache.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["final_training_latest"] == "latest_final_training_cache.json"
        
        # Note: get_cache_file_path uses cache_strategies.latest.filename first,
        # then falls back to files.cache.final_training_latest
        # Since cache_strategies is not set, it will use the default fallback
        # This test verifies the config option exists and is loaded correctly

    def test_files_cache_final_training_index(self, tmp_path):
        """Test files.cache.final_training_index."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  final_training: "final_training"
files:
  cache:
    final_training_index: "final_training_index.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["final_training_index"] == "final_training_index.json"
        
        path = get_cache_file_path(
            tmp_path, config_dir, "final_training", file_type="index"
        )
        assert path.name.endswith(".json")

    def test_files_cache_best_model_selection_latest(self, tmp_path):
        """Test files.cache.best_model_selection_latest."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_model_selection: "best_model_selection"
files:
  cache:
    best_model_selection_latest: "latest_best_model_selection_cache.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["best_model_selection_latest"] == "latest_best_model_selection_cache.json"
        
        path = get_cache_file_path(
            tmp_path, config_dir, "best_model_selection", file_type="latest"
        )
        assert path.name.endswith(".json")

    def test_files_cache_best_model_selection_index(self, tmp_path):
        """Test files.cache.best_model_selection_index."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_model_selection: "best_model_selection"
files:
  cache:
    best_model_selection_index: "best_model_selection_index.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["best_model_selection_index"] == "best_model_selection_index.json"
        
        path = get_cache_file_path(
            tmp_path, config_dir, "best_model_selection", file_type="index"
        )
        assert path.name.endswith(".json")

    def test_files_cache_conversion_cache(self, tmp_path):
        """Test files.cache.conversion_cache."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
files:
  cache:
    conversion_cache: "conversion_cache.json"
""")
        config = load_paths_config(config_dir)
        assert config["files"]["cache"]["conversion_cache"] == "conversion_cache.json"


class TestCacheStrategiesConfiguration:
    """Test cache_strategies configuration (lines 283-364)."""

    def test_cache_strategies_best_configurations_timestamped_enabled(self, tmp_path):
        """Test cache_strategies.best_configurations.timestamped.enabled."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
cache_strategies:
  best_configurations:
    strategy: "dual"
    timestamped:
      enabled: true
      pattern: "best_config_{backbone}_{trial}_{timestamp}.json"
      keep_all: true
      max_files: null
""")
        config = load_paths_config(config_dir)
        strategy = get_cache_strategy_config(config_dir, "best_configurations")
        assert strategy["timestamped"]["enabled"] is True
        assert strategy["timestamped"]["keep_all"] is True
        assert strategy["timestamped"]["max_files"] is None

    def test_cache_strategies_best_configurations_latest_include_timestamped_ref(self, tmp_path):
        """Test cache_strategies.best_configurations.latest.include_timestamped_ref."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
cache_strategies:
  best_configurations:
    strategy: "dual"
    latest:
      enabled: true
      filename: "latest_best_configuration.json"
      include_timestamped_ref: true
""")
        config = load_paths_config(config_dir)
        strategy = get_cache_strategy_config(config_dir, "best_configurations")
        assert strategy["latest"]["include_timestamped_ref"] is True

    def test_cache_strategies_best_configurations_index_max_entries(self, tmp_path):
        """Test cache_strategies.best_configurations.index.max_entries."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
cache_strategies:
  best_configurations:
    strategy: "dual"
    index:
      enabled: true
      filename: "index.json"
      max_entries: 20
      include_metadata: true
""")
        config = load_paths_config(config_dir)
        strategy = get_cache_strategy_config(config_dir, "best_configurations")
        assert strategy["index"]["max_entries"] == 20
        assert strategy["index"]["include_metadata"] is True

    def test_cache_strategies_final_training_all_options(self, tmp_path):
        """Test cache_strategies.final_training with all options."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  final_training: "final_training"
cache_strategies:
  final_training:
    strategy: "dual"
    timestamped:
      enabled: true
      pattern: "final_training_{backbone}_{run_id}_{timestamp}.json"
      keep_all: true
      max_files: null
    latest:
      enabled: true
      filename: "latest_final_training_cache.json"
      include_timestamped_ref: true
    index:
      enabled: true
      filename: "final_training_index.json"
      max_entries: 20
      include_metadata: true
""")
        config = load_paths_config(config_dir)
        strategy = get_cache_strategy_config(config_dir, "final_training")
        assert strategy["strategy"] == "dual"
        assert strategy["timestamped"]["pattern"] == "final_training_{backbone}_{run_id}_{timestamp}.json"
        assert strategy["latest"]["filename"] == "latest_final_training_cache.json"
        assert strategy["index"]["max_entries"] == 20

    def test_cache_strategies_best_model_selection_all_options(self, tmp_path):
        """Test cache_strategies.best_model_selection with all options."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_model_selection: "best_model_selection"
cache_strategies:
  best_model_selection:
    strategy: "dual"
    timestamped:
      enabled: true
      pattern: "best_model_selection_{backbone}_{identifier}_{timestamp}.json"
      keep_all: true
      max_files: null
    latest:
      enabled: true
      filename: "latest_best_model_selection_cache.json"
      include_timestamped_ref: true
    index:
      enabled: true
      filename: "best_model_selection_index.json"
      max_entries: 20
      include_metadata: true
""")
        config = load_paths_config(config_dir)
        strategy = get_cache_strategy_config(config_dir, "best_model_selection")
        assert strategy["strategy"] == "dual"
        assert strategy["timestamped"]["pattern"] == "best_model_selection_{backbone}_{identifier}_{timestamp}.json"
        assert strategy["latest"]["filename"] == "latest_best_model_selection_cache.json"
        assert strategy["index"]["max_entries"] == 20


class TestDriveConfiguration:
    """Test drive configuration (lines 397-409)."""

    def test_drive_mount_point(self, tmp_path):
        """Test drive.mount_point."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
""")
        config = load_paths_config(config_dir)
        assert config["drive"]["mount_point"] == "/content/drive"
        
        drive_base = get_drive_backup_base(config_dir)
        assert "/content/drive" in str(drive_base)

    def test_drive_backup_base_dir(self, tmp_path):
        """Test drive.backup_base_dir."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
""")
        config = load_paths_config(config_dir)
        assert config["drive"]["backup_base_dir"] == "resume-ner-azureml"
        
        drive_base = get_drive_backup_base(config_dir)
        assert "resume-ner-azureml" in str(drive_base)

    def test_drive_auto_restore_on_startup(self, tmp_path):
        """Test drive.auto_restore_on_startup."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
  auto_restore_on_startup: false
""")
        config = load_paths_config(config_dir)
        assert config["drive"]["auto_restore_on_startup"] is False

    def test_drive_auto_restore_on_startup_true(self, tmp_path):
        """Test drive.auto_restore_on_startup = true."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
  auto_restore_on_startup: true
""")
        config = load_paths_config(config_dir)
        assert config["drive"]["auto_restore_on_startup"] is True


class TestNormalizePathsConfiguration:
    """Test normalize_paths configuration (lines 414-429)."""

    def test_normalize_paths_replace(self, tmp_path):
        """Test normalize_paths.replace configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
normalize_paths:
  replace:
    "/": "_"
    "\\\\": "_"
    "-": "_"
    " ": "_"
    ":": "_"
    "*": "_"
    "?": "_"
    "\\"": "_"
    "<": "_"
    ">": "_"
    "|": "_"
""")
        config = load_paths_config(config_dir)
        replace_rules = config["normalize_paths"]["replace"]
        assert replace_rules["/"] == "_"
        assert replace_rules.get("\\") == "_" or "\\" in replace_rules
        assert replace_rules["-"] == "_"
        assert replace_rules[" "] == "_"
        
        # Test that normalization is applied
        normalized, _ = normalize_for_path("test/path-with:invalid*chars", config["normalize_paths"])
        assert "/" not in normalized
        assert "-" not in normalized
        assert ":" not in normalized
        assert "*" not in normalized

    def test_normalize_paths_lowercase(self, tmp_path):
        """Test normalize_paths.lowercase."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
normalize_paths:
  lowercase: false
""")
        config = load_paths_config(config_dir)
        assert config["normalize_paths"]["lowercase"] is False
        
        # Test that lowercase is not applied
        normalized, _ = normalize_for_path("TestPath", config["normalize_paths"])
        assert "TestPath" in normalized or normalized == "TestPath"

    def test_normalize_paths_lowercase_true(self, tmp_path):
        """Test normalize_paths.lowercase = true."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
normalize_paths:
  lowercase: true
""")
        config = load_paths_config(config_dir)
        assert config["normalize_paths"]["lowercase"] is True
        
        # Test that lowercase is applied
        normalized, _ = normalize_for_path("TestPath", config["normalize_paths"])
        assert normalized == "testpath"

    def test_normalize_paths_max_component_length(self, tmp_path):
        """Test normalize_paths.max_component_length."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
normalize_paths:
  max_component_length: 255
""")
        config = load_paths_config(config_dir)
        assert config["normalize_paths"]["max_component_length"] == 255
        
        # Test that max_component_length is applied
        long_string = "a" * 300
        normalized, warnings = normalize_for_path(long_string, config["normalize_paths"])
        assert len(normalized) <= 255
        assert any("Truncated" in w for w in warnings)

    def test_normalize_paths_max_path_length(self, tmp_path):
        """Test normalize_paths.max_path_length."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
normalize_paths:
  max_path_length: 260
""")
        config = load_paths_config(config_dir)
        assert config["normalize_paths"]["max_path_length"] == 260
        # Note: max_path_length is typically used at path construction time, not in normalize_for_path


class TestPatternsConfiguration:
    """Test patterns configuration (lines 197-244)."""

    def test_patterns_best_config_file(self, tmp_path):
        """Test patterns.best_config_file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
patterns:
  best_config_file: "best_config_{backbone}_{trial}_{timestamp}.json"
""")
        config = load_paths_config(config_dir)
        assert config["patterns"]["best_config_file"] == "best_config_{backbone}_{trial}_{timestamp}.json"
        
        # Verify it's used in get_timestamped_cache_filename
        filename = get_timestamped_cache_filename(
            config_dir, "best_configurations", "distilbert", "trial_2", "20251227_220407"
        )
        assert "best_config_" in filename
        assert "distilbert" in filename
        assert "trial_2" in filename
        assert "20251227_220407" in filename

    def test_patterns_final_training_cache_file(self, tmp_path):
        """Test patterns.final_training_cache_file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
patterns:
  final_training_cache_file: "final_training_{backbone}_{run_id}_{timestamp}.json"
""")
        config = load_paths_config(config_dir)
        assert config["patterns"]["final_training_cache_file"] == "final_training_{backbone}_{run_id}_{timestamp}.json"
        
        filename = get_timestamped_cache_filename(
            config_dir, "final_training", "distilbert", "20251227_220407", "20251227_220500"
        )
        assert "final_training_" in filename
        assert "distilbert" in filename

    def test_patterns_best_model_selection_cache_file(self, tmp_path):
        """Test patterns.best_model_selection_cache_file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
patterns:
  best_model_selection_cache_file: "best_model_selection_{backbone}_{identifier}_{timestamp}.json"
""")
        config = load_paths_config(config_dir)
        assert config["patterns"]["best_model_selection_cache_file"] == "best_model_selection_{backbone}_{identifier}_{timestamp}.json"
        
        filename = get_timestamped_cache_filename(
            config_dir, "best_model_selection", "distilbert", "experiment_cachekey", "20251228_001000"
        )
        assert "best_model_selection_" in filename
        assert "distilbert" in filename
        assert "experiment_cachekey" in filename

