"""Unit tests for experiment configuration files.

Tests coverage for all experiment configuration options in config/experiment/*.yaml files.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from infrastructure.config.loader import (
    load_experiment_config,
    ExperimentConfig,
    load_all_configs,
)


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory structure."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create subdirectories
    (config_dir / "experiment").mkdir()
    (config_dir / "data").mkdir()
    (config_dir / "model").mkdir()
    (config_dir / "hpo").mkdir()
    (config_dir / "env").mkdir()
    
    # Create minimal config files
    (config_dir / "data" / "resume_tiny.yaml").write_text("dataset_path: test")
    (config_dir / "model" / "distilbert.yaml").write_text("backbone: distilbert-base-uncased")
    (config_dir / "train.yaml").write_text("training: {}")
    (config_dir / "hpo" / "prod.yaml").write_text("max_trials: 10")
    (config_dir / "hpo" / "smoke.yaml").write_text("max_trials: 2")
    (config_dir / "env" / "azure.yaml").write_text("platform: azureml")
    (config_dir / "benchmark.yaml").write_text("iterations: 100")
    
    return config_dir


class TestExperimentConfigLoading:
    """Test loading experiment configuration files."""

    def test_load_complete_experiment_config(self, tmp_config_dir):
        """Test loading complete experiment config matching resume_ner_baseline.yaml."""
        experiment_yaml = tmp_config_dir / "experiment" / "test_experiment.yaml"
        experiment_yaml.write_text("""
experiment_name: "test_experiment"
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
benchmark_config: "benchmark.yaml"
stages:
  smoke:
    aml_experiment: "resume-ner-smoke"
    hpo_config: "hpo/smoke.yaml"
  hpo:
    aml_experiment: "resume-ner-hpo"
    hpo_config: "hpo/smoke.yaml"
  training:
    aml_experiment: "resume-ner-train"
    backbones:
      - "distilbert"
naming:
  include_backbone_in_experiment: true
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test_experiment")
        
        assert exp_cfg.name == "test_experiment"
        assert exp_cfg.data_config == tmp_config_dir / "data" / "resume_tiny.yaml"
        assert exp_cfg.model_config == tmp_config_dir / "model" / "distilbert.yaml"
        assert exp_cfg.train_config == tmp_config_dir / "train.yaml"
        assert exp_cfg.hpo_config == tmp_config_dir / "hpo" / "prod.yaml"
        assert exp_cfg.env_config == tmp_config_dir / "env" / "azure.yaml"
        assert exp_cfg.benchmark_config == tmp_config_dir / "benchmark.yaml"
        
        # Check stages
        assert "smoke" in exp_cfg.stages
        assert "hpo" in exp_cfg.stages
        assert "training" in exp_cfg.stages
        
        # Check naming
        assert "include_backbone_in_experiment" in exp_cfg.naming
        assert exp_cfg.naming["include_backbone_in_experiment"] is True

    def test_load_experiment_config_with_defaults(self, tmp_config_dir):
        """Test loading experiment config with default benchmark_config."""
        experiment_yaml = tmp_config_dir / "experiment" / "minimal.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "minimal")
        
        # Should use experiment name if experiment_name not in YAML
        assert exp_cfg.name == "minimal"
        # Should default to benchmark.yaml if not specified
        assert exp_cfg.benchmark_config == tmp_config_dir / "benchmark.yaml"
        # Should default to empty dicts
        assert exp_cfg.stages == {}
        assert exp_cfg.naming == {}


class TestExperimentConfigOptions:
    """Test all experiment configuration options."""

    def test_experiment_name_option(self, tmp_config_dir):
        """Test experiment_name option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
experiment_name: "custom_experiment_name"
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.name == "custom_experiment_name"

    def test_experiment_name_fallback(self, tmp_config_dir):
        """Test that experiment_name falls back to filename if not specified."""
        experiment_yaml = tmp_config_dir / "experiment" / "fallback_test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "fallback_test")
        
        assert exp_cfg.name == "fallback_test"

    def test_data_config_option(self, tmp_config_dir):
        """Test data_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.data_config == tmp_config_dir / "data" / "resume_tiny.yaml"

    def test_model_config_option(self, tmp_config_dir):
        """Test model_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.model_config == tmp_config_dir / "model" / "distilbert.yaml"

    def test_train_config_option(self, tmp_config_dir):
        """Test train_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.train_config == tmp_config_dir / "train.yaml"

    def test_hpo_config_option(self, tmp_config_dir):
        """Test hpo_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.hpo_config == tmp_config_dir / "hpo" / "prod.yaml"

    def test_env_config_option(self, tmp_config_dir):
        """Test env_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.env_config == tmp_config_dir / "env" / "azure.yaml"

    def test_benchmark_config_option(self, tmp_config_dir):
        """Test benchmark_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
benchmark_config: "benchmark.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.benchmark_config == tmp_config_dir / "benchmark.yaml"

    def test_benchmark_config_default(self, tmp_config_dir):
        """Test benchmark_config defaults to benchmark.yaml if not specified."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.benchmark_config == tmp_config_dir / "benchmark.yaml"


class TestStagesConfiguration:
    """Test stages configuration options."""

    def test_stages_smoke_aml_experiment(self, tmp_config_dir):
        """Test stages.smoke.aml_experiment option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  smoke:
    aml_experiment: "custom-smoke-experiment"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "smoke" in exp_cfg.stages
        assert exp_cfg.stages["smoke"]["aml_experiment"] == "custom-smoke-experiment"

    def test_stages_smoke_hpo_config(self, tmp_config_dir):
        """Test stages.smoke.hpo_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  smoke:
    hpo_config: "hpo/smoke.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "smoke" in exp_cfg.stages
        assert exp_cfg.stages["smoke"]["hpo_config"] == "hpo/smoke.yaml"

    def test_stages_smoke_both_options(self, tmp_config_dir):
        """Test stages.smoke with both aml_experiment and hpo_config."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  smoke:
    aml_experiment: "resume-ner-smoke"
    hpo_config: "hpo/smoke.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.stages["smoke"]["aml_experiment"] == "resume-ner-smoke"
        assert exp_cfg.stages["smoke"]["hpo_config"] == "hpo/smoke.yaml"

    def test_stages_hpo_aml_experiment(self, tmp_config_dir):
        """Test stages.hpo.aml_experiment option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  hpo:
    aml_experiment: "custom-hpo-experiment"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "hpo" in exp_cfg.stages
        assert exp_cfg.stages["hpo"]["aml_experiment"] == "custom-hpo-experiment"

    def test_stages_hpo_hpo_config(self, tmp_config_dir):
        """Test stages.hpo.hpo_config option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  hpo:
    hpo_config: "hpo/smoke.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "hpo" in exp_cfg.stages
        assert exp_cfg.stages["hpo"]["hpo_config"] == "hpo/smoke.yaml"

    def test_stages_hpo_both_options(self, tmp_config_dir):
        """Test stages.hpo with both aml_experiment and hpo_config."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  hpo:
    aml_experiment: "resume-ner-hpo"
    hpo_config: "hpo/smoke.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.stages["hpo"]["aml_experiment"] == "resume-ner-hpo"
        assert exp_cfg.stages["hpo"]["hpo_config"] == "hpo/smoke.yaml"

    def test_stages_training_aml_experiment(self, tmp_config_dir):
        """Test stages.training.aml_experiment option."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  training:
    aml_experiment: "custom-training-experiment"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "training" in exp_cfg.stages
        assert exp_cfg.stages["training"]["aml_experiment"] == "custom-training-experiment"

    def test_stages_training_backbones_single(self, tmp_config_dir):
        """Test stages.training.backbones with single backbone."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  training:
    backbones:
      - "distilbert"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "training" in exp_cfg.stages
        assert exp_cfg.stages["training"]["backbones"] == ["distilbert"]
        assert isinstance(exp_cfg.stages["training"]["backbones"], list)

    def test_stages_training_backbones_multiple(self, tmp_config_dir):
        """Test stages.training.backbones with multiple backbones."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  training:
    backbones:
      - "distilbert"
      - "distilroberta"
      - "deberta"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.stages["training"]["backbones"] == ["distilbert", "distilroberta", "deberta"]
        assert len(exp_cfg.stages["training"]["backbones"]) == 3

    def test_stages_training_both_options(self, tmp_config_dir):
        """Test stages.training with both aml_experiment and backbones."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  training:
    aml_experiment: "resume-ner-train"
    backbones:
      - "distilbert"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.stages["training"]["aml_experiment"] == "resume-ner-train"
        assert exp_cfg.stages["training"]["backbones"] == ["distilbert"]

    def test_stages_all_stages_together(self, tmp_config_dir):
        """Test all stages configured together."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  smoke:
    aml_experiment: "resume-ner-smoke"
    hpo_config: "hpo/smoke.yaml"
  hpo:
    aml_experiment: "resume-ner-hpo"
    hpo_config: "hpo/smoke.yaml"
  training:
    aml_experiment: "resume-ner-train"
    backbones:
      - "distilbert"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert "smoke" in exp_cfg.stages
        assert "hpo" in exp_cfg.stages
        assert "training" in exp_cfg.stages
        assert exp_cfg.stages["smoke"]["aml_experiment"] == "resume-ner-smoke"
        assert exp_cfg.stages["hpo"]["aml_experiment"] == "resume-ner-hpo"
        assert exp_cfg.stages["training"]["aml_experiment"] == "resume-ner-train"
        assert exp_cfg.stages["training"]["backbones"] == ["distilbert"]


class TestNamingConfiguration:
    """Test naming configuration options."""

    def test_naming_include_backbone_in_experiment_true(self, tmp_config_dir):
        """Test naming.include_backbone_in_experiment = true."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
naming:
  include_backbone_in_experiment: true
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.naming["include_backbone_in_experiment"] is True
        assert isinstance(exp_cfg.naming["include_backbone_in_experiment"], bool)

    def test_naming_include_backbone_in_experiment_false(self, tmp_config_dir):
        """Test naming.include_backbone_in_experiment = false."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
naming:
  include_backbone_in_experiment: false
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.naming["include_backbone_in_experiment"] is False

    def test_naming_missing_defaults_to_empty(self, tmp_config_dir):
        """Test that missing naming section defaults to empty dict."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        assert exp_cfg.naming == {}


class TestExperimentConfigIntegration:
    """Test experiment config integration with config loading."""

    def test_load_all_configs_with_experiment_config(self, tmp_config_dir):
        """Test that load_all_configs works with ExperimentConfig."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        all_configs = load_all_configs(exp_cfg)
        
        assert "data" in all_configs
        assert "model" in all_configs
        assert "train" in all_configs
        assert "hpo" in all_configs
        assert "env" in all_configs
        assert "benchmark" in all_configs

    def test_experiment_config_stages_preserved(self, tmp_config_dir):
        """Test that stages are preserved in ExperimentConfig."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
stages:
  smoke:
    aml_experiment: "test-smoke"
  training:
    backbones: ["distilbert"]
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        # Stages should be preserved as-is
        assert exp_cfg.stages["smoke"]["aml_experiment"] == "test-smoke"
        assert exp_cfg.stages["training"]["backbones"] == ["distilbert"]

    def test_experiment_config_naming_preserved(self, tmp_config_dir):
        """Test that naming is preserved in ExperimentConfig."""
        experiment_yaml = tmp_config_dir / "experiment" / "test.yaml"
        experiment_yaml.write_text("""
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/prod.yaml"
env_config: "env/azure.yaml"
naming:
  include_backbone_in_experiment: true
  custom_option: "custom_value"
""")
        
        exp_cfg = load_experiment_config(tmp_config_dir, "test")
        
        # Naming should be preserved as-is
        assert exp_cfg.naming["include_backbone_in_experiment"] is True
        assert exp_cfg.naming["custom_option"] == "custom_value"


class TestExperimentConfigRealFile:
    """Test loading actual experiment config file."""

    def test_load_real_resume_ner_baseline_config(self):
        """Test loading real resume_ner_baseline.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if not (config_dir / "experiment" / "resume_ner_baseline.yaml").exists():
            pytest.skip("resume_ner_baseline.yaml not found")
        
        exp_cfg = load_experiment_config(config_dir, "resume_ner_baseline")
        
        # Verify required fields
        assert exp_cfg.name == "resume_ner_baseline"
        assert exp_cfg.data_config.exists() or exp_cfg.data_config.name == "resume_tiny.yaml"
        assert exp_cfg.model_config.exists() or exp_cfg.model_config.name == "distilbert.yaml"
        assert exp_cfg.train_config.exists() or exp_cfg.train_config.name == "train.yaml"
        assert exp_cfg.hpo_config.exists() or "hpo" in str(exp_cfg.hpo_config)
        assert exp_cfg.env_config.exists() or "env" in str(exp_cfg.env_config)
        
        # Verify stages if present
        if exp_cfg.stages:
            assert isinstance(exp_cfg.stages, dict)
            if "smoke" in exp_cfg.stages:
                assert isinstance(exp_cfg.stages["smoke"], dict)
            if "hpo" in exp_cfg.stages:
                assert isinstance(exp_cfg.stages["hpo"], dict)
            if "training" in exp_cfg.stages:
                assert isinstance(exp_cfg.stages["training"], dict)
        
        # Verify naming if present
        if exp_cfg.naming:
            assert isinstance(exp_cfg.naming, dict)

    def test_real_resume_ner_baseline_has_all_sections(self):
        """Test that real resume_ner_baseline.yaml has all expected sections."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        experiment_file = config_dir / "experiment" / "resume_ner_baseline.yaml"
        
        if not experiment_file.exists():
            pytest.skip("resume_ner_baseline.yaml not found")
        
        from common.shared.yaml_utils import load_yaml
        config = load_yaml(experiment_file)
        
        # Required fields
        assert "data_config" in config
        assert "model_config" in config
        assert "train_config" in config
        assert "hpo_config" in config
        assert "env_config" in config
        
        # Optional but expected fields
        if "stages" in config:
            assert isinstance(config["stages"], dict)
            if "smoke" in config["stages"]:
                assert isinstance(config["stages"]["smoke"], dict)
            if "hpo" in config["stages"]:
                assert isinstance(config["stages"]["hpo"], dict)
            if "training" in config["stages"]:
                assert isinstance(config["stages"]["training"], dict)
                if "backbones" in config["stages"]["training"]:
                    assert isinstance(config["stages"]["training"]["backbones"], list)
        
        if "naming" in config:
            assert isinstance(config["naming"], dict)

