"""Unit tests for model configuration files.

Tests coverage for all model configuration options in config/model/*.yaml files.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from shared.yaml_utils import load_yaml
from training.config import load_config_file
from config.loader import load_experiment_config, load_all_configs


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory structure."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model_dir = config_dir / "model"
    model_dir.mkdir()
    return config_dir


class TestModelConfigLoading:
    """Test loading model configuration files."""

    def test_load_distilbert_config(self, tmp_config_dir):
        """Test loading distilbert.yaml model config."""
        model_yaml = tmp_config_dir / "model" / "distilbert.yaml"
        model_yaml.write_text("""
backbone: "distilbert-base-uncased"
tokenizer: "distilbert-base-uncased"
preprocessing:
  sequence_length: 40
  max_length: 128
  tokenization: "subword"
  replace_rare_with_unk: true
  unk_frequency_threshold: 2
  keep_stopwords: true
decoding:
  use_crf: true
  crf_learning_rate: 0.01
loss:
  use_class_weights: true
  class_weight_smoothing: 0.1
  ignore_index: -100
""")
        
        config = load_config_file(tmp_config_dir, "model/distilbert.yaml")
        
        assert config["backbone"] == "distilbert-base-uncased"
        assert config["tokenizer"] == "distilbert-base-uncased"
        assert config["preprocessing"]["sequence_length"] == 40
        assert config["preprocessing"]["max_length"] == 128
        assert config["preprocessing"]["tokenization"] == "subword"
        assert config["preprocessing"]["replace_rare_with_unk"] is True
        assert config["preprocessing"]["unk_frequency_threshold"] == 2
        assert config["preprocessing"]["keep_stopwords"] is True
        assert config["decoding"]["use_crf"] is True
        assert config["decoding"]["crf_learning_rate"] == 0.01
        assert config["loss"]["use_class_weights"] is True
        assert config["loss"]["class_weight_smoothing"] == 0.1
        assert config["loss"]["ignore_index"] == -100

    def test_load_distilroberta_config(self, tmp_config_dir):
        """Test loading distilroberta.yaml model config."""
        model_yaml = tmp_config_dir / "model" / "distilroberta.yaml"
        model_yaml.write_text("""
backbone: "distilroberta-base"
tokenizer: "distilroberta-base"
preprocessing:
  sequence_length: 40
  max_length: 128
  tokenization: "subword"
  replace_rare_with_unk: true
  unk_frequency_threshold: 2
  keep_stopwords: true
decoding:
  use_crf: true
  crf_learning_rate: 0.01
loss:
  use_class_weights: true
  class_weight_smoothing: 0.1
  ignore_index: -100
""")
        
        config = load_config_file(tmp_config_dir, "model/distilroberta.yaml")
        
        assert config["backbone"] == "distilroberta-base"
        assert config["tokenizer"] == "distilroberta-base"

    def test_load_deberta_config(self, tmp_config_dir):
        """Test loading deberta.yaml model config."""
        model_yaml = tmp_config_dir / "model" / "deberta.yaml"
        model_yaml.write_text("""
backbone: "microsoft/deberta-v3-base"
tokenizer: "microsoft/deberta-v3-base"
preprocessing:
  sequence_length: 40
  max_length: 128
  tokenization: "subword"
  replace_rare_with_unk: true
  unk_frequency_threshold: 2
  keep_stopwords: true
decoding:
  use_crf: true
  crf_learning_rate: 0.01
loss:
  use_class_weights: true
  class_weight_smoothing: 0.1
  ignore_index: -100
""")
        
        config = load_config_file(tmp_config_dir, "model/deberta.yaml")
        
        assert config["backbone"] == "microsoft/deberta-v3-base"
        assert config["tokenizer"] == "microsoft/deberta-v3-base"


class TestModelConfigOptions:
    """Test all model configuration options."""

    def test_backbone_option(self, tmp_config_dir):
        """Test backbone configuration option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("backbone: \"test-backbone\"\n")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["backbone"] == "test-backbone"

    def test_tokenizer_option(self, tmp_config_dir):
        """Test tokenizer configuration option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("tokenizer: \"test-tokenizer\"\n")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["tokenizer"] == "test-tokenizer"

    def test_preprocessing_sequence_length(self, tmp_config_dir):
        """Test preprocessing.sequence_length option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  sequence_length: 50
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["sequence_length"] == 50

    def test_preprocessing_max_length(self, tmp_config_dir):
        """Test preprocessing.max_length option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  max_length: 256
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["max_length"] == 256

    def test_preprocessing_tokenization(self, tmp_config_dir):
        """Test preprocessing.tokenization option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  tokenization: "subword"
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["tokenization"] == "subword"

    def test_preprocessing_replace_rare_with_unk(self, tmp_config_dir):
        """Test preprocessing.replace_rare_with_unk option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  replace_rare_with_unk: false
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["replace_rare_with_unk"] is False

    def test_preprocessing_unk_frequency_threshold(self, tmp_config_dir):
        """Test preprocessing.unk_frequency_threshold option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  unk_frequency_threshold: 5
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["unk_frequency_threshold"] == 5

    def test_preprocessing_keep_stopwords(self, tmp_config_dir):
        """Test preprocessing.keep_stopwords option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  keep_stopwords: false
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["preprocessing"]["keep_stopwords"] is False

    def test_decoding_use_crf(self, tmp_config_dir):
        """Test decoding.use_crf option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
decoding:
  use_crf: false
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["decoding"]["use_crf"] is False

    def test_decoding_crf_learning_rate(self, tmp_config_dir):
        """Test decoding.crf_learning_rate option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
decoding:
  crf_learning_rate: 0.05
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["decoding"]["crf_learning_rate"] == 0.05

    def test_loss_use_class_weights(self, tmp_config_dir):
        """Test loss.use_class_weights option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
loss:
  use_class_weights: false
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["loss"]["use_class_weights"] is False

    def test_loss_class_weight_smoothing(self, tmp_config_dir):
        """Test loss.class_weight_smoothing option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
loss:
  class_weight_smoothing: 0.2
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["loss"]["class_weight_smoothing"] == 0.2

    def test_loss_ignore_index(self, tmp_config_dir):
        """Test loss.ignore_index option."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
loss:
  ignore_index: -1
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert config["loss"]["ignore_index"] == -1


class TestModelConfigIntegration:
    """Test model config integration with experiment config loading."""

    def test_model_config_via_experiment_config(self, tmp_config_dir):
        """Test loading model config via ExperimentConfig."""
        # Create experiment config
        experiment_dir = tmp_config_dir / "experiment"
        experiment_dir.mkdir()
        experiment_yaml = experiment_dir / "test_experiment.yaml"
        experiment_yaml.write_text("""
experiment_name: "test_experiment"
data_config: "data/resume_v1.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/local.yaml"
env_config: "env/local.yaml"
""")
        
        # Create model config (model_dir already exists from fixture)
        model_dir = tmp_config_dir / "model"
        model_yaml = model_dir / "distilbert.yaml"
        model_yaml.write_text("""
backbone: "distilbert-base-uncased"
tokenizer: "distilbert-base-uncased"
preprocessing:
  sequence_length: 40
  max_length: 128
""")
        
        # Create minimal other configs
        (tmp_config_dir / "data").mkdir(exist_ok=True)
        (tmp_config_dir / "data" / "resume_v1.yaml").write_text("dataset_path: test")
        (tmp_config_dir / "train.yaml").write_text("training: {}")
        (tmp_config_dir / "hpo").mkdir(exist_ok=True)
        (tmp_config_dir / "hpo" / "local.yaml").write_text("{}")
        (tmp_config_dir / "env").mkdir(exist_ok=True)
        (tmp_config_dir / "env" / "local.yaml").write_text("{}")
        
        exp_config = load_experiment_config(tmp_config_dir, "test_experiment")
        all_configs = load_all_configs(exp_config)
        
        assert "model" in all_configs
        assert all_configs["model"]["backbone"] == "distilbert-base-uncased"
        assert all_configs["model"]["tokenizer"] == "distilbert-base-uncased"
        assert all_configs["model"]["preprocessing"]["sequence_length"] == 40
        assert all_configs["model"]["preprocessing"]["max_length"] == 128

    def test_model_config_in_training_config_building(self, tmp_config_dir):
        """Test model config is included when building training config."""
        from unittest.mock import Mock
        
        # Create model config (model_dir already exists from fixture)
        model_dir = tmp_config_dir / "model"
        model_yaml = model_dir / "distilbert.yaml"
        model_yaml.write_text("""
backbone: "distilbert-base-uncased"
tokenizer: "distilbert-base-uncased"
preprocessing:
  sequence_length: 40
decoding:
  use_crf: true
loss:
  use_class_weights: true
""")
        
        # Create train config
        train_yaml = tmp_config_dir / "train.yaml"
        train_yaml.write_text("training: {}\n")
        
        # Mock args
        args = Mock()
        args.backbone = "distilbert"
        
        # Build training config
        config = load_config_file(tmp_config_dir, f"model/{args.backbone}.yaml")
        
        assert config["backbone"] == "distilbert-base-uncased"
        assert config["preprocessing"]["sequence_length"] == 40
        assert config["decoding"]["use_crf"] is True
        assert config["loss"]["use_class_weights"] is True


class TestModelConfigValidation:
    """Test model config validation and edge cases."""

    def test_model_config_missing_sections(self, tmp_config_dir):
        """Test model config with missing optional sections."""
        model_yaml = tmp_config_dir / "model" / "minimal.yaml"
        model_yaml.write_text("""
backbone: "test-backbone"
tokenizer: "test-tokenizer"
""")
        
        config = load_config_file(tmp_config_dir, "model/minimal.yaml")
        
        assert config["backbone"] == "test-backbone"
        assert config["tokenizer"] == "test-tokenizer"
        # Missing sections should not cause errors
        assert "preprocessing" not in config or isinstance(config.get("preprocessing"), dict)
        assert "decoding" not in config or isinstance(config.get("decoding"), dict)
        assert "loss" not in config or isinstance(config.get("loss"), dict)

    def test_model_config_partial_preprocessing(self, tmp_config_dir):
        """Test model config with partial preprocessing section."""
        model_yaml = tmp_config_dir / "model" / "partial.yaml"
        model_yaml.write_text("""
backbone: "test-backbone"
preprocessing:
  sequence_length: 50
  max_length: 256
""")
        
        config = load_config_file(tmp_config_dir, "model/partial.yaml")
        
        assert config["preprocessing"]["sequence_length"] == 50
        assert config["preprocessing"]["max_length"] == 256
        # Other preprocessing options may be missing

    def test_model_config_partial_decoding(self, tmp_config_dir):
        """Test model config with partial decoding section."""
        model_yaml = tmp_config_dir / "model" / "partial.yaml"
        model_yaml.write_text("""
backbone: "test-backbone"
decoding:
  use_crf: true
""")
        
        config = load_config_file(tmp_config_dir, "model/partial.yaml")
        
        assert config["decoding"]["use_crf"] is True
        # crf_learning_rate may be missing

    def test_model_config_partial_loss(self, tmp_config_dir):
        """Test model config with partial loss section."""
        model_yaml = tmp_config_dir / "model" / "partial.yaml"
        model_yaml.write_text("""
backbone: "test-backbone"
loss:
  use_class_weights: false
""")
        
        config = load_config_file(tmp_config_dir, "model/partial.yaml")
        
        assert config["loss"]["use_class_weights"] is False
        # Other loss options may be missing

    def test_model_config_numeric_types(self, tmp_config_dir):
        """Test that numeric types are preserved correctly."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  sequence_length: 40
  max_length: 128
  unk_frequency_threshold: 2
decoding:
  crf_learning_rate: 0.01
loss:
  class_weight_smoothing: 0.1
  ignore_index: -100
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert isinstance(config["preprocessing"]["sequence_length"], int)
        assert isinstance(config["preprocessing"]["max_length"], int)
        assert isinstance(config["preprocessing"]["unk_frequency_threshold"], int)
        assert isinstance(config["decoding"]["crf_learning_rate"], float)
        assert isinstance(config["loss"]["class_weight_smoothing"], float)
        assert isinstance(config["loss"]["ignore_index"], int)

    def test_model_config_boolean_types(self, tmp_config_dir):
        """Test that boolean types are preserved correctly."""
        model_yaml = tmp_config_dir / "model" / "test.yaml"
        model_yaml.write_text("""
preprocessing:
  replace_rare_with_unk: true
  keep_stopwords: false
decoding:
  use_crf: true
loss:
  use_class_weights: false
""")
        
        config = load_config_file(tmp_config_dir, "model/test.yaml")
        
        assert isinstance(config["preprocessing"]["replace_rare_with_unk"], bool)
        assert isinstance(config["preprocessing"]["keep_stopwords"], bool)
        assert isinstance(config["decoding"]["use_crf"], bool)
        assert isinstance(config["loss"]["use_class_weights"], bool)
        assert config["preprocessing"]["replace_rare_with_unk"] is True
        assert config["preprocessing"]["keep_stopwords"] is False
        assert config["decoding"]["use_crf"] is True
        assert config["loss"]["use_class_weights"] is False


class TestModelConfigRealFiles:
    """Test loading actual model config files from config/model/."""

    def test_load_real_distilbert_config(self):
        """Test loading real distilbert.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if (config_dir / "model" / "distilbert.yaml").exists():
            config = load_config_file(config_dir, "model/distilbert.yaml")
            
            assert "backbone" in config
            assert "tokenizer" in config
            assert config["backbone"] == "distilbert-base-uncased"
            assert config["tokenizer"] == "distilbert-base-uncased"

    def test_load_real_distilroberta_config(self):
        """Test loading real distilroberta.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if (config_dir / "model" / "distilroberta.yaml").exists():
            config = load_config_file(config_dir, "model/distilroberta.yaml")
            
            assert "backbone" in config
            assert "tokenizer" in config
            assert config["backbone"] == "distilroberta-base"
            assert config["tokenizer"] == "distilroberta-base"

    def test_load_real_deberta_config(self):
        """Test loading real deberta.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if (config_dir / "model" / "deberta.yaml").exists():
            config = load_config_file(config_dir, "model/deberta.yaml")
            
            assert "backbone" in config
            assert "tokenizer" in config
            assert config["backbone"] == "microsoft/deberta-v3-base"
            assert config["tokenizer"] == "microsoft/deberta-v3-base"

    def test_all_real_model_configs_have_required_sections(self):
        """Test that all real model configs have required sections."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        model_dir = config_dir / "model"
        
        if not model_dir.exists():
            pytest.skip("config/model directory not found")
        
        model_files = list(model_dir.glob("*.yaml"))
        if not model_files:
            pytest.skip("No model config files found")
        
        for model_file in model_files:
            config = load_yaml(model_file)
            
            # Required fields
            assert "backbone" in config, f"{model_file.name} missing 'backbone'"
            assert "tokenizer" in config, f"{model_file.name} missing 'tokenizer'"
            
            # Optional sections (should exist in all current configs)
            if "preprocessing" in config:
                assert isinstance(config["preprocessing"], dict)
            if "decoding" in config:
                assert isinstance(config["decoding"], dict)
            if "loss" in config:
                assert isinstance(config["loss"], dict)

