"""Unit tests for data configuration files.

Tests coverage for all data configuration options in config/data/*.yaml files.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List

from shared.yaml_utils import load_yaml
from training.config import load_config_file
from config.loader import load_experiment_config, load_all_configs
from training.data import build_label_list


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory structure."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create subdirectories
    (config_dir / "data").mkdir()
    (config_dir / "model").mkdir()
    (config_dir / "experiment").mkdir()
    
    return config_dir


class TestDataConfigLoading:
    """Test loading data configuration files."""

    def test_load_complete_data_config(self, tmp_config_dir):
        """Test loading complete data config matching resume_tiny.yaml."""
        data_yaml = tmp_config_dir / "data" / "resume_tiny.yaml"
        data_yaml.write_text("""
name: resume-ner-data-tiny-short
version: v3
description: "Tiny smoke-test subset of Resume NER dataset"
local_path: ../dataset_tiny
seed: 0
splitting:
  train_test_ratio: 0.8
  stratified: false
  random_seed: 42
schema:
  format: json
  annotation_format: character_spans
  entity_types:
    - SKILL
    - EDUCATION
    - DESIGNATION
    - EXPERIENCE
    - NAME
    - EMAIL
    - PHONE
    - LOCATION
  stats:
    median_sentence_length: 19
    mean_sentence_length: 20
    p95_sentence_length: 40
    suggested_sequence_length: 40
    entity_density: 0.35
""")
        
        config = load_config_file(tmp_config_dir, "data/resume_tiny.yaml")
        
        assert config["name"] == "resume-ner-data-tiny-short"
        assert config["version"] == "v3"
        assert config["description"] == "Tiny smoke-test subset of Resume NER dataset"
        assert config["local_path"] == "../dataset_tiny"
        assert config["seed"] == 0
        assert config["splitting"]["train_test_ratio"] == 0.8
        assert config["splitting"]["stratified"] is False
        assert config["splitting"]["random_seed"] == 42
        assert config["schema"]["format"] == "json"
        assert config["schema"]["annotation_format"] == "character_spans"
        assert len(config["schema"]["entity_types"]) == 8
        assert config["schema"]["stats"]["median_sentence_length"] == 19
        assert config["schema"]["stats"]["mean_sentence_length"] == 20
        assert config["schema"]["stats"]["p95_sentence_length"] == 40
        assert config["schema"]["stats"]["suggested_sequence_length"] == 40
        assert config["schema"]["stats"]["entity_density"] == 0.35


class TestDataConfigOptions:
    """Test all data configuration options."""

    def test_name_option(self, tmp_config_dir):
        """Test name option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("name: test-dataset\n")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["name"] == "test-dataset"

    def test_version_option(self, tmp_config_dir):
        """Test version option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("version: v1\n")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["version"] == "v1"

    def test_description_option(self, tmp_config_dir):
        """Test description option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text('description: "Test dataset description"\n')
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["description"] == "Test dataset description"

    def test_local_path_option(self, tmp_config_dir):
        """Test local_path option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("local_path: ../dataset\n")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["local_path"] == "../dataset"

    def test_seed_option(self, tmp_config_dir):
        """Test seed option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("seed: 1\n")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["seed"] == 1
        assert isinstance(config["seed"], int)

    def test_splitting_train_test_ratio(self, tmp_config_dir):
        """Test splitting.train_test_ratio option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
splitting:
  train_test_ratio: 0.9
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["splitting"]["train_test_ratio"] == 0.9
        assert isinstance(config["splitting"]["train_test_ratio"], float)

    def test_splitting_stratified_true(self, tmp_config_dir):
        """Test splitting.stratified = true."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
splitting:
  stratified: true
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["splitting"]["stratified"] is True
        assert isinstance(config["splitting"]["stratified"], bool)

    def test_splitting_stratified_false(self, tmp_config_dir):
        """Test splitting.stratified = false."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
splitting:
  stratified: false
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["splitting"]["stratified"] is False

    def test_splitting_random_seed(self, tmp_config_dir):
        """Test splitting.random_seed option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
splitting:
  random_seed: 123
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["splitting"]["random_seed"] == 123
        assert isinstance(config["splitting"]["random_seed"], int)

    def test_schema_format(self, tmp_config_dir):
        """Test schema.format option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  format: json
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["format"] == "json"

    def test_schema_annotation_format(self, tmp_config_dir):
        """Test schema.annotation_format option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  annotation_format: character_spans
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["annotation_format"] == "character_spans"

    def test_schema_entity_types(self, tmp_config_dir):
        """Test schema.entity_types option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  entity_types:
    - SKILL
    - EDUCATION
    - NAME
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert isinstance(config["schema"]["entity_types"], list)
        assert len(config["schema"]["entity_types"]) == 3
        assert "SKILL" in config["schema"]["entity_types"]
        assert "EDUCATION" in config["schema"]["entity_types"]
        assert "NAME" in config["schema"]["entity_types"]

    def test_schema_stats_median_sentence_length(self, tmp_config_dir):
        """Test schema.stats.median_sentence_length option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  stats:
    median_sentence_length: 25
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["stats"]["median_sentence_length"] == 25
        assert isinstance(config["schema"]["stats"]["median_sentence_length"], int)

    def test_schema_stats_mean_sentence_length(self, tmp_config_dir):
        """Test schema.stats.mean_sentence_length option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  stats:
    mean_sentence_length: 26.5
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["stats"]["mean_sentence_length"] == 26.5
        assert isinstance(config["schema"]["stats"]["mean_sentence_length"], (int, float))

    def test_schema_stats_p95_sentence_length(self, tmp_config_dir):
        """Test schema.stats.p95_sentence_length option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  stats:
    p95_sentence_length: 50
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["stats"]["p95_sentence_length"] == 50
        assert isinstance(config["schema"]["stats"]["p95_sentence_length"], int)

    def test_schema_stats_suggested_sequence_length(self, tmp_config_dir):
        """Test schema.stats.suggested_sequence_length option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  stats:
    suggested_sequence_length: 50
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["stats"]["suggested_sequence_length"] == 50
        assert isinstance(config["schema"]["stats"]["suggested_sequence_length"], int)

    def test_schema_stats_entity_density(self, tmp_config_dir):
        """Test schema.stats.entity_density option."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  stats:
    entity_density: 0.4
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert config["schema"]["stats"]["entity_density"] == 0.4
        assert isinstance(config["schema"]["stats"]["entity_density"], float)


class TestDataConfigIntegration:
    """Test data config integration with other systems."""

    def test_data_config_via_experiment_config(self, tmp_config_dir):
        """Test loading data config via ExperimentConfig."""
        # Create experiment config (experiment_dir already exists from fixture)
        experiment_dir = tmp_config_dir / "experiment"
        experiment_yaml = experiment_dir / "test_experiment.yaml"
        experiment_yaml.write_text("""
experiment_name: "test_experiment"
data_config: "data/resume_tiny.yaml"
model_config: "model/distilbert.yaml"
train_config: "train.yaml"
hpo_config: "hpo/local.yaml"
env_config: "env/local.yaml"
""")
        
        # Create data config
        data_yaml = tmp_config_dir / "data" / "resume_tiny.yaml"
        data_yaml.write_text("""
name: resume-ner-data-tiny-short
version: v3
local_path: ../dataset_tiny
seed: 0
splitting:
  train_test_ratio: 0.8
  stratified: false
  random_seed: 42
schema:
  format: json
  annotation_format: character_spans
  entity_types:
    - SKILL
    - EDUCATION
""")
        
        # Create minimal other configs (directories already exist from fixture)
        (tmp_config_dir / "model" / "distilbert.yaml").write_text("backbone: distilbert-base-uncased")
        (tmp_config_dir / "train.yaml").write_text("training: {}")
        (tmp_config_dir / "hpo").mkdir(exist_ok=True)
        (tmp_config_dir / "hpo" / "local.yaml").write_text("{}")
        (tmp_config_dir / "env").mkdir(exist_ok=True)
        (tmp_config_dir / "env" / "local.yaml").write_text("{}")
        
        exp_config = load_experiment_config(tmp_config_dir, "test_experiment")
        all_configs = load_all_configs(exp_config)
        
        assert "data" in all_configs
        assert all_configs["data"]["name"] == "resume-ner-data-tiny-short"
        assert all_configs["data"]["version"] == "v3"
        assert all_configs["data"]["local_path"] == "../dataset_tiny"
        assert all_configs["data"]["splitting"]["train_test_ratio"] == 0.8
        assert all_configs["data"]["schema"]["format"] == "json"

    def test_build_label_list_from_data_config(self, tmp_config_dir):
        """Test that build_label_list works with data config."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  entity_types:
    - SKILL
    - EDUCATION
    - NAME
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        labels = build_label_list(config)
        
        assert isinstance(labels, list)
        assert "O" in labels
        assert "EDUCATION" in labels
        assert "SKILL" in labels
        assert "NAME" in labels
        # Labels should be sorted (except "O" which is first)
        assert labels[0] == "O"
        assert labels[1:] == sorted(labels[1:])


class TestDataConfigValidation:
    """Test data config validation and edge cases."""

    def test_data_config_missing_optional_sections(self, tmp_config_dir):
        """Test data config with missing optional sections."""
        data_yaml = tmp_config_dir / "data" / "minimal.yaml"
        data_yaml.write_text("""
name: minimal-dataset
local_path: ../dataset
""")
        
        config = load_config_file(tmp_config_dir, "data/minimal.yaml")
        
        assert config["name"] == "minimal-dataset"
        assert config["local_path"] == "../dataset"
        # Missing sections should not cause errors
        assert "version" not in config or isinstance(config.get("version"), str)
        assert "splitting" not in config or isinstance(config.get("splitting"), dict)
        assert "schema" not in config or isinstance(config.get("schema"), dict)

    def test_data_config_partial_splitting(self, tmp_config_dir):
        """Test data config with partial splitting section."""
        data_yaml = tmp_config_dir / "data" / "partial.yaml"
        data_yaml.write_text("""
name: partial-dataset
splitting:
  train_test_ratio: 0.7
""")
        
        config = load_config_file(tmp_config_dir, "data/partial.yaml")
        
        assert config["splitting"]["train_test_ratio"] == 0.7
        # Other splitting options may be missing

    def test_data_config_partial_schema(self, tmp_config_dir):
        """Test data config with partial schema section."""
        data_yaml = tmp_config_dir / "data" / "partial.yaml"
        data_yaml.write_text("""
name: partial-dataset
schema:
  format: json
  entity_types:
    - SKILL
""")
        
        config = load_config_file(tmp_config_dir, "data/partial.yaml")
        
        assert config["schema"]["format"] == "json"
        assert "SKILL" in config["schema"]["entity_types"]
        # Other schema options may be missing

    def test_data_config_partial_stats(self, tmp_config_dir):
        """Test data config with partial stats section."""
        data_yaml = tmp_config_dir / "data" / "partial.yaml"
        data_yaml.write_text("""
name: partial-dataset
schema:
  stats:
    median_sentence_length: 20
    mean_sentence_length: 21
""")
        
        config = load_config_file(tmp_config_dir, "data/partial.yaml")
        
        assert config["schema"]["stats"]["median_sentence_length"] == 20
        assert config["schema"]["stats"]["mean_sentence_length"] == 21
        # Other stats options may be missing

    def test_data_config_numeric_types(self, tmp_config_dir):
        """Test that numeric types are preserved correctly."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
seed: 0
splitting:
  train_test_ratio: 0.8
  random_seed: 42
schema:
  stats:
    median_sentence_length: 19
    mean_sentence_length: 20.5
    p95_sentence_length: 40
    suggested_sequence_length: 40
    entity_density: 0.35
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert isinstance(config["seed"], int)
        assert isinstance(config["splitting"]["train_test_ratio"], float)
        assert isinstance(config["splitting"]["random_seed"], int)
        assert isinstance(config["schema"]["stats"]["median_sentence_length"], int)
        assert isinstance(config["schema"]["stats"]["mean_sentence_length"], (int, float))
        assert isinstance(config["schema"]["stats"]["p95_sentence_length"], int)
        assert isinstance(config["schema"]["stats"]["suggested_sequence_length"], int)
        assert isinstance(config["schema"]["stats"]["entity_density"], float)

    def test_data_config_boolean_types(self, tmp_config_dir):
        """Test that boolean types are preserved correctly."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
splitting:
  stratified: true
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert isinstance(config["splitting"]["stratified"], bool)
        assert config["splitting"]["stratified"] is True

    def test_data_config_entity_types_list(self, tmp_config_dir):
        """Test that entity_types is a list."""
        data_yaml = tmp_config_dir / "data" / "test.yaml"
        data_yaml.write_text("""
schema:
  entity_types:
    - SKILL
    - EDUCATION
    - DESIGNATION
    - EXPERIENCE
    - NAME
    - EMAIL
    - PHONE
    - LOCATION
""")
        
        config = load_config_file(tmp_config_dir, "data/test.yaml")
        
        assert isinstance(config["schema"]["entity_types"], list)
        assert len(config["schema"]["entity_types"]) == 8
        assert all(isinstance(et, str) for et in config["schema"]["entity_types"])


class TestDataConfigRealFiles:
    """Test loading actual data config files from config/data/."""

    def test_load_real_resume_tiny_config(self):
        """Test loading real resume_tiny.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if (config_dir / "data" / "resume_tiny.yaml").exists():
            config = load_config_file(config_dir, "data/resume_tiny.yaml")
            
            assert "name" in config
            assert "version" in config
            assert "local_path" in config
            assert config["name"] == "resume-ner-data-tiny-short"
            assert config["version"] == "v3"

    def test_load_real_resume_v1_config(self):
        """Test loading real resume_v1.yaml from config directory."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        if (config_dir / "data" / "resume_v1.yaml").exists():
            config = load_config_file(config_dir, "data/resume_v1.yaml")
            
            assert "name" in config
            assert "version" in config
            assert "local_path" in config
            assert config["name"] == "resume-ner-data"

    def test_all_real_data_configs_have_required_sections(self):
        """Test that all real data configs have required sections."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        data_dir = config_dir / "data"
        
        if not data_dir.exists():
            pytest.skip("config/data directory not found")
        
        data_files = list(data_dir.glob("*.yaml"))
        if not data_files:
            pytest.skip("No data config files found")
        
        for data_file in data_files:
            config = load_yaml(data_file)
            
            # Required fields (based on usage in codebase)
            assert "name" in config, f"{data_file.name} missing 'name'"
            assert "local_path" in config, f"{data_file.name} missing 'local_path'"
            
            # Optional sections (should exist in all current configs)
            if "splitting" in config:
                assert isinstance(config["splitting"], dict)
            if "schema" in config:
                assert isinstance(config["schema"], dict)
                if "entity_types" in config["schema"]:
                    assert isinstance(config["schema"]["entity_types"], list)
                if "stats" in config["schema"]:
                    assert isinstance(config["schema"]["stats"], dict)

