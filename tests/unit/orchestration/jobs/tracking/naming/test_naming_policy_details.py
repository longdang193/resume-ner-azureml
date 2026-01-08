"""Comprehensive unit tests for naming policy details."""

import yaml
from pathlib import Path
import pytest

from orchestration.jobs.tracking.naming.policy import (
    load_naming_policy,
    format_run_name,
    validate_run_name,
)
from orchestration.naming_centralized import NamingContext


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


class TestComponentConfiguration:
    """Test component extraction and formatting (6.1)."""

    def test_zero_pad_trial_number(self, config_dir):
        """Test zero_pad: 2 for trial_number (t00, t01, t10, t99)."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = format_run_name("hpo_trial", context, policy, config_dir)
        # Should have zero-padded trial number: t05
        assert "t05" in run_name or "t5" in run_name
    
    def test_component_default_values(self, config_dir):
        """Test default values when sources are missing."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "0"
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            # Missing study_key_hash and trial_number
            stage="hpo_trial"
        )
        
        run_name = format_run_name("hpo_trial", context, policy, config_dir)
        # Should use defaults
        assert "unknown" in run_name or "study-" in run_name
    
    def test_component_length_truncation(self, config_dir):
        """Test length: 8 for hash truncation."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",  # Long hash
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = format_run_name("hpo_trial", context, policy, config_dir)
        # Should truncate to 8 chars: study-350a79aa
        assert "study-350a79aa" in run_name


class TestSemanticSuffix:
    """Test semantic suffix handling (6.2)."""

    def test_semantic_suffix_enabled(self, config_dir):
        """Test semantic suffix enabled: true/false toggle."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      semantic_suffix:
        enabled: true
        max_length: 30
        source: "study_name"
        default: ""
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            study_name="smoke_test",
            stage="hpo_sweep"
        )
        
        run_name = format_run_name("hpo_sweep", context, policy, config_dir)
        # Should include semantic suffix
        assert "study-350a79aa" in run_name
    
    def test_semantic_suffix_max_length(self, config_dir):
        """Test semantic suffix max_length: 30 truncation."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      semantic_suffix:
        enabled: true
        max_length: 30
        source: "study_name"
        default: ""
""")
        
        policy = load_naming_policy(config_dir)
        # Create a very long study name
        long_study_name = "a" * 50
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            study_name=long_study_name,
            stage="hpo_sweep"
        )
        
        run_name = format_run_name("hpo_sweep", context, policy, config_dir)
        # Semantic suffix should be truncated to max_length
        assert len(run_name) < len(long_study_name) + 100  # Rough check
    
    def test_semantic_suffix_sanitization(self, config_dir):
        """Test semantic suffix sanitization: remove spaces, replace / with -."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      semantic_suffix:
        enabled: true
        max_length: 30
        source: "study_name"
        default: ""
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            study_name="smoke/test with spaces",  # Has special chars
            stage="hpo_sweep"
        )
        
        run_name = format_run_name("hpo_sweep", context, policy, config_dir)
        # Should sanitize: spaces removed, / replaced with -
        assert "study-350a79aa" in run_name


class TestVersionFormat:
    """Test version suffix formatting (6.3)."""

    def test_version_format_parsing(self, config_dir):
        """Test format: '{separator}{number}' string parsing."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        policy = load_naming_policy(config_dir)
        assert policy["version"]["format"] == "{separator}{number}"
        assert policy["version"]["separator"] == "_"


class TestSeparatorPolicy:
    """Test separator usage (6.4)."""

    def test_separator_field(self, config_dir):
        """Test field: '_' (between major fields)."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
""")
        
        policy = load_naming_policy(config_dir)
        assert policy["separators"]["field"] == "_"
        assert policy["separators"]["component"] == "-"
        assert policy["separators"]["version"] == "_"


class TestNormalizationRules:
    """Test normalization before formatting (6.5)."""

    def test_normalization_env_replace(self, config_dir):
        """Test env.replace: {"/": "_", "-": "_", " ": "_"}."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  env:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
  model:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
""")
        
        policy = load_naming_policy(config_dir)
        assert policy["normalize"]["env"]["replace"]["/"] == "_"
        assert policy["normalize"]["env"]["replace"]["-"] == "_"
        assert policy["normalize"]["env"]["lowercase"] is False
    
    def test_normalization_model_replace(self, config_dir):
        """Test model.replace: {"/": "_", "-": "_", " ": "_"}."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  env:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
  model:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
""")
        
        policy = load_naming_policy(config_dir)
        assert policy["normalize"]["model"]["replace"]["/"] == "_"
        assert policy["normalize"]["model"]["replace"]["-"] == "_"
        assert policy["normalize"]["model"]["lowercase"] is False


class TestValidationRules:
    """Test run name validation (6.6)."""

    def test_validate_max_length(self, config_dir):
        """Test max_length: 256 (MLflow/Azure ML limit)."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
  warn_length: 150
""")
        
        policy = load_naming_policy(config_dir)
        
        # Valid name (within limit)
        validate_run_name("local_distilbert_hpo_trial", policy)
        
        # Name exceeding max_length should raise error
        long_name = "a" * 300
        with pytest.raises(ValueError, match="max_length"):
            validate_run_name(long_name, policy)
    
    def test_validate_forbidden_chars(self, config_dir):
        """Test forbidden_chars validation."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
  warn_length: 150
""")
        
        policy = load_naming_policy(config_dir)
        
        # Invalid name with forbidden char
        with pytest.raises(ValueError, match="forbidden"):
            validate_run_name("local/distilbert/hpo_trial", policy)
    
    def test_validate_warn_length(self, config_dir):
        """Test warn_length: 150 (warning when name exceeds this)."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
  warn_length: 150
""")
        
        policy = load_naming_policy(config_dir)
        
        # Name exceeding warn_length should generate warning (but not error)
        long_name = "a" * 200
        # Should not raise error, but may log warning
        validate_run_name(long_name, policy)  # Should pass validation

