"""Explicit tests for all naming.yaml configuration options (lines 1-156).

This test file ensures every single config option in naming.yaml is explicitly tested.
"""

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


class TestSchemaVersion:
    """Test schema_version (line 5)."""

    def test_schema_version_is_loaded(self, config_dir):
        """Test schema_version: 1 is loaded correctly."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["schema_version"] == 1


class TestSeparatorsExplicit:
    """Test separators section (lines 8-11)."""

    def test_separator_field(self, config_dir):
        """Test separators.field: '_'."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["separators"]["field"] == "_"

    def test_separator_component(self, config_dir):
        """Test separators.component: '-'."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["separators"]["component"] == "-"

    def test_separator_version(self, config_dir):
        """Test separators.version: '_'."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["separators"]["version"] == "_"


class TestVersionFormatExplicit:
    """Test version section (lines 137-139)."""

    def test_version_format(self, config_dir):
        """Test version.format: '{separator}{number}'."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
version:
  format: "{separator}{number}"
  separator: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial{version}"
""")
        policy = load_naming_policy(config_dir)
        assert policy["version"]["format"] == "{separator}{number}"

    def test_version_separator(self, config_dir):
        """Test version.separator: '_'."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
version:
  format: "{separator}{number}"
  separator: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial{version}"
""")
        policy = load_naming_policy(config_dir)
        assert policy["version"]["separator"] == "_"


class TestNormalizeExplicit:
    """Test normalize section (lines 142-148)."""

    def test_normalize_env_replace(self, config_dir):
        """Test normalize.env.replace: {"/": "_", "-": "_", " ": "_"}."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  env:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        replace = policy["normalize"]["env"]["replace"]
        assert replace["/"] == "_"
        assert replace["-"] == "_"
        assert replace[" "] == "_"

    def test_normalize_env_lowercase(self, config_dir):
        """Test normalize.env.lowercase: false."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  env:
    replace: {}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["normalize"]["env"]["lowercase"] is False

    def test_normalize_model_replace(self, config_dir):
        """Test normalize.model.replace: {"/": "_", "-": "_", " ": "_"}."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  model:
    replace: {"/": "_", "-": "_", " ": "_"}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        replace = policy["normalize"]["model"]["replace"]
        assert replace["/"] == "_"
        assert replace["-"] == "_"
        assert replace[" "] == "_"

    def test_normalize_model_lowercase(self, config_dir):
        """Test normalize.model.lowercase: false."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
normalize:
  model:
    replace: {}
    lowercase: false
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial"
""")
        policy = load_naming_policy(config_dir)
        assert policy["normalize"]["model"]["lowercase"] is False


class TestValidateExplicit:
    """Test validate section (lines 151-154)."""

    def test_validate_max_length(self, config_dir):
        """Test validate.max_length: 256."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\\\", ":", "*", "?", '"', "<", ">", "|"]
  warn_length: 150
""")
        policy = load_naming_policy(config_dir)
        assert policy["validate"]["max_length"] == 256

    def test_validate_forbidden_chars(self, config_dir):
        """Test validate.forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\\\", ":", "*", "?", '"', "<", ">", "|"]
  warn_length: 150
""")
        policy = load_naming_policy(config_dir)
        forbidden = policy["validate"]["forbidden_chars"]
        assert "/" in forbidden
        assert "\\" in forbidden
        assert ":" in forbidden
        assert "*" in forbidden
        assert "?" in forbidden
        assert '"' in forbidden
        assert "<" in forbidden
        assert ">" in forbidden
        assert "|" in forbidden

    def test_validate_warn_length(self, config_dir):
        """Test validate.warn_length: 150."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\\\", ":", "*", "?", '"', "<", ">", "|"]
  warn_length: 150
""")
        policy = load_naming_policy(config_dir)
        assert policy["validate"]["warn_length"] == 150


class TestRunNamesComponentOptions:
    """Test all component options for each process type."""

    def test_hpo_trial_component_options(self, config_dir):
        """Test hpo_trial component options: length, source, default, format, zero_pad."""
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
        hpo_trial = policy["run_names"]["hpo_trial"]
        
        # Check study_hash options
        assert hpo_trial["components"]["study_hash"]["length"] == 8
        assert hpo_trial["components"]["study_hash"]["source"] == "study_key_hash"
        assert hpo_trial["components"]["study_hash"]["default"] == "unknown"
        
        # Check trial_number options
        assert hpo_trial["components"]["trial_number"]["format"] == "{number}"
        assert hpo_trial["components"]["trial_number"]["zero_pad"] == 2
        assert hpo_trial["components"]["trial_number"]["source"] == "trial_number"
        assert hpo_trial["components"]["trial_number"]["default"] == "unknown"

    def test_hpo_trial_fold_component_options(self, config_dir):
        """Test hpo_trial_fold component options including fold_idx."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial_fold:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}_fold{fold_idx}"
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
      fold_idx:
        format: "{number}"
        source: "fold_idx"
        default: "0"
""")
        policy = load_naming_policy(config_dir)
        hpo_trial_fold = policy["run_names"]["hpo_trial_fold"]
        
        # Check fold_idx options
        assert hpo_trial_fold["components"]["fold_idx"]["format"] == "{number}"
        assert hpo_trial_fold["components"]["fold_idx"]["source"] == "fold_idx"
        assert hpo_trial_fold["components"]["fold_idx"]["default"] == "0"

    def test_hpo_refit_component_options(self, config_dir):
        """Test hpo_refit component options including trial_hash."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_refit:
    pattern: "{env}_{model}_hpo_refit_study-{study_hash}_trial-{trial_hash}_t{trial_number}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_hash:
        length: 8
        source: "trial_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
""")
        policy = load_naming_policy(config_dir)
        hpo_refit = policy["run_names"]["hpo_refit"]
        
        # Check trial_hash options
        assert hpo_refit["components"]["trial_hash"]["length"] == 8
        assert hpo_refit["components"]["trial_hash"]["source"] == "trial_key_hash"
        assert hpo_refit["components"]["trial_hash"]["default"] == "unknown"

    def test_hpo_sweep_semantic_suffix_options(self, config_dir):
        """Test hpo_sweep semantic_suffix options: enabled, max_length, source, default."""
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
        hpo_sweep = policy["run_names"]["hpo_sweep"]
        
        # Check semantic_suffix options
        assert hpo_sweep["components"]["semantic_suffix"]["enabled"] is True
        assert hpo_sweep["components"]["semantic_suffix"]["max_length"] == 30
        assert hpo_sweep["components"]["semantic_suffix"]["source"] == "study_name"
        assert hpo_sweep["components"]["semantic_suffix"]["default"] == ""

    def test_final_training_component_options(self, config_dir):
        """Test final_training component options: spec_hash, exec_hash, variant."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  final_training:
    pattern: "{env}_{model}_final_training_spec-{spec_hash}_exec-{exec_hash}_v{variant}{version}"
    components:
      spec_hash:
        length: 8
        source: "spec_fp"
        default: "unknown"
      exec_hash:
        length: 8
        source: "exec_fp"
        default: "unknown"
      variant:
        format: "{number}"
        source: "variant"
        default: "1"
""")
        policy = load_naming_policy(config_dir)
        final_training = policy["run_names"]["final_training"]
        
        # Check spec_hash options
        assert final_training["components"]["spec_hash"]["length"] == 8
        assert final_training["components"]["spec_hash"]["source"] == "spec_fp"
        assert final_training["components"]["spec_hash"]["default"] == "unknown"
        
        # Check exec_hash options
        assert final_training["components"]["exec_hash"]["length"] == 8
        assert final_training["components"]["exec_hash"]["source"] == "exec_fp"
        assert final_training["components"]["exec_hash"]["default"] == "unknown"
        
        # Check variant options
        assert final_training["components"]["variant"]["format"] == "{number}"
        assert final_training["components"]["variant"]["source"] == "variant"
        assert final_training["components"]["variant"]["default"] == "1"

    def test_benchmarking_component_options(self, config_dir):
        """Test benchmarking component options: study_hash, trial_hash, bench_hash."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  benchmarking:
    pattern: "{env}_{model}_benchmark_study-{study_hash}_trial-{trial_hash}_bench-{bench_hash}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_hash:
        length: 8
        source: "trial_key_hash"
        default: "unknown"
      bench_hash:
        length: 8
        source: "benchmark_config_hash"
        default: "unknown"
""")
        policy = load_naming_policy(config_dir)
        benchmarking = policy["run_names"]["benchmarking"]
        
        # Check bench_hash options
        assert benchmarking["components"]["bench_hash"]["length"] == 8
        assert benchmarking["components"]["bench_hash"]["source"] == "benchmark_config_hash"
        assert benchmarking["components"]["bench_hash"]["default"] == "unknown"

    def test_conversion_component_options(self, config_dir):
        """Test conversion component options: spec_hash, exec_hash, variant, conv_hash."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  conversion:
    pattern: "{env}_{model}_conversion_spec-{spec_hash}_exec-{exec_hash}_v{variant}_conv-{conv_hash}{version}"
    components:
      spec_hash:
        length: 8
        source: "parent_training_id"
        default: "unknown"
      exec_hash:
        length: 8
        source: "parent_training_id"
        default: "unknown"
      variant:
        format: "{number}"
        source: "parent_training_id"
        default: "1"
      conv_hash:
        length: 8
        source: "conv_fp"
        default: "unknown"
""")
        policy = load_naming_policy(config_dir)
        conversion = policy["run_names"]["conversion"]
        
        # Check conv_hash options
        assert conversion["components"]["conv_hash"]["length"] == 8
        assert conversion["components"]["conv_hash"]["source"] == "conv_fp"
        assert conversion["components"]["conv_hash"]["default"] == "unknown"
        
        # Check that parent_training_id is used as source for spec_hash, exec_hash, variant
        assert conversion["components"]["spec_hash"]["source"] == "parent_training_id"
        assert conversion["components"]["exec_hash"]["source"] == "parent_training_id"
        assert conversion["components"]["variant"]["source"] == "parent_training_id"

