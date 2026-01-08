"""Comprehensive unit tests for MLflow tags."""

import yaml
from pathlib import Path
import pytest

from orchestration.jobs.tracking.naming.tags_registry import (
    TagKeyError,
    TagsRegistry,
    load_tags_registry,
)
from orchestration.jobs.tracking.naming.tags import (
    get_tag_key,
    sanitize_tag_value,
    build_mlflow_tags,
)
from orchestration.naming_centralized import NamingContext


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_tags_yaml():
    """Sample tags.yaml content."""
    return {
        "schema_version": 1,
        "grouping": {
            "study_key_hash": "code.study_key_hash",
            "trial_key_hash": "code.trial_key_hash",
            "parent_run_id": "code.parent_run_id",
            "study_family_hash": "code.study_family_hash",
            "run_key_hash": "code.run_key_hash",
            "group_id": "code.group_id",
            "grouping_schema_version": "code.grouping_schema_version",
        },
        "process": {
            "stage": "code.stage",
            "project": "code.project",
            "backbone": "code.backbone",
            "model": "code.model",
            "run_type": "code.run_type",
            "env": "code.env",
            "storage_env": "code.storage_env",
            "execution_platform": "code.execution_platform",
            "created_by": "code.created_by",
        },
        "training": {
            "trained_on_full_data": "code.trained_on_full_data",
            "source_training_run": "source_training_run",
            "refit": "code.refit",
            "refit_has_validation": "code.refit_has_validation",
            "interrupted": "code.interrupted",
        },
        "hpo": {
            "trial_number": "code.hpo.trial_number",
            "best_trial_run_id": "best_trial_run_id",
            "best_trial_number": "best_trial_number",
            "refit_planned": "code.refit_planned",
        },
        "lineage": {
            "source": "code.lineage.source",
            "hpo_study_key_hash": "code.lineage.hpo_study_key_hash",
            "hpo_trial_key_hash": "code.lineage.hpo_trial_key_hash",
            "hpo_trial_run_id": "code.lineage.hpo_trial_run_id",
            "hpo_refit_run_id": "code.lineage.hpo_refit_run_id",
            "hpo_sweep_run_id": "code.lineage.hpo_sweep_run_id",
            "parent_training_run_id": "code.lineage.parent_training_run_id",
        },
        "paths": {
            "spec_fp": "code.spec_fp",
            "exec_fp": "code.exec_fp",
            "conv_fp": "code.conv_fp",
            "output_dir": "code.output_dir",
            "refit_protocol_fp": "code.refit_protocol_fp",
        },
        "azureml": {
            "run_type": "azureml.runType",
            "sweep": "azureml.sweep",
        },
        "mlflow": {
            "run_type": "mlflow.runType",
            "parent_run_id": "mlflow.parentRunId",
        },
        "legacy": {
            "trial_number": "trial_number",
            "trial_id": "code.trial_id",
            "parent_training_id": "code.parent_training_id",
            "variant": "code.variant",
            "benchmark_config_hash": "code.benchmark_config_hash",
            "run_id_prefix": "code.run_id_prefix",
        },
    }


class TestTagsRegistry:
    """Test tags registry (4.1)."""

    def test_load_tags_registry_from_file(self, config_dir, sample_tags_yaml):
        """Test loading registry from tags.yaml file."""
        tags_file = config_dir / "tags.yaml"
        with tags_file.open("w", encoding="utf-8") as f:
            yaml.dump(sample_tags_yaml, f)
        
        registry = load_tags_registry(config_dir)
        
        assert registry.schema_version == 1
        assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
        assert registry.key("process", "stage") == "code.stage"
    
    def test_load_tags_registry_fallback_to_defaults(self, config_dir):
        """Test falls back to defaults when file missing."""
        registry = load_tags_registry(config_dir)
        
        # Should still work with defaults
        assert registry.schema_version == 0  # Default schema version
        assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
        assert registry.key("process", "stage") == "code.stage"
    
    def test_load_tags_registry_merges_with_defaults(self, config_dir):
        """Test merges loaded data with defaults (loaded takes precedence)."""
        partial_config = {
            "schema_version": 1,
            "grouping": {
                "study_key_hash": "custom.study_key_hash",  # Override default
                # Missing trial_key_hash - should use default
            },
            "process": {
                "stage": "custom.stage",  # Override default
            },
        }
        
        tags_file = config_dir / "tags.yaml"
        with tags_file.open("w", encoding="utf-8") as f:
            yaml.dump(partial_config, f)
        
        registry = load_tags_registry(config_dir)
        
        # Overridden values
        assert registry.key("grouping", "study_key_hash") == "custom.study_key_hash"
        assert registry.key("process", "stage") == "custom.stage"
        
        # Default values (from merge)
        assert registry.key("grouping", "trial_key_hash") == "code.trial_key_hash"
        assert registry.key("process", "project") == "code.project"
    
    def test_load_tags_registry_module_level_caching(self, config_dir, sample_tags_yaml):
        """Test module-level caching (returns same instance)."""
        tags_file = config_dir / "tags.yaml"
        with tags_file.open("w", encoding="utf-8") as f:
            yaml.dump(sample_tags_yaml, f)
        
        registry1 = load_tags_registry(config_dir)
        registry2 = load_tags_registry(config_dir)
        
        # Should return the same instance (cached)
        assert registry1 is registry2
    
    def test_load_tags_registry_schema_version_defaults_to_0(self, config_dir):
        """Test schema version handling (defaults to 0 if missing)."""
        config_no_schema = {
            "grouping": {
                "study_key_hash": "code.study_key_hash",
                "trial_key_hash": "code.trial_key_hash",
            },
            "process": {
                "stage": "code.stage",
                "project": "code.project",
            },
        }
        
        tags_file = config_dir / "tags.yaml"
        with tags_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_no_schema, f)
        
        registry = load_tags_registry(config_dir)
        assert registry.schema_version == 0
    
    def test_tags_registry_key_access_all_sections(self, sample_tags_yaml):
        """Test access all tag keys from all sections."""
        registry = TagsRegistry(raw=sample_tags_yaml, schema_version=1)
        
        # Grouping
        assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
        assert registry.key("grouping", "trial_key_hash") == "code.trial_key_hash"
        assert registry.key("grouping", "parent_run_id") == "code.parent_run_id"
        assert registry.key("grouping", "study_family_hash") == "code.study_family_hash"
        assert registry.key("grouping", "run_key_hash") == "code.run_key_hash"
        assert registry.key("grouping", "group_id") == "code.group_id"
        assert registry.key("grouping", "grouping_schema_version") == "code.grouping_schema_version"
        
        # Process
        assert registry.key("process", "stage") == "code.stage"
        assert registry.key("process", "project") == "code.project"
        assert registry.key("process", "backbone") == "code.backbone"
        assert registry.key("process", "model") == "code.model"
        assert registry.key("process", "run_type") == "code.run_type"
        assert registry.key("process", "env") == "code.env"
        assert registry.key("process", "storage_env") == "code.storage_env"
        assert registry.key("process", "execution_platform") == "code.execution_platform"
        assert registry.key("process", "created_by") == "code.created_by"
        
        # Training
        assert registry.key("training", "trained_on_full_data") == "code.trained_on_full_data"
        assert registry.key("training", "source_training_run") == "source_training_run"
        assert registry.key("training", "refit") == "code.refit"
        assert registry.key("training", "refit_has_validation") == "code.refit_has_validation"
        assert registry.key("training", "interrupted") == "code.interrupted"
        
        # HPO
        assert registry.key("hpo", "trial_number") == "code.hpo.trial_number"
        assert registry.key("hpo", "best_trial_run_id") == "best_trial_run_id"
        assert registry.key("hpo", "best_trial_number") == "best_trial_number"
        assert registry.key("hpo", "refit_planned") == "code.refit_planned"
        
        # Lineage
        assert registry.key("lineage", "source") == "code.lineage.source"
        assert registry.key("lineage", "hpo_study_key_hash") == "code.lineage.hpo_study_key_hash"
        assert registry.key("lineage", "hpo_trial_key_hash") == "code.lineage.hpo_trial_key_hash"
        assert registry.key("lineage", "hpo_trial_run_id") == "code.lineage.hpo_trial_run_id"
        assert registry.key("lineage", "hpo_refit_run_id") == "code.lineage.hpo_refit_run_id"
        assert registry.key("lineage", "hpo_sweep_run_id") == "code.lineage.hpo_sweep_run_id"
        assert registry.key("lineage", "parent_training_run_id") == "code.lineage.parent_training_run_id"
        
        # Paths
        assert registry.key("paths", "spec_fp") == "code.spec_fp"
        assert registry.key("paths", "exec_fp") == "code.exec_fp"
        assert registry.key("paths", "conv_fp") == "code.conv_fp"
        assert registry.key("paths", "output_dir") == "code.output_dir"
        assert registry.key("paths", "refit_protocol_fp") == "code.refit_protocol_fp"
        
        # Azure ML
        assert registry.key("azureml", "run_type") == "azureml.runType"
        assert registry.key("azureml", "sweep") == "azureml.sweep"
        
        # MLflow
        assert registry.key("mlflow", "run_type") == "mlflow.runType"
        assert registry.key("mlflow", "parent_run_id") == "mlflow.parentRunId"
        
        # Legacy
        assert registry.key("legacy", "trial_number") == "trial_number"
        assert registry.key("legacy", "trial_id") == "code.trial_id"
        assert registry.key("legacy", "parent_training_id") == "code.parent_training_id"
        assert registry.key("legacy", "variant") == "code.variant"
        assert registry.key("legacy", "benchmark_config_hash") == "code.benchmark_config_hash"
        assert registry.key("legacy", "run_id_prefix") == "code.run_id_prefix"
    
    def test_tags_registry_raises_tagkeyerror_for_missing_keys(self, sample_tags_yaml):
        """Test raises TagKeyError for missing keys."""
        registry = TagsRegistry(raw=sample_tags_yaml, schema_version=1)
        
        with pytest.raises(TagKeyError, match="Missing tag key: grouping.missing_key"):
            registry.key("grouping", "missing_key")
        
        with pytest.raises(TagKeyError, match="Missing tag key: missing_section.key"):
            registry.key("missing_section", "key")
    
    def test_tags_registry_handles_invalid_section_types(self):
        """Test handles invalid section types."""
        invalid_config = {
            "schema_version": 1,
            "grouping": "not_a_dict",  # Should be a dict
            "process": {
                "stage": "code.stage",
                "project": "code.project",
            },
        }
        
        registry = TagsRegistry(raw=invalid_config, schema_version=1)
        
        with pytest.raises(TagKeyError, match="Section 'grouping' is not a dictionary"):
            registry.key("grouping", "study_key_hash")
    
    def test_tags_registry_handles_invalid_key_value_types(self):
        """Test handles invalid key value types."""
        invalid_config = {
            "schema_version": 1,
            "grouping": {
                "study_key_hash": 123,  # Should be a string
            },
            "process": {
                "stage": "code.stage",
                "project": "code.project",
            },
        }
        
        registry = TagsRegistry(raw=invalid_config, schema_version=1)
        
        with pytest.raises(TagKeyError, match="Tag key 'grouping.study_key_hash' is not a string"):
            registry.key("grouping", "study_key_hash")


class TestTagBuilding:
    """Test tag building (4.2)."""

    def test_build_mlflow_tags_minimal_tags(self, config_dir):
        """Test minimal tags (always set)."""
        tags = build_mlflow_tags(config_dir=config_dir)
        
        assert "code.stage" in tags
        assert tags["code.stage"] == "unknown"
        assert "code.model" in tags
        assert tags["code.model"] == "unknown"
        assert "code.env" in tags
        assert "code.created_by" in tags
        assert "code.project" in tags
    
    def test_build_mlflow_tags_hpo_process(self, config_dir):
        """Test HPO process tags."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            trial_number=5
        )
        
        tags = build_mlflow_tags(
            context=context,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            config_dir=config_dir
        )
        
        assert tags["code.stage"] == "hpo"
        assert tags["code.model"] == "distilbert"
        assert tags["code.hpo.trial_number"] == "5"
        assert tags["code.study_key_hash"] == "350a79aa1234567890abcdef"
        assert tags["code.trial_key_hash"] == "747428f2abcdef1234567890"
        assert tags["code.grouping_schema_version"] == "1"
    
    def test_build_mlflow_tags_hpo_refit_process(self, config_dir):
        """Test HPO refit process tags."""
        context = NamingContext(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            storage_env="local"
        )
        
        tags = build_mlflow_tags(context=context, config_dir=config_dir)
        
        assert tags["code.stage"] == "hpo_refit"
        assert tags["code.run_type"] == "refit"
    
    def test_build_mlflow_tags_benchmarking_process(self, config_dir):
        """Test benchmarking process tags."""
        context = NamingContext(
            process_type="benchmarking",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            benchmark_config_hash="abc12345abcdef1234567890"
        )
        
        tags = build_mlflow_tags(
            context=context,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            config_dir=config_dir
        )
        
        assert tags["code.stage"] == "benchmarking"
        assert tags["code.study_key_hash"] == "350a79aa1234567890abcdef"
        assert tags["code.trial_key_hash"] == "747428f2abcdef1234567890"
        assert tags["code.benchmark_config_hash"] == "abc12345abcdef1234567890"
        assert tags["code.grouping_schema_version"] == "1"
    
    def test_build_mlflow_tags_final_training_process(self, config_dir):
        """Test final training process tags."""
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        output_dir = Path("/outputs/final_training")
        tags = build_mlflow_tags(
            context=context,
            output_dir=output_dir,
            config_dir=config_dir
        )
        
        assert tags["code.stage"] == "final_training"
        assert tags["code.spec_fp"] == "abc123def4567890"
        assert tags["code.exec_fp"] == "xyz789abc1234567"
        assert tags["code.variant"] == "1"
        assert tags["code.output_dir"] == str(output_dir)
    
    def test_build_mlflow_tags_conversion_process(self, config_dir):
        """Test conversion process tags."""
        context = NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local",
            storage_env="local",
            parent_training_id="spec-abc12345_exec-xyz789ab/v1",
            conv_fp="conv1234567890123"
        )
        
        tags = build_mlflow_tags(
            context=context,
            parent_run_id="parent_run_123",
            config_dir=config_dir
        )
        
        assert tags["code.stage"] == "conversion"
        assert tags["code.parent_training_id"] == "spec-abc12345_exec-xyz789ab/v1"
        assert tags["code.conv_fp"] == "conv1234567890123"
        assert tags["code.parent_run_id"] == "parent_run_123"
    
    def test_build_mlflow_tags_optional_tags(self, config_dir):
        """Test optional tags (set when provided)."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local"
        )
        
        tags = build_mlflow_tags(
            context=context,
            parent_run_id="parent_123",
            group_id="group_456",
            refit_protocol_fp="refit_protocol_fp",
            run_key_hash="run_key_hash",
            config_dir=config_dir
        )
        
        assert tags["code.parent_run_id"] == "parent_123"
        assert tags["code.group_id"] == "group_456"
        assert tags["code.refit_protocol_fp"] == "refit_protocol_fp"
        assert tags["code.run_key_hash"] == "run_key_hash"


class TestTagSanitization:
    """Test tag sanitization (4.3)."""

    def test_sanitize_tag_value_truncates_exceeding_max_length(self, config_dir):
        """Test truncates values exceeding max_length (default 250)."""
        long_value = "a" * 300
        sanitized = sanitize_tag_value(long_value, max_length=250, config_dir=config_dir)
        
        assert len(sanitized) == 250
        assert sanitized.endswith("...")
    
    def test_sanitize_tag_value_adds_indicator_when_truncated(self, config_dir):
        """Test adds '...' indicator when truncated."""
        long_value = "a" * 300
        sanitized = sanitize_tag_value(long_value, max_length=250, config_dir=config_dir)
        
        assert sanitized.endswith("...")
        assert len(sanitized) == 250
    
    def test_sanitize_tag_value_preserves_values_within_limit(self, config_dir):
        """Test preserves values within limit."""
        short_value = "short_value"
        sanitized = sanitize_tag_value(short_value, max_length=250, config_dir=config_dir)
        
        assert sanitized == short_value
        assert len(sanitized) == len(short_value)
    
    def test_sanitize_tag_value_handles_empty_strings(self, config_dir):
        """Test handles empty strings."""
        sanitized = sanitize_tag_value("", max_length=250, config_dir=config_dir)
        assert sanitized == ""
    
    def test_sanitize_tag_value_uses_max_length_from_config(self, config_dir):
        """Test uses max_length from config (mlflow.yaml)."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  tags:
    max_length: 100
""")
        
        long_value = "a" * 150
        sanitized = sanitize_tag_value(long_value, max_length=100, config_dir=config_dir)
        
        assert len(sanitized) == 100
        assert sanitized.endswith("...")


class TestTagKeyResolution:
    """Test tag key resolution (4.4)."""

    def test_get_tag_key_loads_from_registry(self, config_dir):
        """Test loads from registry (tags.yaml)."""
        tags_yaml = config_dir / "tags.yaml"
        tags_yaml.write_text("""
grouping:
  study_key_hash: "code.study_key_hash"
process:
  stage: "code.stage"
""")
        
        key = get_tag_key("grouping", "study_key_hash", config_dir=config_dir)
        assert key == "code.study_key_hash"
    
    def test_get_tag_key_falls_back_to_fallback(self, config_dir):
        """Test falls back to provided fallback constant."""
        # Don't create tags.yaml
        key = get_tag_key(
            "grouping", "study_key_hash",
            config_dir=config_dir,
            fallback="code.study_key_hash"
        )
        assert key == "code.study_key_hash"
    
    def test_get_tag_key_raises_when_missing_and_no_fallback(self, config_dir):
        """Test raises TagKeyError when key missing and no fallback."""
        # Don't create tags.yaml
        with pytest.raises(TagKeyError):
            get_tag_key("grouping", "study_key_hash", config_dir=config_dir, fallback=None)
    
    def test_get_tag_key_handles_registry_loading_failures(self, config_dir):
        """Test handles registry loading failures gracefully."""
        # Create invalid tags.yaml
        tags_yaml = config_dir / "tags.yaml"
        tags_yaml.write_text("invalid: yaml: content: [")
        
        # Should fall back to fallback if provided
        key = get_tag_key(
            "grouping", "study_key_hash",
            config_dir=config_dir,
            fallback="code.study_key_hash"
        )
        assert key == "code.study_key_hash"

