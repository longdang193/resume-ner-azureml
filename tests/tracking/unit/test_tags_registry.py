"""Tests for tags registry."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from infrastructure.naming.mlflow.tags_registry import (
    TagKeyError,
    TagsRegistry,
    load_tags_registry,
)


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with tags.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_tags_yaml() -> Dict[str, Any]:
    """Sample tags.yaml content."""
    return {
        "schema_version": 1,
        "grouping": {
            "study_key_hash": "code.study_key_hash",
            "trial_key_hash": "code.trial_key_hash",
            "parent_run_id": "code.parent_run_id",
        },
        "process": {
            "stage": "code.stage",
            "project": "code.project",
            "backbone": "code.backbone",
        },
        "training": {
            "trained_on_full_data": "code.trained_on_full_data",
        },
    }


def test_tags_registry_key_access(sample_tags_yaml: Dict[str, Any]):
    """Test TagsRegistry.key() accessor."""
    registry = TagsRegistry(raw=sample_tags_yaml, schema_version=1)
    
    assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
    assert registry.key("process", "stage") == "code.stage"
    assert registry.key("training", "trained_on_full_data") == "code.trained_on_full_data"


def test_tags_registry_missing_key(sample_tags_yaml: Dict[str, Any]):
    """Test TagsRegistry raises TagKeyError for missing keys."""
    registry = TagsRegistry(raw=sample_tags_yaml, schema_version=1)
    
    with pytest.raises(TagKeyError, match="Missing tag key: grouping.missing_key"):
        registry.key("grouping", "missing_key")
    
    with pytest.raises(TagKeyError, match="Missing tag key: missing_section.key"):
        registry.key("missing_section", "key")


def test_tags_registry_validation_required_keys():
    """Test TagsRegistry validates keys lazily on access (not on initialization)."""
    # TagsRegistry uses lazy validation - it only validates when keys are accessed
    # This allows creating registries with incomplete configs for testing
    incomplete_config = {
        "schema_version": 1,
        "grouping": {
            "trial_key_hash": "code.trial_key_hash",
            # Missing study_key_hash
        },
    }
    
    # Registry can be created with incomplete config (no validation on init)
    registry = TagsRegistry(raw=incomplete_config, schema_version=1)
    
    # Validation happens lazily when accessing missing keys
    with pytest.raises(TagKeyError, match="Missing tag key: grouping.study_key_hash"):
        registry.key("grouping", "study_key_hash")
    
    # But existing keys work fine
    assert registry.key("grouping", "trial_key_hash") == "code.trial_key_hash"


def test_load_tags_registry_from_file(temp_config_dir: Path, sample_tags_yaml: Dict[str, Any]):
    """Test loading registry from tags.yaml file."""
    tags_file = temp_config_dir / "tags.yaml"
    with tags_file.open("w", encoding="utf-8") as f:
        yaml.dump(sample_tags_yaml, f)
    
    registry = load_tags_registry(temp_config_dir)
    
    assert registry.schema_version == 1
    assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
    assert registry.key("process", "stage") == "code.stage"


def test_load_tags_registry_missing_file(temp_config_dir: Path):
    """Test loading registry falls back to defaults when file is missing."""
    # Don't create tags.yaml file
    registry = load_tags_registry(temp_config_dir)
    
    # Should still work with defaults
    assert registry.schema_version == 0  # Default schema version
    assert registry.key("grouping", "study_key_hash") == "code.study_key_hash"
    assert registry.key("process", "stage") == "code.stage"


def test_load_tags_registry_caching(temp_config_dir: Path, sample_tags_yaml: Dict[str, Any]):
    """Test that registry is cached."""
    tags_file = temp_config_dir / "tags.yaml"
    with tags_file.open("w", encoding="utf-8") as f:
        yaml.dump(sample_tags_yaml, f)
    
    registry1 = load_tags_registry(temp_config_dir)
    registry2 = load_tags_registry(temp_config_dir)
    
    # Should return the same instance (cached)
    assert registry1 is registry2


def test_load_tags_registry_merge_with_defaults(temp_config_dir: Path):
    """Test that loaded config merges with defaults."""
    # Create a partial config (missing some keys)
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
    
    tags_file = temp_config_dir / "tags.yaml"
    with tags_file.open("w", encoding="utf-8") as f:
        yaml.dump(partial_config, f)
    
    registry = load_tags_registry(temp_config_dir)
    
    # Overridden values
    assert registry.key("grouping", "study_key_hash") == "custom.study_key_hash"
    assert registry.key("process", "stage") == "custom.stage"
    
    # Default values (from merge)
    assert registry.key("grouping", "trial_key_hash") == "code.trial_key_hash"
    assert registry.key("process", "project") == "code.project"


def test_load_tags_registry_default_schema_version(temp_config_dir: Path):
    """Test that missing schema_version defaults to 0."""
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
    
    tags_file = temp_config_dir / "tags.yaml"
    with tags_file.open("w", encoding="utf-8") as f:
        yaml.dump(config_no_schema, f)
    
    registry = load_tags_registry(temp_config_dir)
    assert registry.schema_version == 0


def test_tags_registry_invalid_section_type():
    """Test TagsRegistry handles invalid section types."""
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


def test_tags_registry_invalid_key_type():
    """Test TagsRegistry handles invalid key value types."""
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




