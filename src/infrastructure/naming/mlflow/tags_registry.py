from __future__ import annotations

"""
@meta
name: naming_mlflow_tags_registry
type: utility
domain: naming
responsibility:
  - Manage centralized MLflow tag key registry
  - Load tag keys from config/tags.yaml
  - Provide tag key accessors with validation
inputs:
  - Configuration directories
outputs:
  - TagsRegistry instances
tags:
  - utility
  - naming
  - mlflow
  - tags
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Tags registry for centralized MLflow tag key management."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from common.shared.yaml_utils import load_yaml
from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Module-level cache for loaded registry
_registry_cache: Optional[TagsRegistry] = None
_registry_cache_path: Optional[Path] = None

class TagKeyError(KeyError):
    """Raised when a tag key is missing from the registry."""

    pass

@dataclass(frozen=True)
class TagsRegistry:
    """
    Registry for MLflow tag keys loaded from config/tags.yaml.
    
    Provides strict accessor for tag keys with validation.
    """

    raw: Dict[str, Any]
    schema_version: int

    def __post_init__(self):
        """Initialize registry (validation is lazy, done on key access)."""
        # Note: Required keys are validated lazily when accessed via self.key()
        # This allows TagsRegistry to be created with incomplete configs for testing
        # Validation happens on first access to required keys
        pass

    def key(self, section: str, name: str) -> str:
        """
        Get tag key for a given section and name.
        
        Args:
            section: Section name (e.g., "grouping", "process", "training")
            name: Key name within the section
            
        Returns:
            Tag key string (e.g., "code.study_key_hash")
            
        Raises:
            TagKeyError: If the section or key is missing or invalid
        """
        # Check if section exists
        if section not in self.raw:
            raise TagKeyError(f"Missing tag key: {section}.{name}")
        
        section_data = self.raw[section]
        if not isinstance(section_data, dict):
            raise TagKeyError(f"Section '{section}' is not a dictionary")
        
        # Check if key exists in section
        if name not in section_data:
            raise TagKeyError(f"Missing tag key: {section}.{name}")
        
        value = section_data[name]
        if not isinstance(value, str):
            raise TagKeyError(
                f"Tag key '{section}.{name}' is not a string: {type(value)}"
            )
        return value

def _get_default_tag_keys() -> Dict[str, Any]:
    """
    Get default tag keys extracted from hardcoded constants.
    
    This provides fallback values when tags.yaml is missing or incomplete.
    """
    return {
        "schema_version": 0,  # Default to 0 if not specified
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

def _deep_merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with overrides taking precedence.
    
    Args:
        defaults: Default dictionary
        overrides: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_tags_registry(config_dir: Optional[Path] = None) -> TagsRegistry:
    """
    Load tags registry from config/tags.yaml with caching and fallback to defaults.
    
    Uses module-level caching to avoid repeated file reads. If the file is missing
    or incomplete, falls back to hardcoded defaults to ensure backward compatibility.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        
    Returns:
        TagsRegistry instance with tag keys from config or defaults.
    """
    global _registry_cache, _registry_cache_path

    if config_dir is None:
        config_dir = Path.cwd() / "config"

    config_path = config_dir / "tags.yaml"

    # Check cache
    if _registry_cache is not None and _registry_cache_path == config_path:
        return _registry_cache

    # Load defaults first
    defaults = _get_default_tag_keys()

    # Try to load from file
    loaded_data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded_data = load_yaml(config_path)
        except Exception as e:
            logger.warning(
                f"[Tags Registry] Failed to load config from {config_path}: {e}. "
                f"Using defaults.",
                exc_info=True
            )
            loaded_data = {}
    else:
        logger.info(
            f"[Tags Registry] Config file not found at {config_path}, using defaults"
        )

    # Merge loaded data with defaults (loaded data takes precedence)
    merged_data = _deep_merge(defaults, loaded_data)

    # Ensure schema_version exists
    if "schema_version" not in merged_data:
        merged_data["schema_version"] = 0
        logger.debug(
            "[Tags Registry] schema_version not found in config, defaulting to 0"
        )

    schema_version = int(merged_data.get("schema_version", 0))

    # Create registry
    registry = TagsRegistry(raw=merged_data, schema_version=schema_version)

    # Update cache
    _registry_cache = registry
    _registry_cache_path = config_path

    return registry

__all__ = ["TagKeyError", "TagsRegistry", "load_tags_registry"]

