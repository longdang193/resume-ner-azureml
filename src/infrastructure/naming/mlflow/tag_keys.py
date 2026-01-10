from __future__ import annotations

"""
@meta
name: naming_mlflow_tag_keys
type: utility
domain: naming
responsibility:
  - Provide centralized tag key definitions
  - Map tag sections and names to config values
inputs:
  - Configuration directories
outputs:
  - Tag key tuples
tags:
  - utility
  - naming
  - mlflow
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Centralized tag key definitions and helpers.

This module provides a single source of truth for all tag key mappings,
eliminating duplication of hardcoded fallback strings throughout the codebase.

All tag key values come from config/tags.yaml via TagsRegistry.
This module only defines the structure (section, name) mappings.
"""
from pathlib import Path
from typing import Optional, Tuple

from .tags import get_tag_key
from .tags_registry import (
    TagKeyError,
    load_tags_registry,
)

# ---------------------------------------------------------------------------
# Tag Key Mappings: (section, name) tuples
# These map to entries in config/tags.yaml
# ---------------------------------------------------------------------------

# Grouping tags
GROUPING_STUDY_KEY_HASH: Tuple[str, str] = ("grouping", "study_key_hash")
GROUPING_TRIAL_KEY_HASH: Tuple[str, str] = ("grouping", "trial_key_hash")
GROUPING_PARENT_RUN_ID: Tuple[str, str] = ("grouping", "parent_run_id")
GROUPING_STUDY_FAMILY_HASH: Tuple[str, str] = ("grouping", "study_family_hash")
GROUPING_RUN_KEY_HASH: Tuple[str, str] = ("grouping", "run_key_hash")
GROUPING_GROUP_ID: Tuple[str, str] = ("grouping", "group_id")
GROUPING_GROUPING_SCHEMA_VERSION: Tuple[str, str] = ("grouping", "grouping_schema_version")

# Process tags
PROCESS_STAGE: Tuple[str, str] = ("process", "stage")
PROCESS_PROJECT: Tuple[str, str] = ("process", "project")
PROCESS_BACKBONE: Tuple[str, str] = ("process", "backbone")
PROCESS_MODEL: Tuple[str, str] = ("process", "model")
PROCESS_RUN_TYPE: Tuple[str, str] = ("process", "run_type")
PROCESS_ENV: Tuple[str, str] = ("process", "env")
PROCESS_STORAGE_ENV: Tuple[str, str] = ("process", "storage_env")
PROCESS_EXECUTION_PLATFORM: Tuple[str, str] = ("process", "execution_platform")
PROCESS_CREATED_BY: Tuple[str, str] = ("process", "created_by")

# Training tags
TRAINING_TRAINED_ON_FULL_DATA: Tuple[str, str] = ("training", "trained_on_full_data")
TRAINING_SOURCE_TRAINING_RUN: Tuple[str, str] = ("training", "source_training_run")
TRAINING_REFIT: Tuple[str, str] = ("training", "refit")
TRAINING_REFIT_HAS_VALIDATION: Tuple[str, str] = ("training", "refit_has_validation")
TRAINING_INTERRUPTED: Tuple[str, str] = ("training", "interrupted")

# HPO tags
HPO_TRIAL_NUMBER: Tuple[str, str] = ("hpo", "trial_number")
HPO_BEST_TRIAL_RUN_ID: Tuple[str, str] = ("hpo", "best_trial_run_id")
HPO_BEST_TRIAL_NUMBER: Tuple[str, str] = ("hpo", "best_trial_number")
HPO_REFIT_PLANNED: Tuple[str, str] = ("hpo", "refit_planned")

# Lineage tags
LINEAGE_SOURCE: Tuple[str, str] = ("lineage", "source")
LINEAGE_HPO_STUDY_KEY_HASH: Tuple[str, str] = ("lineage", "hpo_study_key_hash")
LINEAGE_HPO_TRIAL_KEY_HASH: Tuple[str, str] = ("lineage", "hpo_trial_key_hash")
LINEAGE_HPO_TRIAL_RUN_ID: Tuple[str, str] = ("lineage", "hpo_trial_run_id")
LINEAGE_HPO_REFIT_RUN_ID: Tuple[str, str] = ("lineage", "hpo_refit_run_id")
LINEAGE_HPO_SWEEP_RUN_ID: Tuple[str, str] = ("lineage", "hpo_sweep_run_id")
LINEAGE_PARENT_TRAINING_RUN_ID: Tuple[str, str] = ("lineage", "parent_training_run_id")

# Path tags
PATHS_SPEC_FP: Tuple[str, str] = ("paths", "spec_fp")
PATHS_EXEC_FP: Tuple[str, str] = ("paths", "exec_fp")
PATHS_CONV_FP: Tuple[str, str] = ("paths", "conv_fp")
PATHS_OUTPUT_DIR: Tuple[str, str] = ("paths", "output_dir")
PATHS_REFIT_PROTOCOL_FP: Tuple[str, str] = ("paths", "refit_protocol_fp")

# Azure ML tags
AZUREML_RUN_TYPE: Tuple[str, str] = ("azureml", "run_type")
AZUREML_SWEEP: Tuple[str, str] = ("azureml", "sweep")

# MLflow tags
MLFLOW_RUN_TYPE: Tuple[str, str] = ("mlflow", "run_type")
MLFLOW_PARENT_RUN_ID: Tuple[str, str] = ("mlflow", "parent_run_id")

# Legacy tags
LEGACY_TRIAL_NUMBER: Tuple[str, str] = ("legacy", "trial_number")
LEGACY_TRIAL_ID: Tuple[str, str] = ("legacy", "trial_id")
LEGACY_PARENT_TRAINING_ID: Tuple[str, str] = ("legacy", "parent_training_id")
LEGACY_VARIANT: Tuple[str, str] = ("legacy", "variant")
LEGACY_BENCHMARK_CONFIG_HASH: Tuple[str, str] = ("legacy", "benchmark_config_hash")
LEGACY_RUN_ID_PREFIX: Tuple[str, str] = ("legacy", "run_id_prefix")

# ---------------------------------------------------------------------------
# Helper Functions - Direct access to commonly used tags
# ---------------------------------------------------------------------------

def _get_tag_key_from_mapping(
    mapping: Tuple[str, str], config_dir: Optional[Path] = None
) -> str:
    """
    Get tag key from registry using a mapping tuple.
    
    Args:
        mapping: Tuple of (section, name)
        config_dir: Optional config directory
        
    Returns:
        Tag key string from registry
        
    Raises:
        TagKeyError: If key is missing from registry
    """
    section, name = mapping
    registry = load_tags_registry(config_dir)
    return registry.key(section, name)

# Grouping tags
def get_study_key_hash(config_dir: Optional[Path] = None) -> str:
    """Get study key hash tag key."""
    return _get_tag_key_from_mapping(GROUPING_STUDY_KEY_HASH, config_dir)

def get_trial_key_hash(config_dir: Optional[Path] = None) -> str:
    """Get trial key hash tag key."""
    return _get_tag_key_from_mapping(GROUPING_TRIAL_KEY_HASH, config_dir)

def get_parent_run_id(config_dir: Optional[Path] = None) -> str:
    """Get parent run ID tag key."""
    return _get_tag_key_from_mapping(GROUPING_PARENT_RUN_ID, config_dir)

def get_run_key_hash(config_dir: Optional[Path] = None) -> str:
    """Get run key hash tag key."""
    return _get_tag_key_from_mapping(GROUPING_RUN_KEY_HASH, config_dir)

def get_group_id(config_dir: Optional[Path] = None) -> str:
    """Get group ID tag key."""
    return _get_tag_key_from_mapping(GROUPING_GROUP_ID, config_dir)

# Training tags
def get_trained_on_full_data(config_dir: Optional[Path] = None) -> str:
    """Get trained on full data tag key."""
    return _get_tag_key_from_mapping(TRAINING_TRAINED_ON_FULL_DATA, config_dir)

def get_source_training_run(config_dir: Optional[Path] = None) -> str:
    """Get source training run tag key."""
    return _get_tag_key_from_mapping(TRAINING_SOURCE_TRAINING_RUN, config_dir)

def get_refit(config_dir: Optional[Path] = None) -> str:
    """Get refit tag key."""
    return _get_tag_key_from_mapping(TRAINING_REFIT, config_dir)

def get_refit_has_validation(config_dir: Optional[Path] = None) -> str:
    """Get refit has validation tag key."""
    return _get_tag_key_from_mapping(TRAINING_REFIT_HAS_VALIDATION, config_dir)

def get_interrupted(config_dir: Optional[Path] = None) -> str:
    """Get interrupted tag key."""
    return _get_tag_key_from_mapping(TRAINING_INTERRUPTED, config_dir)

# HPO tags
def get_hpo_trial_number(config_dir: Optional[Path] = None) -> str:
    """Get HPO trial number tag key."""
    return _get_tag_key_from_mapping(HPO_TRIAL_NUMBER, config_dir)

def get_hpo_best_trial_run_id(config_dir: Optional[Path] = None) -> str:
    """Get HPO best trial run ID tag key."""
    return _get_tag_key_from_mapping(HPO_BEST_TRIAL_RUN_ID, config_dir)

def get_hpo_best_trial_number(config_dir: Optional[Path] = None) -> str:
    """Get HPO best trial number tag key."""
    return _get_tag_key_from_mapping(HPO_BEST_TRIAL_NUMBER, config_dir)

def get_hpo_refit_planned(config_dir: Optional[Path] = None) -> str:
    """Get HPO refit planned tag key."""
    return _get_tag_key_from_mapping(HPO_REFIT_PLANNED, config_dir)

# Lineage tags
def get_lineage_source(config_dir: Optional[Path] = None) -> str:
    """Get lineage source tag key."""
    return _get_tag_key_from_mapping(LINEAGE_SOURCE, config_dir)

def get_lineage_hpo_study_key_hash(config_dir: Optional[Path] = None) -> str:
    """Get lineage HPO study key hash tag key."""
    return _get_tag_key_from_mapping(LINEAGE_HPO_STUDY_KEY_HASH, config_dir)

def get_lineage_hpo_trial_key_hash(config_dir: Optional[Path] = None) -> str:
    """Get lineage HPO trial key hash tag key."""
    return _get_tag_key_from_mapping(LINEAGE_HPO_TRIAL_KEY_HASH, config_dir)

def get_lineage_hpo_trial_run_id(config_dir: Optional[Path] = None) -> str:
    """Get lineage HPO trial run ID tag key."""
    return _get_tag_key_from_mapping(LINEAGE_HPO_TRIAL_RUN_ID, config_dir)

def get_lineage_hpo_refit_run_id(config_dir: Optional[Path] = None) -> str:
    """Get lineage HPO refit run ID tag key."""
    return _get_tag_key_from_mapping(LINEAGE_HPO_REFIT_RUN_ID, config_dir)

def get_lineage_hpo_sweep_run_id(config_dir: Optional[Path] = None) -> str:
    """Get lineage HPO sweep run ID tag key."""
    return _get_tag_key_from_mapping(LINEAGE_HPO_SWEEP_RUN_ID, config_dir)

def get_lineage_parent_training_run_id(config_dir: Optional[Path] = None) -> str:
    """Get lineage parent training run ID tag key."""
    return _get_tag_key_from_mapping(LINEAGE_PARENT_TRAINING_RUN_ID, config_dir)

# MLflow tags
def get_mlflow_run_type(config_dir: Optional[Path] = None) -> str:
    """Get MLflow run type tag key."""
    return _get_tag_key_from_mapping(MLFLOW_RUN_TYPE, config_dir)

def get_mlflow_parent_run_id(config_dir: Optional[Path] = None) -> str:
    """Get MLflow parent run ID tag key."""
    return _get_tag_key_from_mapping(MLFLOW_PARENT_RUN_ID, config_dir)

# Azure ML tags
def get_azureml_run_type(config_dir: Optional[Path] = None) -> str:
    """Get Azure ML run type tag key."""
    return _get_tag_key_from_mapping(AZUREML_RUN_TYPE, config_dir)

def get_azureml_sweep(config_dir: Optional[Path] = None) -> str:
    """Get Azure ML sweep tag key."""
    return _get_tag_key_from_mapping(AZUREML_SWEEP, config_dir)

# Process tags
def get_process_stage(config_dir: Optional[Path] = None) -> str:
    """Get process stage tag key."""
    return _get_tag_key_from_mapping(PROCESS_STAGE, config_dir)

def get_process_project(config_dir: Optional[Path] = None) -> str:
    """Get process project tag key."""
    return _get_tag_key_from_mapping(PROCESS_PROJECT, config_dir)

def get_process_backbone(config_dir: Optional[Path] = None) -> str:
    """Get process backbone tag key."""
    return _get_tag_key_from_mapping(PROCESS_BACKBONE, config_dir)

def get_process_model(config_dir: Optional[Path] = None) -> str:
    """Get process model tag key."""
    return _get_tag_key_from_mapping(PROCESS_MODEL, config_dir)

# Legacy tags
def get_legacy_trial_number(config_dir: Optional[Path] = None) -> str:
    """Get legacy trial number tag key."""
    return _get_tag_key_from_mapping(LEGACY_TRIAL_NUMBER, config_dir)

__all__ = [
    # Mappings
    "GROUPING_STUDY_KEY_HASH",
    "GROUPING_TRIAL_KEY_HASH",
    "GROUPING_PARENT_RUN_ID",
    "GROUPING_RUN_KEY_HASH",
    "GROUPING_GROUP_ID",
    "TRAINING_TRAINED_ON_FULL_DATA",
    "TRAINING_REFIT",
    "TRAINING_REFIT_HAS_VALIDATION",
    "TRAINING_INTERRUPTED",
    "HPO_TRIAL_NUMBER",
    "HPO_BEST_TRIAL_RUN_ID",
    "HPO_BEST_TRIAL_NUMBER",
    "HPO_REFIT_PLANNED",
    "LINEAGE_SOURCE",
    "LINEAGE_HPO_STUDY_KEY_HASH",
    "LINEAGE_HPO_TRIAL_KEY_HASH",
    "LINEAGE_HPO_TRIAL_RUN_ID",
    "LINEAGE_HPO_REFIT_RUN_ID",
    "LINEAGE_HPO_SWEEP_RUN_ID",
    "LINEAGE_PARENT_TRAINING_RUN_ID",
    "MLFLOW_RUN_TYPE",
    "MLFLOW_PARENT_RUN_ID",
    "AZUREML_RUN_TYPE",
    "AZUREML_SWEEP",
    # Helper functions
    "get_study_key_hash",
    "get_trial_key_hash",
    "get_parent_run_id",
    "get_run_key_hash",
    "get_group_id",
    "get_trained_on_full_data",
    "get_source_training_run",
    "get_refit",
    "get_refit_has_validation",
    "get_interrupted",
    "get_hpo_trial_number",
    "get_hpo_best_trial_run_id",
    "get_hpo_best_trial_number",
    "get_hpo_refit_planned",
    "get_lineage_source",
    "get_lineage_hpo_study_key_hash",
    "get_lineage_hpo_trial_key_hash",
    "get_lineage_hpo_trial_run_id",
    "get_lineage_hpo_refit_run_id",
    "get_lineage_hpo_sweep_run_id",
    "get_lineage_parent_training_run_id",
    "get_mlflow_run_type",
    "get_mlflow_parent_run_id",
    "get_azureml_run_type",
    "get_azureml_sweep",
    "get_process_stage",
    "get_process_project",
    "get_process_backbone",
    "get_process_model",
    "get_legacy_trial_number",
]

