"""Legacy facade for naming module (backward compatibility).

This module re-exports all functions from the naming module for backward compatibility.
New code should import directly from naming module:
    from naming import NamingContext, create_naming_context, format_run_name

Deprecation: This facade will be removed in a future release.
"""

import warnings

# Re-export all public functions from naming module
from infrastructure.naming import (
    # Context
    NamingContext,
    create_naming_context,
    build_token_values,
    # Display Policy
    load_naming_policy,
    format_run_name,
    validate_naming_policy,
    validate_run_name,
    parse_parent_training_id,
    # Experiments
    get_stage_config,
    build_aml_experiment_name,
    build_mlflow_experiment_name,
    # MLflow Config
    load_mlflow_config,
    get_naming_config,
    get_index_config,
    get_run_finder_config,
    get_auto_increment_config,
    get_tracking_config,
    # MLflow Run Keys
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
    build_counter_key,
    # MLflow Run Names
    build_mlflow_run_name,
    # MLflow Tags
    build_mlflow_tags,
    sanitize_tag_value,
    # MLflow Tags Registry
    TagKeyError,
    TagsRegistry,
    load_tags_registry,
    # MLflow HPO Keys
    build_hpo_study_key,
    build_hpo_study_key_hash,
    build_hpo_study_family_key,
    build_hpo_study_family_hash,
    build_hpo_trial_key,
    build_hpo_trial_key_hash,
    # MLflow Refit Keys
    compute_refit_protocol_fp,
)

# Note: Deprecation warnings are added via module-level warning
# Users importing from orchestration.naming will see deprecation notices
# when the module is imported. For function-level warnings, we rely on
# the functions themselves to emit warnings if needed.

__all__ = [
    # Context
    "NamingContext",
    "create_naming_context",
    "build_token_values",
    # Display Policy
    "load_naming_policy",
    "format_run_name",
    "validate_naming_policy",
    "validate_run_name",
    "parse_parent_training_id",
    # Experiments
    "get_stage_config",
    "build_aml_experiment_name",
    "build_mlflow_experiment_name",
    # MLflow Config
    "load_mlflow_config",
    "get_naming_config",
    "get_index_config",
    "get_run_finder_config",
    "get_auto_increment_config",
    "get_tracking_config",
    # MLflow Run Keys
    "build_mlflow_run_key",
    "build_mlflow_run_key_hash",
    "build_counter_key",
    # MLflow Run Names
    "build_mlflow_run_name",
    # MLflow Tags
    "build_mlflow_tags",
    "sanitize_tag_value",
    # MLflow Tags Registry
    "TagKeyError",
    "TagsRegistry",
    "load_tags_registry",
    # MLflow HPO Keys
    "build_hpo_study_key",
    "build_hpo_study_key_hash",
    "build_hpo_study_family_key",
    "build_hpo_study_family_hash",
    "build_hpo_trial_key",
    "build_hpo_trial_key_hash",
    # MLflow Refit Keys
    "compute_refit_protocol_fp",
]
