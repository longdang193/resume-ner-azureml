from __future__ import annotations

"""
@meta
name: naming_mlflow_tags
type: utility
domain: naming
responsibility:
  - Build MLflow tags from naming contexts
  - Construct tag dictionaries for MLflow runs
inputs:
  - Naming contexts
  - Configuration directories
outputs:
  - MLflow tag dictionaries
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

"""MLflow tag construction utilities."""
import os
from pathlib import Path
from typing import Dict, Optional

from common.shared.platform_detection import detect_platform
from ..context import NamingContext
from .config import get_naming_config
from .tags_registry import (
    TagKeyError,
    load_tags_registry,
)

# ---------------------------------------------------------------------------
# Centralized tag keys
# ---------------------------------------------------------------------------
CODE_STAGE = "code.stage"
CODE_MODEL = "code.model"
CODE_ENV = "code.env"  # legacy; prefer CODE_STORAGE_ENV / CODE_EXECUTION_PLATFORM
CODE_STORAGE_ENV = "code.storage_env"
CODE_EXECUTION_PLATFORM = "code.execution_platform"
CODE_CREATED_BY = "code.created_by"
CODE_PROJECT = "code.project"
CODE_SPEC_FP = "code.spec_fp"
CODE_EXEC_FP = "code.exec_fp"
CODE_VARIANT = "code.variant"
CODE_TRIAL_ID = "code.trial_id"  # legacy; prefer study/trial hashes
CODE_PARENT_TRAINING_ID = "code.parent_training_id"
CODE_CONV_FP = "code.conv_fp"
CODE_STUDY_KEY_HASH = "code.study_key_hash"
CODE_STUDY_FAMILY_HASH = "code.study_family_hash"
CODE_TRIAL_KEY_HASH = "code.trial_key_hash"
CODE_HPO_TRIAL_NUMBER = "code.hpo.trial_number"
CODE_BENCHMARK_CONFIG_HASH = "code.benchmark_config_hash"
CODE_OUTPUT_DIR = "code.output_dir"
CODE_PARENT_RUN_ID = "code.parent_run_id"
CODE_GROUP_ID = "code.group_id"
CODE_GROUPING_SCHEMA_VERSION = "code.grouping_schema_version"
CODE_REFIT_PROTOCOL_FP = "code.refit_protocol_fp"
CODE_RUN_KEY_HASH = "code.run_key_hash"
CODE_RUN_ID_PREFIX = "code.run_id_prefix"
CODE_LINEAGE_PARENT_TRAINING_RUN_ID = "code.lineage.parent_training_run_id"
CODE_LINEAGE_HPO_REFIT_RUN_ID = "code.lineage.hpo_refit_run_id"

def get_tag_key(
    section: str,
    name: str,
    config_dir: Optional[Path] = None,
    fallback: Optional[str] = None,
) -> str:
    """
    Get tag key from registry with fallback to constant.
    
    Tries to load tag key from registry (config/tags.yaml). If registry is unavailable
    or key is missing, falls back to the provided fallback value or raises an error.
    
    Args:
        section: Section name (e.g., "grouping", "process", "training")
        name: Key name within the section
        config_dir: Optional config directory for loading registry
        fallback: Optional fallback value if registry fails
        
    Returns:
        Tag key string (e.g., "code.study_key_hash")
        
    Raises:
        TagKeyError: If key is missing and no fallback provided
    """
    try:
        registry = load_tags_registry(config_dir)
        return registry.key(section, name)
    except TagKeyError:
        if fallback is not None:
            return fallback
        raise
    except Exception:
        # For other exceptions (e.g., file not found), use fallback if available
        if fallback is not None:
            return fallback
        # Re-raise the original exception
        raise

def sanitize_tag_value(
    value: str,
    max_length: int = 250,
    config_dir: Optional[Path] = None,
) -> str:
    """
    Sanitize tag value to fit MLflow tag length limits.

    Args:
        value: Tag value string.
        max_length: Maximum tag length (default: 250, MLflow limit).
        config_dir: Optional config directory for loading naming config.

    Returns:
        Sanitized tag value (truncated if needed).
    """
    if not value:
        return ""

    if len(value) <= max_length:
        return value

    # Truncate and add indicator
    truncated = value[:max_length - 3]
    return f"{truncated}..."

def build_mlflow_tags(
    context: Optional[NamingContext] = None,
    output_dir: Optional[Path] = None,
    parent_run_id: Optional[str] = None,
    group_id: Optional[str] = None,
    project_name: Optional[str] = None,
    config_dir: Optional[Path] = None,
    study_key_hash: Optional[str] = None,
    study_family_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
    refit_protocol_fp: Optional[str] = None,
    run_key_hash: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build code.* tags dictionary for MLflow runs.

    Always sets minimal tags (even without context):
    - code.stage: Process type or "unknown"
    - code.model: Model name or "unknown"
    - code.env: Environment or detected
    - code.created_by: User/system identifier
    - code.project: Project name or default

    Args:
        context: Optional NamingContext (provides process_type, model, environment).
        output_dir: Optional output directory path.
        parent_run_id: Optional parent run ID.
        group_id: Optional group ID.
        project_name: Optional project name (overrides config).
        config_dir: Optional config directory.
        study_key_hash: Optional study key hash (for grouping tags).
        study_family_hash: Optional study family hash (for grouping tags).
        trial_key_hash: Optional trial key hash (for grouping tags).
        refit_protocol_fp: Optional refit protocol fingerprint.
        run_key_hash: Optional run key hash (for cleanup and run finding).

    Returns:
        Dictionary of MLflow tags.
    """
    # Try to load registry, but don't fail if it's unavailable
    try:
        registry = load_tags_registry(config_dir)
        use_registry = True
    except Exception:
        use_registry = False

    def _get_key(section: str, name: str, fallback: str) -> str:
        """Helper to get tag key from registry or fallback."""
        if use_registry:
            try:
                return registry.key(section, name)
            except TagKeyError:
                return fallback
        return fallback

    tags = {}
    sanitize_tags = True
    tag_max_length = 250

    # Get tag keys (from registry or fallback to constants)
    TAG_STAGE = _get_key("process", "stage", CODE_STAGE)
    TAG_MODEL = _get_key("process", "model", CODE_MODEL)
    TAG_ENV = _get_key("process", "env", CODE_ENV)
    TAG_STORAGE_ENV = _get_key("process", "storage_env", CODE_STORAGE_ENV)
    TAG_EXECUTION_PLATFORM = _get_key("process", "execution_platform", CODE_EXECUTION_PLATFORM)
    TAG_CREATED_BY = _get_key("process", "created_by", CODE_CREATED_BY)
    TAG_PROJECT = _get_key("process", "project", CODE_PROJECT)
    TAG_RUN_TYPE = _get_key("process", "run_type", "code.run_type")
    TAG_SPEC_FP = _get_key("paths", "spec_fp", CODE_SPEC_FP)
    TAG_EXEC_FP = _get_key("paths", "exec_fp", CODE_EXEC_FP)
    TAG_CONV_FP = _get_key("paths", "conv_fp", CODE_CONV_FP)
    TAG_OUTPUT_DIR = _get_key("paths", "output_dir", CODE_OUTPUT_DIR)
    TAG_REFIT_PROTOCOL_FP = _get_key("paths", "refit_protocol_fp", CODE_REFIT_PROTOCOL_FP)
    TAG_VARIANT = _get_key("legacy", "variant", CODE_VARIANT)
    TAG_TRIAL_ID = _get_key("legacy", "trial_id", CODE_TRIAL_ID)
    TAG_PARENT_TRAINING_ID = _get_key("legacy", "parent_training_id", CODE_PARENT_TRAINING_ID)
    TAG_STUDY_KEY_HASH = _get_key("grouping", "study_key_hash", CODE_STUDY_KEY_HASH)
    TAG_STUDY_FAMILY_HASH = _get_key("grouping", "study_family_hash", CODE_STUDY_FAMILY_HASH)
    TAG_TRIAL_KEY_HASH = _get_key("grouping", "trial_key_hash", CODE_TRIAL_KEY_HASH)
    TAG_PARENT_RUN_ID = _get_key("grouping", "parent_run_id", CODE_PARENT_RUN_ID)
    TAG_GROUP_ID = _get_key("grouping", "group_id", CODE_GROUP_ID)
    TAG_GROUPING_SCHEMA_VERSION = _get_key("grouping", "grouping_schema_version", CODE_GROUPING_SCHEMA_VERSION)
    TAG_RUN_KEY_HASH = _get_key("grouping", "run_key_hash", CODE_RUN_KEY_HASH)
    TAG_HPO_TRIAL_NUMBER = _get_key("hpo", "trial_number", CODE_HPO_TRIAL_NUMBER)
    TAG_BENCHMARK_CONFIG_HASH = _get_key("legacy", "benchmark_config_hash", CODE_BENCHMARK_CONFIG_HASH)

    # Always set minimal tags
    if context:
        if sanitize_tags:
            # For hpo_refit, set code.stage to "hpo_refit" and code.run_type to "refit"
            if context.process_type == "hpo_refit":
                tags[TAG_STAGE] = sanitize_tag_value(
                    "hpo_refit", max_length=tag_max_length, config_dir=config_dir)
                tags[TAG_RUN_TYPE] = sanitize_tag_value(
                    "refit", max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[TAG_STAGE] = sanitize_tag_value(
                    context.process_type, max_length=tag_max_length, config_dir=config_dir)
            tags[TAG_MODEL] = sanitize_tag_value(
                context.model, max_length=tag_max_length, config_dir=config_dir)
            # Prefer storage_env; keep legacy env for backward compatibility
            storage_env = getattr(context, "storage_env", context.environment)
            tags[TAG_STORAGE_ENV] = sanitize_tag_value(
                storage_env, max_length=tag_max_length, config_dir=config_dir)
            tags[TAG_ENV] = sanitize_tag_value(
                context.environment, max_length=tag_max_length, config_dir=config_dir)
            tags[TAG_EXECUTION_PLATFORM] = sanitize_tag_value(
                context.environment, max_length=tag_max_length, config_dir=config_dir)
        else:
            # For hpo_refit, set code.stage to "hpo_refit" and code.run_type to "refit"
            if context.process_type == "hpo_refit":
                tags[TAG_STAGE] = "hpo_refit"
                tags[TAG_RUN_TYPE] = "refit"
            else:
                tags[TAG_STAGE] = context.process_type
            tags[TAG_MODEL] = context.model
            tags[TAG_ENV] = context.environment
    else:
        env = detect_platform()
        if sanitize_tags:
            tags[TAG_STAGE] = "unknown"
            tags[TAG_MODEL] = "unknown"
            tags[TAG_ENV] = sanitize_tag_value(
                env, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_STAGE] = "unknown"
            tags[TAG_MODEL] = "unknown"
            tags[TAG_ENV] = env

    # Created by (user or system)
    created_by = os.environ.get("USER", os.environ.get("USERNAME", "system"))
    if sanitize_tags:
        tags[TAG_CREATED_BY] = sanitize_tag_value(
            created_by, max_length=tag_max_length, config_dir=config_dir)
    else:
        tags[TAG_CREATED_BY] = created_by

    # Project name
    if project_name:
        if sanitize_tags:
            tags[TAG_PROJECT] = sanitize_tag_value(
                project_name, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_PROJECT] = project_name
    else:
        naming_config = get_naming_config(config_dir)
        default_project = naming_config.get("project_name", "resume-ner")
        if sanitize_tags:
            tags[TAG_PROJECT] = sanitize_tag_value(
                default_project, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_PROJECT] = default_project

    # Optional context fields
    if context:
        if context.spec_fp:
            if sanitize_tags:
                tags[TAG_SPEC_FP] = sanitize_tag_value(
                    context.spec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[TAG_SPEC_FP] = context.spec_fp
        if context.exec_fp:
            if sanitize_tags:
                tags[TAG_EXEC_FP] = sanitize_tag_value(
                    context.exec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[TAG_EXEC_FP] = context.exec_fp
        if context.variant:
            tags[TAG_VARIANT] = str(context.variant)
            if context.trial_id:
                if sanitize_tags:
                    tags[TAG_TRIAL_ID] = sanitize_tag_value(
                        context.trial_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[TAG_TRIAL_ID] = context.trial_id
            if context.parent_training_id:
                if sanitize_tags:
                    tags[TAG_PARENT_TRAINING_ID] = sanitize_tag_value(
                        context.parent_training_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[TAG_PARENT_TRAINING_ID] = context.parent_training_id
            if context.conv_fp:
                if sanitize_tags:
                    tags[TAG_CONV_FP] = sanitize_tag_value(
                        context.conv_fp, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[TAG_CONV_FP] = context.conv_fp

    if output_dir:
        if sanitize_tags:
            tags[TAG_OUTPUT_DIR] = sanitize_tag_value(
                str(output_dir), max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_OUTPUT_DIR] = str(output_dir)

    if parent_run_id:
        if sanitize_tags:
            tags[TAG_PARENT_RUN_ID] = sanitize_tag_value(
                parent_run_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_PARENT_RUN_ID] = parent_run_id

    if group_id:
        if sanitize_tags:
            tags[TAG_GROUP_ID] = sanitize_tag_value(
                group_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_GROUP_ID] = group_id

    # Grouping tags (always set schema version if any grouping tag is present)
    # This allows safe interpretation of hash meanings even if only some hashes are present
    has_grouping_tags = study_key_hash or study_family_hash or trial_key_hash
    if has_grouping_tags:
        tags[TAG_GROUPING_SCHEMA_VERSION] = "1"

    if study_key_hash:
        tags[TAG_STUDY_KEY_HASH] = study_key_hash
    if study_family_hash:
        tags[TAG_STUDY_FAMILY_HASH] = study_family_hash
    if trial_key_hash:
        tags[TAG_TRIAL_KEY_HASH] = trial_key_hash

    # HPO trial number (explicit Optuna trial number for human readability)
    if context and context.process_type == "hpo":
        trial_number = getattr(context, "trial_number", None)
        if trial_number is not None:
            tags[TAG_HPO_TRIAL_NUMBER] = str(int(trial_number))

    # Refit protocol fingerprint
    if refit_protocol_fp:
        tags[TAG_REFIT_PROTOCOL_FP] = refit_protocol_fp

    # Run key hash (for cleanup and run finding)
    if run_key_hash:
        tags[TAG_RUN_KEY_HASH] = run_key_hash

    # Benchmark config hash (for benchmarking processes)
    if context and hasattr(context, "benchmark_config_hash") and context.benchmark_config_hash:
        if sanitize_tags:
            tags[TAG_BENCHMARK_CONFIG_HASH] = sanitize_tag_value(
                context.benchmark_config_hash, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[TAG_BENCHMARK_CONFIG_HASH] = context.benchmark_config_hash

    return tags

