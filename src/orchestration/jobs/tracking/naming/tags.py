"""MLflow tag construction utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from shared.platform_detection import detect_platform
from orchestration.naming_centralized import NamingContext
from orchestration.jobs.tracking.config.loader import get_naming_config

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
    tags = {}
    sanitize_tags = True
    tag_max_length = 250

    # Always set minimal tags
    if context:
        if sanitize_tags:
            # For hpo_refit, set code.stage to "hpo_refit" and code.run_type to "refit"
            if context.process_type == "hpo_refit":
                tags[CODE_STAGE] = sanitize_tag_value(
                    "hpo_refit", max_length=tag_max_length, config_dir=config_dir)
                tags["code.run_type"] = sanitize_tag_value(
                    "refit", max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[CODE_STAGE] = sanitize_tag_value(
                    context.process_type, max_length=tag_max_length, config_dir=config_dir)
            tags[CODE_MODEL] = sanitize_tag_value(
                context.model, max_length=tag_max_length, config_dir=config_dir)
            # Prefer storage_env; keep legacy env for backward compatibility
            storage_env = getattr(context, "storage_env", context.environment)
            tags[CODE_STORAGE_ENV] = sanitize_tag_value(
                storage_env, max_length=tag_max_length, config_dir=config_dir)
            tags[CODE_ENV] = sanitize_tag_value(
                context.environment, max_length=tag_max_length, config_dir=config_dir)
            tags[CODE_EXECUTION_PLATFORM] = sanitize_tag_value(
                context.environment, max_length=tag_max_length, config_dir=config_dir)
        else:
            # For hpo_refit, set code.stage to "hpo_refit" and code.run_type to "refit"
            if context.process_type == "hpo_refit":
                tags["code.stage"] = "hpo_refit"
                tags["code.run_type"] = "refit"
            else:
                tags["code.stage"] = context.process_type
            tags["code.model"] = context.model
            tags["code.env"] = context.environment
    else:
        env = detect_platform()
        if sanitize_tags:
            tags["code.stage"] = "unknown"
            tags["code.model"] = "unknown"
            tags["code.env"] = sanitize_tag_value(
                env, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.stage"] = "unknown"
            tags["code.model"] = "unknown"
            tags["code.env"] = env

    # Created by (user or system)
    created_by = os.environ.get("USER", os.environ.get("USERNAME", "system"))
    if sanitize_tags:
        tags[CODE_CREATED_BY] = sanitize_tag_value(
            created_by, max_length=tag_max_length, config_dir=config_dir)
    else:
        tags[CODE_CREATED_BY] = created_by

    # Project name
    if project_name:
        if sanitize_tags:
            tags[CODE_PROJECT] = sanitize_tag_value(
                project_name, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[CODE_PROJECT] = project_name
    else:
        naming_config = get_naming_config(config_dir)
        default_project = naming_config.get("project_name", "resume-ner")
        if sanitize_tags:
            tags[CODE_PROJECT] = sanitize_tag_value(
                default_project, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[CODE_PROJECT] = default_project

    # Optional context fields
    if context:
        if context.spec_fp:
            if sanitize_tags:
                tags[CODE_SPEC_FP] = sanitize_tag_value(
                    context.spec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[CODE_SPEC_FP] = context.spec_fp
        if context.exec_fp:
            if sanitize_tags:
                tags[CODE_EXEC_FP] = sanitize_tag_value(
                    context.exec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags[CODE_EXEC_FP] = context.exec_fp
        if context.variant:
            tags[CODE_VARIANT] = str(context.variant)
            if context.trial_id:
                if sanitize_tags:
                    tags[CODE_TRIAL_ID] = sanitize_tag_value(
                        context.trial_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[CODE_TRIAL_ID] = context.trial_id
            if context.parent_training_id:
                if sanitize_tags:
                    tags[CODE_PARENT_TRAINING_ID] = sanitize_tag_value(
                        context.parent_training_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[CODE_PARENT_TRAINING_ID] = context.parent_training_id
            if context.conv_fp:
                if sanitize_tags:
                    tags[CODE_CONV_FP] = sanitize_tag_value(
                        context.conv_fp, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags[CODE_CONV_FP] = context.conv_fp

    if output_dir:
        if sanitize_tags:
            tags[CODE_OUTPUT_DIR] = sanitize_tag_value(
                str(output_dir), max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[CODE_OUTPUT_DIR] = str(output_dir)

    if parent_run_id:
        if sanitize_tags:
            tags[CODE_PARENT_RUN_ID] = sanitize_tag_value(
                parent_run_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[CODE_PARENT_RUN_ID] = parent_run_id

    if group_id:
        if sanitize_tags:
            tags[CODE_GROUP_ID] = sanitize_tag_value(
                group_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags[CODE_GROUP_ID] = group_id

    # Grouping tags (always set schema version if any grouping tag is present)
    # This allows safe interpretation of hash meanings even if only some hashes are present
    has_grouping_tags = study_key_hash or study_family_hash or trial_key_hash
    if has_grouping_tags:
        tags[CODE_GROUPING_SCHEMA_VERSION] = "1"

    if study_key_hash:
        tags[CODE_STUDY_KEY_HASH] = study_key_hash
    if study_family_hash:
        tags[CODE_STUDY_FAMILY_HASH] = study_family_hash
    if trial_key_hash:
        tags[CODE_TRIAL_KEY_HASH] = trial_key_hash

    # Refit protocol fingerprint
    if refit_protocol_fp:
        tags[CODE_REFIT_PROTOCOL_FP] = refit_protocol_fp

    # Run key hash (for cleanup and run finding)
    if run_key_hash:
        tags[CODE_RUN_KEY_HASH] = run_key_hash

    return tags

