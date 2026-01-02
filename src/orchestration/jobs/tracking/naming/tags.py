"""MLflow tag construction utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from shared.platform_detection import detect_platform
from orchestration.naming_centralized import NamingContext
from orchestration.jobs.tracking.config.loader import get_naming_config


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
                tags["code.stage"] = sanitize_tag_value(
                    "hpo_refit", max_length=tag_max_length, config_dir=config_dir)
                tags["code.run_type"] = sanitize_tag_value(
                    "refit", max_length=tag_max_length, config_dir=config_dir)
            else:
                tags["code.stage"] = sanitize_tag_value(
                    context.process_type, max_length=tag_max_length, config_dir=config_dir)
            tags["code.model"] = sanitize_tag_value(
                context.model, max_length=tag_max_length, config_dir=config_dir)
            tags["code.env"] = sanitize_tag_value(
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
        tags["code.created_by"] = sanitize_tag_value(
            created_by, max_length=tag_max_length, config_dir=config_dir)
    else:
        tags["code.created_by"] = created_by

    # Project name
    if project_name:
        if sanitize_tags:
            tags["code.project"] = sanitize_tag_value(
                project_name, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.project"] = project_name
    else:
        naming_config = get_naming_config(config_dir)
        default_project = naming_config.get("project_name", "resume-ner")
        if sanitize_tags:
            tags["code.project"] = sanitize_tag_value(
                default_project, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.project"] = default_project

    # Optional context fields
    if context:
        if context.spec_fp:
            if sanitize_tags:
                tags["code.spec_fp"] = sanitize_tag_value(
                    context.spec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags["code.spec_fp"] = context.spec_fp
        if context.exec_fp:
            if sanitize_tags:
                tags["code.exec_fp"] = sanitize_tag_value(
                    context.exec_fp, max_length=tag_max_length, config_dir=config_dir)
            else:
                tags["code.exec_fp"] = context.exec_fp
        if context.variant:
            tags["code.variant"] = str(context.variant)
            if context.trial_id:
                if sanitize_tags:
                    tags["code.trial_id"] = sanitize_tag_value(
                        context.trial_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags["code.trial_id"] = context.trial_id
            if context.parent_training_id:
                if sanitize_tags:
                    tags["code.parent_training_id"] = sanitize_tag_value(
                        context.parent_training_id, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags["code.parent_training_id"] = context.parent_training_id
            if context.conv_fp:
                if sanitize_tags:
                    tags["code.conv_fp"] = sanitize_tag_value(
                        context.conv_fp, max_length=tag_max_length, config_dir=config_dir)
                else:
                    tags["code.conv_fp"] = context.conv_fp

    if output_dir:
        if sanitize_tags:
            tags["code.output_dir"] = sanitize_tag_value(
                str(output_dir), max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.output_dir"] = str(output_dir)

    if parent_run_id:
        if sanitize_tags:
            tags["code.parent_run_id"] = sanitize_tag_value(
                parent_run_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.parent_run_id"] = parent_run_id

    if group_id:
        if sanitize_tags:
            tags["code.group_id"] = sanitize_tag_value(
                group_id, max_length=tag_max_length, config_dir=config_dir)
        else:
            tags["code.group_id"] = group_id

    # Grouping tags (always set schema version if any grouping tag is present)
    # This allows safe interpretation of hash meanings even if only some hashes are present
    has_grouping_tags = study_key_hash or study_family_hash or trial_key_hash
    if has_grouping_tags:
        tags["code.grouping_schema_version"] = "1"

    if study_key_hash:
        tags["code.study_key_hash"] = study_key_hash
    if study_family_hash:
        tags["code.study_family_hash"] = study_family_hash
    if trial_key_hash:
        tags["code.trial_key_hash"] = trial_key_hash

    # Refit protocol fingerprint
    if refit_protocol_fp:
        tags["code.refit_protocol_fp"] = refit_protocol_fp

    # Run key hash (for cleanup and run finding)
    if run_key_hash:
        tags["code.run_key_hash"] = run_key_hash

    return tags

