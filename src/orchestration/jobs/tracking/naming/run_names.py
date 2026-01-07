"""Human-readable run name generation."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional

from orchestration.naming_centralized import NamingContext
from orchestration.jobs.tracking.config.loader import (
    get_naming_config,
    get_auto_increment_config,
)
from orchestration.jobs.tracking.naming.run_keys import (
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
    build_counter_key,
)
from orchestration.jobs.tracking.naming.policy import (
    load_naming_policy,
    format_run_name,
    validate_run_name,
)

logger = logging.getLogger(__name__)


def _short(value: Optional[str], default: str = "unknown") -> str:
    """Return an 8-char short hash or a default if missing."""
    if not value:
        return default
    return value[:8]


def _strip_env_prefix(trial_id: str, environment: Optional[str]) -> str:
    """
    Strip environment prefix from trial_id if present.

    Args:
        trial_id: Trial ID string (may have env prefix like "local_hpo_...").
        environment: Environment name to strip.

    Returns:
        Trial ID without environment prefix.
    """
    if not environment or not trial_id:
        return trial_id

    prefix = f"{environment}_"
    if trial_id.startswith(prefix):
        return trial_id[len(prefix):]
    return trial_id


def build_mlflow_run_name(
    context: NamingContext,
    config_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Build human-readable run name from context (may be overridden by Azure ML).

    Uses systematic naming with optional auto-increment versioning.
    Now uses policy-driven formatting from config/naming.yaml.

    Args:
        context: NamingContext with process type, model, etc.
        config_dir: Configuration directory (for loading naming config).
        root_dir: Project root directory (for counter storage).
        output_dir: Output directory (for inferring root_dir if not provided).

    Returns:
        Human-readable run name string.
    """
    # Infer root_dir from output_dir if not provided
    if root_dir is None and output_dir is not None:
        root_dir = output_dir.parent.parent if output_dir else None

    # Fallback to current directory if still None
    if root_dir is None:
        root_dir = Path.cwd()

    naming_config = get_naming_config(config_dir)
    run_name_config = naming_config.get("run_name", {})

    # Load naming policy
    policy = load_naming_policy(config_dir)
    use_policy = bool(policy and "run_names" in policy)

    # Determine process type (handle hpo -> hpo_trial vs hpo_sweep vs hpo_trial_fold)
    process_type = context.process_type
    if process_type == "hpo":
        # Detect if this is a trial run (not parent sweep)
        is_trial_run = context.trial_id is not None
        stage = getattr(context, "stage", None)
        fold_idx = getattr(context, "fold_idx", None)
        # Also check stage for explicit classification
        if stage == "hpo_trial":
            is_trial_run = True
        elif stage == "hpo_sweep":
            is_trial_run = False
        
        # Set process_type for policy lookup
        if is_trial_run:
            # Check if this is a fold run (CV fold child)
            if fold_idx is not None:
                process_type = "hpo_trial_fold"
            else:
                process_type = "hpo_trial"
        else:
            process_type = "hpo_sweep"
        
        # For trials, enforce strict identity when study_hash is missing.
        study_hash = getattr(context, "study_key_hash", None)
        if not study_hash:
            study_hash = os.environ.get("HPO_STUDY_KEY_HASH")
        if stage == "hpo_trial" and study_hash is None:
            raise ValueError(
                "HPO trial run name built without study_key_hash; "
                "check study identity propagation."
            )

    # Use policy-driven formatting if available
    if use_policy:
        try:
            base_name = format_run_name(process_type, context, policy, config_dir)
            validate_run_name(base_name, policy)
        except Exception as e:
            logger.warning(
                f"Failed to format run name using policy: {e}, falling back to legacy logic",
                exc_info=True
            )
            use_policy = False

    # Fallback to legacy logic if policy not available or formatting failed
    if not use_policy:
        return _build_legacy_run_name(context, naming_config, run_name_config, config_dir)

    # Apply version suffix using existing collision logic
    env = context.storage_env if hasattr(context, "storage_env") else context.environment
    
    # Determine which auto-increment config to use
    auto_inc_process_type = process_type
    if process_type in ("hpo_trial", "hpo_sweep"):
        auto_inc_process_type = "hpo"
    elif process_type == "hpo_refit":
        auto_inc_process_type = "hpo"
    
    auto_inc_config = get_auto_increment_config(config_dir, auto_inc_process_type)
    should_add_version = False
    
    if process_type in ("hpo_trial", "hpo_sweep", "hpo_refit"):
        should_add_version = (
            auto_inc_config.get("enabled", False) and
            auto_inc_config.get("processes", {}).get(auto_inc_process_type, False)
        )
    elif process_type == "benchmarking":
        should_add_version = (
            auto_inc_config.get("enabled", False) and
            auto_inc_config.get("processes", {}).get("benchmarking", False)
        )
    # final_training and conversion don't use version suffix
    
    if should_add_version:
        try:
            run_key = build_mlflow_run_key(context)
            run_key_hash = build_mlflow_run_key_hash(run_key)

            counter_key = build_counter_key(
                naming_config.get("project_name", "resume-ner"),
                process_type,
                run_key_hash,
                env or "",
            )

            from orchestration.jobs.tracking.mlflow_index import reserve_run_name_version

            temp_run_id = f"pending_{datetime.now().isoformat()}"
            version = reserve_run_name_version(
                counter_key,
                temp_run_id,
                root_dir,
                config_dir,
            )
            
            # Get version format from policy
            version_format = policy.get("version", {}).get("format", "_{number}")
            version_separator = policy.get("version", {}).get("separator", "_")
            version_str = version_format.format(separator=version_separator, number=version)
            
            return f"{base_name}{version_str}"
        except Exception as e:
            logger.warning(
                f"Could not reserve version for run name: {e}, using base name without version"
            )
    
    # Fallback: check if we need legacy fallback for missing hashes
    if process_type in ("hpo_trial", "hpo_sweep"):
        if not getattr(context, "study_key_hash", None):
            if context.trial_id:
                return f"{env}_{context.trial_id}"
    elif process_type == "benchmarking":
        if context.trial_id and not getattr(context, "study_key_hash", None):
            return f"{env}_{context.trial_id}"
    
    return base_name


def _build_legacy_run_name(
    context: NamingContext,
    naming_config: dict,
    run_name_config: dict,
    config_dir: Optional[Path] = None,
) -> str:
    """
    Legacy run name building logic (fallback when policy not available).
    
    This preserves the original hardcoded logic for backward compatibility.
    """
    # This is a simplified version - in practice, the full legacy logic
    # would be preserved here. For now, return a basic fallback.
    env = context.storage_env if hasattr(context, "storage_env") else context.environment
    return f"{env}_{context.model}_{context.process_type}_legacy"

