from __future__ import annotations

"""
@meta
name: naming_mlflow_run_keys
type: utility
domain: naming
responsibility:
  - Build stable run_key identifiers from contexts
  - Compute run key hashes
inputs:
  - Naming contexts
outputs:
  - Run key strings and hashes
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

"""Run key building and hashing utilities."""
import hashlib

from ..context import NamingContext

def build_mlflow_run_key(context: NamingContext) -> str:
    """
    Build stable run_key identifier from context using stage-specific templates.

    Templates:
    - HPO trial: "hpo:{model}:{trial_id}"
    - HPO parent: "hpo:{model}:study_{study_key_hash|study_unknown}"
    - Benchmarking: "benchmark:{model}:{trial_id}"
    - Final Training: "final_training:{model}:spec_{spec_fp}:exec_{exec_fp}:v{variant}"
    - Conversion: "conversion:{model}:{parent_training_id}:conv_{conv_fp}"

    Args:
        context: NamingContext with all required information.

    Returns:
        Canonical run_key string.

    Raises:
        ValueError: If required fields are missing for the process type.
    """
    if context.process_type == "hpo":
        # Trial-level HPO run: identified by concrete trial_id
        if context.trial_id:
            return f"hpo:{context.model}:{context.trial_id}"

        # Parent/sweep HPO run: use stable study identity when available
        study_hash = getattr(context, "study_key_hash", None)
        if study_hash:
            return f"hpo:{context.model}:study_{study_hash}"

        # Last resort: still return a key so auto-increment can function,
        # but mark identity as unknown for debugging.
        return f"hpo:{context.model}:study_unknown"

    elif context.process_type == "hpo_refit":
        if not context.trial_id:
            raise ValueError("HPO refit requires trial_id for run_key")
        return f"hpo_refit:{context.model}:{context.trial_id}"

    elif context.process_type == "benchmarking":
        if not context.trial_id:
            raise ValueError("Benchmarking requires trial_id for run_key")
        return f"benchmark:{context.model}:{context.trial_id}"

    elif context.process_type == "final_training":
        if not context.spec_fp or not context.exec_fp:
            raise ValueError(
                "Final training requires spec_fp and exec_fp for run_key")
        return f"final_training:{context.model}:spec_{context.spec_fp}:exec_{context.exec_fp}:v{context.variant}"

    elif context.process_type == "conversion":
        if not context.parent_training_id or not context.conv_fp:
            raise ValueError(
                "Conversion requires parent_training_id and conv_fp for run_key")
        return f"conversion:{context.model}:{context.parent_training_id}:conv_{context.conv_fp}"

    else:
        # Fallback for unknown process types
        return f"{context.process_type}:{context.model}:unknown"

def build_mlflow_run_key_hash(run_key: str) -> str:
    """
    Build SHA256 hash of run_key for tag storage.

    MLflow tags have length limits (typically 250 chars), so we hash
    long run_keys to ensure they fit in tags.

    Args:
        run_key: Canonical run_key string.

    Returns:
        SHA256 hash hex string (always 64 characters).
    """
    return hashlib.sha256(run_key.encode('utf-8')).hexdigest()

def build_counter_key(
    project_name: str,
    process_type: str,
    run_key_hash: str,
    environment: str,
) -> str:
    """
    Build counter key for auto-increment versioning.

    Format: "{project}:{process_type}:{run_key_hash}:{env}"

    Args:
        project_name: Project name (e.g., "resume-ner").
        process_type: Process type (e.g., "hpo", "benchmarking").
        run_key_hash: Hash of run_key (64 hex chars).
        environment: Environment name (e.g., "local", "kaggle").

    Returns:
        Counter key string.
    """
    return f"{project_name}:{process_type}:{run_key_hash}:{environment}"

