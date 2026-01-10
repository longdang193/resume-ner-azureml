"""
@meta
name: naming_context_tokens
type: utility
domain: naming
responsibility:
  - Expand NamingContext into token dictionary
  - Generate short hash forms (8 chars) for paths and names
inputs:
  - NamingContext objects
outputs:
  - Token value dictionaries
tags:
  - utility
  - naming
  - tokens
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Expand NamingContext into token dict for path and naming patterns."""

from typing import Dict

from .context import NamingContext


def build_token_values(context: NamingContext) -> Dict[str, str]:
    """
    Expand NamingContext into a dictionary of token values.

    This consolidates ALL token expansion logic (hash slicing, etc.) into a single place.
    Used by both paths/build_output_path() and naming/format_run_name().

    Args:
        context: NamingContext instance

    Returns:
        Dictionary with all token values:
        - Full values: spec_fp, exec_fp, conv_fp, study_key_hash, trial_key_hash, benchmark_config_hash
        - Short forms (8 chars): spec8, exec8, conv8, study8, trial8, bench8
        - Name-only short forms: spec_hash, exec_hash, conv_hash, study_hash, trial_hash, bench_hash
        - Other fields: environment, storage_env, model, variant, trial_id, parent_training_id, etc.
    """
    values: Dict[str, str] = {
        # Basic fields
        "environment": context.environment,
        "storage_env": getattr(context, "storage_env", context.environment),
        "model": context.model,
        "variant": str(context.variant),
        "trial_id": context.trial_id or "",
        "parent_training_id": context.parent_training_id or "",
        # Full fingerprints/hashes
        "spec_fp": context.spec_fp or "",
        "exec_fp": context.exec_fp or "",
        "conv_fp": context.conv_fp or "",
        "study_key_hash": context.study_key_hash or "",
        "trial_key_hash": context.trial_key_hash or "",
        "benchmark_config_hash": context.benchmark_config_hash or "",
    }

    # Short fingerprint helpers (8 characters) - for paths and names
    values["spec8"] = (context.spec_fp or "")[:8] if context.spec_fp else ""
    values["exec8"] = (context.exec_fp or "")[:8] if context.exec_fp else ""
    values["conv8"] = (context.conv_fp or "")[:8] if context.conv_fp else ""
    values["study8"] = (context.study_key_hash or "")[:8] if context.study_key_hash else ""
    values["trial8"] = (context.trial_key_hash or "")[:8] if context.trial_key_hash else ""
    values["bench8"] = (context.benchmark_config_hash or "")[:8] if context.benchmark_config_hash else ""

    # Name-only short forms (for display names, not paths)
    # These are typically 8 chars but can be shorter if the full hash is short
    spec_hash_full = context.spec_fp or ""
    exec_hash_full = context.exec_fp or ""
    conv_hash_full = context.conv_fp or ""
    
    values["spec_hash"] = spec_hash_full[:8] if len(spec_hash_full) > 8 else spec_hash_full
    values["exec_hash"] = exec_hash_full[:8] if len(exec_hash_full) > 8 else exec_hash_full
    values["conv_hash"] = conv_hash_full[:8] if len(conv_hash_full) > 8 else conv_hash_full
    values["study_hash"] = context.study_key_hash or ""
    values["trial_hash"] = context.trial_key_hash or ""
    values["bench_hash"] = context.benchmark_config_hash or ""

    # Additional fields that might be used in patterns
    if context.stage:
        values["stage"] = context.stage
    if context.study_name:
        values["study_name"] = context.study_name
    if context.trial_number is not None:
        values["trial_number"] = str(context.trial_number)
    if context.fold_idx is not None:
        values["fold_idx"] = str(context.fold_idx)

    return values

