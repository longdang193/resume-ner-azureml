"""
@meta
name: paths_resolve
type: utility
domain: paths
responsibility:
  - Resolve all output paths (single authority for filesystem layout)
  - Build paths from naming contexts
  - Apply path patterns and normalization
inputs:
  - Naming contexts
  - Configuration directories
outputs:
  - Resolved output paths
tags:
  - utility
  - paths
  - filesystem
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Resolve all output paths (single authority for filesystem layout)."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.normalize import normalize_for_path
from core.placeholders import extract_placeholders
from .config import apply_env_overrides, load_paths_config
from .validation import validate_output_path

logger = logging.getLogger(__name__)

# Centralized constant: Map process_type to pattern key in paths.yaml
PROCESS_PATTERN_KEYS: Dict[str, str] = {
    "final_training": "final_training_v2",
    "conversion": "conversion_v2",
    "hpo": "hpo_v2",
    "benchmarking": "benchmarking_v2",
}


def resolve_output_path(
    root_dir: Path,
    config_dir: Path,
    category: str,
    **kwargs
) -> Path:
    """
    Resolve output path from config (legacy function for backward compatibility).

    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        category: Output category (e.g., "hpo", "final_training", "cache").
        **kwargs: Additional path components or pattern replacements.

    Returns:
        Resolved path.

    Examples:
        resolve_output_path(ROOT_DIR, CONFIG_DIR, "hpo")
        # -> outputs/hpo

        resolve_output_path(ROOT_DIR, CONFIG_DIR, "cache", subcategory="best_configurations")
        # -> outputs/cache/best_configurations

        resolve_output_path(ROOT_DIR, CONFIG_DIR, "final_training", 
                          backbone="distilbert", run_id="20251227_220407")
        # -> outputs/final_training/distilbert_20251227_220407
    """
    paths_config = load_paths_config(config_dir)
    base_outputs = paths_config["base"]["outputs"]
    base_outputs_path = Path(base_outputs)
    base_dir = base_outputs_path if base_outputs_path.is_absolute() else root_dir / \
        base_outputs

    if category == "cache" and "subcategory" in kwargs:
        # Special handling for cache subdirectories
        cache_sub = kwargs.pop("subcategory")
        cache_dir = paths_config["cache"][cache_sub]
        return base_dir / paths_config["outputs"]["cache"] / cache_dir

    output_dir = paths_config["outputs"].get(category, category)
    path = base_dir / output_dir

    # Apply pattern replacements if provided
    if kwargs:
        pattern = paths_config.get("patterns", {}).get(category, "")
        if pattern:
            for key, value in kwargs.items():
                pattern = pattern.replace(f"{{{key}}}", str(value))
            path = path / pattern
        else:
            # Append as subdirectories
            for value in kwargs.values():
                path = path / str(value)

    return path


def _get_pattern_key(process_type: str) -> Optional[str]:
    """Map process_type to paths.yaml pattern key."""
    mapping = {
        "hpo": "hpo_v2",
        "hpo_refit": "hpo_v2",  # Use same pattern as hpo, refit is subdirectory
        "benchmarking": "benchmarking_v2",
        "final_training": "final_training_v2",
        "conversion": "conversion_v2",
        "best_configurations": "best_config_v2",
    }
    return mapping.get(process_type)


def _validate_output_path_internal(path: Path, process_type: str) -> None:
    """
    Basic validation of output path (sanity check).

    Since paths are constructed from validated NamingContext and patterns,
    this is primarily a defense-in-depth check for programming errors.

    Args:
        path: Path to validate
        process_type: Process type for error messages

    Raises:
        ValueError: If path is invalid
    """
    # Basic sanity check - paths from build_output_path should never be None/empty
    if not path or not str(path):
        raise ValueError(f"Invalid {process_type} output path: {path}")

    path_str = str(path)
    if not path_str or path_str in (".", ".."):
        raise ValueError(f"Invalid {process_type} output path: {path_str}")

    # Note: Removed version number check - root cause (pip install command) is fixed.
    # Paths from build_output_path always have structure like outputs/category/.../...
    # so they will never be just a version number.


def _build_output_path_fallback(
    root_dir: Path,
    context: Any,  # NamingContext from naming module
    base_outputs: str = "outputs"
) -> Path:
    """
    Fallback path building logic (hardcoded, used when patterns not available).

    This is used when paths.yaml config is missing or when required hashes are unavailable.
    For v2 processes (hpo, benchmarking), requires study_key_hash and trial_key_hash.
    """
    base_path = root_dir / base_outputs

    if context.process_type == "hpo":
        # For v2 HPO, prefer hashes but fallback to legacy pattern if missing
        if context.study_key_hash and context.trial_key_hash:
            study8 = context.study_key_hash[:8]
            trial8 = context.trial_key_hash[:8]
            final_path = base_path / "hpo" / context.storage_env / \
                context.model / f"study-{study8}" / f"trial-{trial8}"
        else:
            # Legacy pattern: use trial_id if available, otherwise simple structure
            if context.trial_id:
                final_path = base_path / "hpo" / context.storage_env / \
                    context.model / context.trial_id
            else:
                final_path = base_path / "hpo" / context.storage_env / context.model

    elif context.process_type == "hpo_refit":
        # For v2 HPO refit, require hashes
        if not context.study_key_hash or not context.trial_key_hash:
            raise ValueError(
                f"HPO refit v2 requires study_key_hash and trial_key_hash. "
                f"Got study_key_hash={'present' if context.study_key_hash else 'missing'}, "
                f"trial_key_hash={'present' if context.trial_key_hash else 'missing'}"
            )
        study8 = context.study_key_hash[:8]
        trial8 = context.trial_key_hash[:8]
        final_path = base_path / "hpo" / context.storage_env / \
            context.model / f"study-{study8}" / f"trial-{trial8}" / "refit"

    elif context.process_type == "benchmarking":
        # For v2 benchmarking, require hashes
        if not context.study_key_hash or not context.trial_key_hash:
            raise ValueError(
                f"Benchmarking v2 requires study_key_hash and trial_key_hash. "
                f"Got study_key_hash={'present' if context.study_key_hash else 'missing'}, "
                f"trial_key_hash={'present' if context.trial_key_hash else 'missing'}"
            )
        study8 = context.study_key_hash[:8]
        trial8 = context.trial_key_hash[:8]
        bench8 = (context.benchmark_config_hash or "")[
            :8] if context.benchmark_config_hash else ""
        if bench8:
            final_path = base_path / "benchmarking" / context.storage_env / \
                context.model / f"study-{study8}" / \
                f"trial-{trial8}" / f"bench-{bench8}"
        else:
            final_path = base_path / "benchmarking" / context.storage_env / \
                context.model / f"study-{study8}" / f"trial-{trial8}"

    elif context.process_type == "final_training":
        # Format: spec_<spec_fp>_exec_<exec_fp>/v<variant>
        spec_exec_dir = f"spec_{context.spec_fp}_exec_{context.exec_fp}"
        variant_dir = f"v{context.variant}"
        final_path = base_path / "final_training" / context.storage_env / \
            context.model / spec_exec_dir / variant_dir

    elif context.process_type == "conversion":
        # Format: <parent_training_id>/conv_<conv_fp>
        conv_dir = f"conv_{context.conv_fp}"
        final_path = base_path / "conversion" / context.storage_env / \
            context.model / context.parent_training_id / conv_dir

    elif context.process_type == "best_configurations":
        # Format: <model>/spec_<spec_fp>
        spec_dir = f"spec_{context.spec_fp}"
        final_path = base_path / "cache" / "best_configurations" / context.model / spec_dir

    else:
        raise ValueError(f"Unknown process_type: {context.process_type}")

    # CRITICAL: Validate final path to prevent creating invalid files like '1.0.0'
    _validate_output_path_internal(final_path, context.process_type)

    return final_path


def build_output_path(
    root_dir: Path,
    context: Any,  # NamingContext from naming module
    base_outputs: str = "outputs",
    config_dir: Optional[Path] = None
) -> Path:
    """
    Build output path following new centralized structure (v2) - ONLY v2 entrypoint.

    Paths are generated using patterns from config/paths.yaml (v2 patterns).
    This ensures path structures are configurable and maintainable.

    Path structures (from paths.yaml patterns):
    - HPO v2:
        outputs/hpo/{storage_env}/{model}/study_{study8}/trial_{trial8}/...
    - Benchmarking v2:
        outputs/benchmarking/{storage_env}/{model}/study_{study8}/trial_{trial8}/bench_{bench8}/...
    - Final training:
        outputs/final_training/{storage_env}/{model}/spec_{spec_fp}_exec_{exec_fp}/v{variant}/
    - Conversion:
        outputs/conversion/{storage_env}/{model}/{parent_training_id}/conv_{conv_fp}/
    - Best config:
        outputs/cache/best_configurations/{model}/spec_{spec_fp}/

    Args:
        root_dir: Project root directory.
        context: Naming context with all required information.
        base_outputs: Base outputs directory name (default: "outputs", overridden by config if available).
        config_dir: Configuration directory (default: root_dir / "config").

    Returns:
        Full path to output directory.
    """
    # Determine config directory
    if config_dir is None:
        config_dir = root_dir / "config"

    # Try to load paths config
    try:
        storage_env = getattr(context, "storage_env", context.environment)
        paths_config = load_paths_config(config_dir, storage_env=storage_env)
    except Exception as e:
        # Use DEBUG level for expected fallback scenarios in tests
        # (missing pattern keys, YAML syntax errors in test fixtures)
        # Use WARNING for unexpected errors in production
        error_msg = str(e)
        is_expected_fallback = (
            "Missing required pattern keys" in error_msg or
            "schema_version" in error_msg or
            "while scanning" in error_msg or  # YAML parsing errors in test fixtures
            "could not find expected" in error_msg  # YAML parsing errors
        )
        if is_expected_fallback:
            logger.debug(
                f"Failed to load paths.yaml config: {e}. Using fallback logic.")
        else:
            logger.warning(
                f"Failed to load paths.yaml config: {e}. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Get base outputs from config (or use provided/default)
    base_outputs = paths_config.get("base", {}).get("outputs", base_outputs)
    base_outputs_path = Path(base_outputs)
    if base_outputs_path.is_absolute():
        base_path = base_outputs_path
    else:
        base_path = root_dir / base_outputs

    # Map process_type to output category
    category_map = {
        "hpo": "hpo",
        "hpo_refit": "hpo",  # Refit is part of HPO output structure
        "benchmarking": "benchmarking",
        "final_training": "final_training",
        "conversion": "conversion",
        "best_configurations": "cache",  # Special case
    }

    category = category_map.get(context.process_type)
    if category is None:
        raise ValueError(f"Unknown process_type: {context.process_type}")

    # Get pattern key
    pattern_key = _get_pattern_key(context.process_type)
    if not pattern_key:
        logger.warning(
            f"No pattern key mapping for process_type: {context.process_type}. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Get pattern from config
    patterns = paths_config.get("patterns", {})
    pattern = patterns.get(pattern_key)

    if not pattern:
        logger.warning(
            f"Pattern '{pattern_key}' not found in paths.yaml. Using fallback logic.")
        return _build_output_path_fallback(root_dir, context, base_outputs)

    # Check if v2 pattern requires hashes that are missing
    # For hpo/hpo_refit/benchmarking v2 patterns, use fallback if hashes unavailable
    if pattern_key in ("hpo_v2", "benchmarking_v2"):
        requires_study_hash = "{study8}" in pattern
        requires_trial_hash = "{trial8}" in pattern
        if requires_study_hash and not context.study_key_hash:
            logger.warning(
                f"Pattern '{pattern_key}' requires study_key_hash but it's missing. Using fallback logic.")
            return _build_output_path_fallback(root_dir, context, base_outputs)
        if requires_trial_hash and not context.trial_key_hash:
            logger.warning(
                f"Pattern '{pattern_key}' requires trial_key_hash but it's missing. Using fallback logic.")
            return _build_output_path_fallback(root_dir, context, base_outputs)

    # Extract values from context
    # NOTE: storage_env defaults to environment in NamingContext.__post_init__
    # TODO: Use naming/context_tokens.py::build_token_values() when that module is created
    values = {
        "environment": context.environment,
        "storage_env": getattr(context, "storage_env", context.environment),
        "model": context.model,
        # Full fingerprints
        "spec_fp": context.spec_fp or "",
        "exec_fp": context.exec_fp or "",
        "variant": context.variant,
        "trial_id": context.trial_id or "",
        "parent_training_id": context.parent_training_id or "",
        "conv_fp": context.conv_fp or "",
        # Short fingerprint helpers for lineage-based patterns
        "spec8": (context.spec_fp or "")[:8] if context.spec_fp else "",
        "exec8": (context.exec_fp or "")[:8] if context.exec_fp else "",
        "conv8": (context.conv_fp or "")[:8] if context.conv_fp else "",
        # Optional short forms for HPO/benchmark v2 layouts
        "study8": (context.study_key_hash or "")[:8] if context.study_key_hash else "",
        "trial8": (context.trial_key_hash or "")[:8] if context.trial_key_hash else "",
        "bench8": (context.benchmark_config_hash or "")[
            :8
        ]
        if context.benchmark_config_hash
        else "",
    }

    # Optional path normalization (configurable in paths.yaml under normalize_paths)
    path_norm_rules = paths_config.get("normalize_paths")
    if path_norm_rules:
        normalized_values = {}
        for key, value in values.items():
            normalized_value, warn_msgs = normalize_for_path(
                value, path_norm_rules)
            if warn_msgs:
                for msg in warn_msgs:
                    logger.warning(
                        f"[build_output_path] Normalized '{key}' for path: {msg}"
                    )
            normalized_values[key] = normalized_value
        values = normalized_values

    # Resolve pattern by replacing placeholders
    resolved_pattern = pattern
    for key, value in values.items():
        resolved_pattern = resolved_pattern.replace(f"{{{key}}}", str(value))

    # Build final path
    if context.process_type == "best_configurations":
        # Special handling: cache/best_configurations/{model}/spec_{spec_fp}/
        # Pattern is relative to cache/best_configurations, not outputs
        final_path = base_path / "cache" / \
            "best_configurations" / Path(resolved_pattern)
    else:
        # Get output directory from config
        output_dir = paths_config.get("outputs", {}).get(category, category)
        # Handle nested paths (e.g., "spec_abc_exec_xyz/v1")
        # Split by "/" and create path components
        pattern_parts = resolved_pattern.split("/")
        # Filter out empty parts (can occur if a placeholder resolved to empty string)
        pattern_parts = [part for part in pattern_parts if part]
        base_output_path = base_path / output_dir / Path(*pattern_parts)

        # For hpo_refit, append "refit" subdirectory
        if context.process_type == "hpo_refit":
            final_path = base_output_path / "refit"
        else:
            final_path = base_output_path

    # CRITICAL: Validate final path to prevent creating invalid files like '1.0.0'
    # Use public validation function from paths/validation.py
    try:
        validate_output_path(final_path)
    except ValueError as e:
        # Re-raise with process_type context
        raise ValueError(f"Invalid {context.process_type} output path: {e}") from e

    return final_path

