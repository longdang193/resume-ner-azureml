"""Legacy facade for paths module (backward compatibility).

This module re-exports all functions from the paths module for backward compatibility.
New code should import directly from paths module:
    from paths import load_paths_config, build_output_path

Deprecation: This facade will be removed in a future release.
"""

import warnings
from pathlib import Path
from typing import Optional

# Re-export all public functions from infrastructure.paths module
from infrastructure.paths import (
    # Config
    apply_env_overrides,
    load_paths_config,
    validate_paths_config,
    # Resolve
    PROCESS_PATTERN_KEYS,
    build_output_path,
    resolve_output_path,
    # Validation
    validate_path_before_mkdir,
    validate_output_path,
    # Cache
    get_cache_file_path,
    get_cache_strategy_config,
    get_timestamped_cache_filename,
    load_cache_file,
    save_cache_with_dual_strategy,
    # Drive
    get_drive_backup_base,
    get_drive_backup_path,
    # Parse
    find_study_by_hash,
    find_trial_by_hash,
    is_v2_path,
    parse_hpo_path_v2,
)

# Import NamingContext for resolve_output_path_v2 wrapper
from infrastructure.naming.context import NamingContext


def resolve_output_path_v2(
    root_dir: Path,
    context: NamingContext,
    base_outputs: str = "outputs",
    config_dir: Optional[Path] = None
) -> Path:
    """
    Legacy wrapper for build_output_path (v2 path entrypoint).

    This function is deprecated. Use paths.build_output_path() directly:
        from paths import build_output_path
        path = build_output_path(root_dir, context, base_outputs, config_dir)

    Args:
        root_dir: Project root directory.
        context: NamingContext with all required information.
        base_outputs: Base outputs directory name (default: "outputs").
        config_dir: Configuration directory (default: root_dir / "config").

    Returns:
        Full path to output directory.
    """
    warnings.warn(
        "resolve_output_path_v2() is deprecated. "
        "Use 'from paths import build_output_path' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return build_output_path(root_dir, context, base_outputs, config_dir)


__all__ = [
    # Config
    "load_paths_config",
    "apply_env_overrides",
    "validate_paths_config",
    # Resolve
    "PROCESS_PATTERN_KEYS",
    "resolve_output_path",
    "resolve_output_path_v2",  # Legacy wrapper
    "build_output_path",
    # Validation
    "validate_path_before_mkdir",
    "validate_output_path",
    # Cache
    "get_cache_file_path",
    "get_timestamped_cache_filename",
    "get_cache_strategy_config",
    "save_cache_with_dual_strategy",
    "load_cache_file",
    # Drive
    "get_drive_backup_base",
    "get_drive_backup_path",
    # Parse
    "parse_hpo_path_v2",
    "is_v2_path",
    "find_study_by_hash",
    "find_trial_by_hash",
]
