"""Filesystem path management (single authority)."""

from paths.cache import (
    get_cache_file_path,
    get_cache_strategy_config,
    get_timestamped_cache_filename,
    load_cache_file,
    save_cache_with_dual_strategy,
)
from paths.config import (
    apply_env_overrides,
    load_paths_config,
    validate_paths_config,
)
from paths.drive import (
    get_drive_backup_base,
    get_drive_backup_path,
    resolve_output_path_for_colab,
)
from paths.parse import (
    find_study_by_hash,
    find_trial_by_hash,
    is_v2_path,
    parse_hpo_path_v2,
)
from paths.resolve import (
    PROCESS_PATTERN_KEYS,
    build_output_path,
    resolve_output_path,
)
from paths.utils import find_project_root
from paths.validation import (
    validate_output_path,
    validate_path_before_mkdir,
)

__all__ = [
    # Config
    "load_paths_config",
    "apply_env_overrides",
    "validate_paths_config",
    # Resolve
    "PROCESS_PATTERN_KEYS",
    "resolve_output_path",
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
    "resolve_output_path_for_colab",
    # Parse
    "parse_hpo_path_v2",
    "is_v2_path",
    "find_study_by_hash",
    "find_trial_by_hash",
    # Utils
    "find_project_root",
]

