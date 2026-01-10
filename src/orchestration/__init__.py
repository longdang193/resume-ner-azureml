"""Legacy orchestration module facade.

This module provides backward compatibility by re-exporting functions from
the new modular structure. All imports from this module are deprecated.

New modules:
- infrastructure.config: Configuration management
- infrastructure.fingerprints: Fingerprint computation
- infrastructure.metadata: Metadata and index management
- common.constants: Shared constants
- infrastructure.platform.azureml: Azure ML utilities
- infrastructure.storage: Storage and backup utilities
- selection: Configuration selection
- benchmarking: Benchmarking orchestration
- conversion: Conversion execution
- training_exec: Final training execution

This module will be removed in 2 releases.
"""

import warnings
from typing import Any

# Emit deprecation warning
warnings.warn(
    "orchestration module is deprecated. "
    "Use 'infrastructure', 'common', or 'data' modules instead. "
    "This will be removed in 2 releases.",
    DeprecationWarning,
    stacklevel=2
)

# Constants - moved to common.constants module
from common.constants import (
    STAGE_SMOKE,
    STAGE_HPO,
    STAGE_TRAINING,
    EXPERIMENT_NAME,
    MODEL_NAME,
    PROD_STAGE,
    CONVERSION_JOB_NAME,
    METRICS_FILENAME,
    BENCHMARK_FILENAME,
    CHECKPOINT_DIRNAME,
    OUTPUTS_DIRNAME,
    MLRUNS_DIRNAME,
    DEFAULT_RANDOM_SEED,
    DEFAULT_K_FOLDS,
)

# Paths - these remain in orchestration/paths.py (facade to paths module)
from .paths import (
    load_paths_config,
    resolve_output_path,
    resolve_output_path_v2,
    get_cache_file_path,
    get_timestamped_cache_filename,
    get_cache_strategy_config,
    save_cache_with_dual_strategy,
    load_cache_file,
    get_drive_backup_base,
    get_drive_backup_path,
)

# Storage - moved to infrastructure.storage module
from infrastructure.storage import (
    DriveBackupStore,
    BackupResult,
    BackupAction,
    EnsureLocalOptions,
    mount_colab_drive,
    create_colab_store,
)

# Naming - these remain in orchestration/naming.py (facade to naming module)
from .naming import get_stage_config, build_aml_experiment_name, build_mlflow_experiment_name

# Fingerprints - moved to infrastructure.fingerprints module
from infrastructure.fingerprints import (
    compute_spec_fp,
    compute_exec_fp,
    compute_conv_fp,
    compute_bench_fp,
    compute_hardware_fp,
)

# Naming centralized - these remain in orchestration/naming_centralized.py (facade to naming module)
from .naming_centralized import (
    NamingContext,
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)

# Metadata - moved to infrastructure.metadata module
from infrastructure.metadata import (
    update_index,
    find_by_spec_fp,
    find_by_env,
    find_by_model,
    find_by_spec_and_env,
    get_latest_entry,
)

# MLflow utils - moved to tracking/mlflow/setup.py (check if exists, otherwise keep here)
try:
    from infrastructure.tracking.mlflow.setup import setup_mlflow_for_stage
except ImportError:
    # Fallback to local implementation if not in tracking module
    from .mlflow_utils import setup_mlflow_for_stage

# Benchmarking - moved to benchmarking module
from benchmarking.utils import run_benchmarking

# Config - moved to config module
try:
    from config.training import load_final_training_config
except (ImportError, ModuleNotFoundError):
    # PyTorch or training module not available - skip this import
    load_final_training_config = None


def _deprecation_warning(name: str, new_module: str) -> None:
    """Issue deprecation warning for moved functions."""
    warnings.warn(
        f"Importing '{name}' from 'orchestration' is deprecated. "
        f"Please import from '{new_module}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Issue deprecation warnings for moved modules
_deprecation_warning("constants", "constants")
_deprecation_warning("fingerprints", "fingerprints")
_deprecation_warning("metadata/index_manager", "metadata")
_deprecation_warning("metadata/metadata_manager", "metadata")
_deprecation_warning("drive_backup", "storage")
_deprecation_warning("benchmark_utils", "benchmarking.utils")
_deprecation_warning("config_loader", "config.loader")
_deprecation_warning("conversion_config", "config.conversion")
_deprecation_warning("final_training_config", "config.training")
_deprecation_warning("environment", "config.environment")
_deprecation_warning("config_compat", "config.validation")
_deprecation_warning("data_assets", "azureml.data_assets")
_deprecation_warning("normalize", "core.normalize")
_deprecation_warning("tokens", "core.tokens")

__all__ = [
    "STAGE_SMOKE",
    "STAGE_HPO",
    "STAGE_TRAINING",
    "EXPERIMENT_NAME",
    "MODEL_NAME",
    "PROD_STAGE",
    "CONVERSION_JOB_NAME",
    "METRICS_FILENAME",
    "BENCHMARK_FILENAME",
    "CHECKPOINT_DIRNAME",
    "OUTPUTS_DIRNAME",
    "MLRUNS_DIRNAME",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_K_FOLDS",
    # Path resolution exports
    "load_paths_config",
    "resolve_output_path",
    "resolve_output_path_v2",
    "get_cache_file_path",
    "get_timestamped_cache_filename",
    "get_cache_strategy_config",
    "save_cache_with_dual_strategy",
    "load_cache_file",
    "get_drive_backup_base",
    "get_drive_backup_path",
    # Drive backup exports
    "DriveBackupStore",
    "BackupResult",
    "BackupAction",
    "EnsureLocalOptions",
    "mount_colab_drive",
    "create_colab_store",
    # Other exports
    "get_stage_config",
    "build_aml_experiment_name",
    "build_mlflow_experiment_name",
    # Fingerprints
    "compute_spec_fp",
    "compute_exec_fp",
    "compute_conv_fp",
    "compute_bench_fp",
    "compute_hardware_fp",
    # Centralized naming
    "NamingContext",
    "create_naming_context",
    "build_output_path",
    "build_parent_training_id",
    # Index management
    "update_index",
    "find_by_spec_fp",
    "find_by_env",
    "find_by_model",
    "find_by_spec_and_env",
    "get_latest_entry",
    "setup_mlflow_for_stage",
    "run_benchmarking",
    # Final training config (may be None if PyTorch not available)
    "load_final_training_config",
]
