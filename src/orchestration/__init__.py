from .constants import (
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
from .drive_backup import (
    DriveBackupStore,
    BackupResult,
    BackupAction,
    EnsureLocalOptions,
    mount_colab_drive,
    create_colab_store,
)
from .naming import get_stage_config, build_aml_experiment_name, build_mlflow_experiment_name
from .fingerprints import (
    compute_spec_fp,
    compute_exec_fp,
    compute_conv_fp,
    compute_bench_fp,
    compute_hardware_fp,
)
from .naming_centralized import (
    NamingContext,
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)
from .index_manager import (
    update_index,
    find_by_spec_fp,
    find_by_env,
    find_by_model,
    find_by_spec_and_env,
    get_latest_entry,
)
from .mlflow_utils import setup_mlflow_for_stage
from .benchmark_utils import run_benchmarking

# Lazy import for final_training_config to avoid PyTorch dependency when only constants are needed
try:
    from .final_training_config import load_final_training_config
except (ImportError, ModuleNotFoundError):
    # PyTorch or training module not available - skip this import
    load_final_training_config = None

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
