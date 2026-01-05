"""Load and resolve final training configuration from YAML."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from shared.yaml_utils import load_yaml
from shared.platform_detection import detect_platform
from training.checkpoint_loader import validate_checkpoint
from orchestration.config_loader import load_all_configs, ExperimentConfig
from orchestration.naming_centralized import build_output_path, create_naming_context


def load_final_training_config(
    root_dir: Path,
    config_dir: Path,
    best_config: Dict[str, Any],
    experiment_config: ExperimentConfig,
    train_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load and resolve final training configuration from YAML.
    
    This function:
    1. Loads config/final_training.yaml
    2. Resolves dataset config (auto-detect or explicit)
    3. Computes spec_fp and exec_fp based on identity controls
    4. Resolves checkpoint path (auto-detect from fingerprints/metadata or explicit)
    5. Computes variant number (auto-increment or explicit) based on run.mode
    6. Merges with best_config and train.yaml defaults
    7. Returns resolved config dict compatible with existing code
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        best_config: Best selected configuration from Step P1-3.6 (contains backbone, hyperparameters).
                     This configuration is selected after considering both HPO and benchmarking results.
        experiment_config: Experiment configuration (contains data_config, etc.).
        train_config: Optional base training config (loads train.yaml if None).
    
    Returns:
        Resolved final training config dict with all parameters set.
    """
    # Load final_training.yaml
    final_training_yaml_path = config_dir / "final_training.yaml"
    
    if not final_training_yaml_path.exists():
        # Backward compatibility: fall back to inline config building
        warnings.warn(
            "config/final_training.yaml not found. Falling back to inline config building. "
            "This is deprecated. Please create config/final_training.yaml.",
            DeprecationWarning,
            stacklevel=2
        )
        return _build_final_training_config_inline(
            best_config, experiment_config, train_config, config_dir
        )
    
    final_training_config = load_yaml(final_training_yaml_path)
    
    # Load train_config if not provided
    if train_config is None:
        train_config = load_yaml(experiment_config.train_config)
    
    # Resolve dataset config
    data_config = _resolve_dataset_config(
        final_training_config.get("dataset", {}),
        experiment_config,
        config_dir
    )
    
    # Load all configs for fingerprint computation
    all_configs = load_all_configs(experiment_config)
    # Override data_config if resolved differently
    if data_config:
        all_configs["data"] = data_config
    
    # Resolve seed
    seed = _resolve_seed(
        final_training_config.get("seed", {}),
        train_config,
        best_config
    )
    
    # Compute fingerprints
    spec_fp, exec_fp = _compute_fingerprints(
        root_dir,
        all_configs,
        seed,
        final_training_config.get("identity", {})
    )
    
    # Resolve checkpoint based on source.type
    checkpoint_path = _resolve_checkpoint(
        root_dir,
        config_dir,
        final_training_config.get("source", {}),
        final_training_config.get("checkpoint", {}),
        spec_fp,
        exec_fp,
        best_config
    )
    
    # Compute variant based on run.mode
    run_mode = final_training_config.get("run", {}).get("mode", "reuse_if_exists")
    variant = _resolve_variant(
        root_dir,
        config_dir,
        final_training_config.get("variant", {}),
        run_mode,
        spec_fp,
        exec_fp,
        best_config.get("backbone", "unknown")
    )
    
    # Merge configs: train.yaml defaults -> best_config -> final_training.yaml overrides
    merged_config = _merge_configs(
        train_config,
        best_config,
        final_training_config,
        data_config,
        seed,
        checkpoint_path
    )
    
    # Add computed values
    merged_config["spec_fp"] = spec_fp
    merged_config["exec_fp"] = exec_fp
    merged_config["variant"] = variant
    
    # Set CHECKPOINT_PATH environment variable if checkpoint is resolved
    if checkpoint_path:
        os.environ["CHECKPOINT_PATH"] = str(checkpoint_path)
    
    return merged_config


def _resolve_dataset_config(
    dataset_config: Dict[str, Any],
    experiment_config: ExperimentConfig,
    config_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Resolve dataset configuration.
    
    Args:
        dataset_config: Dataset config from final_training.yaml.
        experiment_config: Experiment configuration.
        config_dir: Config directory.
    
    Returns:
        Resolved data config dict or None if using local_path_override.
    """
    # Check for local_path_override first
    local_path_override = dataset_config.get("local_path_override")
    if local_path_override:
        # Return None to indicate using local path directly
        return None
    
    # Check for explicit data_config
    data_config_path = dataset_config.get("data_config")
    if data_config_path:
        # Load explicit config file
        if not Path(data_config_path).is_absolute():
            data_config_path = config_dir / data_config_path
        else:
            data_config_path = Path(data_config_path)
        return load_yaml(data_config_path)
    
    # Auto-detect: use experiment_config.data_config
    return load_yaml(experiment_config.data_config)


def _resolve_seed(
    seed_config: Dict[str, Any],
    train_config: Dict[str, Any],
    best_config: Dict[str, Any]
) -> int:
    """
    Resolve random seed.
    
    Args:
        seed_config: Seed config from final_training.yaml.
        train_config: Base training config.
        best_config: Best selected configuration (from Step P1-3.6).
    
    Returns:
        Resolved random seed value.
    """
    random_seed = seed_config.get("random_seed")
    if random_seed is not None:
        return int(random_seed)
    
    # Try train_config
    train_seed = train_config.get("training", {}).get("random_seed")
    if train_seed is not None:
        return int(train_seed)
    
    # Default
    return 42


def _compute_fingerprints(
    root_dir: Path,
    all_configs: Dict[str, Any],
    seed: int,
    identity_config: Dict[str, Any]
) -> tuple[str, str]:
    """
    Compute spec_fp and exec_fp based on identity controls.
    
    Args:
        root_dir: Project root directory.
        all_configs: All loaded configs (data, model, train, env).
        seed: Random seed.
        identity_config: Identity controls from final_training.yaml.
    
    Returns:
        Tuple of (spec_fp, exec_fp).
    """
    try:
        from orchestration.fingerprints import compute_spec_fp, compute_exec_fp
        
        # Compute spec_fp (always includes: dataset, train, model, seed)
        spec_fp = compute_spec_fp(
            model_config=all_configs.get("model", {}),
            data_config=all_configs.get("data", {}),
            train_config=all_configs.get("train", {}),
            seed=seed
        )
        
        # Compute exec_fp based on identity controls
        include_code_fp = identity_config.get("include_code_fp", True)
        include_precision_fp = identity_config.get("include_precision_fp", True)
        include_determinism_fp = identity_config.get("include_determinism_fp", False)
        
        # Get git SHA if needed
        git_sha = None
        if include_code_fp:
            try:
                import subprocess
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root_dir,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_sha = None
        
        exec_fp = compute_exec_fp(
            git_sha=git_sha if include_code_fp else None,
            env_config=all_configs.get("env", {}),
            include_precision=include_precision_fp,
            include_determinism=include_determinism_fp
        )
        
        return spec_fp, exec_fp
        
    except ImportError:
        # Fingerprint functions not available - return placeholder
        warnings.warn(
            "Fingerprint computation functions not available. Using placeholder fingerprints.",
            RuntimeWarning,
            stacklevel=2
        )
        return "placeholder_spec_fp", "placeholder_exec_fp"


def _resolve_checkpoint(
    root_dir: Path,
    config_dir: Path,
    source_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
    spec_fp: str,
    exec_fp: str,
    best_config: Dict[str, Any]
) -> Optional[Path]:
    """
    Resolve checkpoint path based on source.type.
    
    Source type semantics:
    - scratch: No checkpoint, start training from scratch
    - best_selected: Use checkpoint from best selected configuration (Step P1-3.6)
                     This configuration is selected after considering both HPO and benchmarking results
    - final_training: Continue from previous final training run (specified via source.parent)
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        source_config: Source config from final_training.yaml.
        checkpoint_config: Checkpoint config from final_training.yaml.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        best_config: Best selected configuration (from Step P1-3.6, considers HPO + benchmarking).
    
    Returns:
        Resolved checkpoint path or None.
    """
    source_type = source_config.get("type", "scratch")
    
    # If scratch, no checkpoint
    if source_type == "scratch":
        return None
    
    # Auto-derive checkpoint.load from source.type if not explicitly set
    # Explicit value can override but we warn if it conflicts
    explicit_load = checkpoint_config.get("load")
    if explicit_load is None:
        # Auto-derive: True for best_selected or final_training, False for scratch
        should_load = source_type in ("best_selected", "final_training")
        checkpoint_load = should_load
    else:
        # Use explicit value, but warn if it conflicts with source.type
        checkpoint_load = explicit_load
        should_load = source_type in ("best_selected", "final_training")
        if checkpoint_load != should_load:
            import warnings
            warnings.warn(
                f"checkpoint.load={checkpoint_load} conflicts with source.type={source_type}. "
                f"Expected checkpoint.load={should_load} for source.type={source_type}. "
                f"Using explicit checkpoint.load value, but this may cause unexpected behavior.",
                UserWarning,
                stacklevel=3
            )
    
    # Check if checkpoint.load is False (either explicit or auto-derived)
    if not checkpoint_load:
        return None
    
    # Get checkpoint source
    checkpoint_source = checkpoint_config.get("source")
    
    # Handle source.type-specific logic
    if source_type == "best_selected":
        # For best_selected, checkpoint should come from the best_configuration
        # If checkpoint.source is not explicitly set, try to resolve from best_config metadata
        if checkpoint_source is None:
            # Try to get checkpoint from best_configuration metadata/cache
            checkpoint_path = _resolve_checkpoint_from_best_config(
                root_dir, config_dir, best_config
            )
            if checkpoint_path:
                return checkpoint_path
            # Fall back to fingerprint-based resolution
            return _resolve_checkpoint_from_fingerprints(
                root_dir, config_dir, spec_fp, exec_fp
            )
    elif source_type == "final_training":
        # For final_training, use source.parent if specified
        parent = source_config.get("parent")
        if parent:
            checkpoint_source = parent
    
    # Resolve checkpoint source
    if checkpoint_source is None:
        # Auto-detect from fingerprints/metadata
        return _resolve_checkpoint_from_fingerprints(
            root_dir, config_dir, spec_fp, exec_fp
        )
    elif isinstance(checkpoint_source, str):
        # Explicit path
        checkpoint_path = Path(checkpoint_source)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        
        # Validate if requested
        if checkpoint_config.get("validate", True):
            if validate_checkpoint(checkpoint_path):
                return checkpoint_path
            return None
        return checkpoint_path
    elif isinstance(checkpoint_source, dict):
        # Dict with spec_fp/exec_fp/variant
        parent_spec_fp = checkpoint_source.get("spec_fp", spec_fp)
        parent_exec_fp = checkpoint_source.get("exec_fp", exec_fp)
        parent_variant = checkpoint_source.get("variant", 1)
        
        # Build path from fingerprints
        environment = detect_platform()
        backbone = best_config.get("backbone", "unknown")
        backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
        
        from orchestration.naming_centralized import create_naming_context
        
        parent_context = create_naming_context(
            process_type="final_training",
            model=backbone_name,
            spec_fp=parent_spec_fp,
            exec_fp=parent_exec_fp,
            environment=environment,
            variant=parent_variant
        )
        
        parent_output_dir = build_output_path(root_dir, parent_context)
        checkpoint_path = parent_output_dir / "checkpoint"
        
        if checkpoint_config.get("validate", True):
            if validate_checkpoint(checkpoint_path):
                return checkpoint_path
            return None
        return checkpoint_path
    
    # If checkpoint_source is still None at this point, try auto-detection
    if checkpoint_source is None:
        return _resolve_checkpoint_from_fingerprints(
            root_dir, config_dir, spec_fp, exec_fp
        )
    
    return None


def _resolve_checkpoint_from_best_config(
    root_dir: Path,
    config_dir: Path,
    best_config: Dict[str, Any]
) -> Optional[Path]:
    """
    Resolve checkpoint path from best selected configuration.
    
    The best configuration (from Step P1-3.6) may contain checkpoint information
    in its metadata or cache. This function attempts to locate the checkpoint
    associated with the best selected configuration.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        best_config: Best selected configuration dict (from Step P1-3.6).
    
    Returns:
        Resolved checkpoint path or None if not found.
    """
    # Try to get checkpoint path from best_config metadata
    checkpoint_path_str = best_config.get("checkpoint_path")
    if checkpoint_path_str:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.is_absolute():
            checkpoint_path = checkpoint_path.resolve()
        else:
            checkpoint_path = (root_dir / checkpoint_path).resolve()
        
        if validate_checkpoint(checkpoint_path):
            return checkpoint_path
    
    # Try to get from output_dir in best_config
    output_dir_str = best_config.get("output_dir")
    if output_dir_str:
        output_dir = Path(output_dir_str)
        if not output_dir.is_absolute():
            output_dir = root_dir / output_dir
        checkpoint_path = output_dir.resolve() / "checkpoint"
        if validate_checkpoint(checkpoint_path):
            return checkpoint_path
    
    # Try to get from trial_id/run_id and construct path
    trial_id = best_config.get("trial_id") or best_config.get("run_id")
    backbone = best_config.get("backbone", "unknown")
    backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
    
    if trial_id:
        # Try old-style path structure
        old_checkpoint = root_dir / "outputs" / "hpo" / backbone_name / trial_id / "checkpoint"
        if validate_checkpoint(old_checkpoint):
            return old_checkpoint
        
        # Try benchmarking path structure
        bench_checkpoint = root_dir / "outputs" / "benchmarking" / backbone_name / trial_id / "checkpoint"
        if validate_checkpoint(bench_checkpoint):
            return bench_checkpoint
    
    return None


def _resolve_checkpoint_from_fingerprints(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str
) -> Optional[Path]:
    """
    Auto-detect checkpoint path from fingerprints/metadata/index.
    
    Resolution priority:
    1. Index lookup (by spec_fp and environment)
    2. Metadata lookup (by spec_fp)
    3. Cache lookup (latest training cache)
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
    
    Returns:
        Resolved checkpoint path or None.
    """
    environment = detect_platform()
    
    # Priority 1: Index lookup
    try:
        from orchestration.index_manager import find_by_spec_and_env, get_latest_entry
        
        entries = find_by_spec_and_env(root_dir, spec_fp, environment, "final_training")
        if entries:
            # Get latest entry
            latest = get_latest_entry(
                root_dir, "final_training",
                spec_fp=spec_fp,
                environment=environment
            )
            if latest:
                path_str = latest.get("path")
                if path_str:
                    checkpoint_path = Path(path_str) / "checkpoint"
                    if validate_checkpoint(checkpoint_path):
                        return checkpoint_path
    except ImportError:
        pass
    
    # Priority 2: Metadata lookup (scan cache directory)
    try:
        from orchestration.paths import resolve_output_path
        from shared.json_cache import load_json
        
        cache_dir = resolve_output_path(
            root_dir, config_dir, "cache", subcategory="final_training"
        )
        
        # Look for metadata files with matching spec_fp
        if cache_dir.exists():
            for metadata_file in cache_dir.glob("*_metadata.json"):
                metadata = load_json(metadata_file, default={})
                if metadata.get("spec_fp") == spec_fp:
                    # Try to get checkpoint path from metadata
                    checkpoint_path_str = metadata.get("status", {}).get("training", {}).get("checkpoint_path")
                    if checkpoint_path_str:
                        checkpoint_path = Path(checkpoint_path_str)
                        if validate_checkpoint(checkpoint_path):
                            return checkpoint_path
    except Exception:
        pass
    
    # Priority 3: Cache lookup (latest training cache)
    try:
        from orchestration.paths import resolve_output_path
        from shared.json_cache import load_json
        
        cache_dir = resolve_output_path(
            root_dir, config_dir, "cache", subcategory="final_training"
        )
        
        # Look for latest cache file
        latest_cache = cache_dir / "latest_final_training_cache.json"
        if latest_cache.exists():
            cache_data = load_json(latest_cache, default={})
            output_dir = cache_data.get("output_dir")
            if output_dir:
                checkpoint_path = Path(output_dir) / "checkpoint"
                if validate_checkpoint(checkpoint_path):
                    return checkpoint_path
    except Exception:
        pass
    
    return None


def _resolve_variant(
    root_dir: Path,
    config_dir: Path,
    variant_config: Dict[str, Any],
    run_mode: str,
    spec_fp: str,
    exec_fp: str,
    backbone: str
) -> int:
    """
    Resolve variant number based on run.mode.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        variant_config: Variant config from final_training.yaml.
        run_mode: Run mode (reuse_if_exists, force_new, etc.).
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        backbone: Model backbone name.
    
    Returns:
        Resolved variant number.
    """
    explicit_variant = variant_config.get("number")
    
    # If run_mode is force_new, always increment (ignore explicit variant)
    if run_mode == "force_new":
        return _compute_next_variant(root_dir, config_dir, spec_fp, exec_fp, backbone)
    
    # If explicit variant is set, use it (unless mode requires increment)
    if explicit_variant is not None:
        return int(explicit_variant)
    
    # Auto-increment: find next available variant
    if run_mode == "reuse_if_exists":
        # Check if variant exists and is complete
        existing_variant = _find_existing_variant(
            root_dir, config_dir, spec_fp, exec_fp, backbone
        )
        if existing_variant and _is_variant_complete(
            root_dir, config_dir, spec_fp, exec_fp, backbone, existing_variant
        ):
            return existing_variant
    
    # Increment to next available
    return _compute_next_variant(root_dir, config_dir, spec_fp, exec_fp, backbone)


def _find_existing_variant(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    backbone: str
) -> Optional[int]:
    """
    Find existing variant number for given fingerprints.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        backbone: Model backbone name.
    
    Returns:
        Existing variant number or None.
    """
    environment = detect_platform()
    backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
    
    try:
        from orchestration.index_manager import find_by_spec_and_env
        
        entries = find_by_spec_and_env(root_dir, spec_fp, environment, "final_training")
        if entries:
            # Get highest variant
            variants = [e.get("variant", 1) for e in entries if e.get("exec_fp") == exec_fp]
            if variants:
                return max(variants)
    except ImportError:
        pass
    
    # Fallback: scan output directories
    try:
        from orchestration.naming_centralized import build_output_path, create_naming_context
        
        variants = []
        for variant_num in range(1, 100):  # Reasonable limit
            context = create_naming_context(
                process_type="final_training",
                model=backbone_name,
                spec_fp=spec_fp,
                exec_fp=exec_fp,
                environment=environment,
                variant=variant_num
            )
            output_dir = build_output_path(root_dir, context)
            if output_dir.exists():
                variants.append(variant_num)
        
        if variants:
            return max(variants)
    except Exception:
        pass
    
    return None


def _is_variant_complete(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    backbone: str,
    variant: int
) -> bool:
    """
    Check if variant is marked as complete.
    
    Checks both metadata.json completion flag and presence of valid checkpoint files.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        backbone: Model backbone name.
        variant: Variant number.
    
    Returns:
        True if variant is complete, False otherwise.
    """
    environment = detect_platform()
    backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
    
    try:
        from orchestration.naming_centralized import build_output_path, create_naming_context
        
        context = create_naming_context(
            process_type="final_training",
            model=backbone_name,
            spec_fp=spec_fp,
            exec_fp=exec_fp,
            environment=environment,
            variant=variant
        )
        output_dir = build_output_path(root_dir, context)
        metadata_file = output_dir / "metadata.json"
        checkpoint_dir = output_dir / "checkpoint"
        
        # First check: metadata.json with completion flag
        if metadata_file.exists():
            try:
                from shared.json_cache import load_json
                metadata = load_json(metadata_file, default={})
                if metadata.get("status", {}).get("training", {}).get("completed", False):
                    return True
            except Exception:
                pass
        
        # Fallback: check if checkpoint exists and has model files (indicates training completed)
        if checkpoint_dir.exists():
            # Check for key model files that indicate successful training
            config_file = checkpoint_dir / "config.json"
            model_files = list(checkpoint_dir.glob("model.*"))
            required_files = ["model.safetensors"]
            
            if config_file.exists() and (model_files or any(
                (checkpoint_dir / f).exists() for f in required_files
            )):
                return True
    except Exception:
        pass
    
    return False


def _compute_next_variant(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    backbone: str
) -> int:
    """
    Compute next available variant number.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        backbone: Model backbone name.
    
    Returns:
        Next available variant number (starts at 1 if none exist).
    """
    existing = _find_existing_variant(root_dir, config_dir, spec_fp, exec_fp, backbone)
    if existing is None:
        return 1
    return existing + 1


def _merge_configs(
    train_config: Dict[str, Any],
    best_config: Dict[str, Any],
    final_training_config: Dict[str, Any],
    data_config: Optional[Dict[str, Any]],
    seed: int,
    checkpoint_path: Optional[Path]
) -> Dict[str, Any]:
    """
    Merge configs: train.yaml defaults -> best_config -> final_training.yaml overrides.
    
    Args:
        train_config: Base training config.
        best_config: Best selected configuration (from Step P1-3.6, considers HPO + benchmarking).
        final_training_config: Final training config from YAML.
        data_config: Resolved data config.
        seed: Resolved seed.
        checkpoint_path: Resolved checkpoint path.
    
    Returns:
        Merged config dict compatible with existing build_final_training_config() output.
    """
    training_defaults = train_config.get("training", {})
    hyperparameters = best_config.get("hyperparameters", {})
    training_overrides = final_training_config.get("training", {})
    
    # Start with train.yaml defaults
    merged = {
        "backbone": best_config.get("backbone", "unknown"),
        "learning_rate": training_defaults.get("learning_rate", 2e-5),
        "dropout": training_defaults.get("dropout", 0.1),
        "weight_decay": training_defaults.get("weight_decay", 0.01),
        "batch_size": training_defaults.get("batch_size", 16),
        "epochs": training_defaults.get("epochs", 5),
        "random_seed": seed,
        "early_stopping_enabled": training_defaults.get("early_stopping", {}).get("enabled", False),
    }
    
    # Apply best_config hyperparameters
    if "learning_rate" in hyperparameters:
        merged["learning_rate"] = hyperparameters["learning_rate"]
    if "dropout" in hyperparameters:
        merged["dropout"] = hyperparameters["dropout"]
    if "weight_decay" in hyperparameters:
        merged["weight_decay"] = hyperparameters["weight_decay"]
    
    # Apply final_training.yaml overrides (non-null values only)
    if training_overrides.get("learning_rate") is not None:
        merged["learning_rate"] = training_overrides["learning_rate"]
    if training_overrides.get("epochs") is not None:
        merged["epochs"] = training_overrides["epochs"]
    if training_overrides.get("batch_size") is not None:
        merged["batch_size"] = training_overrides["batch_size"]
    if training_overrides.get("dropout") is not None:
        merged["dropout"] = training_overrides["dropout"]
    if training_overrides.get("weight_decay") is not None:
        merged["weight_decay"] = training_overrides["weight_decay"]
    
    # Apply other training overrides
    for key in ["gradient_accumulation_steps", "warmup_steps", "max_grad_norm"]:
        if training_overrides.get(key) is not None:
            merged[key] = training_overrides[key]
        elif key in training_defaults:
            merged[key] = training_defaults[key]
    
    # Early stopping overrides
    early_stopping_overrides = training_overrides.get("early_stopping", {})
    if early_stopping_overrides.get("enabled") is not None:
        merged["early_stopping_enabled"] = early_stopping_overrides["enabled"]
    
    # Add checkpoint path if resolved
    if checkpoint_path:
        merged["checkpoint_path"] = str(checkpoint_path)
    
    # Add MLflow config (metadata only, not used in fingerprints)
    mlflow_config = final_training_config.get("mlflow", {})
    if mlflow_config:
        merged["mlflow"] = mlflow_config
    
    return merged


def _build_final_training_config_inline(
    best_config: Dict[str, Any],
    experiment_config: ExperimentConfig,
    train_config: Optional[Dict[str, Any]],
    config_dir: Path
) -> Dict[str, Any]:
    """
    Backward compatibility: build config inline (deprecated).
    
    This function replicates the old inline build_final_training_config() behavior.
    """
    if train_config is None:
        train_config = load_yaml(experiment_config.train_config)
    
    hyperparameters = best_config.get("hyperparameters", {})
    training_defaults = train_config.get("training", {})
    
    return {
        "backbone": best_config.get("backbone", "unknown"),
        "learning_rate": hyperparameters.get("learning_rate", training_defaults.get("learning_rate", 2e-5)),
        "dropout": hyperparameters.get("dropout", training_defaults.get("dropout", 0.1)),
        "weight_decay": hyperparameters.get("weight_decay", training_defaults.get("weight_decay", 0.01)),
        "batch_size": training_defaults.get("batch_size", 16),
        "epochs": training_defaults.get("epochs", 5),
        "random_seed": training_defaults.get("random_seed", 42),
        "early_stopping_enabled": False,
        "use_combined_data": True,
        "use_all_data": True,
    }

