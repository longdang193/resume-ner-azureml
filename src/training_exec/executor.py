from __future__ import annotations

"""
@meta
name: final_training_executor
type: script
domain: training
responsibility:
  - Execute final training with best HPO configuration
  - Load final training configuration
  - Build training context and output directory
  - Execute training subprocess
  - Set lineage tags
inputs:
  - Best model configuration
  - Final training configuration
  - Experiment configuration
outputs:
  - Final training checkpoint directory
  - MLflow training run
tags:
  - orchestration
  - training
  - final_training
ci:
  runnable: true
  needs_gpu: true
  needs_cloud: false
lifecycle:
  status: active
"""

"""Final training execution module."""
import subprocess
from pathlib import Path
from typing import Any, Dict

import mlflow
from training.execution import (
    MLflowConfig,
    TrainingOptions,
    build_training_command,
    create_training_mlflow_run,
    execute_training_subprocess,
    setup_training_environment,
)

from infrastructure.config.loader import ExperimentConfig, load_all_configs
# azureml.data_assets imported lazily when needed
from infrastructure.config.training import load_final_training_config

# Try to import resolve_dataset_path, fallback to local implementation if Azure ML not available
try:
    from infrastructure.platform.azureml.data_assets import resolve_dataset_path
except ImportError:
    # Fallback implementation if Azure ML is not available
    def resolve_dataset_path(data_config: Dict[str, Any]) -> Path:
        """Resolve dataset path from data config (fallback implementation)."""
        local_path = data_config.get("local_path", "../dataset")
        if not isinstance(local_path, str):
            raise ValueError(
                f"data_config['local_path'] must be a string, got {type(local_path).__name__}"
            )
        dataset_path = Path(local_path)
        # Check if seed-based dataset structure
        seed = data_config.get("seed")
        if seed is not None and "dataset_tiny" in str(dataset_path):
            dataset_path = dataset_path / f"seed{seed}"
        return dataset_path
from infrastructure.fingerprints.compute import compute_exec_fp, compute_spec_fp
from infrastructure.tracking.mlflow.naming import (
    build_mlflow_run_name,
    build_mlflow_tags,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
from infrastructure.tracking.mlflow.index import update_mlflow_index
from infrastructure.naming import create_naming_context
from infrastructure.paths import build_output_path
from common.shared.platform_detection import detect_platform
from common.shared.yaml_utils import load_yaml

def execute_final_training(
    root_dir: Path,
    config_dir: Path,
    best_model: Dict[str, Any],
    experiment_config: ExperimentConfig,
    lineage: Dict[str, Any],
    training_experiment_name: str,
    platform: str,
) -> Path:
    """
    Execute final training with best configuration.

    This function:
    1. Loads final training config from final_training.yaml using load_final_training_config()
    2. Builds training context and output directory
    3. Executes training as subprocess
    4. Sets lineage tags after training completes
    5. Returns checkpoint directory path

    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        best_model: Best selected model dictionary from model selection.
                   Expected keys: backbone, params (with hyperparameters), tags
        experiment_config: Experiment configuration (contains data_config, etc.).
        lineage: Lineage dictionary from extract_lineage_from_best_model().
        training_experiment_name: MLflow experiment name for training runs.
        platform: Platform name (local, colab, kaggle).

    Returns:
        Path to final training checkpoint directory.

    Raises:
        RuntimeError: If training subprocess fails.
        ValueError: If required configuration is missing.
    """
    # Prepare best_config dict for load_final_training_config
    # The function expects best_config with backbone and hyperparameters
    best_config = {
        "backbone": best_model.get("backbone"),
        "hyperparameters": best_model.get("params", {}),
    }

    # Load final training config (uses final_training.yaml)
    final_training_config = load_final_training_config(
        root_dir=root_dir,
        config_dir=config_dir,
        best_config=best_config,
        experiment_config=experiment_config,
    )

    # Build training context and output directory
    all_configs = load_all_configs(experiment_config)
    environment = detect_platform()

    # Get fingerprints from config (already computed by load_final_training_config)
    spec_fp = final_training_config.get("spec_fp")
    exec_fp = final_training_config.get("exec_fp")
    variant = final_training_config.get("variant", 1)

    # If not in config, compute them
    if not spec_fp or not exec_fp:
        spec_fp = compute_spec_fp(
            model_config=all_configs.get("model", {}),
            data_config=all_configs.get("data", {}),
            train_config=all_configs.get("train", {}),
            seed=int(best_model.get("params", {}).get("random_seed", 42)),
        )

        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root_dir,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_sha = None

        exec_fp = compute_exec_fp(
            git_sha=git_sha,
            env_config=all_configs.get("env", {}),
        )

    # Create training context
    backbone_name = final_training_config.get("backbone", "distilbert")
    if "-" in backbone_name:
        backbone_name = backbone_name.split("-")[0]

    training_context = create_naming_context(
        process_type="final_training",
        model=backbone_name,
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        environment=environment,
        variant=variant,
    )

    final_output_dir = build_output_path(root_dir, training_context)

    print(f"✓ Final training config loaded from final_training.yaml")
    print(f"✓ Output directory: {final_output_dir}")

    # Resolve dataset path from final_training.yaml config
    # Check for local_path_override first (from final_training.yaml)
    final_training_yaml = load_yaml(config_dir / "final_training.yaml")

    # Check if run should be skipped based on run.mode
    run_mode = final_training_yaml.get(
        "run", {}).get("mode", "reuse_if_exists")
    final_checkpoint_dir = final_output_dir / "checkpoint"

    if run_mode == "reuse_if_exists":
        # Helper function to check if a checkpoint is complete
        def is_checkpoint_complete(checkpoint_dir: Path, metadata_file: Path) -> bool:
            """Check if checkpoint is complete (has metadata with completion flag, or valid checkpoint files)."""
            # First check: metadata.json with completion flag
            if metadata_file.exists():
                try:
                    from common.shared.json_cache import load_json
                    metadata = load_json(metadata_file, default={})
                    if metadata.get("status", {}).get("training", {}).get("completed", False):
                        return True
                except Exception:
                    pass

            # Fallback: check if checkpoint exists and has model files (indicates training completed)
            if checkpoint_dir.exists():
                # Check for key model files that indicate successful training
                required_files = ["config.json", "model.safetensors"]
                # Also accept .bin or .pt files as alternatives
                model_files = list(checkpoint_dir.glob("model.*"))
                config_file = checkpoint_dir / "config.json"

                if config_file.exists() and (model_files or any(
                    (checkpoint_dir / f).exists() for f in required_files
                )):
                    return True

            return False

        # First check: exact match (resolved variant)
        metadata_file = final_output_dir / "metadata.json"
        if is_checkpoint_complete(final_checkpoint_dir, metadata_file):
            print(f"✓ Found existing completed final training run")
            print(f"  Output directory: {final_output_dir}")
            print(f"  Checkpoint: {final_checkpoint_dir}")
            print(f"  Reusing existing checkpoint (run.mode: reuse_if_exists)")
            return final_checkpoint_dir

        # Second check: search for any complete variant with same spec_fp + exec_fp
        # This handles cases where _resolve_variant didn't find the complete variant
        try:
            from infrastructure.paths import resolve_output_path
            base_output_dir = resolve_output_path(
                root_dir, config_dir, "final_training")
            final_training_base = base_output_dir / environment / backbone_name

            if final_training_base.exists():
                # Look for directories matching spec_{spec_fp}_exec_{exec_fp}
                spec_exec_pattern = f"spec_{spec_fp}_exec_{exec_fp}"
                for spec_exec_dir in final_training_base.iterdir():
                    if spec_exec_dir.is_dir() and spec_exec_dir.name == spec_exec_pattern:
                        # Check all variants in this directory (highest first)
                        variant_dirs = sorted(
                            [d for d in spec_exec_dir.iterdir() if d.is_dir()
                             and d.name.startswith("v")],
                            key=lambda d: int(
                                d.name[1:]) if d.name[1:].isdigit() else 0,
                            reverse=True  # Check highest variant first
                        )
                        for variant_dir in variant_dirs:
                            variant_checkpoint = variant_dir / "checkpoint"
                            variant_metadata = variant_dir / "metadata.json"

                            if is_checkpoint_complete(variant_checkpoint, variant_metadata):
                                print(
                                    f"✓ Found existing completed final training run")
                                print(
                                    f"  Output directory: {variant_dir}")
                                print(
                                    f"  Checkpoint: {variant_checkpoint}")
                                print(
                                    f"  Reusing existing checkpoint (run.mode: reuse_if_exists)")
                                return variant_checkpoint
                        break  # Only check the matching spec_exec directory
        except Exception as e:
            print(f"⚠ Warning: Could not search for existing runs: {e}")
            # Continue with training if search fails
    dataset_config_yaml = final_training_yaml.get("dataset", {})
    local_path_override = dataset_config_yaml.get("local_path_override")

    if local_path_override:
        # Use the override path directly
        dataset_local_path = Path(local_path_override) if Path(
            local_path_override).is_absolute() else root_dir / local_path_override
    else:
        # Get dataset path from data config
        # Try to get from all_configs first (which may have been resolved by load_final_training_config)
        data_config = all_configs.get("data", {})
        if not data_config:
            # Fallback: resolve data config using same logic as load_final_training_config
            # Check for explicit data_config in final_training.yaml
            data_config_path = dataset_config_yaml.get("data_config")
            if data_config_path:
                if not Path(data_config_path).is_absolute():
                    data_config_path = config_dir / data_config_path
                else:
                    data_config_path = Path(data_config_path)
                data_config = load_yaml(data_config_path)
            else:
                # Use experiment_config.data_config
                if experiment_config.data_config:
                    data_config = load_yaml(experiment_config.data_config)
                else:
                    data_config = {}

        # Get dataset path from data config using resolve_dataset_path
        # This handles seed-based dataset structures (e.g., dataset_tiny/seed0/)
        # resolve_dataset_path returns a Path relative to the config directory
        dataset_path_from_config = resolve_dataset_path(data_config)

        # Resolve to absolute path relative to config directory
        if dataset_path_from_config.is_absolute():
            dataset_local_path = dataset_path_from_config
        else:
            # Resolve relative to config directory (e.g., "../dataset_tiny" -> root_dir/dataset_tiny)
            dataset_local_path = (
                config_dir / dataset_path_from_config).resolve()

    # Validate dataset path exists
    if not dataset_local_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_local_path}\n"
            f"Please check:\n"
            f"  1. final_training.yaml dataset.local_path_override (if set)\n"
            f"  2. Data config file dataset_path setting\n"
            f"  3. That the dataset directory exists at the specified path"
        )

    # Build training command arguments using shared infrastructure
    training_options = TrainingOptions(
        epochs=final_training_config["epochs"],
        early_stopping_enabled=final_training_config.get("early_stopping_enabled", False),
        use_combined_data=final_training_config.get("use_combined_data", True),
        random_seed=final_training_config["random_seed"],
    )
    hyperparameters = {
        "learning_rate": final_training_config["learning_rate"],
        "batch_size": final_training_config["batch_size"],
        "dropout": final_training_config["dropout"],
        "weight_decay": final_training_config["weight_decay"],
    }
    training_args = build_training_command(
        backbone=final_training_config["backbone"],
        dataset_path=dataset_local_path,
        config_dir=config_dir,
        hyperparameters=hyperparameters,
        training_options=training_options,
    )

    # Set up environment variables using shared infrastructure
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    mlflow_config = MLflowConfig(
        experiment_name=training_experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_id=None,  # Will be set after run creation
    )
    training_env = setup_training_environment(
        root_dir=root_dir,
        src_dir=root_dir / "src",
        output_dir=final_output_dir,
        mlflow_config=mlflow_config,
    )

    # Create MLflow run in parent process (no active context) using shared infrastructure
    # Build systematic run name
    run_name = build_mlflow_run_name(
        context=training_context,
        config_dir=config_dir,
        root_dir=root_dir,
        output_dir=final_output_dir,
    )

    # Build tags using build_mlflow_tags + add training-specific and lineage tags
    tags = build_mlflow_tags(
        context=training_context,
        output_dir=final_output_dir,
        parent_run_id=None,
        group_id=None,
        config_dir=config_dir,
    )
    # Get tag keys from registry (using centralized helpers)
    from infrastructure.naming.mlflow.tag_keys import (
        get_lineage_hpo_refit_run_id,
        get_lineage_hpo_study_key_hash,
        get_lineage_hpo_sweep_run_id,
        get_lineage_hpo_trial_key_hash,
        get_lineage_hpo_trial_run_id,
        get_lineage_parent_training_run_id,
        get_lineage_source,
        get_mlflow_run_type,
        get_study_key_hash,
        get_trained_on_full_data,
        get_trial_key_hash,
    )
    
    mlflow_run_type_tag = get_mlflow_run_type(config_dir)
    trained_on_full_data_tag = get_trained_on_full_data(config_dir)
    study_key_hash_tag = get_study_key_hash(config_dir)
    trial_key_hash_tag = get_trial_key_hash(config_dir)
    lineage_source_tag = get_lineage_source(config_dir)
    lineage_hpo_study_key_hash_tag = get_lineage_hpo_study_key_hash(config_dir)
    lineage_hpo_trial_key_hash_tag = get_lineage_hpo_trial_key_hash(config_dir)
    lineage_hpo_trial_run_id_tag = get_lineage_hpo_trial_run_id(config_dir)
    lineage_hpo_refit_run_id_tag = get_lineage_hpo_refit_run_id(config_dir)
    lineage_hpo_sweep_run_id_tag = get_lineage_hpo_sweep_run_id(config_dir)
    
    tags[mlflow_run_type_tag] = "training"
    tags["training_type"] = "final"
    tags[trained_on_full_data_tag] = "true"
    tags["mlflow.runName"] = run_name  # Ensure run name is set

    # Add lineage tags (keep code.study_key_hash and code.trial_key_hash as primary,
    # also add code.lineage.* for explicit lineage tracking)
    if lineage.get("hpo_study_key_hash"):
        # Primary grouping tags (for consistency with rest of system)
        tags[study_key_hash_tag] = lineage["hpo_study_key_hash"]
        # Lineage tags (additional, for explicit lineage tracking)
        tags[lineage_hpo_study_key_hash_tag] = lineage["hpo_study_key_hash"]
        tags[lineage_source_tag] = "hpo_best_selected"

    if lineage.get("hpo_trial_key_hash"):
        tags[trial_key_hash_tag] = lineage["hpo_trial_key_hash"]
        tags[lineage_hpo_trial_key_hash_tag] = lineage["hpo_trial_key_hash"]

    if lineage.get("hpo_trial_run_id"):
        tags[lineage_hpo_trial_run_id_tag] = lineage["hpo_trial_run_id"]
    if lineage.get("hpo_refit_run_id"):
        tags[lineage_hpo_refit_run_id_tag] = lineage["hpo_refit_run_id"]
    if lineage.get("hpo_sweep_run_id"):
        tags[lineage_hpo_sweep_run_id_tag] = lineage["hpo_sweep_run_id"]

    # Create run WITHOUT starting it (no active context) using shared infrastructure
    run_id = None
    experiment_id = None
    try:
        run_id, created_run = create_training_mlflow_run(
            experiment_name=training_experiment_name,
            run_name=run_name,
            tags=tags,
            root_dir=root_dir,
            config_dir=config_dir,
            context=training_context,
            tracking_uri=mlflow_tracking_uri,
        )
        experiment_id = created_run.info.experiment_id
        print(f"✓ Created MLflow run: {run_name} ({run_id[:12]}...)")

        # Pass run_id to subprocess
        training_env["MLFLOW_RUN_ID"] = run_id
    except Exception as e:
        print(f"⚠ Could not create MLflow run: {e}")
        import traceback
        traceback.print_exc()
        # Continue without MLflow run (training will still work, just no tracking)
        run_id = None

    # Execute training using shared infrastructure
    print("🔄 Running final training...")
    try:
        result = execute_training_subprocess(
            command=training_args,
            cwd=root_dir,
            env=training_env,
        )
        # Subprocess should have ended the run, but verify it's terminated
        if run_id:
            from infrastructure.tracking.mlflow import ensure_run_terminated
            ensure_run_terminated(run_id, expected_status="FINISHED")
    except RuntimeError as e:
        # Handle subprocess failure - ensure run is marked as FAILED
        if run_id:
            from infrastructure.tracking.mlflow import terminate_run_safe
            terminate_run_safe(run_id, status="FAILED", check_status=True)
        raise

    # Find final checkpoint directory
    final_checkpoint_dir = final_output_dir / "checkpoint"
    if not final_checkpoint_dir.exists():
        # Try actual checkpoint location
        actual_checkpoint = root_dir / "outputs" / "checkpoint"
        if actual_checkpoint.exists():
            final_checkpoint_dir = actual_checkpoint

    # Save metadata.json with completion status
    try:
        from infrastructure.metadata.training import save_metadata_with_fingerprints

        # Prepare MLflow info (use variables from outer scope)
        mlflow_info_dict = None
        if run_id:
            mlflow_info_dict = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "tracking_uri": mlflow_tracking_uri,
            }

        # Prepare status updates
        status_updates = {
            "training": {
                "completed": True,
                "checkpoint_path": str(final_checkpoint_dir),
            }
        }

        # Save metadata (mlflow_info is passed as keyword argument)
        metadata_path = save_metadata_with_fingerprints(
            root_dir=root_dir,
            config_dir=config_dir,
            context=training_context,
            metadata_content={
                "backbone": backbone_name,
            },
            status_updates=status_updates,
            mlflow_info=mlflow_info_dict,  # Pass as keyword argument
        )
        print(f"✓ Saved metadata to: {metadata_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save metadata.json: {e}")
        import traceback
        traceback.print_exc()
        # Continue even if metadata save fails

    print(f"✓ Final training completed. Checkpoint: {final_checkpoint_dir}")
    if run_id:
        print(f"✓ MLflow run: {run_id[:12]}...")

    # Tags are already set during run creation, no need to apply lineage tags post-hoc
    return final_checkpoint_dir
