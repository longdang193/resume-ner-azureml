"""Final training execution module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import mlflow
from mlflow.tracking import MlflowClient

from orchestration.config_loader import ExperimentConfig, load_all_configs
from orchestration.data_assets import resolve_dataset_path
from orchestration.final_training_config import load_final_training_config
from orchestration.fingerprints import compute_exec_fp, compute_spec_fp
from orchestration.jobs.tracking.mlflow_naming import (
    build_mlflow_run_name,
    build_mlflow_tags,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
from orchestration.jobs.tracking.mlflow_index import update_mlflow_index
from orchestration.naming_centralized import build_output_path, create_naming_context
from shared.platform_detection import detect_platform
from shared.yaml_utils import load_yaml


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
            dataset_local_path = (config_dir / dataset_path_from_config).resolve()

    # Validate dataset path exists
    if not dataset_local_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_local_path}\n"
            f"Please check:\n"
            f"  1. final_training.yaml dataset.local_path_override (if set)\n"
            f"  2. Data config file dataset_path setting\n"
            f"  3. That the dataset directory exists at the specified path"
        )

    # Build training command arguments
    training_args = [
        sys.executable,
        "-m",
        "training.train",
        "--data-asset",
        str(dataset_local_path),
        "--config-dir",
        str(config_dir),
        "--backbone",
        final_training_config["backbone"],
        "--learning-rate",
        str(final_training_config["learning_rate"]),
        "--batch-size",
        str(final_training_config["batch_size"]),
        "--dropout",
        str(final_training_config["dropout"]),
        "--weight-decay",
        str(final_training_config["weight_decay"]),
        "--epochs",
        str(final_training_config["epochs"]),
        "--random-seed",
        str(final_training_config["random_seed"]),
        "--early-stopping-enabled",
        str(final_training_config.get("early_stopping_enabled", False)).lower(),
        "--use-combined-data",
        str(final_training_config.get("use_combined_data", True)).lower(),
    ]

    # Set up environment variables
    training_env = os.environ.copy()
    training_env["AZURE_ML_OUTPUT_checkpoint"] = str(final_output_dir)

    # Add src directory to PYTHONPATH
    src_dir = root_dir / "src"
    pythonpath = training_env.get("PYTHONPATH", "")
    if pythonpath:
        training_env["PYTHONPATH"] = f"{str(src_dir)}{os.pathsep}{pythonpath}"
    else:
        training_env["PYTHONPATH"] = str(src_dir)

    # Set MLflow tracking environment
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if mlflow_tracking_uri:
        training_env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    training_env["MLFLOW_EXPERIMENT_NAME"] = training_experiment_name

    # Create MLflow run in parent process (no active context)
    # Build systematic run name
    run_name = build_mlflow_run_name(
        context=training_context,
        config_dir=config_dir,
        root_dir=root_dir,
        output_dir=final_output_dir,
    )

    # Get or create experiment
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(training_experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(training_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        # Fallback: use mlflow API
        mlflow.set_experiment(training_experiment_name)
        experiment = mlflow.get_experiment_by_name(training_experiment_name)
        if experiment is None:
            raise RuntimeError(f"Could not get or create experiment: {training_experiment_name}") from e
        experiment_id = experiment.experiment_id

    # Build tags using build_mlflow_tags + add training-specific and lineage tags
    tags = build_mlflow_tags(
        context=training_context,
        output_dir=final_output_dir,
        parent_run_id=None,
        group_id=None,
        config_dir=config_dir,
    )
    tags["mlflow.runType"] = "training"
    tags["training_type"] = "final"
    tags["code.trained_on_full_data"] = "true"
    tags["mlflow.runName"] = run_name  # Ensure run name is set

    # Add lineage tags (keep code.study_key_hash and code.trial_key_hash as primary,
    # also add code.lineage.* for explicit lineage tracking)
    if lineage.get("hpo_study_key_hash"):
        # Primary grouping tags (for consistency with rest of system)
        tags["code.study_key_hash"] = lineage["hpo_study_key_hash"]
        # Lineage tags (additional, for explicit lineage tracking)
        tags["code.lineage.hpo_study_key_hash"] = lineage["hpo_study_key_hash"]
        tags["code.lineage.source"] = "hpo_best_selected"

    if lineage.get("hpo_trial_key_hash"):
        tags["code.trial_key_hash"] = lineage["hpo_trial_key_hash"]
        tags["code.lineage.hpo_trial_key_hash"] = lineage["hpo_trial_key_hash"]

    if lineage.get("hpo_trial_run_id"):
        tags["code.lineage.hpo_trial_run_id"] = lineage["hpo_trial_run_id"]
    if lineage.get("hpo_refit_run_id"):
        tags["code.lineage.hpo_refit_run_id"] = lineage["hpo_refit_run_id"]
    if lineage.get("hpo_sweep_run_id"):
        tags["code.lineage.hpo_sweep_run_id"] = lineage["hpo_sweep_run_id"]

    # Create run WITHOUT starting it (no active context)
    try:
        created_run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        )
        run_id = created_run.info.run_id

        # Update local index
        try:
            run_key = build_mlflow_run_key(training_context)
            run_key_hash = build_mlflow_run_key_hash(run_key)
            update_mlflow_index(
                root_dir=root_dir,
                run_key_hash=run_key_hash,
                run_id=run_id,
                experiment_id=experiment_id,
                tracking_uri=mlflow_tracking_uri or mlflow.get_tracking_uri(),
                config_dir=config_dir,
            )
        except Exception as e:
            print(f"⚠ Could not update MLflow index: {e}")

        print(f"✓ Created MLflow run: {run_name} ({run_id[:12]}...)")

        # Pass run_id to subprocess
        training_env["MLFLOW_RUN_ID"] = run_id
    except Exception as e:
        print(f"⚠ Could not create MLflow run: {e}")
        import traceback
        traceback.print_exc()
        # Continue without MLflow run (training will still work, just no tracking)
        run_id = None

    # Execute training
    print("🔄 Running final training...")
    result = subprocess.run(
        training_args,
        cwd=root_dir,
        env=training_env,
        capture_output=True,
        text=True,
    )

    # Handle subprocess failure - ensure run is marked as FAILED
    if result.returncode != 0:
        if run_id:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception as e:
                print(f"⚠ Could not mark run as FAILED: {e}")
        raise RuntimeError(
            f"Final training failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        # Subprocess should have ended the run, but verify it's terminated
        if run_id:
            try:
                run_info = client.get_run(run_id)
                if run_info.info.status == "RUNNING":
                    # Subprocess didn't end the run, mark it as finished
                    client.set_terminated(run_id, status="FINISHED")
            except Exception as e:
                print(f"⚠ Could not verify run status: {e}")

    # Find final checkpoint directory
    final_checkpoint_dir = final_output_dir / "checkpoint"
    if not final_checkpoint_dir.exists():
        # Try actual checkpoint location
        actual_checkpoint = root_dir / "outputs" / "checkpoint"
        if actual_checkpoint.exists():
            final_checkpoint_dir = actual_checkpoint

    print(f"✓ Final training completed. Checkpoint: {final_checkpoint_dir}")
    if run_id:
        print(f"✓ MLflow run: {run_id[:12]}...")

    # Tags are already set during run creation, no need to apply lineage tags post-hoc
    return final_checkpoint_dir
