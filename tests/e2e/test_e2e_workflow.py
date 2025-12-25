#!/usr/bin/env python3
"""End-to-end workflow test script mimicking the Colab notebook.

This script validates the complete training pipeline using tiny datasets:
1. Load centralized configs
2. Verify dataset
3. Setup MLflow
4. Run smoke HPO sweep
5. Select best configuration
6. Run final training (optional)
7. Validate outputs

Usage:
    python tests/e2e/test_e2e_workflow.py [--skip-training] [--skip-benchmark] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tests/e2e/ -> tests/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import orchestration modules
from orchestration import (
    STAGE_HPO,
    STAGE_TRAINING,
    EXPERIMENT_NAME,
    METRICS_FILENAME,
    CHECKPOINT_DIRNAME,
    DEFAULT_RANDOM_SEED,
    build_mlflow_experiment_name,
)
from orchestration.config_loader import (
    ExperimentConfig,
    load_experiment_config,
    load_all_configs,
    compute_config_hashes,
    create_config_metadata,
)
from orchestration.jobs.local_sweeps import run_local_hpo_sweep
from orchestration.jobs.local_selection import select_best_configuration_across_studies


def setup_paths() -> Tuple[Path, Path, Path]:
    """Setup project paths and verify structure."""
    root_dir = PROJECT_ROOT
    src_dir = root_dir / "src"
    config_dir = root_dir / "config"

    # Verify required directories exist
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    print(f"✓ Project root: {root_dir}")
    print(f"✓ Source directory: {src_dir}")
    print(f"✓ Config directory: {config_dir}")

    return root_dir, src_dir, config_dir


def load_configs(config_dir: Path, experiment_name: str = EXPERIMENT_NAME) -> Tuple[ExperimentConfig, Dict[str, Any]]:
    """Load experiment config and all domain configs."""
    print(f"\n[Step 1] Loading centralized configs...")
    print(f"  Experiment: {experiment_name}")

    experiment_config = load_experiment_config(config_dir, experiment_name)
    configs = load_all_configs(experiment_config)
    config_hashes = compute_config_hashes(configs)
    config_metadata = create_config_metadata(configs, config_hashes)

    print(f"✓ Loaded experiment: {experiment_config.name}")
    print(f"✓ Config domains: {sorted(configs.keys())}")
    print(f"✓ Config hashes: {config_hashes}")

    return experiment_config, configs


def verify_dataset(data_config: Dict[str, Any], config_dir: Path) -> Path:
    """Verify that the dataset directory exists and contains required files."""
    print(f"\n[Step 2] Verifying dataset...")

    local_path_str = data_config.get("local_path", "../dataset")
    dataset_path = (config_dir / local_path_str).resolve()

    # Check if seed-based dataset structure (for dataset_tiny with seed subdirectories)
    seed = data_config.get("seed")
    if seed is not None and "dataset_tiny" in str(dataset_path):
        dataset_path = dataset_path / f"seed{seed}"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}\n"
            f"This path comes from the data config's 'local_path' field.\n"
            f"If you need to create the dataset, run: notebooks/00_make_tiny_dataset.ipynb"
        )

    # Check required file
    train_file = dataset_path / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {train_file}\n"
            f"This path comes from the data config's 'local_path' field."
        )

    print(f"✓ Dataset directory: {dataset_path}")
    print(f"  (from data config: {data_config.get('name', 'unknown')} v{data_config.get('version', 'unknown')})")
    if seed is not None:
        print(f"  Using seed: {seed}")

    train_size = train_file.stat().st_size
    print(f"  ✓ train.json ({train_size:,} bytes)")

    # Check optional validation file
    val_file = dataset_path / "validation.json"
    if val_file.exists():
        val_size = val_file.stat().st_size
        print(f"  ✓ validation.json ({val_size:,} bytes)")
    else:
        print(f"  ⚠ validation.json not found (optional)")

    return dataset_path


def setup_mlflow(root_dir: Path) -> None:
    """Setup MLflow tracking with local file store."""
    print(f"\n[Step 3] Setting up MLflow...")

    import mlflow

    mlflow_dir = root_dir / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    mlflow_tracking_uri = mlflow_dir.as_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    print(f"✓ MLflow tracking URI: {mlflow_tracking_uri}")


def setup_kfold_splits(
    dataset_path: Path,
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """Setup k-fold splits if enabled in HPO config."""
    k_fold_config = hpo_config.get("k_fold", {})
    k_folds_enabled = k_fold_config.get("enabled", False)

    if not k_folds_enabled:
        print("\n[Step 4] K-fold CV disabled - using single train/validation split")
        return None

    from training.cv_utils import create_kfold_splits, save_fold_splits, validate_splits
    from training.data import load_dataset

    n_splits = k_fold_config.get("n_splits", 3)
    random_seed = k_fold_config.get("random_seed", DEFAULT_RANDOM_SEED)
    shuffle = k_fold_config.get("shuffle", True)
    stratified = k_fold_config.get("stratified", False)
    entity_types = data_config.get("schema", {}).get("entity_types", [])

    print(f"\n[Step 4] Setting up {n_splits}-fold cross-validation splits...")

    full_dataset = load_dataset(str(dataset_path))
    train_data = full_dataset.get("train", [])

    fold_splits = create_kfold_splits(
        dataset=train_data,
        k=n_splits,
        random_seed=random_seed,
        shuffle=shuffle,
        stratified=stratified,
        entity_types=entity_types,
    )

    # Optional validation
    validate_splits(train_data, fold_splits, entity_types=entity_types)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_splits_file = output_dir / "fold_splits.json"
    save_fold_splits(
        fold_splits,
        fold_splits_file,
        metadata={
            "k": n_splits,
            "random_seed": random_seed,
            "shuffle": shuffle,
            "stratified": stratified,
            "dataset_path": str(dataset_path),
        },
    )

    print(f"✓ K-fold splits saved to: {fold_splits_file}")
    return fold_splits_file


def run_hpo_sweep(
    dataset_path: Path,
    config_dir: Path,
    experiment_config: ExperimentConfig,
    configs: Dict[str, Any],
    output_dir: Path,
    fold_splits_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run HPO sweep for all backbones in the search space."""
    print(f"\n[Step 5] Running HPO sweep...")

    hpo_config = configs["hpo"]
    train_config = configs["train"]
    backbone_values = hpo_config["search_space"]["backbone"]["values"]

    print(f"  Backbones: {backbone_values}")
    print(f"  Max trials: {hpo_config['sampling']['max_trials']}")
    print(f"  Objective: {hpo_config['objective']['metric']} ({hpo_config['objective']['goal']})")

    k_fold_config = hpo_config.get("k_fold", {})
    k_folds_enabled = k_fold_config.get("enabled", False)
    k_folds_param = k_fold_config.get("n_splits", 3) if k_folds_enabled else None

    studies = {}

    for backbone in backbone_values:
        mlflow_experiment_name = build_mlflow_experiment_name(
            experiment_config.name, STAGE_HPO, backbone
        )
        backbone_output_dir = output_dir / backbone

        print(f"\n  Running HPO for {backbone}...")
        study = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone=backbone,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=backbone_output_dir,
            mlflow_experiment_name=mlflow_experiment_name,
            k_folds=k_folds_param,
            fold_splits_file=fold_splits_file,
        )

        studies[backbone] = study

        if study.trials:
            best_trial = study.best_trial
            objective_metric = hpo_config["objective"]["metric"]
            print(f"  ✓ {backbone}: {len(study.trials)} trials completed")
            print(f"    Best {objective_metric}: {best_trial.value:.4f}")

    return studies


def select_best_config(
    studies: Dict[str, Any],
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    hpo_output_dir: Path,
) -> Dict[str, Any]:
    """Select best configuration across all backbones."""
    print(f"\n[Step 6] Selecting best configuration...")

    dataset_version = data_config.get("version", "unknown")

    best_configuration = select_best_configuration_across_studies(
        studies=studies,
        hpo_config=hpo_config,
        dataset_version=dataset_version,
        hpo_output_dir=hpo_output_dir,
    )

    objective_metric = hpo_config["objective"]["metric"]
    print(f"✓ Best configuration selected:")
    print(f"  Backbone: {best_configuration.get('backbone')}")
    print(f"  Trial: {best_configuration.get('trial_name')}")
    print(f"  Best {objective_metric}: {best_configuration.get('selection_criteria', {}).get('best_value'):.4f}")

    selection_criteria = best_configuration.get("selection_criteria", {})
    if "reason" in selection_criteria:
        print(f"  Selection reason: {selection_criteria['reason']}")

    return best_configuration


def run_final_training(
    dataset_path: Path,
    config_dir: Path,
    best_config: Dict[str, Any],
    train_config: Dict[str, Any],
    experiment_config: ExperimentConfig,
    output_dir: Path,
    root_dir: Path,
) -> Path:
    """Run final training with best configuration."""
    print(f"\n[Step 7] Running final training...")

    import mlflow

    backbone = best_config["backbone"]
    hyperparameters = best_config.get("hyperparameters", {})
    training_defaults = train_config.get("training", {})

    # Build final training config
    final_config = {
        "backbone": backbone,
        "learning_rate": hyperparameters.get("learning_rate", training_defaults.get("learning_rate", 2e-5)),
        "dropout": hyperparameters.get("dropout", training_defaults.get("dropout", 0.1)),
        "weight_decay": hyperparameters.get("weight_decay", training_defaults.get("weight_decay", 0.01)),
        "batch_size": training_defaults.get("batch_size", 16),
        "epochs": training_defaults.get("epochs", 5),
        "random_seed": DEFAULT_RANDOM_SEED,
        "early_stopping_enabled": False,  # Disable for final training
        "use_combined_data": True,
    }

    print(f"  Backbone: {backbone}")
    print(f"  Learning rate: {final_config['learning_rate']}")
    print(f"  Batch size: {final_config['batch_size']}")
    print(f"  Epochs: {final_config['epochs']}")

    mlflow_experiment_name = build_mlflow_experiment_name(
        experiment_config.name, STAGE_TRAINING, backbone
    )
    final_output_dir = output_dir / backbone
    final_output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(mlflow_experiment_name)

    # Build training command - run as module to allow relative imports
    args = [
        sys.executable,
        "-m",
        "training.train",
        "--data-asset",
        str(dataset_path),
        "--config-dir",
        str(config_dir),
        "--backbone",
        backbone,
        "--learning-rate",
        str(final_config["learning_rate"]),
        "--batch-size",
        str(final_config["batch_size"]),
        "--dropout",
        str(final_config["dropout"]),
        "--weight-decay",
        str(final_config["weight_decay"]),
        "--epochs",
        str(final_config["epochs"]),
        "--random-seed",
        str(final_config["random_seed"]),
        "--early-stopping-enabled",
        "false",
        "--use-combined-data",
        "true",
    ]

    # Set environment variables
    env = os.environ.copy()
    env["AZURE_ML_OUTPUT_checkpoint"] = str(final_output_dir)
    env["AZURE_ML_OUTPUT_CHECKPOINT"] = str(final_output_dir)

    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

    # Add src directory to PYTHONPATH to allow module imports
    src_dir = str(root_dir / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = src_dir

    print(f"  Running training...")
    result = subprocess.run(
        args,
        cwd=root_dir,
        env=env,
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Final training failed with return code {result.returncode}")

    # Check for checkpoint (may be in outputs/checkpoint or final_output_dir/checkpoint)
    actual_checkpoint = root_dir / "outputs" / "checkpoint"
    expected_checkpoint = final_output_dir / "checkpoint"

    if expected_checkpoint.exists() and any(expected_checkpoint.iterdir()):
        checkpoint_dir = expected_checkpoint
    elif actual_checkpoint.exists() and any(actual_checkpoint.iterdir()):
        checkpoint_dir = actual_checkpoint
    else:
        raise FileNotFoundError(
            f"Training completed but no checkpoint found.\n"
            f"  Expected: {expected_checkpoint}\n"
            f"  Actual: {actual_checkpoint}"
        )

    print(f"✓ Training completed. Checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def validate_outputs(
    hpo_output_dir: Path,
    best_config: Dict[str, Any],
    final_checkpoint_dir: Optional[Path] = None,
) -> None:
    """Validate that required output files exist."""
    print(f"\n[Step 8] Validating outputs...")

    backbone = best_config["backbone"]
    trial_name = best_config.get("trial_name", "unknown")

    # Validate HPO outputs
    trial_dir = hpo_output_dir / backbone / trial_name
    if not trial_dir.exists():
        # Try with fold suffix (for k-fold CV)
        trial_dirs = list((hpo_output_dir / backbone).glob(f"{trial_name}_fold*"))
        if trial_dirs:
            trial_dir = trial_dirs[0]
        else:
            raise FileNotFoundError(f"HPO trial directory not found: {trial_dir}")

    metrics_file = trial_dir / METRICS_FILENAME
    if not metrics_file.exists():
        raise FileNotFoundError(f"HPO metrics file not found: {metrics_file}")

    print(f"✓ HPO outputs validated:")
    print(f"  Trial directory: {trial_dir}")
    print(f"  Metrics file: {metrics_file}")

    # Validate final training outputs (if training was run)
    if final_checkpoint_dir:
        if not final_checkpoint_dir.exists():
            raise FileNotFoundError(f"Final checkpoint directory not found: {final_checkpoint_dir}")

        checkpoint_files = list(final_checkpoint_dir.glob("*.bin")) + list(final_checkpoint_dir.glob("*.safetensors"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in: {final_checkpoint_dir}")

        print(f"✓ Final training outputs validated:")
        print(f"  Checkpoint directory: {final_checkpoint_dir}")
        print(f"  Checkpoint files: {len(checkpoint_files)}")


def main():
    """Main entry point for end-to-end workflow test."""
    parser = argparse.ArgumentParser(
        description="End-to-end workflow test script for training pipeline validation"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip final training step (only run HPO)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmarking step (not implemented yet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results (default: outputs/e2e_test)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"Experiment name (default: {EXPERIMENT_NAME})",
    )

    args = parser.parse_args()

    try:
        # Setup
        root_dir, src_dir, config_dir = setup_paths()

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = root_dir / "outputs" / "e2e_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

        # Load configs
        experiment_config, configs = load_configs(config_dir, args.experiment)

        # Verify dataset
        dataset_path = verify_dataset(configs["data"], config_dir)

        # Setup MLflow
        setup_mlflow(root_dir)

        # Setup k-fold splits if enabled
        hpo_output_dir = output_dir / "hpo"
        fold_splits_file = setup_kfold_splits(
            dataset_path,
            configs["hpo"],
            configs["data"],
            hpo_output_dir,
        )

        # Run HPO sweep
        studies = run_hpo_sweep(
            dataset_path,
            config_dir,
            experiment_config,
            configs,
            hpo_output_dir,
            fold_splits_file,
        )

        # Select best configuration
        best_config = select_best_config(
            studies,
            configs["hpo"],
            configs["data"],
            hpo_output_dir,
        )

        # Save best config for reference
        best_config_file = output_dir / "best_configuration.json"
        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"✓ Best configuration saved to: {best_config_file}")

        # Run final training (optional)
        final_checkpoint_dir = None
        if not args.skip_training:
            final_training_output_dir = output_dir / "final_training"
            final_checkpoint_dir = run_final_training(
                dataset_path,
                config_dir,
                best_config,
                configs["train"],
                experiment_config,
                final_training_output_dir,
                root_dir,
            )
        else:
            print(f"\n[Step 7] Skipping final training (--skip-training)")

        # Validate outputs
        validate_outputs(hpo_output_dir, best_config, final_checkpoint_dir)

        print(f"\n{'='*60}")
        print(f"✓ End-to-end workflow test completed successfully!")
        print(f"{'='*60}")
        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ End-to-end workflow test failed: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

