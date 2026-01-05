from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import Environment, Job, Data


DEFAULT_RANDOM_SEED = 42


def build_final_training_config(
    best_config: Dict[str, Any],
    train_config: Dict[str, Any],
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Build final training configuration by merging best HPO config with train.yaml defaults.

    Args:
        best_config: Best configuration from HPO selection (must have 'backbone' and 'hyperparameters').
        train_config: Training defaults from train.yaml.
        random_seed: Random seed for reproducibility.

    Returns:
        Final training configuration dictionary.
    """
    hyperparameters = best_config.get("hyperparameters", {})
    training_defaults = train_config.get("training", {})

    # Final training should reuse global defaults for core schedule/throughput
    # settings (batch_size, epochs), and only override the most important
    # HPO-tuned knobs (learning rate / regularisation).
    return {
        "backbone": best_config["backbone"],
        # Override from HPO when present:
        "learning_rate": hyperparameters.get("learning_rate", training_defaults.get("learning_rate", 2e-5)),
        "dropout": hyperparameters.get("dropout", training_defaults.get("dropout", 0.1)),
        "weight_decay": hyperparameters.get("weight_decay", training_defaults.get("weight_decay", 0.01)),
        # Always use train.yaml defaults for these:
        "batch_size": training_defaults.get("batch_size", 16),
        "epochs": training_defaults.get("epochs", 5),
        "random_seed": random_seed,
        "early_stopping_enabled": False,
        "use_combined_data": True,
        "use_all_data": True,  # Final training uses all data without validation split
    }


def validate_final_training_job(job: Job) -> None:
    """
    Validate final training job completed successfully.

    Args:
        job: Completed job instance.

    Raises:
        ValueError: If job did not complete successfully.
    """
    if job.status != "Completed":
        raise ValueError(f"Final training job failed with status: {job.status}")


def _build_data_input_from_asset(data_asset: Data) -> Input:
    """
    Build a standard Azure ML ``Input`` for a ``uri_folder`` data asset.

    Args:
        data_asset: Registered Azure ML data asset.

    Returns:
        Input pointing at the asset, mounted as a folder.
    """
    return Input(
        type="uri_folder",
        path=f"azureml:{data_asset.name}:{data_asset.version}",
        mode="mount",
    )


def create_final_training_job(
    script_path: Path,
    data_asset: Data,
    environment: Environment,
    compute_cluster: str,
    final_config: Dict[str, Any],
    aml_experiment_name: str,
    tags: Dict[str, str],
) -> Any:
    """
    Build the final, single-run training job using the best HPO config.

    The resulting command job is responsible for training the production
    model and producing the artefacts (metrics, model files) used for
    deployment or registration.

    Args:
        script_path: Path to the training script within the repo.
        data_asset: Registered data asset providing training data.
        environment: Azure ML environment to run the job in.
        compute_cluster: Name of the compute cluster to target.
        final_config: Selected hyperparameter configuration.
        aml_experiment_name: AML experiment name for this stage/backbone.
        tags: Fully prepared tags dictionary for the job.

    Returns:
        Configured command job ready for submission.

    Raises:
        FileNotFoundError: If the training script is missing.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    args = (
        f"--data-asset ${{{{inputs.data}}}} "
        f"--config-dir config "
        f"--backbone {final_config['backbone']} "
        f"--learning-rate {final_config['learning_rate']} "
        f"--batch-size {final_config['batch_size']} "
        f"--dropout {final_config['dropout']} "
        f"--weight-decay {final_config['weight_decay']} "
        f"--epochs {final_config['epochs']} "
        f"--random-seed {final_config['random_seed']} "
        f"--early-stopping-enabled {str(final_config['early_stopping_enabled']).lower()} "
        f"--use-combined-data {str(final_config['use_combined_data']).lower()}"
    )

    data_input = _build_data_input_from_asset(data_asset)

    # Use the project root as code snapshot so both `src/` and `config/` are included.
    # Azure ML automatically sets AZURE_ML_OUTPUT_checkpoint for the named "checkpoint"
    # output, which the training script will use to save model artefacts.
    return command(
        code="..",
        command=f"python src/{script_path.name} {args}",
        inputs={"data": data_input},
        outputs={
            "checkpoint": Output(type="uri_folder"),
        },
        environment=environment,
        compute=compute_cluster,
        experiment_name=aml_experiment_name,
        tags=tags,
        display_name="final-training",
        description="Final production training with best HPO configuration",
    )


