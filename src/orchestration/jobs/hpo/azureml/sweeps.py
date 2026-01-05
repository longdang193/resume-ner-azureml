from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import Input, command, MLClient
from azure.ai.ml.entities import Environment, Data, Job
from azure.ai.ml.sweep import (
    SweepJob,
    Objective,
    SweepJobLimits,
)

from ..search_space import create_search_space


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


def create_dry_run_sweep_job_for_backbone(
    script_path: Path,
    data_asset: Data,
    environment: Environment,
    compute_cluster: str,
    backbone: str,
    smoke_hpo_config: Dict[str, Any],
    configs: Dict[str, Any],
    config_metadata: Dict[str, str],
    aml_experiment_name: str,
    stage: str,
) -> SweepJob:
    """
    Build a small HPO sweep job used as a smoke test for a backbone.

    The dry run uses a reduced search space (no backbone dimension and
    fewer trials) to validate that data access, training, and metrics
    wiring all function correctly before launching the full HPO sweep.

    Args:
        script_path: Path to the training script within the repo.
        data_asset: Registered data asset providing training data.
        environment: Azure ML environment to run the sweep in.
        compute_cluster: Name of the compute cluster to target.
        backbone: Backbone identifier (e.g. ``distilbert``).
        smoke_hpo_config: Parsed smoke HPO configuration.
        configs: Global configuration mapping (for context only).
        config_metadata: Precomputed configuration metadata for tagging.
        aml_experiment_name: AML experiment name for this stage/backbone.
        stage: Logical experiment stage (e.g. ``smoke``).

    Returns:
        Configured :class:`SweepJob` ready for submission.

    Raises:
        FileNotFoundError: If the training script is missing.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    reduced = {
        "search_space": {
            k: v for k, v in smoke_hpo_config["search_space"].items() if k != "backbone"
        }
    }
    search_space = create_search_space(reduced)

    trials = max(2, smoke_hpo_config["sampling"]["max_trials"] // 2)

    cmd_args = (
        f"--data-asset ${{{{inputs.data}}}} "
        f"--config-dir config "
        f"--backbone {backbone} "
        f"--learning-rate ${{{{search_space.learning_rate}}}} "
        f"--batch-size ${{{{search_space.batch_size}}}} "
        f"--dropout ${{{{search_space.dropout}}}} "
        f"--weight-decay ${{{{search_space.weight_decay}}}}"
    )

    data_input = _build_data_input_from_asset(data_asset)

    trial_job = command(
        code="..",
        command=f"python src/{script_path.name} {cmd_args}",
        inputs={"data": data_input},
        environment=environment,
        compute=compute_cluster,
    )

    objective = Objective(
        goal=smoke_hpo_config["objective"]["goal"],
        primary_metric=smoke_hpo_config["objective"]["metric"],
    )
    timeout_seconds = smoke_hpo_config["sampling"]["timeout_minutes"] * 60
    limits = SweepJobLimits(max_total_trials=trials, timeout=timeout_seconds)

    return SweepJob(
        trial=trial_job,
        search_space=search_space,
        sampling_algorithm=smoke_hpo_config["sampling"]["algorithm"],
        objective=objective,
        limits=limits,
        inputs={"data": data_input},
        experiment_name=aml_experiment_name,
        tags={
            **config_metadata,
            "job_type": "dry_run_sweep",
            "backbone": backbone,
            "stage": stage,
        },
        display_name=f"dry-run-sweep-{backbone}",
        description=f"Dry run sweep for {backbone}",
    )


def create_hpo_sweep_job_for_backbone(
    script_path: Path,
    data_asset: Data,
    environment: Environment,
    compute_cluster: str,
    hpo_config: Dict[str, Any],
    backbone: str,
    configs: Dict[str, Any],
    config_metadata: Dict[str, str],
    aml_experiment_name: str,
    stage: str,
) -> SweepJob:
    """
    Build a production HPO sweep job for a specific backbone model.

    The production sweep typically uses a richer search space and more
    trials than the dry run, and is the primary source for selecting the
    best configuration.

    Args:
        script_path: Path to the training script within the repo.
        data_asset: Registered data asset providing training data.
        environment: Azure ML environment to run the sweep in.
        compute_cluster: Name of the compute cluster to target.
        hpo_config: Parsed HPO configuration.
        backbone: Backbone identifier (e.g. ``distilbert``).
        configs: Global configuration mapping (for context only).
        config_metadata: Precomputed configuration metadata for tagging.
        aml_experiment_name: AML experiment name for this stage/backbone.
        stage: Logical experiment stage (e.g. ``hpo``).

    Returns:
        Configured :class:`SweepJob` ready for submission.

    Raises:
        FileNotFoundError: If the training script is missing.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    reduced = {
        "search_space": {
            k: v for k, v in hpo_config["search_space"].items() if k != "backbone"
        }
    }
    search_space = create_search_space(reduced)

    cmd_args = (
        f"--data-asset ${{{{inputs.data}}}} "
        f"--config-dir config "
        f"--backbone {backbone} "
        f"--learning-rate ${{{{search_space.learning_rate}}}} "
        f"--batch-size ${{{{search_space.batch_size}}}} "
        f"--dropout ${{{{search_space.dropout}}}} "
        f"--weight-decay ${{{{search_space.weight_decay}}}}"
    )

    data_input = _build_data_input_from_asset(data_asset)
    trial_job = command(
        code="..",
        command=f"python src/{script_path.name} {cmd_args}",
        inputs={"data": data_input},
        environment=environment,
        compute=compute_cluster,
    )

    objective = Objective(
        goal=hpo_config["objective"]["goal"],
        primary_metric=hpo_config["objective"]["metric"],
    )
    timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60
    limits = SweepJobLimits(
        max_total_trials=hpo_config["sampling"]["max_trials"],
        timeout=timeout_seconds,
    )

    early_termination = None
    if "early_termination" in hpo_config:
        from azure.ai.ml.sweep import BanditPolicy

        et_cfg = hpo_config["early_termination"]
        if et_cfg.get("policy") == "bandit":
            early_termination = BanditPolicy(
                evaluation_interval=et_cfg["evaluation_interval"],
                slack_factor=et_cfg["slack_factor"],
                delay_evaluation=et_cfg["delay_evaluation"],
            )

    return SweepJob(
        trial=trial_job,
        search_space=search_space,
        sampling_algorithm=hpo_config["sampling"]["algorithm"],
        objective=objective,
        limits=limits,
        early_termination=early_termination,
        compute=compute_cluster,
        inputs={"data": data_input},
        experiment_name=aml_experiment_name,
        tags={
            **config_metadata,
            "job_type": "hpo_sweep",
            "backbone": backbone,
            "stage": stage,
        },
        display_name=f"hpo-sweep-{backbone}",
        description=f"Production HPO sweep for {backbone}",
    )


def _validate_job_status(job: Job, job_type: str, backbone: str) -> None:
    """
    Validate that a job has completed successfully.

    Args:
        job: Job instance to validate
        job_type: Type of job for error messages (e.g., "Dry run sweep", "HPO sweep")
        backbone: Backbone model name for error messages

    Raises:
        ValueError: If job status is not "Completed"
    """
    if job.status != "Completed":
        raise ValueError(
            f"{job_type} job for {backbone} failed with status: {job.status}")


def _get_trial_count(job: Job, ml_client: MLClient | None = None) -> int | None:
    """
    Get trial count from job, with fallback to API call if needed.

    Args:
        job: Job instance to check
        ml_client: Optional ML client for fallback API call

    Returns:
        Trial count if available, None otherwise
    """
    # Check trial_count first (fast, no API call needed)
    if hasattr(job, "trial_count") and job.trial_count and job.trial_count > 0:
        return job.trial_count

    # Only make expensive API call if trial_count is not available and ml_client provided
    if ml_client is not None:
        try:
            children = list(ml_client.jobs.list(parent_job_name=job.name))
            return len(children) if children else None
        except Exception:
            return None

    return None


def validate_sweep_job(
    job: Job,
    backbone: str,
    job_type: str = "Sweep",
    min_expected_trials: int | None = None,
    ml_client: MLClient | None = None,
) -> None:
    """
    Validate sweep job completed successfully with required trials.

    This is the unified validation function for all sweep job types (dry run, HPO, etc.).

    Args:
        job: Completed sweep job instance
        backbone: Backbone model name for error messages
        job_type: Type of job for error messages (e.g., "Dry run sweep", "HPO sweep")
        min_expected_trials: Minimum number of trials expected. If None, only checks for > 0 trials.
        ml_client: Optional ML client for fallback trial count retrieval

    Raises:
        ValueError: If validation fails
    """
    _validate_job_status(job, job_type, backbone)

    trial_count = _get_trial_count(job, ml_client)

    if trial_count is None or trial_count == 0:
        error_msg = f"{job_type} job for {backbone} produced no trials"
        if job_type == "Dry run sweep":
            error_msg += f" (parent run: {job.name}). Check sweep logs and child runs in portal."
        raise ValueError(error_msg)

    if min_expected_trials is not None and trial_count < min_expected_trials:
        raise ValueError(
            f"{job_type} job for {backbone} only produced {trial_count} trial(s), "
            f"expected at least {min_expected_trials}"
        )


from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import Input, command, MLClient
from azure.ai.ml.entities import Environment, Data, Job
from azure.ai.ml.sweep import (
    SweepJob,
    Objective,
    SweepJobLimits,
)

from ..search_space import create_search_space


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


def create_dry_run_sweep_job_for_backbone(
    script_path: Path,
    data_asset: Data,
    environment: Environment,
    compute_cluster: str,
    backbone: str,
    smoke_hpo_config: Dict[str, Any],
    configs: Dict[str, Any],
    config_metadata: Dict[str, str],
    aml_experiment_name: str,
    stage: str,
) -> SweepJob:
    """
    Build a small HPO sweep job used as a smoke test for a backbone.

    The dry run uses a reduced search space (no backbone dimension and
    fewer trials) to validate that data access, training, and metrics
    wiring all function correctly before launching the full HPO sweep.

    Args:
        script_path: Path to the training script within the repo.
        data_asset: Registered data asset providing training data.
        environment: Azure ML environment to run the sweep in.
        compute_cluster: Name of the compute cluster to target.
        backbone: Backbone identifier (e.g. ``distilbert``).
        smoke_hpo_config: Parsed smoke HPO configuration.
        configs: Global configuration mapping (for context only).
        config_metadata: Precomputed configuration metadata for tagging.
        aml_experiment_name: AML experiment name for this stage/backbone.
        stage: Logical experiment stage (e.g. ``smoke``).

    Returns:
        Configured :class:`SweepJob` ready for submission.

    Raises:
        FileNotFoundError: If the training script is missing.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    reduced = {
        "search_space": {
            k: v for k, v in smoke_hpo_config["search_space"].items() if k != "backbone"
        }
    }
    search_space = create_search_space(reduced)

    trials = max(2, smoke_hpo_config["sampling"]["max_trials"] // 2)

    cmd_args = (
        f"--data-asset ${{{{inputs.data}}}} "
        f"--config-dir config "
        f"--backbone {backbone} "
        f"--learning-rate ${{{{search_space.learning_rate}}}} "
        f"--batch-size ${{{{search_space.batch_size}}}} "
        f"--dropout ${{{{search_space.dropout}}}} "
        f"--weight-decay ${{{{search_space.weight_decay}}}}"
    )

    data_input = _build_data_input_from_asset(data_asset)

    trial_job = command(
        code="..",
        command=f"python src/{script_path.name} {cmd_args}",
        inputs={"data": data_input},
        environment=environment,
        compute=compute_cluster,
    )

    objective = Objective(
        goal=smoke_hpo_config["objective"]["goal"],
        primary_metric=smoke_hpo_config["objective"]["metric"],
    )
    timeout_seconds = smoke_hpo_config["sampling"]["timeout_minutes"] * 60
    limits = SweepJobLimits(max_total_trials=trials, timeout=timeout_seconds)

    return SweepJob(
        trial=trial_job,
        search_space=search_space,
        sampling_algorithm=smoke_hpo_config["sampling"]["algorithm"],
        objective=objective,
        limits=limits,
        inputs={"data": data_input},
        experiment_name=aml_experiment_name,
        tags={
            **config_metadata,
            "job_type": "dry_run_sweep",
            "backbone": backbone,
            "stage": stage,
        },
        display_name=f"dry-run-sweep-{backbone}",
        description=f"Dry run sweep for {backbone}",
    )


def create_hpo_sweep_job_for_backbone(
    script_path: Path,
    data_asset: Data,
    environment: Environment,
    compute_cluster: str,
    hpo_config: Dict[str, Any],
    backbone: str,
    configs: Dict[str, Any],
    config_metadata: Dict[str, str],
    aml_experiment_name: str,
    stage: str,
) -> SweepJob:
    """
    Build a production HPO sweep job for a specific backbone model.

    The production sweep typically uses a richer search space and more
    trials than the dry run, and is the primary source for selecting the
    best configuration.

    Args:
        script_path: Path to the training script within the repo.
        data_asset: Registered data asset providing training data.
        environment: Azure ML environment to run the sweep in.
        compute_cluster: Name of the compute cluster to target.
        hpo_config: Parsed HPO configuration.
        backbone: Backbone identifier (e.g. ``distilbert``).
        configs: Global configuration mapping (for context only).
        config_metadata: Precomputed configuration metadata for tagging.
        aml_experiment_name: AML experiment name for this stage/backbone.
        stage: Logical experiment stage (e.g. ``hpo``).

    Returns:
        Configured :class:`SweepJob` ready for submission.

    Raises:
        FileNotFoundError: If the training script is missing.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    reduced = {
        "search_space": {
            k: v for k, v in hpo_config["search_space"].items() if k != "backbone"
        }
    }
    search_space = create_search_space(reduced)

    cmd_args = (
        f"--data-asset ${{{{inputs.data}}}} "
        f"--config-dir config "
        f"--backbone {backbone} "
        f"--learning-rate ${{{{search_space.learning_rate}}}} "
        f"--batch-size ${{{{search_space.batch_size}}}} "
        f"--dropout ${{{{search_space.dropout}}}} "
        f"--weight-decay ${{{{search_space.weight_decay}}}}"
    )

    data_input = _build_data_input_from_asset(data_asset)
    trial_job = command(
        code="..",
        command=f"python src/{script_path.name} {cmd_args}",
        inputs={"data": data_input},
        environment=environment,
        compute=compute_cluster,
    )

    objective = Objective(
        goal=hpo_config["objective"]["goal"],
        primary_metric=hpo_config["objective"]["metric"],
    )
    timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60
    limits = SweepJobLimits(
        max_total_trials=hpo_config["sampling"]["max_trials"],
        timeout=timeout_seconds,
    )

    early_termination = None
    if "early_termination" in hpo_config:
        from azure.ai.ml.sweep import BanditPolicy

        et_cfg = hpo_config["early_termination"]
        if et_cfg.get("policy") == "bandit":
            early_termination = BanditPolicy(
                evaluation_interval=et_cfg["evaluation_interval"],
                slack_factor=et_cfg["slack_factor"],
                delay_evaluation=et_cfg["delay_evaluation"],
            )

    return SweepJob(
        trial=trial_job,
        search_space=search_space,
        sampling_algorithm=hpo_config["sampling"]["algorithm"],
        objective=objective,
        limits=limits,
        early_termination=early_termination,
        compute=compute_cluster,
        inputs={"data": data_input},
        experiment_name=aml_experiment_name,
        tags={
            **config_metadata,
            "job_type": "hpo_sweep",
            "backbone": backbone,
            "stage": stage,
        },
        display_name=f"hpo-sweep-{backbone}",
        description=f"Production HPO sweep for {backbone}",
    )


def _validate_job_status(job: Job, job_type: str, backbone: str) -> None:
    """
    Validate that a job has completed successfully.

    Args:
        job: Job instance to validate
        job_type: Type of job for error messages (e.g., "Dry run sweep", "HPO sweep")
        backbone: Backbone model name for error messages

    Raises:
        ValueError: If job status is not "Completed"
    """
    if job.status != "Completed":
        raise ValueError(
            f"{job_type} job for {backbone} failed with status: {job.status}")


def _get_trial_count(job: Job, ml_client: MLClient | None = None) -> int | None:
    """
    Get trial count from job, with fallback to API call if needed.

    Args:
        job: Job instance to check
        ml_client: Optional ML client for fallback API call

    Returns:
        Trial count if available, None otherwise
    """
    # Check trial_count first (fast, no API call needed)
    if hasattr(job, "trial_count") and job.trial_count and job.trial_count > 0:
        return job.trial_count

    # Only make expensive API call if trial_count is not available and ml_client provided
    if ml_client is not None:
        try:
            children = list(ml_client.jobs.list(parent_job_name=job.name))
            return len(children) if children else None
        except Exception:
            return None

    return None


def validate_sweep_job(
    job: Job,
    backbone: str,
    job_type: str = "Sweep",
    min_expected_trials: int | None = None,
    ml_client: MLClient | None = None,
) -> None:
    """
    Validate sweep job completed successfully with required trials.

    This is the unified validation function for all sweep job types (dry run, HPO, etc.).

    Args:
        job: Completed sweep job instance
        backbone: Backbone model name for error messages
        job_type: Type of job for error messages (e.g., "Dry run sweep", "HPO sweep")
        min_expected_trials: Minimum number of trials expected. If None, only checks for > 0 trials.
        ml_client: Optional ML client for fallback trial count retrieval

    Raises:
        ValueError: If validation fails
    """
    _validate_job_status(job, job_type, backbone)

    trial_count = _get_trial_count(job, ml_client)

    if trial_count is None or trial_count == 0:
        error_msg = f"{job_type} job for {backbone} produced no trials"
        if job_type == "Dry run sweep":
            error_msg += f" (parent run: {job.name}). Check sweep logs and child runs in portal."
        raise ValueError(error_msg)

    if min_expected_trials is not None and trial_count < min_expected_trials:
        raise ValueError(
            f"{job_type} job for {backbone} only produced {trial_count} trial(s), "
            f"expected at least {min_expected_trials}"
        )

