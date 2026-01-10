from __future__ import annotations

"""
@meta
name: environment_config_builder
type: utility
domain: config
responsibility:
  - Build Azure ML environment configuration from YAML
  - Load conda environment specifications
  - Compute environment hashes
  - Get or create Azure ML environments
  - Prepare environment images with warm-up jobs
inputs:
  - env.yaml configuration
  - Conda environment files
outputs:
  - EnvironmentConfig dataclass
  - Azure ML Environment objects
tags:
  - utility
  - config
  - azureml
  - environment
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: true
lifecycle:
  status: active
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError

from .loader import CONFIG_HASH_LENGTH
from common.shared.yaml_utils import load_yaml

# Centralised defaults for the training environment. These can be overridden
# by providing values in the env.yaml config (see EnvironmentConfig below).
DEFAULT_ENVIRONMENT_NAME = "resume-ner-training"
DEFAULT_CONDA_RELATIVE_PATH = Path("environment/conda.yaml")
DEFAULT_DOCKER_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

# Warm-up job configuration
WARMUP_DISPLAY_NAME = "environment-warmup"
WARMUP_HISTORY_LIMIT = 10

@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Resolved configuration for the Azure ML training environment.

    Instances of this class are typically built from ``env.yaml`` via
    :func:`build_environment_config`. The underlying YAML remains the
    single source of truth; this object just provides a convenient,
    strongly-typed view for the orchestrator and helpers.
    """

    name: str
    conda_path: Path
    docker_image: str
    warmup_display_name: str
    warmup_history_limit: int

def build_environment_config(
    config_root: Path,
    env_settings: Optional[Dict[str, Any]] = None,
) -> EnvironmentConfig:
    """
    Build an :class:`EnvironmentConfig` from ``env.yaml`` settings.

    Expected (optional) structure in ``env.yaml``:

    .. code-block:: yaml

        environment:
          name: "resume-ner-training"
          conda_file: "environment/conda.yaml"
          docker_image: "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
          warmup:
            display_name: "environment-warmup"
            history_limit: 10

    Args:
        config_root: Path to the top-level ``config`` directory.
        env_settings: Parsed ``env.yaml`` dictionary (usually ``configs["env"]``).

    Returns:
        EnvironmentConfig populated from YAML values or sensible defaults.
    """
    env_settings = env_settings or {}
    env_section = env_settings.get("environment", {}) or {}

    name = env_section.get("name", DEFAULT_ENVIRONMENT_NAME)

    conda_relative = env_section.get("conda_file", str(DEFAULT_CONDA_RELATIVE_PATH))
    conda_path = config_root / conda_relative

    docker_image = env_section.get("docker_image", DEFAULT_DOCKER_IMAGE)

    warmup_section = env_section.get("warmup", {}) or {}
    warmup_display = warmup_section.get("display_name", WARMUP_DISPLAY_NAME)
    warmup_limit = int(warmup_section.get("history_limit", WARMUP_HISTORY_LIMIT))

    return EnvironmentConfig(
        name=name,
        conda_path=conda_path,
        docker_image=docker_image,
        warmup_display_name=warmup_display,
        warmup_history_limit=warmup_limit,
    )

def load_conda_environment(path: Path) -> Dict[str, Any]:
    """
    Load a conda environment specification from disk.

    Args:
        path: Path to the conda YAML file.

    Returns:
        Parsed conda environment dictionary.

    Raises:
        FileNotFoundError: If the conda file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    # Centralized YAML parsing (keeps behavior consistent across the repo)
    return load_yaml(path)

def compute_environment_hash(conda_deps: Dict[str, Any], docker_image: str) -> str:
    """
    Compute a deterministic short hash for an environment definition.

    The hash is based on both the conda dependencies and the base Docker
    image to ensure that any meaningful change results in a new
    environment version.

    Args:
        conda_deps: Parsed conda environment specification.
        docker_image: Docker image used as the base for the environment.

    Returns:
        Hex string of length ``CONFIG_HASH_LENGTH`` suitable for use as a
        version suffix.
    """
    env_spec = {"conda_dependencies": conda_deps, "docker_image": docker_image}
    env_str = json.dumps(env_spec, sort_keys=True)
    full_hash = hashlib.sha256(env_str.encode("utf-8")).hexdigest()
    return full_hash[:CONFIG_HASH_LENGTH]

def get_or_create_environment(
    ml_client: MLClient,
    name: str,
    version: str,
    conda_dependencies: Dict[str, Any],
    docker_image: str,
) -> Environment:
    """
    Resolve or create an Azure ML :class:`Environment` asset.

    Args:
        ml_client: Azure ML client used for environment operations.
        name: Logical name of the environment.
        version: Version string (often derived from :func:`compute_environment_hash`).
        conda_dependencies: Conda dependencies to attach to the environment.
        docker_image: Base Docker image used to build the environment.

    Returns:
        The resolved or newly created :class:`Environment` asset.
    """
    try:
        return ml_client.environments.get(name=name, version=version)
    except ResourceNotFoundError:
        environment = Environment(
            name=name,
            version=version,
            conda_file=conda_dependencies,
            image=docker_image,
            description=f"Training environment (hash: {version})",
        )
        return ml_client.environments.create_or_update(environment)

def create_training_environment(
    ml_client: MLClient,
    env_config: EnvironmentConfig,
) -> Environment:
    """
    High-level helper to resolve the training environment from configuration.

    This function encapsulates the workflow of:

    * Loading conda dependencies from disk.
    * Computing a hash-based version string.
    * Creating or fetching the corresponding Azure ML environment.

    Args:
        ml_client: Azure ML client used for environment operations.
        env_config: Resolved environment configuration.

    Returns:
        The resolved or newly created training environment.
    """
    conda_deps = load_conda_environment(env_config.conda_path)
    env_hash = compute_environment_hash(conda_deps, env_config.docker_image)
    version = f"v{env_hash}"
    return get_or_create_environment(
        ml_client=ml_client,
        name=env_config.name,
        version=version,
        conda_dependencies=conda_deps,
        docker_image=env_config.docker_image,
    )

def prepare_environment_image(
    ml_client: MLClient,
    environment: Environment,
    compute_cluster: str,
    env_config: EnvironmentConfig,
) -> None:
    """
    Submit a tiny warm-up job to ensure the environment image is built.

    The first usage of a new environment version can incur a non-trivial
    image build time. Running this warm-up job proactively reduces
    latency and avoids timeouts for subsequent training or sweep jobs.

    Args:
        ml_client: Azure ML client used for job operations.
        environment: Target environment whose image should be materialised.
        compute_cluster: Name of the compute cluster on which to run the warm-up.
        env_config: Environment configuration, providing warm-up settings.

    Raises:
        RuntimeError: If the warm-up job fails to complete successfully.
    """
    try:
        jobs = ml_client.jobs.list(
            display_name=env_config.warmup_display_name,
            list_view_type="All",
        )
        for idx, job in enumerate(jobs):
            if idx >= env_config.warmup_history_limit:
                break
            if (
                job.status == "Completed"
                and getattr(getattr(job, "environment", None), "name", None)
                == environment.name
                and getattr(getattr(job, "environment", None), "version", None)
                == environment.version
            ):
                return
    except Exception:
        # If listing jobs fails we still attempt the warm‑up.
        pass

    warmup_job = command(
        code="../src",
        command="python -c \"print('Environment ready')\"",
        environment=environment,
        compute=compute_cluster,
        display_name=env_config.warmup_display_name,
        description=WARMUP_DISPLAY_NAME,
    )

    submitted = ml_client.jobs.create_or_update(warmup_job)
    ml_client.jobs.stream(submitted.name)
    completed = ml_client.jobs.get(submitted.name)

    if completed.status != "Completed":
        raise RuntimeError(
            f"Environment warm‑up job failed with status {completed.status}. "
            "Inspect job logs for details."
        )
