from __future__ import annotations

"""
@meta
name: conversion_azureml
type: utility
domain: conversion
responsibility:
  - Create Azure ML conversion jobs
  - Extract checkpoint outputs from training jobs
  - Validate conversion job completion
inputs:
  - Training job outputs
  - Azure ML environment and compute
outputs:
  - Azure ML conversion job definitions
tags:
  - utility
  - conversion
  - azureml
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: true
lifecycle:
  status: active
"""

"""Azure ML job creation and validation for model conversion.

This module provides Azure ML-specific functionality for creating and validating
conversion jobs. It handles checkpoint extraction from training jobs, conversion
job creation, and job validation.
"""
from pathlib import Path
from typing import Any, Dict, Optional

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import Environment, Job

from common.constants import CONVERSION_JOB_NAME

def get_checkpoint_output_from_training_job(
    training_job: Job, ml_client: Optional[MLClient] = None
) -> Any:
    """Extract checkpoint output reference from a completed training job.

    Azure ML auto-registers job outputs as data assets. We prefer using the
    actual output reference (which may be a data asset URI like
    ``azureml:azureml_<job_name>_output_data_checkpoint:1``) over constructing
    datastore URIs manually, as it's more reliable and handles edge cases.

    Args:
        training_job: Completed training job with checkpoint output.
        ml_client: Optional MLClient to fetch data asset if output URI is not available.

    Returns:
        Checkpoint output reference (URI string, data asset reference, or Job object).

    Raises:
        ValueError: If training job has no checkpoint output or did not complete successfully.
    """
    # Validate that the training job completed successfully
    if training_job.status != "Completed":
        raise ValueError(
            f"Training job {training_job.name} did not complete successfully. "
            f"Status: {training_job.status}. Cannot use checkpoint output from failed job."
        )
    
    # First, try to extract URI/path from the job's outputs (works if job was just created)
    # This is checked first because it doesn't require ml_client
    if hasattr(training_job, "outputs") and training_job.outputs:
        checkpoint_output = training_job.outputs.get("checkpoint")
        if checkpoint_output is not None:
            # Prefer .uri (data asset reference) over .path (legacy)
            if hasattr(checkpoint_output, "uri") and checkpoint_output.uri:
                return checkpoint_output.uri
            elif hasattr(checkpoint_output, "path") and checkpoint_output.path:
                return checkpoint_output.path
            # If output exists but has no URI/path, we'll need ml_client below
    
    # If job outputs don't have URI/path (e.g., when reloading from cache), we need to
    # construct or fetch the auto-registered data asset. Azure ML auto-registers job outputs
    # as data assets with the pattern: azureml_<job_name>_output_data_<output_name>:1
    
    # Construct the expected data asset name
    data_asset_name = f"azureml_{training_job.name}_output_data_checkpoint"
    
    if ml_client is not None:
        # Try to fetch the data asset to verify it exists and get the exact reference
        try:
            data_asset = ml_client.data.get(name=data_asset_name, version="1")
            # Return the data asset reference in the format Azure ML expects
            asset_ref = f"azureml:{data_asset.name}:{data_asset.version}"
            return asset_ref
        except Exception as e:
            # If fetching fails, construct the reference directly (less reliable but works)
            import warnings
            warnings.warn(
                f"Could not fetch data asset '{data_asset_name}' for training job '{training_job.name}': {e}. "
                "Constructing data asset reference directly. This may cause issues if the asset doesn't exist."
            )
            # Fall through to direct construction
    else:
        # ml_client not provided - construct the data asset reference directly
        # This is less reliable but allows the function to work without ml_client
        import warnings
        warnings.warn(
            f"ml_client not provided for training job '{training_job.name}'. "
            "Constructing data asset reference directly. For best results, provide ml_client."
        )
    
    # Construct the data asset reference directly (works if the asset exists)
    # Azure ML auto-registers with version "1" for the first output
    asset_ref = f"azureml:{data_asset_name}:1"
    return asset_ref

def _get_job_output_reference(
    job: Job,
    output_name: str,
    ml_client: Optional[MLClient] = None,
) -> str:
    """Resolve an Azure ML job output to a stable reference usable as an Input/Model path."""
    if hasattr(job, "outputs") and job.outputs and output_name in job.outputs:
        out = job.outputs[output_name]
        uri = getattr(out, "uri", None)
        if uri:
            return str(uri)
        path = getattr(out, "path", None)
        if path:
            return str(path)

    # Prefer the auto-registered data asset reference (stable even across reloads)
    data_asset_name = f"azureml_{job.name}_output_data_{output_name}"
    if ml_client is not None:
        try:
            data_asset = ml_client.data.get(name=data_asset_name, version="1")
            return f"azureml:{data_asset.name}:{data_asset.version}"
        except Exception:
            pass

    return f"azureml:{data_asset_name}:1"

def create_conversion_job(
    script_path: Path,
    checkpoint_uri: str,
    environment: Environment,
    compute_cluster: str,
    backbone: str,
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Any:
    """Create Azure ML Command Job for model conversion to ONNX with int8 quantization.

    This function assumes:
    - The checkpoint has already been resolved to a valid AML data asset URI or datastore URI.
    - All metadata/tags (config hashes, source job, etc.) are prepared by the caller.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Conversion script not found: {script_path}")

    # Basic validation of checkpoint format
    if not (checkpoint_uri.startswith("azureml:") or checkpoint_uri.startswith("azureml://")):
        raise ValueError(
            f"Unexpected checkpoint URI format: {checkpoint_uri}. "
            "Expected a data asset reference (azureml:...) or datastore URI (azureml://...)."
        )

    checkpoint_input = Input(type="uri_folder", path=checkpoint_uri)

    command_args = (
        f"--checkpoint-path ${{{{inputs.checkpoint}}}} "
        f"--config-dir config "
        f"--backbone {backbone} "
        f"--output-dir ${{{{outputs.onnx_model}}}} "
        f"--quantize-int8 "
        f"--run-smoke-test"
    )

    # Use project root as code snapshot so both `src/` and `config/` are included.
    return command(
        code="..",
        command=f"python -m conversion.execution {command_args}",
        inputs={
            "checkpoint": checkpoint_input,
        },
        outputs={
            "onnx_model": Output(type="uri_folder"),
        },
        environment=environment,
        compute=compute_cluster,
        experiment_name=experiment_name,
        tags=tags or {},
        display_name=CONVERSION_JOB_NAME,
        description="Convert PyTorch checkpoint to optimized ONNX model (int8 quantized)",
    )

def validate_conversion_job(job: Job, ml_client: Optional[MLClient] = None) -> None:
    """Validate conversion job completed successfully with required ONNX model output.

    Args:
        job: Completed job instance
        ml_client: Optional MLClient to resolve the output reference when SDK doesn't populate it.

    Raises:
        ValueError: If validation fails
    """
    if job.status != "Completed":
        raise ValueError(f"Conversion job failed with status: {job.status}")

    if not hasattr(job, "outputs") or not job.outputs:
        raise ValueError("Conversion job produced no outputs")

    if "onnx_model" not in job.outputs:
        raise ValueError("Conversion job missing required output: onnx_model")

    onnx_ref = _get_job_output_reference(job, "onnx_model", ml_client=ml_client)
    if not onnx_ref or not (onnx_ref.startswith("azureml:") or onnx_ref.startswith("azureml://")):
        raise ValueError(f"Invalid ONNX model output reference: {onnx_ref}")

