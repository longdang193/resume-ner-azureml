from __future__ import annotations

"""
@meta
name: conversion_orchestration
type: script
domain: conversion
responsibility:
  - Orchestrate model conversion workflow
  - Load conversion configuration
  - Build output directories
  - Create MLflow runs
  - Execute conversion subprocess
  - Handle errors and cleanup
inputs:
  - conversion.yaml
  - Parent training checkpoint
  - Parent training fingerprints
outputs:
  - ONNX model directory
  - MLflow conversion run
tags:
  - orchestration
  - conversion
  - mlflow
ci:
  runnable: true
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""High-level orchestration for model conversion workflow.

This module orchestrates the conversion workflow from the orchestration layer:
- Loads conversion configuration
- Builds output directories
- Creates MLflow runs
- Executes conversion subprocess
- Handles errors and cleanup
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from infrastructure.config.loader import ExperimentConfig
from infrastructure.config.conversion import load_conversion_config
from infrastructure.naming import create_naming_context
from infrastructure.paths import build_output_path
from infrastructure.tracking.mlflow.naming import (
    build_mlflow_run_name,
    build_mlflow_tags,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
from infrastructure.tracking.mlflow.index import update_mlflow_index
from common.shared.platform_detection import detect_platform
from common.shared.logging_utils import get_script_logger

_log = get_script_logger("conversion.orchestration")

def execute_conversion(
    root_dir: Path,
    config_dir: Path,
    parent_training_output_dir: Path,
    parent_spec_fp: str,
    parent_exec_fp: str,
    experiment_config: ExperimentConfig,
    conversion_experiment_name: str,
    platform: str,
    parent_training_run_id: Optional[str] = None,
) -> Path:
    """
    Execute model conversion to ONNX format.
    
    This function:
    1. Loads conversion config from conversion.yaml using load_conversion_config()
    2. Builds conversion context and output directory
    3. Executes conversion as subprocess
    4. Sets MLflow tracking
    5. Returns ONNX model directory path
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        parent_training_output_dir: Output directory of parent final training run.
        parent_spec_fp: Parent training specification fingerprint.
        parent_exec_fp: Parent training execution fingerprint.
        experiment_config: Experiment configuration.
        conversion_experiment_name: MLflow experiment name for conversion runs.
        platform: Platform name (local, colab, kaggle).
        parent_training_run_id: Optional MLflow run ID of parent training run.
    
    Returns:
        Path to conversion output directory (contains ONNX model).
    
    Raises:
        RuntimeError: If conversion subprocess fails.
        ValueError: If required configuration is missing.
    """
    # Load conversion config (uses conversion.yaml)
    conversion_config = load_conversion_config(
        root_dir=root_dir,
        config_dir=config_dir,
        parent_training_output_dir=parent_training_output_dir,
        parent_spec_fp=parent_spec_fp,
        parent_exec_fp=parent_exec_fp,
        experiment_config=experiment_config,
    )
    
    # Get fingerprints and config from resolved config
    conv_fp = conversion_config.get("conv_fp")
    parent_training_id = conversion_config.get("parent_training_id")
    backbone = conversion_config.get("backbone")  # Canonical backbone from metadata
    
    # Build conversion context (executor computes output dir - single source of truth)
    environment = detect_platform()
    # Use short name only for UI naming, keep canonical backbone for model loading
    if "-" in backbone:
        backbone_short = backbone.split("-")[0]
    else:
        backbone_short = backbone
    
    conversion_context = create_naming_context(
        process_type="conversion",
        model=backbone_short,
        spec_fp=parent_spec_fp,
        exec_fp=parent_exec_fp,
        environment=environment,
        parent_training_id=parent_training_id,
        conv_fp=conv_fp,
    )
    
    # Build output directory (executor is source of truth)
    conversion_output_dir = build_output_path(root_dir, conversion_context)
    conversion_output_dir.mkdir(parents=True, exist_ok=True)
    
    _log.info(f"Output directory: {conversion_output_dir}")
    
    # Get conversion parameters
    checkpoint_path = conversion_config["checkpoint_path"]
    quantization = conversion_config["onnx"]["quantization"]
    opset_version = conversion_config["onnx"]["opset_version"]
    run_smoke_test = conversion_config["onnx"]["run_smoke_test"]
    filename_pattern = conversion_config.get("output", {}).get("filename_pattern", "model_{quantization}.onnx")
    
    # Build conversion command arguments
    conversion_args = [
        sys.executable,
        "-m",
        "conversion.execution",
        "--checkpoint-path",
        checkpoint_path,
        "--config-dir",
        str(config_dir),
        "--backbone",
        backbone,  # Use canonical backbone
        "--output-dir",
        str(conversion_output_dir),
        "--opset-version",
        str(opset_version),  # Pass opset_version
    ]
    
    # Add quantization flag if needed
    if quantization == "int8":
        conversion_args.append("--quantize-int8")
    
    # Add smoke test flag if needed
    if run_smoke_test:
        conversion_args.append("--run-smoke-test")
    
    # Set up environment variables
    conversion_env = os.environ.copy()
    
    # Add src directory to PYTHONPATH
    src_dir = root_dir / "src"
    pythonpath = conversion_env.get("PYTHONPATH", "")
    if pythonpath:
        conversion_env["PYTHONPATH"] = f"{str(src_dir)}{os.pathsep}{pythonpath}"
    else:
        conversion_env["PYTHONPATH"] = str(src_dir)
    
    # Set MLflow tracking environment
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if mlflow_tracking_uri:
        conversion_env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    conversion_env["MLFLOW_EXPERIMENT_NAME"] = conversion_experiment_name
    
    # Create MLflow run in parent process (no active context)
    # Build systematic run name
    run_name = build_mlflow_run_name(
        context=conversion_context,
        config_dir=config_dir,
        root_dir=root_dir,
        output_dir=conversion_output_dir,
    )
    
    # Get or create experiment
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(conversion_experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(conversion_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        # Fallback: use mlflow API
        mlflow.set_experiment(conversion_experiment_name)
        experiment = mlflow.get_experiment_by_name(conversion_experiment_name)
        if experiment is None:
            raise RuntimeError(
                f"Could not get or create experiment: {conversion_experiment_name}") from e
        experiment_id = experiment.experiment_id
    
    # Build tags using build_mlflow_tags
    # NOTE: Do NOT set parent_run_id for cross-experiment lineage (use code.lineage.* only)
    tags = build_mlflow_tags(
        context=conversion_context,
        output_dir=conversion_output_dir,
        parent_run_id=None,  # Fixed: Don't set for cross-experiment
        group_id=None,
        config_dir=config_dir,
    )
    tags["mlflow.runType"] = "conversion"
    tags["conversion.format"] = conversion_config["format"]
    tags["conversion.quantization"] = quantization
    tags["conversion.opset_version"] = str(opset_version)
    tags["mlflow.runName"] = run_name  # Ensure run name is set
    
    # Add lineage tags (link to parent training via code.lineage.* only)
    if parent_training_run_id:
        tags["code.lineage.parent_training_run_id"] = parent_training_run_id
    tags["code.lineage.source"] = "final_training"
    tags["code.lineage.parent_spec_fp"] = parent_spec_fp
    tags["code.lineage.parent_exec_fp"] = parent_exec_fp
    
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
            run_key = build_mlflow_run_key(conversion_context)
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
            _log.debug(f"Could not update MLflow index: {e}")
        
        _log.info(f"Created MLflow run: {run_name} ({run_id[:12]}...)")
        
        # Pass run_id to subprocess (CRITICAL: subprocess must use this)
        conversion_env["MLFLOW_RUN_ID"] = run_id
    except Exception as e:
        _log.warning(f"Could not create MLflow run: {e}")
        # Continue without MLflow run (conversion will still work, just no tracking)
        run_id = None
    
    # Execute conversion (stream output instead of capture_output=True)
    _log.info(f"Running conversion: {' '.join(conversion_args)}")
    process = subprocess.Popen(
        conversion_args,
        cwd=root_dir,
        env=conversion_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr separately for better error reporting
        text=True,
        bufsize=1,  # Line buffered
    )
    
    # Stream output in real-time and capture for error reporting
    stdout_lines = []
    stderr_lines = []
    import threading
    
    def read_stdout():
        """Read stdout in a separate thread."""
        try:
            for line in process.stdout:
                _log.info(line.rstrip())
                stdout_lines.append(line.rstrip())
        except Exception as e:
            _log.debug(f"Error reading stdout: {e}")
    
    def read_stderr():
        """Read stderr in a separate thread."""
        try:
            for line in process.stderr:
                _log.warning(line.rstrip())
                stderr_lines.append(line.rstrip())
        except Exception as e:
            _log.debug(f"Error reading stderr: {e}")
    
    # Start threads to read stdout and stderr concurrently
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process to complete
    returncode = process.wait()
    
    # Wait for threads to finish reading
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    
    # Handle subprocess failure - ensure run is marked as FAILED
    if returncode != 0:
        if run_id:
            from infrastructure.tracking.mlflow import terminate_run_safe
            terminate_run_safe(run_id, status="FAILED", check_status=True)
        
        # Build detailed error message
        error_msg = f"Model conversion failed with return code {returncode}"
        if stderr_lines:
            error_msg += f"\n\nStderr output (last 30 lines):\n" + "\n".join(stderr_lines[-30:])
        if stdout_lines:
            # Check if there are any error-like messages in stdout
            error_lines = [line for line in stdout_lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'fatal'])]
            if error_lines:
                error_msg += f"\n\nError messages from stdout (last 20 lines):\n" + "\n".join(error_lines[-20:])
            else:
                # If no obvious error lines, show last few lines of stdout
                error_msg += f"\n\nLast stdout output (last 10 lines):\n" + "\n".join(stdout_lines[-10:])
        
        raise RuntimeError(error_msg)
    else:
        # Subprocess should have ended the run, but verify it's terminated
        if run_id:
            try:
                from infrastructure.tracking.mlflow import ensure_run_terminated
                ensure_run_terminated(run_id, expected_status="FINISHED")
            except Exception as e:
                _log.debug(f"Could not verify run status: {e}")
    
    # Find ONNX model file (respect filename_pattern)
    onnx_model_path = _find_onnx_model(conversion_output_dir, quantization, filename_pattern)
    
    _log.info(f"Conversion completed. ONNX model: {onnx_model_path}")
    
    return conversion_output_dir

def _find_onnx_model(output_dir: Path, quantization: str, filename_pattern: str) -> Path:
    """
    Find ONNX model file in output directory, respecting filename_pattern.
    
    Args:
        output_dir: Conversion output directory.
        quantization: Quantization type (none, int8, dynamic).
        filename_pattern: Filename pattern from config (e.g., "model_{quantization}.onnx").
    
    Returns:
        Path to ONNX model file.
    """
    # Try filename_pattern first (check both root and onnx_model subdirectory)
    if "{quantization}" in filename_pattern:
        quant_str = "int8" if quantization == "int8" else "fp32"
        expected_name = filename_pattern.format(quantization=quant_str)
        # Check root directory
        expected_path = output_dir / expected_name
        if expected_path.exists():
            return expected_path
        # Check onnx_model subdirectory
        expected_path = output_dir / "onnx_model" / expected_name
        if expected_path.exists():
            return expected_path
    
    # Fallback: try common patterns (check both root and onnx_model subdirectory)
    if quantization == "int8":
        patterns = ["model_int8.onnx", "model.onnx"]
    else:
        patterns = ["model.onnx", "model_fp32.onnx"]
    
    for pattern in patterns:
        # Check root directory
        model_path = output_dir / pattern
        if model_path.exists():
            return model_path
        # Check onnx_model subdirectory
        model_path = output_dir / "onnx_model" / pattern
        if model_path.exists():
            return model_path
    
    # Last resort: find any .onnx file (search recursively, as model may be in subdirectory)
    onnx_files = list(output_dir.rglob("*.onnx"))
    if onnx_files:
        return onnx_files[0]
    
    # If not found, return expected path based on pattern
    if "{quantization}" in filename_pattern:
        quant_str = "int8" if quantization == "int8" else "fp32"
        return output_dir / filename_pattern.format(quantization=quant_str)
    return output_dir / "model.onnx"

