"""
@meta
name: conversion_execution
type: script
domain: conversion
responsibility:
  - Main entry point for conversion subprocess execution
  - Coordinate checkpoint resolution
  - Execute ONNX export
  - Run smoke tests
  - Log results to MLflow
inputs:
  - PyTorch checkpoint directory
  - Conversion configuration
outputs:
  - ONNX model file
  - MLflow metrics and artifacts
tags:
  - entrypoint
  - conversion
  - onnx
  - mlflow
ci:
  runnable: true
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Conversion script entry point for subprocess execution.

This module provides the main entry point for conversion execution when called
as a subprocess from the orchestration layer. It coordinates checkpoint resolution,
ONNX export, and smoke testing.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Import azureml.mlflow BEFORE importing mlflow to register the URI scheme
# MLflow registers URI schemes at import time via entrypoints
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri and "azureml" in tracking_uri.lower():
    try:
        from common.shared.mlflow_setup import _try_import_azureml_mlflow
        _try_import_azureml_mlflow()
    except Exception:
        # If import fails, we'll handle fallback in main() before using MLflow
        pass

import mlflow
from infrastructure.tracking.mlflow import log_artifact_safe  # noqa: F401

from .cli import parse_conversion_arguments
from .export import export_to_onnx
from .testing import run_smoke_test
from infrastructure.platform.adapters import get_platform_adapter
from common.shared.argument_parsing import validate_config_dir
from common.shared.logging_utils import get_script_logger
from common.shared.mlflow_setup import _get_local_tracking_uri

_log = get_script_logger("conversion.execution")


def resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    """Resolve checkpoint path to HF checkpoint directory using platform adapter."""
    platform_adapter = get_platform_adapter()
    checkpoint_resolver = platform_adapter.get_checkpoint_resolver()
    _log.info(f"Resolving checkpoint directory from '{checkpoint_path}'")
    return checkpoint_resolver.resolve_checkpoint_dir(checkpoint_path)


def main() -> None:
    """
    Main conversion entry point for subprocess execution.
    
    This function is called when the conversion script is executed as a subprocess.
    It coordinates checkpoint resolution, ONNX export, smoke testing, and MLflow tracking.
    """
    args = parse_conversion_arguments()
    _log.info(
        f"Starting conversion: checkpoint='{args.checkpoint_path}', "
        f"backbone='{args.backbone}', quantize_int8={args.quantize_int8}, "
        f"opset_version={getattr(args, 'opset_version', 18)}"
    )

    config_dir = validate_config_dir(args.config_dir)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)

    # Use platform adapter for output directory resolution
    platform_adapter = get_platform_adapter(default_output_dir=Path(args.output_dir))
    output_resolver = platform_adapter.get_output_path_resolver()
    output_dir = output_resolver.resolve_output_path(
        "onnx_model", default=Path(args.output_dir))
    output_dir = output_resolver.ensure_output_directory(output_dir)
    _log.info(f"Output directory: '{output_dir}'")

    # Determine conversion parameters
    conversion_target = "onnx_int8" if args.quantize_int8 else "onnx_fp32"
    quantization = "int8" if args.quantize_int8 else "none"
    opset_version = getattr(args, 'opset_version', 18)
    backbone = args.backbone

    # Calculate original checkpoint size for compression ratio
    original_checkpoint_size_mb = None
    try:
        total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
        original_checkpoint_size_mb = total_size / (1024 * 1024)
    except Exception:
        pass

    use_run_id = os.environ.get("MLFLOW_RUN_ID")
    started_run_directly = False
    conversion_success = False
    smoke_test_passed = None
    onnx_path = None

    try:
        # Verify Azure ML scheme is registered before starting run
        current_tracking_uri = mlflow.get_tracking_uri()
        if current_tracking_uri and "azureml" in current_tracking_uri.lower():
            # Ensure azureml.mlflow is imported to register the scheme
            # This should have been done at module import time, but ensure it's done here too
            try:
                from common.shared.mlflow_setup import _try_import_azureml_mlflow
                _try_import_azureml_mlflow()
            except Exception as e:
                _log.debug(f"Could not import azureml.mlflow: {e}")
            
            # Ensure compatibility patch is applied (for artifact logging)
            try:
                from infrastructure.tracking.mlflow.compatibility import apply_azureml_artifact_patch
                apply_azureml_artifact_patch()
            except Exception as e:
                _log.debug(f"Could not apply Azure ML compatibility patch: {e}")
            
            # Check if scheme is registered by trying to get the store builder
            # Note: Even if this check fails, we'll still try to use Azure ML if MLFLOW_RUN_ID is set
            # The check is mainly for warning purposes
            try:
                from mlflow.tracking._tracking_service.registry import _tracking_store_registry
                # Try to get the store builder - this will raise KeyError if scheme not registered
                builder = _tracking_store_registry.get_store_builder("azureml://test")
                if builder is None:
                    raise KeyError("Azure ML scheme builder is None")
            except (KeyError, AttributeError, Exception) as e:
                # Scheme check failed, but if we have a run_id, try to use it anyway
                # The artifact logging compatibility patch might still work
                if use_run_id:
                    _log.warning(
                        f"Azure ML scheme registration check failed (error: {e}), "
                        "but continuing with Azure ML run_id. Artifact logging may fail."
                    )
                    # Don't remove run_id - let's try to use it anyway
                else:
                    # No run_id, so fall back to local tracking
                    _log.warning(
                        f"Azure ML scheme not registered (error: {e}), falling back to local tracking."
                    )
                    local_uri = _get_local_tracking_uri()
                    mlflow.set_tracking_uri(local_uri)
                    os.environ["MLFLOW_TRACKING_URI"] = local_uri

        # Start MLflow run if run_id provided
        if use_run_id:
            _log.info(f"Using MLflow run: {use_run_id[:12]}...")
            mlflow.start_run(run_id=use_run_id)
            started_run_directly = True

            mlflow.log_param("conversion_source", args.checkpoint_path)
            mlflow.log_param("conversion_target", conversion_target)
            mlflow.log_param("onnx_opset_version", opset_version)
            mlflow.log_param("conversion_backbone", backbone)
            mlflow.log_param("quantization", quantization)

        # Perform conversion
        try:
            onnx_path = export_to_onnx(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                quantize_int8=args.quantize_int8,
                opset_version=opset_version,
            )
            _log.info(f"Conversion completed. ONNX model: '{onnx_path}'")
            conversion_success = True
        except Exception as e:
            _log.error(f"Conversion failed: {e}")
            conversion_success = False
            raise

        # Run smoke test if requested
        if args.run_smoke_test and conversion_success:
            try:
                run_smoke_test(onnx_path, checkpoint_dir)
                smoke_test_passed = True
                _log.info("Smoke test passed")
            except Exception as e:
                smoke_test_passed = False
                _log.warning(f"Smoke test failed: {e}")

        # Log conversion results to MLflow if run is active
        if started_run_directly:
            mlflow.log_metric("conversion_success", 1 if conversion_success else 0)
            if onnx_path and onnx_path.exists():
                model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                mlflow.log_metric("onnx_model_size_mb", model_size_mb)
                if original_checkpoint_size_mb:
                    compression_ratio = original_checkpoint_size_mb / model_size_mb
                    mlflow.log_metric("compression_ratio", compression_ratio)
            if smoke_test_passed is not None:
                mlflow.log_metric("smoke_test_passed", 1 if smoke_test_passed else 0)

            # Log ONNX model as artifact
            if onnx_path and onnx_path.exists():
                success = log_artifact_safe(
                    local_path=onnx_path,
                    artifact_path="onnx_model",
                    run_id=None,
                    max_retries=5,
                )
                if success:
                    _log.info(f"Logged ONNX model to MLflow: {onnx_path}")
                else:
                    _log.warning(
                        f"Failed to log ONNX model artifact to MLflow. "
                        f"Model available at: {onnx_path}"
                    )

    finally:
        if started_run_directly:
            mlflow.end_run()


if __name__ == "__main__":
    main()
