"""Convert a trained Hugging Face token-classification checkpoint to ONNX.

This script is executed inside an Azure ML command job and is expected to:

* Read a training checkpoint (a folder created via ``save_pretrained``)
* Export an ONNX model into the provided output folder
* Optionally apply dynamic int8 quantization and run a smoke test
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional

# Import azureml.mlflow early to register the 'azureml' URI scheme before MLflow initializes
# This must happen before mlflow is imported to ensure the scheme is registered
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri and "azureml" in tracking_uri.lower():
    try:
        import azureml.mlflow  # noqa: F401
    except ImportError:
        # If azureml.mlflow is not available, fallback to local tracking
        print(
            "  [Conversion] WARNING: azureml.mlflow not available, but Azure ML URI detected. "
            "Falling back to local tracking. Install azureml-mlflow to use Azure ML tracking.",
            file=sys.stderr, flush=True)
        # Override with local tracking URI
        from shared.mlflow_setup import _get_local_tracking_uri
        tracking_uri = _get_local_tracking_uri()
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        # CRITICAL: Clear MLFLOW_RUN_ID and MLFLOW_USE_RUN_ID if falling back to local
        # This prevents trying to use an Azure ML run ID with a local SQLite store
        if "MLFLOW_RUN_ID" in os.environ:
            del os.environ["MLFLOW_RUN_ID"]
            print("  [Conversion] Cleared MLFLOW_RUN_ID for local fallback.", file=sys.stderr, flush=True)
        if "MLFLOW_USE_RUN_ID" in os.environ:
            del os.environ["MLFLOW_USE_RUN_ID"]
            print("  [Conversion] Cleared MLFLOW_USE_RUN_ID for local fallback.", file=sys.stderr, flush=True)
        print("  [Conversion] A new local run will be created.", file=sys.stderr, flush=True)

import mlflow
# Import tracking.mlflow to ensure Azure ML compatibility patch is applied
from tracking.mlflow import log_artifact_safe  # noqa: F401

from .cli import parse_conversion_arguments
from .onnx_exporter import export_to_onnx
from .smoke_test import run_smoke_test
from platform_adapters import get_platform_adapter
from platform_adapters.checkpoint_resolver import CheckpointResolver
from shared.argument_parsing import validate_config_dir
from shared.logging_utils import get_script_logger

_log = get_script_logger("convert_to_onnx")


def resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    """Resolve checkpoint path to HF checkpoint directory using platform adapter."""
    platform_adapter = get_platform_adapter()
    checkpoint_resolver = platform_adapter.get_checkpoint_resolver()

    _log.info(f"Resolving checkpoint directory from '{checkpoint_path}'")
    return checkpoint_resolver.resolve_checkpoint_dir(checkpoint_path)


def main(tracker: Optional[Any] = None, source_training_run: Optional[str] = None) -> None:
    """
    Main conversion entry point.

    Args:
        tracker: Optional MLflowConversionTracker instance for logging (legacy).
        source_training_run: Optional MLflow run ID of training that produced checkpoint (legacy).
    """
    args = parse_conversion_arguments()
    _log.info(
        "Starting model conversion job with arguments: "
        f"checkpoint_path='{args.checkpoint_path}', "
        f"config_dir='{args.config_dir}', "
        f"backbone='{args.backbone}', "
        f"output_dir='{args.output_dir}', "
        f"quantize_int8={args.quantize_int8}, "
        f"run_smoke_test={args.run_smoke_test}, "
        f"opset_version={getattr(args, 'opset_version', 18)}"
    )

    config_dir = validate_config_dir(args.config_dir)
    _log.info(f"Using config directory: '{config_dir}'")

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)

    # Use platform adapter for output directory resolution
    platform_adapter = get_platform_adapter(
        default_output_dir=Path(args.output_dir))
    output_resolver = platform_adapter.get_output_path_resolver()
    output_dir = output_resolver.resolve_output_path(
        "onnx_model", default=Path(args.output_dir))
    output_dir = output_resolver.ensure_output_directory(output_dir)
    _log.info(
        f"Resolved checkpoint directory to '{checkpoint_dir}', "
        f"output directory to '{output_dir}'"
    )

    # Determine conversion type and parameters
    conversion_target = "onnx_int8" if args.quantize_int8 else "onnx_fp32"
    quantization = "int8" if args.quantize_int8 else "none"
    # Get from args if available
    opset_version = getattr(args, 'opset_version', 18)
    backbone = args.backbone

    # Calculate original checkpoint size for compression ratio
    original_checkpoint_size_mb = None
    try:
        total_size = sum(
            f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
        original_checkpoint_size_mb = total_size / (1024 * 1024)
    except Exception as e:
        _log.debug(f"Could not calculate checkpoint size: {e}")

    # Check if we should use an existing run (from parent process)
    use_run_id = os.environ.get("MLFLOW_RUN_ID")
    started_run_directly = False

    conversion_success = False
    smoke_test_passed = None
    onnx_path = None
    conversion_log_path = None

    try:
        # Start MLflow run if run_id provided (new pattern) or tracker provided (legacy)
        if use_run_id:
            # New pattern: parent process created run, we just start it and log
            _log.info(f"Using existing MLflow run: {use_run_id[:12]}...")
            mlflow.start_run(run_id=use_run_id)
            started_run_directly = True

            # Log conversion parameters
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
                    opset_version=opset_version,  # Pass opset_version
                )
                _log.info(
                    f"Conversion completed. ONNX model written to '{onnx_path}'")
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
            elif not args.run_smoke_test:
                _log.info("Smoke test not requested; skipping")

            # Log conversion results
            mlflow.log_metric("conversion_success",
                              1 if conversion_success else 0)
            if onnx_path and onnx_path.exists():
                model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                mlflow.log_metric("onnx_model_size_mb", model_size_mb)
                if original_checkpoint_size_mb:
                    compression_ratio = original_checkpoint_size_mb / model_size_mb
                    mlflow.log_metric("compression_ratio", compression_ratio)
            if smoke_test_passed is not None:
                mlflow.log_metric("smoke_test_passed",
                                  1 if smoke_test_passed else 0)

            # Log ONNX model as artifact
            if onnx_path and onnx_path.exists():
                success = log_artifact_safe(
                    local_path=onnx_path,
                    artifact_path="onnx_model",
                    run_id=None,  # Use active run
                    max_retries=5,
                )
                if success:
                    _log.info(f"Logged ONNX model to MLflow: {onnx_path}")
                else:
                    _log.warning(
                        f"Failed to log ONNX model artifact to MLflow. "
                        f"Model is still available at: {onnx_path}"
                    )
                    # Don't fail the entire conversion if artifact logging fails
                    # The model file is still created successfully

        elif tracker:
            try:
                with tracker.start_conversion_run(
                    run_name=run_name,
                    conversion_type=conversion_target,
                    source_training_run=source_training_run,
                ):
                    # Log conversion parameters
                    tracker.log_conversion_parameters(
                        checkpoint_path=args.checkpoint_path,
                        conversion_target=conversion_target,
                        quantization=quantization,
                        opset_version=opset_version,
                        backbone=backbone,
                    )

                    # Perform conversion
                    try:
                        onnx_path = export_to_onnx(
                            checkpoint_dir=checkpoint_dir,
                            output_dir=output_dir,
                            quantize_int8=args.quantize_int8,
                        )
                        _log.info(
                            f"Conversion completed. ONNX model written to '{onnx_path}'")
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
                    elif not args.run_smoke_test:
                        _log.info("Smoke test not requested; skipping")

                    # Log conversion results
                    tracker.log_conversion_results(
                        conversion_success=conversion_success,
                        onnx_model_path=onnx_path,
                        original_checkpoint_size=original_checkpoint_size_mb,
                        smoke_test_passed=smoke_test_passed,
                        conversion_log_path=conversion_log_path,
                    )
            except Exception as e:
                _log.warning(
                    f"Could not log conversion results to MLflow: {e}")
                # Still perform conversion even if tracking fails
                if not conversion_success:
                    onnx_path = export_to_onnx(
                        checkpoint_dir=checkpoint_dir,
                        output_dir=output_dir,
                        quantize_int8=args.quantize_int8,
                        opset_version=opset_version,
                    )
                    _log.info(
                        f"Conversion completed. ONNX model written to '{onnx_path}'")
                    if args.run_smoke_test:
                        run_smoke_test(onnx_path, checkpoint_dir)
        else:
            # No MLflow tracking - perform conversion normally
            onnx_path = export_to_onnx(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                quantize_int8=args.quantize_int8,
                opset_version=opset_version,
            )
            _log.info(
                f"Conversion completed. ONNX model written to '{onnx_path}'")

            if args.run_smoke_test:
                run_smoke_test(onnx_path, checkpoint_dir)
            else:
                _log.info("Smoke test not requested; skipping")

    finally:
        # End the run if we started it directly
        if started_run_directly:
            mlflow.end_run()
            _log.info("Ended MLflow run")


if __name__ == "__main__":
    main()
