"""Convert a trained Hugging Face token-classification checkpoint to ONNX.

This script is executed inside an Azure ML command job and is expected to:

* Read a training checkpoint (a folder created via ``save_pretrained``)
* Export an ONNX model into the provided output folder
* Optionally apply dynamic int8 quantization and run a smoke test
"""

from pathlib import Path
from typing import Any, Optional

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
        tracker: Optional MLflowConversionTracker instance for logging.
        source_training_run: Optional MLflow run ID of training that produced checkpoint.
    """
    args = parse_conversion_arguments()
    _log.info(
        "Starting model conversion job with arguments: "
        f"checkpoint_path='{args.checkpoint_path}', "
        f"config_dir='{args.config_dir}', "
        f"backbone='{args.backbone}', "
        f"output_dir='{args.output_dir}', "
        f"quantize_int8={args.quantize_int8}, "
        f"run_smoke_test={args.run_smoke_test}"
    )
    
    config_dir = validate_config_dir(args.config_dir)
    _log.info(f"Using config directory: '{config_dir}'")
    
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_path)
    
    # Use platform adapter for output directory resolution
    platform_adapter = get_platform_adapter(default_output_dir=Path(args.output_dir))
    output_resolver = platform_adapter.get_output_path_resolver()
    output_dir = output_resolver.resolve_output_path("onnx_model", default=Path(args.output_dir))
    output_dir = output_resolver.ensure_output_directory(output_dir)
    _log.info(
        f"Resolved checkpoint directory to '{checkpoint_dir}', "
        f"output directory to '{output_dir}'"
    )
    
    # Determine conversion type and parameters
    conversion_target = "onnx_int8" if args.quantize_int8 else "onnx_fp32"
    quantization = "int8" if args.quantize_int8 else "none"
    opset_version = 18  # Hardcoded in onnx_exporter
    backbone = args.backbone
    
    # Calculate original checkpoint size for compression ratio
    original_checkpoint_size_mb = None
    try:
        total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
        original_checkpoint_size_mb = total_size / (1024 * 1024)
    except Exception as e:
        _log.debug(f"Could not calculate checkpoint size: {e}")
    
    # Create run name
    run_name = f"conversion_{backbone}_{conversion_target}"
    
    # Start MLflow tracking run if tracker provided
    conversion_success = False
    smoke_test_passed = None
    onnx_path = None
    conversion_log_path = None
    
    if tracker:
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
                    _log.info(f"Conversion completed. ONNX model written to '{onnx_path}'")
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
            _log.warning(f"Could not log conversion results to MLflow: {e}")
            # Still perform conversion even if tracking fails
            if not conversion_success:
                onnx_path = export_to_onnx(
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    quantize_int8=args.quantize_int8,
                )
                _log.info(f"Conversion completed. ONNX model written to '{onnx_path}'")
                if args.run_smoke_test:
                    run_smoke_test(onnx_path, checkpoint_dir)
    else:
        # No tracker - perform conversion normally
        onnx_path = export_to_onnx(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            quantize_int8=args.quantize_int8,
        )
        _log.info(f"Conversion completed. ONNX model written to '{onnx_path}'")
        
        if args.run_smoke_test:
            run_smoke_test(onnx_path, checkpoint_dir)
        else:
            _log.info("Smoke test not requested; skipping")


if __name__ == "__main__":
    main()





