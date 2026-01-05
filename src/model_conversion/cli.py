"""Command-line argument parsing for conversion script."""

import argparse

from shared.argument_parsing import (
    add_config_dir_argument,
    add_backbone_argument,
)


def parse_conversion_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for model conversion.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to ONNX")
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint directory",
    )
    add_config_dir_argument(parser)
    add_backbone_argument(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Enable int8 quantization",
    )
    parser.add_argument(
        "--run-smoke-test",
        action="store_true",
        help="Run smoke inference test after conversion",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    
    return parser.parse_args()

