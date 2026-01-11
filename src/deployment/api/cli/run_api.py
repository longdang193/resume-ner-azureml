"""
@meta
name: api_server_cli
type: script
domain: api
responsibility:
  - Start FastAPI server for Resume NER model
  - Parse command-line arguments
  - Configure server settings
inputs:
  - ONNX model file
  - Checkpoint directory
outputs:
  - Running API server
tags:
  - entrypoint
  - api
  - fastapi
ci:
  runnable: true
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""CLI script to start the FastAPI server."""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from ..config import APIConfig


def setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start the Resume NER API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--onnx-model",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (for tokenizer)",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Server host (default: {APIConfig.API_HOST})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Server port (default: {APIConfig.API_PORT})",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: {APIConfig.API_WORKERS})",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {APIConfig.LOG_LEVEL})",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    # Set model paths
    onnx_path = Path(args.onnx_model)
    checkpoint_dir = Path(args.checkpoint)
    
    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    
    APIConfig.set_model_paths(onnx_path, checkpoint_dir)
    
    # Override config with CLI arguments
    if args.host:
        APIConfig.API_HOST = args.host
    if args.port:
        APIConfig.API_PORT = args.port
    if args.workers:
        APIConfig.API_WORKERS = args.workers
    if args.log_level:
        APIConfig.LOG_LEVEL = args.log_level
    
    # Setup logging
    setup_logging(APIConfig.LOG_LEVEL)
    
    # Validate configuration
    try:
        APIConfig.validate()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Start server
    uvicorn.run(
        "src.api.app:app",
        host=APIConfig.API_HOST,
        port=APIConfig.API_PORT,
        workers=APIConfig.API_WORKERS if not args.reload else 1,
        reload=args.reload,
        log_level=APIConfig.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()

