"""Pytest fixtures for API integration tests."""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def onnx_model_path(request) -> str:
    """
    Path to ONNX model file.
    
    Can be provided via command line: pytest --onnx-model <path>
    Or via environment variable: ONNX_MODEL_PATH
    
    Otherwise, searches in common locations:
    1. outputs/final_training/distilroberta/distilroberta_model.onnx
    2. outputs/distilroberta_model.onnx
    """
    # Check command line option first
    onnx_path = request.config.getoption("--onnx-model", default=None)
    if onnx_path:
        return str(Path(onnx_path).resolve())
    
    # Check environment variable
    onnx_path = os.getenv("ONNX_MODEL_PATH")
    if onnx_path:
        return str(Path(onnx_path).resolve())
    
    # Standard structure location
    root_dir = Path(__file__).parent.parent.parent.parent
    standard_path = root_dir / "outputs" / "final_training" / "distilroberta" / "distilroberta_model.onnx"
    
    if standard_path.exists():
        return str(standard_path.resolve())
    
    # Return standard path (for skip message)
    return str(standard_path.resolve())


@pytest.fixture(scope="session")
def checkpoint_dir(request) -> str:
    """
    Path to checkpoint directory.
    
    Can be provided via command line: pytest --checkpoint <path>
    Or via environment variable: CHECKPOINT_DIR
    
    Otherwise, searches in common locations:
    1. outputs/final_training/distilroberta/checkpoint
    2. outputs/distilroberta_checkpoint
    """
    # Check command line option first
    checkpoint = request.config.getoption("--checkpoint", default=None)
    if checkpoint:
        return str(Path(checkpoint).resolve())
    
    # Check environment variable
    checkpoint = os.getenv("CHECKPOINT_DIR")
    if checkpoint:
        return str(Path(checkpoint).resolve())
    
    # Standard structure location
    root_dir = Path(__file__).parent.parent.parent.parent
    standard_path = root_dir / "outputs" / "final_training" / "distilroberta" / "checkpoint"
    
    if standard_path.exists():
        return str(standard_path.resolve())
    
    # Return standard path (for skip message)
    return str(standard_path.resolve())


@pytest.fixture(scope="session")
def sample_text() -> str:
    """Sample text for testing inference."""
    return "John Doe is a software engineer at Google. Email: john.doe@example.com Phone: +1-555-123-4567 Location: Seattle, WA"


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--onnx-model",
        action="store",
        default=None,
        help="Path to ONNX model file for integration tests",
    )
    parser.addoption(
        "--checkpoint",
        action="store",
        default=None,
        help="Path to checkpoint directory for integration tests",
    )

