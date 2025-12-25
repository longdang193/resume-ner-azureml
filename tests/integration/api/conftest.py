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
    """
    # Check command line option first
    onnx_path = request.config.getoption("--onnx-model", default=None)
    if onnx_path:
        return onnx_path
    
    # Check environment variable
    onnx_path = os.getenv("ONNX_MODEL_PATH")
    if onnx_path:
        return onnx_path
    
    # Default path (may not exist)
    return "outputs/final_training/distilroberta/distilroberta_model.onnx"


@pytest.fixture(scope="session")
def checkpoint_dir(request) -> str:
    """
    Path to checkpoint directory.
    
    Can be provided via command line: pytest --checkpoint <path>
    Or via environment variable: CHECKPOINT_DIR
    """
    # Check command line option first
    checkpoint = request.config.getoption("--checkpoint", default=None)
    if checkpoint:
        return checkpoint
    
    # Check environment variable
    checkpoint = os.getenv("CHECKPOINT_DIR")
    if checkpoint:
        return checkpoint
    
    # Default path (may not exist)
    return "outputs/final_training/distilroberta/checkpoint"


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

