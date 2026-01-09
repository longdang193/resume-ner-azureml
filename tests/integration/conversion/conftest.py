"""Shared fixtures for conversion tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any


@pytest.fixture
def sample_conversion_config():
    """Sample conversion.yaml configuration matching the actual file."""
    return {
        "target": {
            "format": "onnx"
        },
        "onnx": {
            "opset_version": 18,
            "quantization": "none",
            "run_smoke_test": True
        },
        "output": {
            "filename_pattern": "model_{quantization}.onnx"
        }
    }


@pytest.fixture
def custom_conversion_config():
    """Custom conversion configuration with non-default values."""
    return {
        "target": {
            "format": "onnx"
        },
        "onnx": {
            "opset_version": 19,
            "quantization": "int8",
            "run_smoke_test": False
        },
        "output": {
            "filename_pattern": "custom_{quantization}_model.onnx"
        }
    }


@pytest.fixture
def resolved_conversion_config():
    """Resolved conversion config as returned by load_conversion_config."""
    return {
        "conv_fp": "test_conv_fp_123",
        "parent_training_id": "spec_test_exec_test/v1",
        "backbone": "distilbert-base-uncased",
        "checkpoint_path": "/path/to/checkpoint",
        "format": "onnx",
        "onnx": {
            "opset_version": 18,
            "quantization": "none",
            "run_smoke_test": True
        },
        "output": {
            "filename_pattern": "model_{quantization}.onnx"
        }
    }

