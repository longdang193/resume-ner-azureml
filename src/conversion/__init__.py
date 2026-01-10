"""Conversion module for model conversion to ONNX format.

This module provides:
- High-level orchestration for conversion workflows
- Conversion script execution entry point
- Core ONNX export functionality
- Smoke testing for exported models
- Azure ML job creation utilities
- CLI argument parsing
"""

from __future__ import annotations

from .orchestration import execute_conversion

# Azure ML functions (optional - only available if azure.ai.ml is installed)
try:
    from .azureml import (
        get_checkpoint_output_from_training_job,
        create_conversion_job,
        validate_conversion_job,
    )
    __all__ = [
        "execute_conversion",
        "get_checkpoint_output_from_training_job",
        "create_conversion_job",
        "validate_conversion_job",
    ]
except ImportError:
    # Azure ML not available - export only orchestration function
    __all__ = [
        "execute_conversion",
    ]
