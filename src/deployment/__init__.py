"""Model deployment module.

This module provides functionality for deploying models, including:
- Conversion: Converting models to ONNX format for production
- API: FastAPI service for serving model predictions
"""

from .conversion import execute_conversion

# Azure ML functions (optional - only available if azure.ai.ml is installed)
try:
    from .conversion import (
        get_checkpoint_output_from_training_job,
        create_conversion_job,
        validate_conversion_job,
    )
    _conversion_exports = [
        "execute_conversion",
        "get_checkpoint_output_from_training_job",
        "create_conversion_job",
        "validate_conversion_job",
    ]
except ImportError:
    # Azure ML not available - export only orchestration function
    _conversion_exports = [
        "execute_conversion",
    ]

# API exports - lazy import to avoid requiring FastAPI dependencies at module load time
# Don't import API at module level to avoid FastAPI dependency issues
# Users should import directly: from deployment.api import app
__all__ = _conversion_exports

