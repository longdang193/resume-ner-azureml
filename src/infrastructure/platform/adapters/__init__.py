"""Platform adapters for Azure ML and local execution.

This module provides a clean separation between platform-specific concerns
(environment variables, output paths, logging) and platform-agnostic core logic.
"""

from .adapters import (
    PlatformAdapter,
    get_platform_adapter,
    AzureMLAdapter,
    LocalAdapter,
)
from .outputs import OutputPathResolver
from .logging_adapter import LoggingAdapter
from .mlflow_context import MLflowContextManager
from .checkpoint_resolver import CheckpointResolver

__all__ = [
    "PlatformAdapter",
    "get_platform_adapter",
    "AzureMLAdapter",
    "LocalAdapter",
    "OutputPathResolver",
    "LoggingAdapter",
    "MLflowContextManager",
    "CheckpointResolver",
]

