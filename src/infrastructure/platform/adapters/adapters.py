"""
@meta
name: platform_adapters
type: utility
domain: platform_adapters
responsibility:
  - Define platform adapter interface for platform-specific operations
  - Implement Azure ML and local adapters
inputs:
  - Platform detection results
outputs:
  - Platform adapter instances
tags:
  - utility
  - platform_adapters
  - adapter_pattern
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Platform adapter interface and implementations.

This module defines the adapter pattern for platform-specific operations,
allowing the core training and conversion logic to run consistently
on both Azure ML and local environments.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .outputs import OutputPathResolver
from .logging_adapter import LoggingAdapter
from .mlflow_context import MLflowContextManager
from .checkpoint_resolver import CheckpointResolver


class PlatformAdapter(ABC):
    """Abstract interface for platform-specific operations."""

    @abstractmethod
    def get_output_path_resolver(self) -> OutputPathResolver:
        """Get the output path resolver for this platform."""
        pass

    @abstractmethod
    def get_logging_adapter(self) -> LoggingAdapter:
        """Get the logging adapter for this platform."""
        pass

    @abstractmethod
    def get_mlflow_context_manager(self) -> MLflowContextManager:
        """Get the MLflow context manager for this platform."""
        pass

    @abstractmethod
    def get_checkpoint_resolver(self) -> CheckpointResolver:
        """Get the checkpoint resolver for this platform."""
        pass

    @abstractmethod
    def is_platform_job(self) -> bool:
        """Check if running in a platform-managed job context."""
        pass


class AzureMLAdapter(PlatformAdapter):
    """Adapter for Azure ML execution environment."""

    def get_output_path_resolver(self) -> OutputPathResolver:
        """Get Azure ML output path resolver."""
        from .outputs import AzureMLOutputPathResolver
        return AzureMLOutputPathResolver()

    def get_logging_adapter(self) -> LoggingAdapter:
        """Get Azure ML logging adapter."""
        from .logging_adapter import AzureMLLoggingAdapter
        return AzureMLLoggingAdapter()

    def get_mlflow_context_manager(self) -> MLflowContextManager:
        """Get Azure ML MLflow context manager."""
        from .mlflow_context import AzureMLMLflowContextManager
        return AzureMLMLflowContextManager()

    def get_checkpoint_resolver(self) -> CheckpointResolver:
        """Get Azure ML checkpoint resolver."""
        from .checkpoint_resolver import AzureMLCheckpointResolver
        return AzureMLCheckpointResolver()

    def is_platform_job(self) -> bool:
        """Check if running in Azure ML job context."""
        return any(key.startswith("AZURE_ML_") for key in os.environ.keys())


class LocalAdapter(PlatformAdapter):
    """Adapter for local execution environment."""

    def __init__(self, default_output_dir: Optional[Path] = None):
        """Initialize local adapter with optional default output directory."""
        self._default_output_dir = default_output_dir or Path("./outputs")

    def get_output_path_resolver(self) -> OutputPathResolver:
        """Get local output path resolver."""
        from .outputs import LocalOutputPathResolver
        return LocalOutputPathResolver(self._default_output_dir)

    def get_logging_adapter(self) -> LoggingAdapter:
        """Get local logging adapter."""
        from .logging_adapter import LocalLoggingAdapter
        return LocalLoggingAdapter()

    def get_mlflow_context_manager(self) -> MLflowContextManager:
        """Get local MLflow context manager."""
        from .mlflow_context import LocalMLflowContextManager
        return LocalMLflowContextManager()

    def get_checkpoint_resolver(self) -> CheckpointResolver:
        """Get local checkpoint resolver."""
        from .checkpoint_resolver import LocalCheckpointResolver
        return LocalCheckpointResolver()

    def is_platform_job(self) -> bool:
        """Local execution is never a platform-managed job."""
        return False


def get_platform_adapter(default_output_dir: Optional[Path] = None) -> PlatformAdapter:
    """
    Detect and return the appropriate platform adapter.

    Args:
        default_output_dir: Default output directory for local execution.

    Returns:
        PlatformAdapter instance (AzureMLAdapter or LocalAdapter).
    """
    # Check for Azure ML environment variables
    if any(key.startswith("AZURE_ML_") for key in os.environ.keys()):
        return AzureMLAdapter()
    return LocalAdapter(default_output_dir)

