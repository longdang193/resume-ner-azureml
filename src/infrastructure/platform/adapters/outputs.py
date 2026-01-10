"""
@meta
name: platform_outputs
type: utility
domain: platform_adapters
responsibility:
  - Resolve output paths for different platforms
  - Handle Azure ML named outputs and local directories
inputs:
  - Output name identifiers
outputs:
  - Resolved output directory paths
tags:
  - utility
  - platform_adapters
  - outputs
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Output path resolution for different platforms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import os


class OutputPathResolver(ABC):
    """Abstract interface for resolving output paths."""

    @abstractmethod
    def resolve_output_path(self, output_name: str, default: Optional[Path] = None) -> Path:
        """
        Resolve the output path for a named output.

        Args:
            output_name: Name of the output (e.g., "checkpoint", "onnx_model").
            default: Default path if resolution fails.

        Returns:
            Resolved Path object.
        """
        pass

    @abstractmethod
    def ensure_output_directory(self, output_path: Path) -> Path:
        """
        Ensure the output directory exists and is ready for writing.

        Args:
            output_path: Path to the output directory.

        Returns:
            Path object (may be the same or a subdirectory).
        """
        pass


class AzureMLOutputPathResolver(OutputPathResolver):
    """Output path resolver for Azure ML jobs."""

    def resolve_output_path(self, output_name: str, default: Optional[Path] = None) -> Path:
        """
        Resolve Azure ML output path from environment variables.

        Azure ML automatically sets AZURE_ML_OUTPUT_<output_name> for each named output.
        Falls back to AZURE_ML_OUTPUT_DIR for backward compatibility.
        """
        # Try named output first (e.g., AZURE_ML_OUTPUT_checkpoint)
        env_var = f"AZURE_ML_OUTPUT_{output_name.upper()}"
        output_path = os.getenv(env_var)

        # Fall back to generic output directory
        if not output_path:
            output_path = os.getenv("AZURE_ML_OUTPUT_DIR")

        # Final fallback to default or current directory
        if not output_path:
            if default:
                output_path = str(default)
            else:
                output_path = "./outputs"

        return Path(output_path)

    def ensure_output_directory(self, output_path: Path) -> Path:
        """
        Ensure output directory exists and create placeholder file if needed.

        Azure ML requires at least one file in a named output directory to
        materialize it in the datastore.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to ensure Azure ML materializes the output
        placeholder = output_path / "output_placeholder.txt"
        if not placeholder.exists():
            placeholder.write_text(
                f"This file ensures the Azure ML output directory is materialized. "
                f"Real outputs are saved in subdirectories."
            )

        return output_path


class LocalOutputPathResolver(OutputPathResolver):
    """Output path resolver for local execution."""

    def __init__(self, default_output_dir: Path):
        """Initialize with default output directory."""
        self._default_output_dir = default_output_dir

    def resolve_output_path(self, output_name: str, default: Optional[Path] = None) -> Path:
        """
        Resolve local output path (simple directory structure).

        Args:
            output_name: Name of the output (used as subdirectory name).
            default: Default path if not provided.

        Returns:
            Path to output directory.
        """
        if default:
            base_dir = default
        else:
            base_dir = self._default_output_dir

        # Create subdirectory for named output
        return base_dir / output_name

    def ensure_output_directory(self, output_path: Path) -> Path:
        """Ensure output directory exists."""
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
