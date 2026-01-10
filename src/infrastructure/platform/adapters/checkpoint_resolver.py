"""
@meta
name: platform_checkpoint_resolver
type: utility
domain: platform_adapters
responsibility:
  - Resolve checkpoint paths for different platforms
  - Handle Azure ML mounted inputs and local paths
inputs:
  - Checkpoint path strings
outputs:
  - Resolved checkpoint directory paths
tags:
  - utility
  - platform_adapters
  - checkpoint
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Checkpoint path resolution for different platforms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class CheckpointResolver(ABC):
    """Abstract interface for resolving checkpoint paths."""

    @abstractmethod
    def resolve_checkpoint_dir(self, checkpoint_path: str) -> Path:
        """
        Resolve a checkpoint path to the actual Hugging Face checkpoint directory.

        Args:
            checkpoint_path: Input checkpoint path (may be a mount point or directory).

        Returns:
            Path to the Hugging Face checkpoint directory.

        Raises:
            FileNotFoundError: If checkpoint directory cannot be found.
        """
        pass


class AzureMLCheckpointResolver(CheckpointResolver):
    """Checkpoint resolver for Azure ML mounted inputs."""

    def resolve_checkpoint_dir(self, checkpoint_path: str) -> Path:
        """
        Resolve an Azure ML mounted input folder to the HF checkpoint directory.

        Training saves into `<output_dir>/checkpoint/` using Hugging Face save_pretrained.
        """
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        # Training saves into `<output_dir>/checkpoint/` using Hugging Face save_pretrained.
        candidates = [root, root / "checkpoint"]
        for d in candidates:
            if not d.exists() or not d.is_dir():
                continue
            # A HF model directory contains at least one of these.
            if (d / "config.json").exists() and (
                (d / "pytorch_model.bin").exists()
                or (d / "model.safetensors").exists()
                or (d / "model.pt").exists()
            ):
                return d

        # Best-effort file listing for diagnostics
        files = self._list_files(root, limit=40)
        raise FileNotFoundError(
            "Could not locate a Hugging Face checkpoint directory under "
            f"'{checkpoint_path}'. Expected a folder created by `save_pretrained` "
            "containing `config.json` and model weights. "
            f"Found files (up to 40): {files}"
        )

    def _list_files(self, root: Path, limit: int = 40) -> list[str]:
        """List files in directory for diagnostics."""
        if not root.exists():
            return []
        files: list[str] = []
        try:
            for item in root.rglob("*"):
                if item.is_file():
                    files.append(str(item.relative_to(root)))
                    if len(files) >= limit:
                        break
        except Exception:
            # Best-effort listing for diagnostics; ignore traversal errors.
            pass
        return files


class LocalCheckpointResolver(CheckpointResolver):
    """Checkpoint resolver for local file system."""

    def resolve_checkpoint_dir(self, checkpoint_path: str) -> Path:
        """
        Resolve a local checkpoint path to the HF checkpoint directory.

        Same logic as Azure ML, but for local file system.
        """
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        # Training saves into `<output_dir>/checkpoint/` using Hugging Face save_pretrained.
        candidates = [root, root / "checkpoint"]
        for d in candidates:
            if not d.exists() or not d.is_dir():
                continue
            # A HF model directory contains at least one of these.
            if (d / "config.json").exists() and (
                (d / "pytorch_model.bin").exists()
                or (d / "model.safetensors").exists()
                or (d / "model.pt").exists()
            ):
                return d

        # Best-effort file listing for diagnostics
        files = self._list_files(root, limit=40)
        raise FileNotFoundError(
            "Could not locate a Hugging Face checkpoint directory under "
            f"'{checkpoint_path}'. Expected a folder created by `save_pretrained` "
            "containing `config.json` and model weights. "
            f"Found files (up to 40): {files}"
        )

    def _list_files(self, root: Path, limit: int = 40) -> list[str]:
        """List files in directory for diagnostics."""
        if not root.exists():
            return []
        files: list[str] = []
        try:
            for item in root.rglob("*"):
                if item.is_file():
                    files.append(str(item.relative_to(root)))
                    if len(files) >= limit:
                        break
        except Exception:
            # Best-effort listing for diagnostics; ignore traversal errors.
            pass
        return files

