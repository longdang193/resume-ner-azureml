"""Artifact upload and checkpoint archive management."""

from __future__ import annotations

import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def should_skip_file(file_path: Path, relative_path: str) -> bool:
    """
    Determine if a file should be skipped when creating archive.

    Args:
        file_path: Absolute path to file.
        relative_path: Relative path within checkpoint directory.

    Returns:
        True if file should be skipped, False otherwise.
    """
    # Skip patterns
    skip_patterns = ['.tmp', '.cache', '__pycache__', '.pyc', '.log']
    skip_extensions = ['.tmp', '.cache']

    name = file_path.name
    if any(pattern in name for pattern in skip_patterns):
        return True

    if file_path.suffix.lower() in skip_extensions:
        return True

    # Skip very large files (>100MB) unless they're model files
    try:
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            if file_path.suffix.lower() not in ['.pt', '.pth', '.onnx', '.bin', '.safetensors']:
                logger.debug(
                    f"Skipping large non-model file: {relative_path} ({file_size / 1024 / 1024:.1f}MB)")
                return True
    except OSError:
        # Can't stat file, skip it
        return True

    return False


def create_checkpoint_archive(
    checkpoint_dir: Path,
    trial_number: int,
    output_path: Optional[Path] = None
) -> tuple[Path, Dict[str, Any]]:
    """
    Create compressed archive from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        trial_number: Trial number for manifest.
        output_path: Optional output path for archive. If None, uses temp file.

    Returns:
        Tuple of (archive_path, manifest_dict).
    """
    if output_path is None:
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tar.gz', prefix='checkpoint_')
        output_path = Path(temp_path)
        # Close the file descriptor, we'll open it with tarfile
        os.close(temp_fd)

    manifest = {
        "trial_number": trial_number,
        "archive_format": "tar.gz",
        "extracted_path": "best_trial_checkpoint",
        "files": [],
        "total_size": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    checkpoint_dir = checkpoint_dir.resolve()
    files_added = 0

    with tarfile.open(output_path, 'w:gz') as tar:
        for root, dirs, files in os.walk(checkpoint_dir):
            # Filter out directories to skip
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in [
                                                  '.tmp', '.cache', '__pycache__'])]

            for file in files:
                file_path = Path(root) / file
                try:
                    file_path = file_path.resolve()

                    if not file_path.exists():
                        continue

                    # Get relative path for archive
                    try:
                        relative_path = file_path.relative_to(checkpoint_dir)
                    except ValueError:
                        # File is outside checkpoint_dir, skip
                        continue

                    # Check if should skip
                    if should_skip_file(file_path, str(relative_path)):
                        continue

                    # Add to archive
                    arcname = f"best_trial_checkpoint/{relative_path}"
                    tar.add(file_path, arcname=arcname, recursive=False)

                    # Update manifest
                    file_size = file_path.stat().st_size
                    manifest["files"].append({
                        "path": str(relative_path),
                        "size": file_size
                    })
                    manifest["total_size"] += file_size
                    files_added += 1

                except Exception as e:
                    logger.warning(f"Error adding {file_path} to archive: {e}")
                    continue

    manifest["file_count"] = files_added
    logger.info(
        f"Created checkpoint archive: {output_path} "
        f"({files_added} files, {manifest['total_size'] / 1024 / 1024:.1f}MB)"
    )

    return output_path, manifest

