"""File utility functions for validation and verification."""

from __future__ import annotations

from pathlib import Path


def verify_output_file(
    output_dir: Path,
    filename: str,
    step_name: str
) -> Path:
    """
    Verify output file exists, raise informative error if not.

    Args:
        output_dir: Directory containing the file.
        filename: Name of the file to verify.
        step_name: Name of the step for error message.

    Returns:
        Path to the verified file.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    file_path = output_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Step {step_name}: Expected output file not found: {file_path}\n"
            f"Please ensure the previous step completed successfully."
        )
    return file_path

