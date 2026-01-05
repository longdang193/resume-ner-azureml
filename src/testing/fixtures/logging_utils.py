"""Logging utilities for test scripts.

This module provides shared logging functionality for capturing stdout/stderr
and writing to both console and log files.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class TeeOutput:
    """Capture stdout/stderr and write to both console and log file."""

    def __init__(self, log_file: Path):
        """
        Initialize TeeOutput.

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file_handle = open(log_file, "a", encoding="utf-8")

    def write(self, message: str):
        """Write to both console and log file."""
        # Write to original stdout (console)
        self.original_stdout.write(message)
        self.original_stdout.flush()
        # Also write to log file (raw output, no timestamp duplication)
        if self.file_handle is not None:
            try:
                self.file_handle.write(message)
                self.file_handle.flush()
            except (ValueError, OSError):
                # File handle may be closed during interpreter shutdown
                pass

    def flush(self):
        """Flush both outputs."""
        self.original_stdout.flush()
        if self.file_handle is not None:
            try:
                self.file_handle.flush()
            except (ValueError, OSError):
                # File handle may be closed during interpreter shutdown
                pass

    def close(self):
        """Close the log file and restore original stdout/stderr."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except (ValueError, OSError):
                pass
            self.file_handle = None
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


def setup_logging(
    output_dir: Path,
    log_file: Optional[Path] = None,
    log_prefix: str = "test",
) -> Tuple[logging.Logger, Path]:
    """
    Setup logging to both console and file.

    Args:
        output_dir: Directory where log file will be created (if log_file not provided)
        log_file: Optional specific log file path
        log_prefix: Prefix for auto-generated log file name

    Returns:
        Tuple of (logger, log_file_path)
    """
    if log_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"{log_prefix}_{timestamp}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    return logger, log_file


def setup_tee_output(log_file: Path) -> TeeOutput:
    """
    Setup TeeOutput to capture stdout/stderr to log file.

    Args:
        log_file: Path to log file

    Returns:
        TeeOutput instance
    """
    tee = TeeOutput(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee
