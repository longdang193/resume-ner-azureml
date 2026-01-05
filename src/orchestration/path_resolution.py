"""Path resolution utilities for Colab, Kaggle, and local environments.

Handles Google Drive path mapping and validation for Colab environments.
"""

import re
from pathlib import Path
from typing import Optional, Callable

from shared.platform_detection import detect_platform
from shared.logging_utils import get_logger

logger = get_logger(__name__)


def validate_path_before_mkdir(path: Path, context: str = "directory") -> Path:
    """
    Validate path before creating directory to prevent creating invalid files.

    Args:
        path: Path to validate
        context: Context string for error messages

    Returns:
        Validated and resolved path

    Raises:
        ValueError: If path is invalid
    """
    if not path or not str(path):
        raise ValueError(f"Invalid {context} path: {path}")

    # Ensure path is absolute
    if not path.is_absolute():
        path = path.resolve()

    path_str = str(path)

    # Basic invalid cases
    if path_str in ("", ".", ".."):
        raise ValueError(
            f"Invalid {context} path (too short or relative): {path_str}"
        )

    # Split path
    path_parts = path_str.replace("\\", "/").split("/")

    # Check if last part looks like a version number (e.g. "1.0.0")
    if path_parts:
        last_part = path_parts[-1]
        if re.match(r"^[\d\.]+$", last_part):
            # Reject single-part paths like "1.0.0"
            if len(path_parts) == 1:
                raise ValueError(
                    f"Invalid {context} path (looks like version number): {path_str}"
                )

    # Validate path has reasonable structure
    if len(path_parts) < 2:
        raise ValueError(
            f"Invalid {context} path (too short, appears to be filename): {path_str}"
        )

    # Safety: path exists but is a file
    if path.exists() and path.is_file():
        logger.error(f"Path exists as file, not directory: {path}")
        raise ValueError(
            f"Cannot create {context}, path exists as file: {path}"
        )

    return path


def resolve_hpo_output_dir(hpo_output_dir: Path) -> Path:
    """
    Resolve HPO output directory, checking Drive on Colab if local doesn't exist or is empty.

    On Colab, if the local path doesn't exist or is empty, check the corresponding Drive path.

    Args:
        hpo_output_dir: Local HPO output directory path

    Returns:
        Resolved path (Drive path on Colab if available and has content, else local path)
    """
    platform = detect_platform()
    if platform == "colab":
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists() and drive_path.is_dir():
            base_str = str(hpo_output_dir)
            if "/resume-ner-azureml" in base_str:
                drive_dir: Optional[Path]
                if base_str.endswith("/resume-ner-azureml") or base_str.endswith("/resume-ner-azureml/"):
                    drive_dir = drive_path / "resume-ner-azureml"
                else:
                    parts = base_str.split("/resume-ner-azureml/", 1)
                    if len(parts) == 2 and parts[1]:
                        drive_dir = drive_path / "resume-ner-azureml" / parts[1]
                    else:
                        drive_dir = None

                if drive_dir and drive_dir.exists():
                    try:
                        has_content = any(item.is_dir() for item in drive_dir.iterdir())
                        if has_content:
                            logger.debug(f"Using Drive path for HPO output: {drive_dir}")
                            return drive_dir
                    except (PermissionError, OSError) as e:
                        logger.debug(f"Could not check Drive directory {drive_dir}: {e}")

    if hpo_output_dir.exists():
        return hpo_output_dir

    return hpo_output_dir


def resolve_output_path_for_colab(
    output_path: Path, validate_path_func: Optional[Callable[[Path, str], Path]] = None
) -> Path:
    """
    Resolve output path, redirecting to Drive on Colab if available.

    Validates input path to prevent creating invalid files like '1.0.0'.

    Args:
        output_path: Path to resolve
        validate_path_func: Optional function to validate path before creating directories.
            If None, uses validate_path_before_mkdir.

    Returns:
        Resolved path (Drive path on Colab if available, else original path)

    Raises:
        ValueError: If path is invalid
    """
    if not output_path or not str(output_path):
        raise ValueError(f"Cannot resolve invalid path: {output_path}")

    if not output_path.is_absolute():
        output_path = output_path.resolve()

    path_str = str(output_path)
    if not path_str or path_str in (".", ".."):
        raise ValueError(f"Cannot resolve invalid path (too short or relative): {path_str}")

    path_parts = path_str.replace("\\", "/").split("/")

    if len(path_parts) > 0 and re.match(r"^[\d\.]+$", path_parts[-1]) and len(path_parts) == 1:
        raise ValueError(f"Invalid path (looks like version number): {path_str}")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid path (too short, appears to be filename): {path_str}")

    platform = detect_platform()
    if platform == "colab":
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists() and drive_path.is_dir():
            base_str = str(output_path)
            if "/resume-ner-azureml" in base_str:
                drive_dir: Optional[Path]
                if base_str.endswith("/resume-ner-azureml") or base_str.endswith("/resume-ner-azureml/"):
                    drive_dir = drive_path / "resume-ner-azureml"
                else:
                    parts = base_str.split("/resume-ner-azureml/", 1)
                    if len(parts) == 2 and parts[1]:
                        relative_path = parts[1].strip("/")
                        if relative_path and not relative_path.startswith("..") and "/.." not in relative_path:
                            drive_dir = drive_path / "resume-ner-azureml" / relative_path
                        else:
                            logger.warning(f"Invalid relative path detected: {relative_path}, returning original path")
                            return output_path
                    else:
                        logger.warning(f"Cannot parse path: {base_str}, returning original path")
                        return output_path

                if "drive_dir" not in locals() or not drive_dir:
                    logger.warning("drive_dir not properly set, returning original path")
                    return output_path

                drive_dir_str = str(drive_dir)
                drive_dir_parts = drive_dir_str.replace("\\", "/").split("/")
                if drive_dir_parts:
                    last_part = drive_dir_parts[-1]
                    if re.match(r"^[\d\.]+$", last_part) and len(drive_dir_parts) == 1:
                        logger.error(f"drive_dir appears to be just a version number: {drive_dir_str}")
                        return output_path

                if not drive_dir_str or len(drive_dir_str) < 10:
                    logger.warning(f"drive_dir validation failed (too short): {drive_dir_str}, returning original path")
                    return output_path

                # Ensure parent directory exists (only if drive_dir is valid and not root)
                if drive_dir and drive_dir != drive_path and drive_dir.parent != drive_dir:
                    try:
                        if drive_dir.parent.exists():
                            if drive_dir.parent.is_file():
                                logger.warning(f"Parent path exists as file, not directory: {drive_dir.parent}")
                                return output_path
                        else:
                            validator = validate_path_func or validate_path_before_mkdir
                            parent_dir = validator(drive_dir.parent, context="directory")
                            parent_dir.mkdir(parents=True, exist_ok=True)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Failed to create parent directory for {drive_dir}: {e}")
                        return output_path

                logger.debug(f"Resolved output path to Drive: {drive_dir}")
                return drive_dir

    # Return original if not Colab or Drive not available
    if output_path and str(output_path) and len(str(output_path)) > 1:
        return output_path

    raise ValueError(f"Invalid output path: {output_path}")

