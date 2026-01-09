"""Drive backup path mapping and Colab-specific path resolution."""

import re
from pathlib import Path
from typing import Optional, Callable

from shared.platform_detection import detect_platform
from shared.logging_utils import get_logger
from paths.config import load_paths_config
from paths.validation import validate_path_before_mkdir

logger = get_logger(__name__)


def get_drive_backup_base(config_dir: Path) -> Optional[Path]:
    """
    Get base Google Drive backup directory from config.

    Args:
        config_dir: Config directory.

    Returns:
        Base Drive backup path (e.g., /content/drive/MyDrive/resume-ner-checkpoints), 
        or None if not configured.

    Examples:
        get_drive_backup_base(CONFIG_DIR)
        # -> Path("/content/drive/MyDrive/resume-ner-checkpoints")
    """
    paths_config = load_paths_config(config_dir)
    drive_config = paths_config.get("drive", {})

    if not drive_config:
        return None

    mount_point = drive_config.get("mount_point", "/content/drive")
    backup_base = drive_config.get("backup_base_dir", "resume-ner-checkpoints")

    return Path(mount_point) / "MyDrive" / backup_base


def get_drive_backup_path(
    root_dir: Path,
    config_dir: Path,
    local_path: Path
) -> Optional[Path]:
    """
    Convert local output path to Drive backup path, mirroring structure.

    Only paths within outputs/ can be backed up. The function automatically
    mirrors the exact same directory structure from outputs/ to Drive.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        local_path: Local file or directory path to backup (must be within outputs/).

    Returns:
        Equivalent Drive backup path, or None if Drive not configured or path outside outputs/.

    Examples:
        Local: outputs/hpo/distilbert/trial_0/checkpoint/
        Drive:  /content/drive/MyDrive/resume-ner-checkpoints/outputs/hpo/distilbert/trial_0/checkpoint/

        Local: outputs/cache/best_configurations/latest_best_configuration.json
        Drive:  /content/drive/MyDrive/resume-ner-checkpoints/outputs/cache/best_configurations/latest_best_configuration.json
    """
    paths_config = load_paths_config(config_dir)
    drive_config = paths_config.get("drive", {})

    if not drive_config:
        return None

    # Get the base outputs directory
    base_outputs = paths_config["base"]["outputs"]
    base_outputs_path = Path(base_outputs)
    outputs_dir = (
        base_outputs_path if base_outputs_path.is_absolute() else root_dir / base_outputs
    )
    base_outputs_name = base_outputs_path.name if base_outputs_path.is_absolute() else base_outputs

    # Check if the local path is within outputs/
    try:
        relative_path = local_path.relative_to(outputs_dir)
    except ValueError:
        # Path is not within outputs/, can't mirror it
        return None

    # Get Drive base directory
    drive_base = get_drive_backup_base(config_dir)
    if not drive_base:
        return None

    # Build Drive path: mount_point/MyDrive/backup_base/outputs/relative_path
    drive_path = drive_base / base_outputs_name / relative_path

    return drive_path


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

