"""HPO-specific path resolution utilities.

This module provides HPO-specific path resolution functions, particularly
for handling Google Drive path mapping in Colab environments.
"""

from pathlib import Path
from typing import Optional

from shared.platform_detection import detect_platform
from shared.logging_utils import get_logger

logger = get_logger(__name__)


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


