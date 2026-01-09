"""Path utility functions."""

from __future__ import annotations

from pathlib import Path

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def find_project_root(config_dir: Path) -> Path:
    """
    Find project root directory by walking up from config_dir.

    Looks for a directory containing both `src/` and `src/training/` subdirectories.

    Args:
        config_dir: Configuration directory path to start searching from.

    Returns:
        Path to project root directory.

    Raises:
        ValueError: If project root cannot be found after maximum depth.
    """
    candidate_root = config_dir.parent
    max_depth = 5
    depth = 0

    while depth < max_depth:
        if (candidate_root / "src").exists() and (candidate_root / "src" / "training").exists():
            logger.debug(
                f"Found project root: {candidate_root} (from config_dir: {config_dir})"
            )
            return candidate_root
        candidate_root = candidate_root.parent
        depth += 1

    # Fallback: assume config_dir is project_root/config
    root_dir = config_dir.parent
    logger.warning(
        f"Could not find project root with src/training/ directory after {max_depth} levels. "
        f"Using {root_dir} as root_dir. Config_dir: {config_dir}"
    )
    return root_dir


