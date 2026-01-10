"""Run ID index management (run_key_hash → run_id mapping)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from common.shared.json_cache import load_json, save_json
from common.shared.logging_utils import get_logger
from orchestration.jobs.tracking.config.loader import get_index_config
from orchestration.jobs.tracking.index.file_locking import acquire_lock, release_lock

logger = get_logger(__name__)


def get_mlflow_index_path(root_dir: Path, config_dir: Optional[Path] = None) -> Path:
    """
    Get path to mlflow_index.json in cache directory.

    Args:
        root_dir: Project root directory.
        config_dir: Optional config directory (defaults to root_dir / "config").

    Returns:
        Path to mlflow_index.json file.
    """
    if config_dir is None:
        config_dir = root_dir / "config"

    # Derive project root from config_dir when available to avoid nesting
    # indexes under stage-specific roots (e.g. outputs/hpo or notebooks).
    project_root = config_dir.parent if config_dir is not None else root_dir

    # Use same cache structure as index_manager under the project root
    cache_dir = project_root / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Read file_name from config
    index_config = get_index_config(config_dir)
    file_name = index_config.get("file_name", "mlflow_index.json")

    return cache_dir / file_name


def update_mlflow_index(
    root_dir: Path,
    run_key_hash: str,
    run_id: str,
    experiment_id: str,
    tracking_uri: str,
    config_dir: Optional[Path] = None,
    max_entries: Optional[int] = None,
) -> Path:
    """
    Update index with new run_key_hash → run_id mapping.

    Uses file locking for concurrency protection. If locking is not available
    (e.g., Windows), falls back to non-atomic write (with warning).

    Args:
        root_dir: Project root directory.
        run_key_hash: SHA256 hash of run_key.
        run_id: MLflow run ID.
        experiment_id: MLflow experiment ID.
        tracking_uri: MLflow tracking URI.
        config_dir: Optional config directory.
        max_entries: Maximum number of entries to keep (LRU eviction). If None, reads from config.

    Returns:
        Path to index file.

    Raises:
        ValueError: If required parameters are missing.
    """
    if not run_key_hash or not run_id or not experiment_id or not tracking_uri:
        raise ValueError(
            "All parameters (run_key_hash, run_id, experiment_id, tracking_uri) are required")

    # Read config for enabled flag and max_entries
    index_config = get_index_config(config_dir)
    enabled = index_config.get("enabled", True)

    if not enabled:
        logger.debug("MLflow index disabled in config, skipping update")
        return get_mlflow_index_path(root_dir, config_dir)

    # Read max_entries from config if not provided
    if max_entries is None:
        max_entries = index_config.get("max_entries", 1000)

    index_path = get_mlflow_index_path(root_dir, config_dir)

    # Acquire lock
    lock_fd = acquire_lock(index_path)
    if lock_fd is None:
        logger.warning(
            f"Could not acquire lock for {index_path}, proceeding with non-atomic write")

    try:
        # Load existing index
        index = load_json(index_path, default={})

        # Update entry
        index[run_key_hash] = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "updated_at": datetime.now().isoformat(),
        }

        # LRU eviction: keep only most recent max_entries
        if len(index) > max_entries:
            # Sort by updated_at (most recent first)
            sorted_entries = sorted(
                index.items(),
                key=lambda x: x[1].get("updated_at", ""),
                reverse=True
            )
            # Keep only max_entries
            index = dict(sorted_entries[:max_entries])
            logger.debug(
                f"Evicted {len(sorted_entries) - max_entries} old entries from MLflow index")

        # Save index atomically: write to temp file, then rename
        temp_path = index_path.with_suffix('.tmp')
        try:
            save_json(temp_path, index)

            # Atomic rename (works on both Unix and Windows)
            if sys.platform == 'win32':
                # Windows: need to remove target first for atomic replace
                if index_path.exists():
                    index_path.unlink()
            # Atomic on Unix, safe on Windows after unlink
            temp_path.replace(index_path)

            logger.debug(
                f"Updated MLflow index: {run_key_hash[:16]}... → {run_id[:12]}...")
        except Exception as e:
            # Clean up temp file on error
            temp_path.unlink(missing_ok=True)
            raise

    finally:
        # Release lock
        release_lock(lock_fd, index_path)

    return index_path


def find_in_mlflow_index(
    root_dir: Path,
    run_key_hash: str,
    tracking_uri: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    """
    Find run_id in local index by run_key_hash.

    Optionally filters by tracking_uri to ensure alignment.

    Args:
        root_dir: Project root directory.
        run_key_hash: SHA256 hash of run_key to search for.
        tracking_uri: Optional tracking URI to verify alignment.
        config_dir: Optional config directory.

    Returns:
        Dictionary with run_id, experiment_id, tracking_uri if found, None otherwise.
    """
    if not run_key_hash:
        return None

    index_path = get_mlflow_index_path(root_dir, config_dir)

    if not index_path.exists():
        return None

    # Load index
    index = load_json(index_path, default={})

    # Lookup
    entry = index.get(run_key_hash)
    if not entry:
        return None

    # Verify tracking URI alignment if provided
    if tracking_uri:
        stored_uri = entry.get("tracking_uri", "")
        if stored_uri != tracking_uri:
            logger.warning(
                f"Tracking URI mismatch in index: stored={stored_uri[:50]}..., "
                f"requested={tracking_uri[:50]}..."
            )
            return None

    return {
        "run_id": entry.get("run_id"),
        "experiment_id": entry.get("experiment_id"),
        "tracking_uri": entry.get("tracking_uri"),
    }

