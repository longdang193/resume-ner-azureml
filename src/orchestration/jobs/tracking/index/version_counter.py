"""Run name version reservation and commit."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.json_cache import load_json, save_json
from shared.logging_utils import get_logger
from orchestration.jobs.tracking.index.file_locking import acquire_lock, release_lock

logger = get_logger(__name__)


def get_run_name_counter_path(root_dir: Path, config_dir: Optional[Path] = None) -> Path:
    """
    Get path to run_name_counter.json in cache directory.

    Args:
        root_dir: Project root directory.
        config_dir: Optional config directory (defaults to root_dir / "config").

    Returns:
        Path to run_name_counter.json file.
    """
    if config_dir is None:
        config_dir = root_dir / "config"

    # Derive project root from config_dir when available to avoid nesting
    # counters under stage-specific roots (e.g. outputs/hpo or notebooks).
    project_root = config_dir.parent if config_dir is not None else root_dir

    # Use same cache structure as index_manager under the project root
    cache_dir = project_root / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / "run_name_counter.json"


def reserve_run_name_version(
    counter_key: str,
    run_id: str,
    root_dir: Path,
    config_dir: Optional[Path] = None,
) -> int:
    """
    Reserve a version number for a run name using atomic counter store.

    Uses reserve/commit pattern to prevent duplicates on crashes.
    Never reuses numbers - always increments from max committed version.

    Args:
        counter_key: Counter key (format: "{project}:{process_type}:{run_key_hash}:{env}").
        run_id: MLflow run ID (or "pending" if not yet created).
        root_dir: Project root directory.
        config_dir: Optional config directory.

    Returns:
        Reserved version number (starts at 1 if key doesn't exist).

    Raises:
        RuntimeError: If lock acquisition fails after timeout.
    """
    counter_path = get_run_name_counter_path(root_dir, config_dir)

    logger.info(
        f"[Reserve Version] Starting reservation: counter_key={counter_key[:60]}..., "
        f"root_dir={root_dir}, config_dir={config_dir}, counter_path={counter_path}"
    )

    # Acquire lock
    lock_fd = acquire_lock(counter_path, timeout=10.0)
    if lock_fd is None:
        logger.warning(
            f"[Reserve Version] Could not acquire lock for {counter_path}, "
            f"proceeding with non-atomic write"
        )

    try:
        # Load existing allocations
        counter_data = load_json(counter_path, default={"allocations": []})
        allocations: List[Dict[str, any]] = counter_data.get("allocations", [])
        logger.info(
            f"[Reserve Version] Loaded {len(allocations)} existing allocations from {counter_path}"
        )

        # Find max committed version for this counter_key
        max_version = 0
        matching_allocations = []
        committed_versions = []
        reserved_versions = []
        expired_versions = []

        for alloc in allocations:
            if alloc.get("counter_key") == counter_key:
                matching_allocations.append(alloc)
                status = alloc.get("status", "unknown")
                version = alloc.get("version", 0)

                if status == "committed":
                    max_version = max(max_version, version)
                    committed_versions.append(version)
                elif status == "reserved":
                    reserved_versions.append(version)
                elif status == "expired":
                    expired_versions.append(version)

        # (a) Deduplicate matching allocations by version, keeping newest (by reserved_at timestamp)
        by_version = {}
        for alloc in matching_allocations:
            v = alloc.get("version")
            if v is None:
                continue
            # Use reserved_at timestamp to determine newest
            reserved_at_str = alloc.get("reserved_at", "")
            existing_ts = by_version.get(v, {}).get("reserved_at", "")
            if v not in by_version or reserved_at_str > existing_ts:
                by_version[v] = alloc

        matching_allocations = list(by_version.values())

        # Recompute lists after deduplication
        committed_versions = [a.get("version") for a in matching_allocations if a.get(
            "status") == "committed"]
        reserved_versions = [
            a.get("version") for a in matching_allocations if a.get("status") == "reserved"]
        expired_versions = [
            a.get("version") for a in matching_allocations if a.get("status") == "expired"]
        max_version = max(committed_versions) if committed_versions else 0

        logger.info(
            f"[Reserve Version] Found {len(matching_allocations)} allocations for counter_key (after deduplication): "
            f"committed={committed_versions}, reserved={reserved_versions}, expired={expired_versions}, "
            f"max_committed_version={max_version}"
        )

        if matching_allocations:
            logger.info(
                f"[Reserve Version] Allocation details: "
                f"{[(a.get('version'), a.get('status'), a.get('run_id', '')[:12]) for a in matching_allocations]}"
            )

        # (b) Skip reserved/expired versions when calculating next version
        reserved_or_expired = {
            a["version"] for a in matching_allocations
            if a.get("status") in {"reserved", "expired"} and "version" in a
        }

        # (c) Make reservation idempotent: check if this run_id already has a reservation
        existing_reservation = None
        for alloc in matching_allocations:
            if (alloc.get("run_id") == run_id and
                alloc.get("status") == "reserved" and
                    alloc.get("counter_key") == counter_key):
                existing_reservation = alloc
                logger.info(
                    f"[Reserve Version] Found existing reservation for run_id={run_id[:12]}...: "
                    f"version={alloc.get('version')}, returning existing reservation"
                )
                break

        if existing_reservation:
            # Return existing reservation (idempotent)
            return existing_reservation.get("version")

        # Increment to get next version (never reuse, skip reserved/expired)
        next_version = max_version + 1
        while next_version in reserved_or_expired:
            next_version += 1
            logger.debug(
                f"[Reserve Version] Skipping version {next_version - 1} (already reserved/expired), "
                f"trying {next_version}"
            )

        logger.info(
            f"[Reserve Version] Reserving next version: {next_version} "
            f"(incremented from max_committed={max_version}, skipped {len(reserved_or_expired)} reserved/expired versions)"
        )

        # Add new reservation entry
        new_allocation = {
            "counter_key": counter_key,
            "version": next_version,
            "run_id": run_id,
            "status": "reserved",
            "reserved_at": datetime.now().isoformat(),
            "committed_at": None,
        }
        allocations.append(new_allocation)

        # Save atomically
        counter_data["allocations"] = allocations
        temp_path = counter_path.with_suffix('.tmp')
        try:
            save_json(temp_path, counter_data)

            # Atomic rename
            if sys.platform == 'win32':
                if counter_path.exists():
                    counter_path.unlink()
            temp_path.replace(counter_path)

            logger.info(
                f"[Reserve Version] ✓ Successfully reserved version {next_version} "
                f"for counter_key {counter_key[:50]}... "
                f"(run_id: {run_id[:12] if run_id != 'pending' else 'pending'}...)"
            )
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            logger.error(
                f"[Reserve Version] ✗ Failed to save reservation: {e}",
                exc_info=True
            )
            raise

        return next_version

    finally:
        release_lock(lock_fd, counter_path)


def commit_run_name_version(
    counter_key: str,
    run_id: str,
    version: int,
    root_dir: Path,
    config_dir: Optional[Path] = None,
) -> bool:
    """
    Commit a reserved version number after MLflow run is successfully created.

    Args:
        counter_key: Counter key (must match reservation).
        run_id: MLflow run ID (must match reservation or be "pending").
        version: Version number to commit (must match reservation).
        root_dir: Project root directory.
        config_dir: Optional config directory.

    Raises:
        RuntimeError: If lock acquisition fails after timeout.
    """
    counter_path = get_run_name_counter_path(root_dir, config_dir)

    logger.info(
        f"[Commit Version] Starting commit: counter_key={counter_key[:60]}..., "
        f"version={version}, run_id={run_id[:12]}..., counter_path={counter_path}"
    )

    # Acquire lock
    lock_fd = acquire_lock(counter_path, timeout=10.0)
    if lock_fd is None:
        logger.warning(
            f"[Commit Version] Could not acquire lock for {counter_path}, proceeding with non-atomic write"
        )

    try:
        # Load existing allocations
        counter_data = load_json(counter_path, default={"allocations": []})
        allocations: List[Dict[str, any]] = counter_data.get("allocations", [])
        logger.info(
            f"[Commit Version] Loaded {len(allocations)} existing allocations from {counter_path}"
        )

        # Find matching reservation entry
        found = False
        matching_reservations = []
        all_matching = []

        for alloc in allocations:
            if alloc.get("counter_key") == counter_key:
                all_matching.append(alloc)
                if alloc.get("version") == version:
                    matching_reservations.append(alloc)
                    if alloc.get("status") == "reserved":
                        # Update to committed
                        old_status = alloc.get("status")
                        alloc["status"] = "committed"
                        alloc["committed_at"] = datetime.now().isoformat()
                        # Update run_id if it was "pending"
                        if alloc.get("run_id") == "pending" or run_id != "pending":
                            alloc["run_id"] = run_id
                        found = True
                        logger.info(
                            f"[Commit Version] ✓ Found and committed reservation: version={version}, "
                            f"status changed from '{old_status}' to 'committed', "
                            f"run_id={run_id[:12]}..., counter_key={counter_key[:50]}..."
                        )
                        break

        if not found:
            logger.warning(
                f"[Commit Version] ✗ Could not find reservation to commit: counter_key={counter_key[:50]}..., "
                f"version={version}, run_id={run_id[:12]}... "
            )
            if matching_reservations:
                logger.warning(
                    f"[Commit Version] Found {len(matching_reservations)} matching allocations with version {version}: "
                    f"{[(a.get('version'), a.get('status'), a.get('run_id', '')[:12]) for a in matching_reservations]}"
                )
            if all_matching:
                logger.warning(
                    f"[Commit Version] All allocations for counter_key: "
                    f"{[(a.get('version'), a.get('status'), a.get('run_id', '')[:12]) for a in all_matching]}"
                )
            # Don't fail - idempotent operation

        # Save atomically
        counter_data["allocations"] = allocations
        temp_path = counter_path.with_suffix('.tmp')
        try:
            save_json(temp_path, counter_data)

            # Atomic rename
            if sys.platform == 'win32':
                if counter_path.exists():
                    counter_path.unlink()
            temp_path.replace(counter_path)

            if found:
                logger.info(
                    f"[Commit Version] ✓ Successfully saved committed version {version} to {counter_path}"
                )
            return found
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            logger.error(
                f"[Commit Version] ✗ Failed to save commit: {e}",
                exc_info=True
            )
            raise

    finally:
        release_lock(lock_fd, counter_path)
    return False


def cleanup_stale_reservations(
    root_dir: Path,
    config_dir: Optional[Path] = None,
    stale_minutes: int = 30,
) -> int:
    """
    Clean up stale "reserved" entries (crashed processes that never committed).

    Marks entries older than stale_minutes as "expired" (or removes them).

    Args:
        root_dir: Project root directory.
        config_dir: Optional config directory.
        stale_minutes: Minutes after which a reservation is considered stale.

    Returns:
        Count of cleaned entries.
    """
    counter_path = get_run_name_counter_path(root_dir, config_dir)

    if not counter_path.exists():
        return 0

    # Acquire lock
    lock_fd = acquire_lock(counter_path, timeout=10.0)
    if lock_fd is None:
        logger.warning(
            f"Could not acquire lock for {counter_path}, skipping cleanup")
        return 0

    try:
        # Load existing allocations
        counter_data = load_json(counter_path, default={"allocations": []})
        allocations: List[Dict[str, any]] = counter_data.get("allocations", [])

        # Find stale reservations
        now = datetime.now()
        cleaned_count = 0
        updated_allocations = []

        for alloc in allocations:
            if alloc.get("status") == "reserved":
                reserved_at_str = alloc.get("reserved_at")
                if reserved_at_str:
                    try:
                        reserved_at = datetime.fromisoformat(reserved_at_str)
                        age_minutes = (
                            now - reserved_at).total_seconds() / 60.0

                        if age_minutes > stale_minutes:
                            # Mark as expired (or remove - we'll mark for now)
                            alloc["status"] = "expired"
                            alloc["expired_at"] = now.isoformat()
                            cleaned_count += 1
                            logger.debug(
                                f"Marked stale reservation as expired: counter_key={alloc.get('counter_key', '')[:50]}..., "
                                f"version={alloc.get('version')}, age={age_minutes:.1f} minutes"
                            )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid reserved_at timestamp in allocation: {e}")

            # Keep all allocations (including expired ones for audit trail)
            updated_allocations.append(alloc)

        if cleaned_count > 0:
            # Save atomically
            counter_data["allocations"] = updated_allocations
            temp_path = counter_path.with_suffix('.tmp')
            try:
                save_json(temp_path, counter_data)

                # Atomic rename
                if sys.platform == 'win32':
                    if counter_path.exists():
                        counter_path.unlink()
                temp_path.replace(counter_path)

                logger.info(f"Cleaned up {cleaned_count} stale reservations")
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                logger.warning(f"Failed to save cleaned allocations: {e}")

        return cleaned_count

    finally:
        release_lock(lock_fd, counter_path)
