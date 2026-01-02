"""Cross-platform file locking utilities."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

# fcntl is Unix-only, handle import gracefully for Windows compatibility
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def acquire_lock(file_path: Path, timeout: float = 10.0) -> Optional[object]:
    """
    Acquire file lock for atomic writes (Unix/Linux).

    Args:
        file_path: Path to file to lock.
        timeout: Maximum time to wait for lock (seconds).

    Returns:
        File handle with lock, or None if lock failed.
    """
    if not HAS_FCNTL:
        # Windows or platform without fcntl - return None (fallback to non-atomic)
        return None

    try:
        lock_file = file_path.with_suffix('.lock')
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        lock_fd = os.open(str(lock_file), os.O_CREAT |
                          os.O_WRONLY | os.O_TRUNC)

        # Try to acquire exclusive lock (non-blocking)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except BlockingIOError:
            # Lock is held, wait with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_fd
                except BlockingIOError:
                    time.sleep(0.1)
            os.close(lock_fd)
            return None
    except (OSError, AttributeError):
        # Windows or lock not available, return None (fallback to non-atomic)
        return None


def release_lock(lock_fd: Optional[object], file_path: Path) -> None:
    """
    Release file lock.

    Args:
        lock_fd: File handle with lock (from acquire_lock).
        file_path: Path to file that was locked.
    """
    if not HAS_FCNTL or lock_fd is None:
        return

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        # Remove lock file
        lock_file = file_path.with_suffix('.lock')
        if lock_file.exists():
            try:
                lock_file.unlink()
            except OSError:
                pass
    except (OSError, AttributeError):
        pass

