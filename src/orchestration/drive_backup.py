"""Legacy facade for drive_backup module.

This module provides backward compatibility by re-exporting from storage.drive.
All imports from this module are deprecated.
"""

import warnings
from pathlib import Path
from storage.drive import (
    BackupAction,
    BackupResult,
    EnsureLocalOptions,
    DriveBackupStore,
    mount_colab_drive,
    create_colab_store,
            )
# Also re-export get_drive_backup_base from paths for backward compatibility
    from paths import get_drive_backup_base

__all__ = [
    "Path",
    "BackupAction",
    "BackupResult",
    "EnsureLocalOptions",
    "DriveBackupStore",
    "mount_colab_drive",
    "create_colab_store",
    "get_drive_backup_base",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'drive_backup' from 'orchestration' is deprecated. "
    "Please import from 'storage.drive' (and 'paths' for get_drive_backup_base) instead.",
    DeprecationWarning,
    stacklevel=2
)
