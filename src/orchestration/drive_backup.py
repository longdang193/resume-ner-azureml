"""Google Drive backup/restore functionality for Colab environments.

This module provides a clean, testable API for backing up and restoring files
from Google Drive when running in Google Colab. It separates core backup logic
from Colab-specific mounting operations.
"""

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional


class BackupAction(str, Enum):
    """Action taken during backup/restore operation."""

    COPIED = "copied"
    SKIPPED = "skipped"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class BackupResult:
    """Structured result from backup/restore operations."""

    ok: bool
    action: BackupAction
    src: Path
    dst: Path
    reason: str = ""
    error: Optional[Exception] = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "✓" if self.ok else "✗"
        return (
            f"{status} [{self.action.value.upper()}] {self.src.name} -> "
            f"{self.dst.name} ({self.reason})"
        )


@dataclass
class EnsureLocalOptions:
    """Options for ensure_local() method."""

    if_missing: bool = True
    if_stale: bool = False
    prefer: Literal["local", "drive", "newer"] = "local"


@dataclass
class DriveBackupStore:
    """
    Core backup/restore operations (environment-agnostic).

    Handles path mapping, file/directory copying, and validation.
    Does NOT handle Colab-specific mounting - that's separate.
    """

    root_dir: Path  # Project root (e.g., /content/resume-ner-azureml)
    backup_root: Path  # Drive backup base (e.g., /content/drive/MyDrive/resume-ner-checkpoints)
    only_outputs: bool = True  # Enforce outputs/ restriction
    dry_run: bool = False  # For testing

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.root_dir.exists():
            raise ValueError(f"root_dir does not exist: {self.root_dir}")

    def drive_path_for(self, local_path: Path) -> Path:
        """
        Convert local path to Drive backup path (mirrors structure).

        Args:
            local_path: Path within root_dir (must be under outputs/ if only_outputs=True)

        Returns:
            Equivalent path in backup_root

        Raises:
            ValueError: If path is outside allowed scope
        """
        # Validate path is within root_dir
        try:
            relative = local_path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(
                f"Path {local_path} is not under root_dir {self.root_dir}"
            )

        # Enforce outputs/ restriction if enabled
        if self.only_outputs and "outputs" not in relative.parts:
            raise ValueError(
                f"Drive backup only supports paths under outputs/. Got: {relative}"
            )

        return self.backup_root / relative

    def backup(
        self, local_path: Path, expect: Literal["file", "dir", "any"] = "any"
    ) -> BackupResult:
        """
        Backup file/directory to Drive.

        Args:
            local_path: Local path to backup (must exist)
            expect: Expected type ("file", "dir", or "any" to infer)

        Returns:
            BackupResult with operation details
        """
        # Validate local_path exists
        if not local_path.exists():
            return BackupResult(
                ok=False,
                action=BackupAction.MISSING,
                src=local_path,
                dst=self.drive_path_for(local_path),
                reason="Local path does not exist",
            )

        # Infer type if needed
        if expect == "any":
            expect = "dir" if local_path.is_dir() else "file"

        # Validate type matches
        is_dir = local_path.is_dir()
        if expect == "file" and is_dir:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=local_path,
                dst=self.drive_path_for(local_path),
                reason="Expected file but got directory",
            )
        elif expect == "dir" and not is_dir:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=local_path,
                dst=self.drive_path_for(local_path),
                reason="Expected directory but got file",
            )

        # Get destination path
        drive_path = self.drive_path_for(local_path)

        if self.dry_run:
            return BackupResult(
                ok=True,
                action=BackupAction.SKIPPED,
                src=local_path,
                dst=drive_path,
                reason="Dry run - would backup",
            )

        # Perform backup
        try:
            drive_path.parent.mkdir(parents=True, exist_ok=True)

            if is_dir:
                if drive_path.exists():
                    shutil.rmtree(drive_path)
                shutil.copytree(local_path, drive_path)
            else:
                shutil.copy2(local_path, drive_path)

            return BackupResult(
                ok=True,
                action=BackupAction.COPIED,
                src=local_path,
                dst=drive_path,
                reason="Backed up successfully",
            )
        except Exception as e:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=local_path,
                dst=drive_path,
                reason=f"Backup failed: {e}",
                error=e,
            )

    def restore(
        self, local_path: Path, expect: Literal["file", "dir", "any"] = "any"
    ) -> BackupResult:
        """
        Restore file/directory from Drive.

        Args:
            local_path: Local path where file should be restored
            expect: Expected type ("file", "dir", or "any" to infer from source)

        Returns:
            BackupResult with operation details
        """
        drive_path = self.drive_path_for(local_path)

        # Check if backup exists
        if not drive_path.exists():
            return BackupResult(
                ok=False,
                action=BackupAction.MISSING,
                src=local_path,
                dst=drive_path,
                reason="Drive backup does not exist",
            )

        # Infer type from source if needed
        if expect == "any":
            expect = "dir" if drive_path.is_dir() else "file"

        # Validate type
        is_dir = drive_path.is_dir()
        if expect == "file" and is_dir:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=local_path,
                dst=drive_path,
                reason="Expected file but backup is directory",
            )
        elif expect == "dir" and not is_dir:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=local_path,
                dst=drive_path,
                reason="Expected directory but backup is file",
            )

        if self.dry_run:
            return BackupResult(
                ok=True,
                action=BackupAction.SKIPPED,
                src=local_path,
                dst=drive_path,
                reason="Dry run - would restore",
            )

        # Perform restore
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if is_dir:
                if local_path.exists():
                    shutil.rmtree(local_path)
                shutil.copytree(drive_path, local_path)
            else:
                shutil.copy2(drive_path, local_path)

            return BackupResult(
                ok=True,
                action=BackupAction.COPIED,
                src=drive_path,
                dst=local_path,
                reason="Restored successfully",
            )
        except Exception as e:
            return BackupResult(
                ok=False,
                action=BackupAction.ERROR,
                src=drive_path,
                dst=local_path,
                reason=f"Restore failed: {e}",
                error=e,
            )

    def ensure_local(
        self, local_path: Path, options: Optional[EnsureLocalOptions] = None
    ) -> BackupResult:
        """
        Ensure file exists locally, restoring from Drive if needed.

        Primary entry point for most use cases.

        Args:
            local_path: Path to ensure exists locally
            options: Options for restore behavior

        Returns:
            BackupResult with operation details
        """
        options = options or EnsureLocalOptions()

        # If local exists and we're not checking staleness, skip
        if local_path.exists() and not options.if_stale:
            return BackupResult(
                ok=True,
                action=BackupAction.SKIPPED,
                src=local_path,
                dst=self.drive_path_for(local_path),
                reason="Local file exists",
            )

        # Local missing or stale - attempt restore
        return self.restore(local_path)

    def backup_exists(self, local_path: Path) -> bool:
        """
        Check if backup exists in Drive.

        Args:
            local_path: Local path to check

        Returns:
            True if backup exists, False otherwise
        """
        drive_path = self.drive_path_for(local_path)
        return drive_path.exists()

    def as_restore_callback(self) -> Callable[[Path], bool]:
        """
        Create callback function for backward compatibility.

        Returns:
            Function that takes Path and returns bool (True if restored)
        """

        def callback(local_path: Path) -> bool:
            result = self.restore(local_path)
            return result.ok

        return callback

    def as_backup_callback(self) -> Callable[[Path], bool]:
        """
        Create backup callback for backward compatibility.

        Returns:
            Function that takes Path and returns bool (True if backed up)
        """

        def callback(local_path: Path) -> bool:
            result = self.backup(local_path)
            return result.ok

        return callback


def mount_colab_drive(mount_point: str = "/content/drive") -> Path:
    """
    Mount Google Drive in Colab environment.

    Args:
        mount_point: Mount point path

    Returns:
        Path to mounted Drive (MyDrive)

    Raises:
        ImportError: If not in Colab environment
        RuntimeError: If mount fails
    """
    try:
        from google.colab import drive
    except ImportError:
        raise ImportError("google.colab.drive not available (not in Colab)")

    drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def create_colab_store(
    root_dir: Path,
    config_dir: Path,
    mount_point: str = "/content/drive",
) -> Optional[DriveBackupStore]:
    """
    Factory function to create DriveBackupStore for Colab environment.

    Handles mounting and configuration reading.

    Args:
        root_dir: Project root directory
        config_dir: Config directory (for paths.yaml)
        mount_point: Drive mount point

    Returns:
        DriveBackupStore if configured, None if backup disabled
    """
    # Mount Drive
    try:
        drive_root = mount_colab_drive(mount_point)
    except (ImportError, RuntimeError) as e:
        print(f"⚠ Warning: Could not mount Drive: {e}")
        return None

    # Get backup base from config
    from orchestration.paths import get_drive_backup_base

    backup_base = get_drive_backup_base(config_dir)

    if not backup_base:
        # Fallback to default (use project name to preserve structure)
        # Try to infer project name from root_dir, default to "resume-ner-azureml"
        project_name = root_dir.name if root_dir.name else "resume-ner-azureml"
        backup_base = drive_root / project_name
        backup_base.mkdir(parents=True, exist_ok=True)
        print(f"Using default backup location: {backup_base}")
    else:
        backup_base.mkdir(parents=True, exist_ok=True)
        print(f"Using configured backup location: {backup_base}")

    return DriveBackupStore(root_dir=root_dir, backup_root=backup_base)


