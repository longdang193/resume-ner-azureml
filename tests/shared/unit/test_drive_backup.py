"""Unit tests for Google Drive backup module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storage.drive import (
    BackupAction,
    BackupResult,
    DriveBackupStore,
    EnsureLocalOptions,
    create_colab_store,
    mount_colab_drive,
)


class TestBackupResult:
    """Test BackupResult dataclass."""

    def test_backup_result_str_success(self):
        """Test string representation for successful backup."""
        result = BackupResult(
            ok=True,
            action=BackupAction.COPIED,
            src=Path("test/file.txt"),
            dst=Path("backup/file.txt"),
            reason="Backed up successfully",
        )
        assert "✓" in str(result)
        assert "COPIED" in str(result)
        assert "Backed up successfully" in str(result)

    def test_backup_result_str_error(self):
        """Test string representation for error."""
        result = BackupResult(
            ok=False,
            action=BackupAction.ERROR,
            src=Path("test/file.txt"),
            dst=Path("backup/file.txt"),
            reason="Backup failed: Permission denied",
        )
        assert "✗" in str(result)
        assert "ERROR" in str(result)


class TestDriveBackupStore:
    """Test DriveBackupStore class."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        root_dir = tmp_path / "project"
        backup_root = tmp_path / "backup"
        root_dir.mkdir()
        backup_root.mkdir()
        return root_dir, backup_root

    @pytest.fixture
    def store(self, temp_dirs):
        """Create DriveBackupStore instance for testing."""
        root_dir, backup_root = temp_dirs
        return DriveBackupStore(root_dir=root_dir, backup_root=backup_root)

    def test_init_validation(self, tmp_path):
        """Test that init validates root_dir exists."""
        backup_root = tmp_path / "backup"
        backup_root.mkdir()
        
        with pytest.raises(ValueError, match="root_dir does not exist"):
            DriveBackupStore(root_dir=tmp_path / "nonexistent", backup_root=backup_root)

    def test_drive_path_for_valid_path(self, store, temp_dirs):
        """Test path mapping for valid paths."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "hpo" / "local" / "distilbert"
        outputs_dir.mkdir(parents=True)
        
        local_path = outputs_dir / "trial_0" / "checkpoint"
        drive_path = store.drive_path_for(local_path)
        
        expected = backup_root / "outputs" / "hpo" / "local" / "distilbert" / "trial_0" / "checkpoint"
        assert drive_path == expected

    def test_drive_path_for_outside_root(self, store, temp_dirs):
        """Test that path outside root_dir raises ValueError."""
        root_dir, _ = temp_dirs
        outside_path = root_dir.parent / "outside" / "file.txt"
        
        with pytest.raises(ValueError, match="not under root_dir"):
            store.drive_path_for(outside_path)

    def test_drive_path_for_outside_outputs(self, store, temp_dirs):
        """Test that path outside outputs/ raises ValueError when only_outputs=True."""
        root_dir, _ = temp_dirs
        (root_dir / "config").mkdir()
        config_file = root_dir / "config" / "test.yaml"
        
        with pytest.raises(ValueError, match="only supports paths under outputs"):
            store.drive_path_for(config_file)

    def test_drive_path_for_allows_outputs_when_disabled(self, temp_dirs):
        """Test that only_outputs=False allows paths outside outputs/."""
        root_dir, backup_root = temp_dirs
        store = DriveBackupStore(root_dir=root_dir, backup_root=backup_root, only_outputs=False)
        
        config_file = root_dir / "config" / "test.yaml"
        config_file.parent.mkdir()
        drive_path = store.drive_path_for(config_file)
        
        expected = backup_root / "config" / "test.yaml"
        assert drive_path == expected

    def test_backup_file(self, store, temp_dirs):
        """Test backing up a file."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        outputs_dir.mkdir(parents=True)
        
        test_file = outputs_dir / "test.txt"
        test_file.write_text("test content")
        
        result = store.backup(test_file)
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED
        assert (backup_root / "outputs" / "test" / "test.txt").exists()
        assert (backup_root / "outputs" / "test" / "test.txt").read_text() == "test content"

    def test_backup_directory(self, store, temp_dirs):
        """Test backing up a directory."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        test_dir = outputs_dir / "checkpoint"
        test_dir.mkdir(parents=True)
        
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        result = store.backup(test_dir)
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED
        backup_dir = backup_root / "outputs" / "test" / "checkpoint"
        assert backup_dir.exists()
        assert (backup_dir / "file1.txt").read_text() == "content1"
        assert (backup_dir / "file2.txt").read_text() == "content2"

    def test_backup_nonexistent_file(self, store, temp_dirs):
        """Test backing up a non-existent file."""
        root_dir, _ = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        outputs_dir.mkdir(parents=True)
        
        nonexistent = outputs_dir / "nonexistent.txt"
        result = store.backup(nonexistent)
        
        assert result.ok is False
        assert result.action == BackupAction.MISSING
        assert "does not exist" in result.reason

    def test_backup_type_mismatch(self, store, temp_dirs):
        """Test backup with type mismatch."""
        root_dir, _ = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        test_file = outputs_dir / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")
        
        result = store.backup(test_file, expect="dir")
        
        assert result.ok is False
        assert result.action == BackupAction.ERROR
        assert "Expected directory but got file" in result.reason

    def test_backup_type_inference(self, store, temp_dirs):
        """Test that backup infers type when expect='any'."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        test_file = outputs_dir / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")
        
        result = store.backup(test_file, expect="any")
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED

    def test_backup_dry_run(self, store, temp_dirs):
        """Test backup in dry_run mode."""
        root_dir, backup_root = temp_dirs
        store.dry_run = True
        outputs_dir = root_dir / "outputs" / "test"
        test_file = outputs_dir / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")
        
        result = store.backup(test_file)
        
        assert result.ok is True
        assert result.action == BackupAction.SKIPPED
        assert "Dry run" in result.reason
        assert not (backup_root / "outputs" / "test" / "test.txt").exists()

    def test_restore_file(self, store, temp_dirs):
        """Test restoring a file."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        backup_outputs = backup_root / "outputs" / "test"
        backup_outputs.mkdir(parents=True)
        
        backup_file = backup_outputs / "test.txt"
        backup_file.write_text("restored content")
        
        local_file = outputs_dir / "test.txt"
        result = store.restore(local_file)
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED
        assert local_file.exists()
        assert local_file.read_text() == "restored content"

    def test_restore_directory(self, store, temp_dirs):
        """Test restoring a directory."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        backup_outputs = backup_root / "outputs" / "test"
        backup_dir = backup_outputs / "checkpoint"
        backup_dir.mkdir(parents=True)
        
        (backup_dir / "file1.txt").write_text("content1")
        (backup_dir / "file2.txt").write_text("content2")
        
        local_dir = outputs_dir / "checkpoint"
        result = store.restore(local_dir)
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED
        assert local_dir.exists()
        assert (local_dir / "file1.txt").read_text() == "content1"
        assert (local_dir / "file2.txt").read_text() == "content2"

    def test_restore_nonexistent_backup(self, store, temp_dirs):
        """Test restoring when backup doesn't exist."""
        root_dir, _ = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        outputs_dir.mkdir(parents=True)
        
        local_file = outputs_dir / "nonexistent.txt"
        result = store.restore(local_file)
        
        assert result.ok is False
        assert result.action == BackupAction.MISSING
        assert "does not exist" in result.reason

    def test_restore_overwrites_existing(self, store, temp_dirs):
        """Test that restore overwrites existing local file."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        outputs_dir.mkdir(parents=True)
        backup_outputs = backup_root / "outputs" / "test"
        backup_outputs.mkdir(parents=True)
        
        # Create existing local file
        local_file = outputs_dir / "test.txt"
        local_file.write_text("old content")
        
        # Create backup file
        backup_file = backup_outputs / "test.txt"
        backup_file.write_text("new content")
        
        result = store.restore(local_file)
        
        assert result.ok is True
        assert local_file.read_text() == "new content"

    def test_ensure_local_exists(self, store, temp_dirs):
        """Test ensure_local when file already exists."""
        root_dir, _ = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        test_file = outputs_dir / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("existing content")
        
        result = store.ensure_local(test_file)
        
        assert result.ok is True
        assert result.action == BackupAction.SKIPPED
        assert "exists" in result.reason.lower()

    def test_ensure_local_restores_if_missing(self, store, temp_dirs):
        """Test ensure_local restores when file is missing."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        backup_outputs = backup_root / "outputs" / "test"
        backup_outputs.mkdir(parents=True)
        
        backup_file = backup_outputs / "test.txt"
        backup_file.write_text("restored content")
        
        local_file = outputs_dir / "test.txt"
        result = store.ensure_local(local_file)
        
        assert result.ok is True
        assert result.action == BackupAction.COPIED
        assert local_file.exists()

    def test_backup_exists_true(self, store, temp_dirs):
        """Test backup_exists when backup exists."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        backup_outputs = backup_root / "outputs" / "test"
        backup_outputs.mkdir(parents=True)
        
        backup_file = backup_outputs / "test.txt"
        backup_file.write_text("content")
        
        local_file = outputs_dir / "test.txt"
        assert store.backup_exists(local_file) is True

    def test_backup_exists_false(self, store, temp_dirs):
        """Test backup_exists when backup doesn't exist."""
        root_dir, _ = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        local_file = outputs_dir / "test.txt"
        assert store.backup_exists(local_file) is False

    def test_as_restore_callback(self, store, temp_dirs):
        """Test restore callback adapter."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        backup_outputs = backup_root / "outputs" / "test"
        backup_outputs.mkdir(parents=True)
        
        backup_file = backup_outputs / "test.txt"
        backup_file.write_text("content")
        
        callback = store.as_restore_callback()
        local_file = outputs_dir / "test.txt"
        
        assert callback(local_file) is True
        assert local_file.exists()

    def test_as_backup_callback(self, store, temp_dirs):
        """Test backup callback adapter."""
        root_dir, backup_root = temp_dirs
        outputs_dir = root_dir / "outputs" / "test"
        test_file = outputs_dir / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")
        
        callback = store.as_backup_callback()
        
        assert callback(test_file) is True
        assert (backup_root / "outputs" / "test" / "test.txt").exists()


class TestEnsureLocalOptions:
    """Test EnsureLocalOptions dataclass."""

    def test_default_options(self):
        """Test default option values."""
        options = EnsureLocalOptions()
        assert options.if_missing is True
        assert options.if_stale is False
        assert options.prefer == "local"


class TestColabMounting:
    """Test Colab-specific mounting functions."""

    @patch("storage.drive.Path")
    def test_mount_colab_drive_success(self, mock_path):
        """Test successful Drive mounting."""
        mock_drive = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__ = MagicMock(return_value=Path("/content/drive/MyDrive"))
        mock_path.return_value = mock_path_instance
        
        with patch.dict("sys.modules", {"google.colab": MagicMock(drive=mock_drive)}):
            result = mount_colab_drive("/content/drive")
            
            mock_drive.mount.assert_called_once_with("/content/drive")
            mock_path.assert_called_once_with("/content/drive")

    def test_mount_colab_drive_import_error(self):
        """Test mount_colab_drive raises ImportError when not in Colab."""
        with pytest.raises(ImportError, match="google.colab.drive not available"):
            mount_colab_drive()

    @patch("storage.drive.mount_colab_drive")
    @patch("paths.get_drive_backup_base")
    def test_create_colab_store_success(self, mock_get_base, mock_mount, tmp_path):
        """Test creating store for Colab environment."""
        root_dir = tmp_path / "project"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        mock_drive_root = tmp_path / "drive" / "MyDrive"
        mock_mount.return_value = mock_drive_root
        mock_get_base.return_value = mock_drive_root / "resume-ner-checkpoints"
        
        store = create_colab_store(root_dir, config_dir)
        
        assert store is not None
        assert store.root_dir == root_dir
        assert store.backup_root == mock_drive_root / "resume-ner-checkpoints"

    @patch("storage.drive.mount_colab_drive")
    def test_create_colab_store_mount_fails(self, mock_mount, tmp_path):
        """Test create_colab_store returns None when mount fails."""
        root_dir = tmp_path / "project"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        mock_mount.side_effect = ImportError("Not in Colab")
        
        store = create_colab_store(root_dir, config_dir)
        
        assert store is None

    @patch("storage.drive.mount_colab_drive")
    @patch("paths.get_drive_backup_base")
    def test_create_colab_store_default_path(self, mock_get_base, mock_mount, tmp_path):
        """Test create_colab_store uses default path when config not found."""
        root_dir = tmp_path / "project"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        mock_drive_root = tmp_path / "drive" / "MyDrive"
        mock_mount.return_value = mock_drive_root
        mock_get_base.return_value = None  # Config not found
        
        store = create_colab_store(root_dir, config_dir)
        
        assert store is not None
        # When config not found, uses root_dir.name as project name
        assert store.backup_root == mock_drive_root / "project"








