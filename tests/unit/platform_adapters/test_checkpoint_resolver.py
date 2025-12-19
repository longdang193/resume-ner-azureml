"""Tests for checkpoint path resolution functionality."""

from pathlib import Path
from unittest.mock import patch
import pytest

from platform_adapters.checkpoint_resolver import (
    AzureMLCheckpointResolver,
    LocalCheckpointResolver,
)


class TestAzureMLCheckpointResolver:
    """Tests for AzureMLCheckpointResolver class."""

    def test_resolve_checkpoint_dir_direct_path(self, temp_dir):
        """Test resolving checkpoint directory when path is direct checkpoint."""
        resolver = AzureMLCheckpointResolver()
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "pytorch_model.bin").write_text("dummy")
        
        result = resolver.resolve_checkpoint_dir(str(checkpoint_dir))
        
        assert result == checkpoint_dir

    def test_resolve_checkpoint_dir_nested_path(self, temp_dir):
        """Test resolving checkpoint directory when checkpoint is nested."""
        resolver = AzureMLCheckpointResolver()
        
        root = temp_dir / "outputs"
        root.mkdir()
        checkpoint_dir = root / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.safetensors").write_text("dummy")
        
        result = resolver.resolve_checkpoint_dir(str(root))
        
        assert result == checkpoint_dir

    def test_resolve_checkpoint_dir_model_pt(self, temp_dir):
        """Test resolving checkpoint directory with model.pt file."""
        resolver = AzureMLCheckpointResolver()
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.pt").write_text("dummy")
        
        result = resolver.resolve_checkpoint_dir(str(checkpoint_dir))
        
        assert result == checkpoint_dir

    def test_resolve_checkpoint_dir_path_not_found(self):
        """Test resolving checkpoint directory when path doesn't exist."""
        resolver = AzureMLCheckpointResolver()
        
        with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
            resolver.resolve_checkpoint_dir("/nonexistent/path")

    def test_resolve_checkpoint_dir_no_valid_checkpoint(self, temp_dir):
        """Test resolving checkpoint directory when no valid checkpoint found."""
        resolver = AzureMLCheckpointResolver()
        
        root = temp_dir / "outputs"
        root.mkdir()
        (root / "some_file.txt").write_text("dummy")
        
        with pytest.raises(FileNotFoundError, match="Could not locate a Hugging Face checkpoint"):
            resolver.resolve_checkpoint_dir(str(root))

    def test_resolve_checkpoint_dir_missing_config(self, temp_dir):
        """Test resolving checkpoint directory when config.json is missing."""
        resolver = AzureMLCheckpointResolver()
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_model.bin").write_text("dummy")
        
        with pytest.raises(FileNotFoundError):
            resolver.resolve_checkpoint_dir(str(checkpoint_dir))

    def test_resolve_checkpoint_dir_missing_model_file(self, temp_dir):
        """Test resolving checkpoint directory when model file is missing."""
        resolver = AzureMLCheckpointResolver()
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        
        with pytest.raises(FileNotFoundError):
            resolver.resolve_checkpoint_dir(str(checkpoint_dir))

    def test_list_files_success(self, temp_dir):
        """Test _list_files method lists files correctly."""
        resolver = AzureMLCheckpointResolver()
        
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("content2")
        
        files = resolver._list_files(temp_dir)
        
        assert len(files) >= 2
        assert "file1.txt" in files
        # Normalize path separators for cross-platform compatibility
        assert any("subdir" in f and "file2.txt" in f for f in files)

    def test_list_files_with_limit(self, temp_dir):
        """Test _list_files method respects limit."""
        resolver = AzureMLCheckpointResolver()
        
        for i in range(50):
            (temp_dir / f"file{i}.txt").write_text("content")
        
        files = resolver._list_files(temp_dir, limit=10)
        
        assert len(files) == 10

    def test_list_files_nonexistent_path(self):
        """Test _list_files method with nonexistent path."""
        resolver = AzureMLCheckpointResolver()
        
        files = resolver._list_files(Path("/nonexistent"))
        
        assert files == []

    def test_list_files_traversal_error(self, temp_dir):
        """Test _list_files method handles traversal errors gracefully."""
        resolver = AzureMLCheckpointResolver()
        
        with patch("platform_adapters.checkpoint_resolver.Path.rglob") as mock_rglob:
            mock_rglob.side_effect = PermissionError("Access denied")
            
            files = resolver._list_files(temp_dir)
            
            assert files == []


class TestLocalCheckpointResolver:
    """Tests for LocalCheckpointResolver class."""

    def test_resolve_checkpoint_dir_direct_path(self, temp_dir):
        """Test resolving checkpoint directory when path is direct checkpoint."""
        resolver = LocalCheckpointResolver()
        
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "pytorch_model.bin").write_text("dummy")
        
        result = resolver.resolve_checkpoint_dir(str(checkpoint_dir))
        
        assert result == checkpoint_dir

    def test_resolve_checkpoint_dir_nested_path(self, temp_dir):
        """Test resolving checkpoint directory when checkpoint is nested."""
        resolver = LocalCheckpointResolver()
        
        root = temp_dir / "outputs"
        root.mkdir()
        checkpoint_dir = root / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.safetensors").write_text("dummy")
        
        result = resolver.resolve_checkpoint_dir(str(root))
        
        assert result == checkpoint_dir

    def test_resolve_checkpoint_dir_path_not_found(self):
        """Test resolving checkpoint directory when path doesn't exist."""
        resolver = LocalCheckpointResolver()
        
        with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
            resolver.resolve_checkpoint_dir("/nonexistent/path")

    def test_resolve_checkpoint_dir_no_valid_checkpoint(self, temp_dir):
        """Test resolving checkpoint directory when no valid checkpoint found."""
        resolver = LocalCheckpointResolver()
        
        root = temp_dir / "outputs"
        root.mkdir()
        (root / "some_file.txt").write_text("dummy")
        
        with pytest.raises(FileNotFoundError, match="Could not locate a Hugging Face checkpoint"):
            resolver.resolve_checkpoint_dir(str(root))

    def test_list_files_success(self, temp_dir):
        """Test _list_files method lists files correctly."""
        resolver = LocalCheckpointResolver()
        
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("content2")
        
        files = resolver._list_files(temp_dir)
        
        assert len(files) >= 2
        assert "file1.txt" in files
        # Normalize path separators for cross-platform compatibility
        assert any("subdir" in f and "file2.txt" in f for f in files)

    def test_list_files_with_limit(self, temp_dir):
        """Test _list_files method respects limit."""
        resolver = LocalCheckpointResolver()
        
        for i in range(50):
            (temp_dir / f"file{i}.txt").write_text("content")
        
        files = resolver._list_files(temp_dir, limit=10)
        
        assert len(files) == 10

    def test_list_files_nonexistent_path(self):
        """Test _list_files method with nonexistent path."""
        resolver = LocalCheckpointResolver()
        
        files = resolver._list_files(Path("/nonexistent"))
        
        assert files == []

    def test_list_files_traversal_error(self, temp_dir):
        """Test _list_files method handles traversal errors gracefully."""
        resolver = LocalCheckpointResolver()
        
        with patch("platform_adapters.checkpoint_resolver.Path.rglob") as mock_rglob:
            mock_rglob.side_effect = PermissionError("Access denied")
            
            files = resolver._list_files(temp_dir)
            
            assert files == []

