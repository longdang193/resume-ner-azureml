"""Tests for data asset registration and path resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from azure.core.exceptions import ResourceNotFoundError
from orchestration.data_assets import (
    resolve_dataset_path,
    register_data_asset,
    ensure_data_asset_uploaded,
    build_data_asset_reference,
)


class TestResolveDatasetPath:
    """Tests for resolve_dataset_path function."""

    def test_resolve_with_local_path(self):
        """Test resolving path when local_path is provided."""
        data_config = {"local_path": "./data/dataset"}
        result = resolve_dataset_path(data_config)
        
        assert isinstance(result, Path)
        # Path normalizes "./data/dataset" to "data/dataset" on some platforms
        # Check that the path ends with "data/dataset" regardless of leading "./"
        path_str = str(result).replace("\\", "/")
        assert path_str.endswith("data/dataset")
        # Verify it's a relative path (doesn't start with absolute path indicators)
        assert not path_str.startswith("/") or path_str.startswith("./")

    def test_resolve_with_default(self):
        """Test resolving path with default when local_path is missing."""
        data_config = {}
        result = resolve_dataset_path(data_config)
        
        assert isinstance(result, Path)
        # Normalize path separators for cross-platform compatibility
        assert str(result).replace("\\", "/") == "../dataset"

    def test_resolve_with_absolute_path(self):
        """Test resolving absolute path."""
        data_config = {"local_path": "/absolute/path/to/dataset"}
        result = resolve_dataset_path(data_config)
        
        assert isinstance(result, Path)
        # Normalize path separators for cross-platform compatibility
        assert str(result).replace("\\", "/") == "/absolute/path/to/dataset"

    def test_resolve_raises_error_for_non_string(self):
        """Test that ValueError is raised when local_path is not a string."""
        data_config = {"local_path": 123}
        
        with pytest.raises(ValueError, match="must be a string"):
            resolve_dataset_path(data_config)

    def test_resolve_raises_error_for_list(self):
        """Test that ValueError is raised when local_path is a list."""
        data_config = {"local_path": ["path1", "path2"]}
        
        with pytest.raises(ValueError, match="must be a string"):
            resolve_dataset_path(data_config)


class TestRegisterDataAsset:
    """Tests for register_data_asset function."""

    def test_register_existing_asset(self):
        """Test returning existing asset when it already exists."""
        mock_ml_client = MagicMock()
        mock_existing_asset = MagicMock()
        mock_ml_client.data.get.return_value = mock_existing_asset
        
        result = register_data_asset(
            ml_client=mock_ml_client,
            name="test-asset",
            version="1",
            uri="/local/path",
            description="Test asset"
        )
        
        assert result == mock_existing_asset
        mock_ml_client.data.get.assert_called_once_with(name="test-asset", version="1")
        mock_ml_client.data.create_or_update.assert_not_called()

    def test_register_new_asset_local_path(self):
        """Test creating new asset with local path."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_new_asset = MagicMock()
        mock_ml_client.data.create_or_update.return_value = mock_new_asset
        
        with patch("orchestration.data_assets.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.return_value = Path("/absolute/path")
            mock_path.return_value = mock_path_instance
            
            result = register_data_asset(
                ml_client=mock_ml_client,
                name="test-asset",
                version="1",
                uri="/local/path",
                description="Test asset"
            )
            
            assert result == mock_new_asset
            mock_ml_client.data.create_or_update.assert_called_once()
            call_args = mock_ml_client.data.create_or_update.call_args[0][0]
            assert call_args.name == "test-asset"
            assert call_args.version == "1"
            assert call_args.description == "Test asset"

    def test_register_new_asset_datastore_uri(self):
        """Test creating new asset with datastore URI (no normalization)."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_new_asset = MagicMock()
        mock_ml_client.data.create_or_update.return_value = mock_new_asset
        
        result = register_data_asset(
            ml_client=mock_ml_client,
            name="test-asset",
            version="1",
            uri="azureml://datastores/workspaceblobstore/paths/dataset",
            description="Test asset"
        )
        
        assert result == mock_new_asset
        call_args = mock_ml_client.data.create_or_update.call_args[0][0]
        assert call_args.path == "azureml://datastores/workspaceblobstore/paths/dataset"

    def test_register_new_asset_http_uri(self):
        """Test creating new asset with HTTP URI (no normalization)."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_new_asset = MagicMock()
        mock_ml_client.data.create_or_update.return_value = mock_new_asset
        
        result = register_data_asset(
            ml_client=mock_ml_client,
            name="test-asset",
            version="1",
            uri="https://example.com/dataset",
            description="Test asset"
        )
        
        assert result == mock_new_asset
        call_args = mock_ml_client.data.create_or_update.call_args[0][0]
        assert call_args.path == "https://example.com/dataset"

    def test_register_new_asset_path_resolution_error(self):
        """Test handling path resolution errors gracefully."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_new_asset = MagicMock()
        mock_ml_client.data.create_or_update.return_value = mock_new_asset
        
        with patch("orchestration.data_assets.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.side_effect = OSError("Cannot resolve path")
            mock_path.return_value = mock_path_instance
            
            result = register_data_asset(
                ml_client=mock_ml_client,
                name="test-asset",
                version="1",
                uri="/invalid/path",
                description="Test asset"
            )
            
            assert result == mock_new_asset
            call_args = mock_ml_client.data.create_or_update.call_args[0][0]
            # On Windows, Path("/invalid/path") becomes "C:/invalid/path", so normalize
            path_str = str(call_args.path).replace("\\", "/")
            # Check that it uses the original path on error (may be normalized on Windows)
            assert path_str.endswith("/invalid/path") or path_str == "/invalid/path"


class TestEnsureDataAssetUploaded:
    """Tests for ensure_data_asset_uploaded function."""

    def test_ensure_existing_asset_datastore_path(self):
        """Test returning existing asset with datastore path."""
        mock_ml_client = MagicMock()
        mock_existing_asset = MagicMock()
        mock_existing_asset.name = "test-asset"
        mock_existing_asset.version = "1"
        mock_existing_asset.path = "azureml://datastores/workspaceblobstore/paths/dataset"
        mock_ml_client.data.get.return_value = mock_existing_asset
        
        local_path = Path("/local/path")
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        result = ensure_data_asset_uploaded(
            ml_client=mock_ml_client,
            data_asset=data_asset,
            local_path=local_path,
            description="Test asset"
        )
        
        assert result == mock_existing_asset
        mock_ml_client.data.create_or_update.assert_not_called()

    def test_ensure_existing_asset_local_path_match(self):
        """Test returning existing asset when local path matches."""
        mock_ml_client = MagicMock()
        mock_existing_asset = MagicMock()
        mock_existing_asset.name = "test-asset"
        mock_existing_asset.version = "1"
        mock_existing_asset.path = "/local/path"
        mock_ml_client.data.get.return_value = mock_existing_asset
        
        local_path = Path("/local/path")
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        with patch("orchestration.data_assets.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.return_value = Path("/local/path")
            mock_path.return_value = mock_path_instance
            
            result = ensure_data_asset_uploaded(
                ml_client=mock_ml_client,
                data_asset=data_asset,
                local_path=local_path,
                description="Test asset"
            )
            
            assert result == mock_existing_asset

    def test_ensure_new_asset_creation(self):
        """Test creating new asset when it doesn't exist."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_new_asset = MagicMock()
        mock_ml_client.data.create_or_update.return_value = mock_new_asset
        
        local_path = Path("/local/path")
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        result = ensure_data_asset_uploaded(
            ml_client=mock_ml_client,
            data_asset=data_asset,
            local_path=local_path,
            description="Test asset"
        )
        
        assert result == mock_new_asset
        mock_ml_client.data.create_or_update.assert_called_once()

    def test_ensure_asset_creation_race_condition(self):
        """Test handling race condition when asset is created between check and create."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = [
            ResourceNotFoundError("Asset not found"),
            MagicMock()  # Asset exists on second call
        ]
        mock_ml_client.data.create_or_update.side_effect = Exception("Asset already exists")
        mock_existing_asset = MagicMock()
        mock_ml_client.data.get.return_value = mock_existing_asset
        
        local_path = Path("/local/path")
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        # Reset get to return existing asset on second call
        def get_side_effect(name, version):
            if mock_ml_client.data.get.call_count == 1:
                raise ResourceNotFoundError("Asset not found")
            return mock_existing_asset
        
        mock_ml_client.data.get.side_effect = get_side_effect
        
        result = ensure_data_asset_uploaded(
            ml_client=mock_ml_client,
            data_asset=data_asset,
            local_path=local_path,
            description="Test asset"
        )
        
        assert result == mock_existing_asset

    def test_ensure_asset_creation_fallback_to_input(self):
        """Test fallback to input data_asset when all operations fail."""
        mock_ml_client = MagicMock()
        mock_ml_client.data.get.side_effect = ResourceNotFoundError("Asset not found")
        mock_ml_client.data.create_or_update.side_effect = Exception("Creation failed")
        
        local_path = Path("/local/path")
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        # Both get calls fail
        def get_side_effect(name, version):
            raise ResourceNotFoundError("Asset not found")
        
        mock_ml_client.data.get.side_effect = get_side_effect
        
        result = ensure_data_asset_uploaded(
            ml_client=mock_ml_client,
            data_asset=data_asset,
            local_path=local_path,
            description="Test asset"
        )
        
        assert result == data_asset

    def test_ensure_asset_path_normalization_string_comparison(self):
        """Test path matching using string normalization when Path.resolve fails."""
        mock_ml_client = MagicMock()
        mock_existing_asset = MagicMock()
        mock_existing_asset.name = "test-asset"
        mock_existing_asset.version = "1"
        mock_existing_asset.path = "C:\\local\\path"  # Windows path
        mock_ml_client.data.get.return_value = mock_existing_asset
        
        local_path = Path("C:/local/path")  # Unix-style path
        data_asset = MagicMock()
        data_asset.name = "test-asset"
        data_asset.version = "1"
        
        with patch("orchestration.data_assets.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.resolve.side_effect = ValueError("Invalid path")
            mock_path.return_value = mock_path_instance
            
            result = ensure_data_asset_uploaded(
                ml_client=mock_ml_client,
                data_asset=data_asset,
                local_path=local_path,
                description="Test asset"
            )
            
            assert result == mock_existing_asset


class TestBuildDataAssetReference:
    """Tests for build_data_asset_reference function."""

    def test_build_reference_with_datastore_path(self):
        """Test building reference when asset path contains /paths/."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "test-asset"
        mock_data_asset.version = "2"
        mock_data_asset.path = "azureml://datastores/workspaceblobstore/paths/dataset/train"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        
        assert result["asset_uri"] == "azureml:test-asset:2"
        assert result["datastore_path"] == "azureml://datastores/workspaceblobstore/paths/dataset/train"

    def test_build_reference_without_datastore_path(self):
        """Test building reference when asset path doesn't contain /paths/."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "test-asset"
        mock_data_asset.version = "2"
        mock_data_asset.path = "azureml://datastores/workspaceblobstore/dataset"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        
        assert result["asset_uri"] == "azureml:test-asset:2"
        assert result["datastore_path"] == "azureml://datastores/workspaceblobstore/dataset"

    def test_build_reference_with_trailing_slash(self):
        """Test building reference when path has trailing slash."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "test-asset"
        mock_data_asset.version = "2"
        mock_data_asset.path = "azureml://datastores/workspaceblobstore/paths/dataset/"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        
        assert result["asset_uri"] == "azureml:test-asset:2"
        assert result["datastore_path"] == "azureml://datastores/workspaceblobstore/paths/dataset"

    def test_build_reference_extracts_relative_path(self):
        """Test that relative path is correctly extracted from full datastore path."""
        mock_ml_client = MagicMock()
        mock_datastore = MagicMock()
        mock_datastore.name = "workspaceblobstore"
        mock_ml_client.datastores.get_default.return_value = mock_datastore
        
        mock_data_asset = MagicMock()
        mock_data_asset.name = "test-asset"
        mock_data_asset.version = "2"
        mock_data_asset.path = "azureml://datastores/otherstore/paths/relative/path"
        
        result = build_data_asset_reference(mock_ml_client, mock_data_asset)
        
        assert result["asset_uri"] == "azureml:test-asset:2"
        assert result["datastore_path"] == "azureml://datastores/workspaceblobstore/paths/relative/path"

